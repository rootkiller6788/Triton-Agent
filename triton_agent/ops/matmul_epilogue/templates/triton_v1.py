"""Matmul Epilogue Triton kernel v1: tiled matmul + GELU fusion."""

import triton
import triton.language as tl


@triton.jit
def _matmul_epilogue_v1_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    batch_idx = pid_b
    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    m_mask = m_offs < M
    n_mask = n_offs[None, :] < N
    k_mask = k_offs[None, :] < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + k_offs
        a_ptrs = a_ptr + batch_idx * (M * K) + m_offs[:, None] * K + k[None, :]
        b_ptrs = b_ptr + k[:, None] * N + n_offs[None, :]

        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask, other=0.0)
        acc += tl.dot(a, b)

    c = acc + tl.load(bias_ptr + n_offs, mask=n_offs[None, :] < N, other=0.0)

    c_fp = c.to(tl.float32)
    gelu = 0.5 * c_fp * (1.0 + tl.tanh(0.79788456 * (c_fp + 0.044715 * c_fp * c_fp * c_fp)))

    c_ptrs = c_ptr + batch_idx * (M * N) + m_offs[:, None] * N + n_offs[None, :]
    tl.store(c_ptrs, gelu, mask=m_mask[:, None] & n_mask)


def matmul_epilogue_v1(
    a, b, bias,
    B: int, M: int, N: int, K: int,
    BLOCK_M: int = 128, BLOCK_N: int = 128, BLOCK_K: int = 64,
    NUM_WARPS: int = 4, NUM_STAGES: int = 3,
):
    c = triton.empty((B, M, N), dtype=a.dtype, device=a.device)
    grid = (B, (M + BLOCK_M - 1) // BLOCK_M)
    _matmul_epilogue_v1_kernel[grid](
        a, b, bias, c,
        B=B, M=M, N=N, K=K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return c
