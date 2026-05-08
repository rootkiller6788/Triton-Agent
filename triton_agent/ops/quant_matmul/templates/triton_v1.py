"""Quantized MatMul Triton kernel template v1: row-wise INT8 quantization."""

import triton
import triton.language as tl


@triton.jit
def _quant_matmul_v1_kernel(
    a_ptr, b_ptr, c_ptr,
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    m_mask = m_offs < M
    n_mask = n_offs < N
    k_mask = k_offs < K

    a_scale = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + k_offs
        a_ptrs = a_ptr + pid_b * (M * K) + m_offs[:, None] * K + k[None, :]
        b_ptrs = b_ptr + k[:, None] * N + n_offs[None, :]

        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_offs[None, :] < (K - k_start), other=0.0)
        b = tl.load(b_ptrs, mask=k_offs[:, None] < (K - k_start) & n_offs[None, :] < N, other=0.0)

        a_max = tl.max(tl.abs(a), axis=0)
        scale = a_max / 127.0
        a_q = tl.round(a / scale)
        a_q = tl.clamp(a_q, -127, 127)

        acc += tl.dot(a_q.to(tl.float32), b)

    c = acc
    c_ptrs = c_ptr + pid_b * (M * N) + m_offs[:, None] * N + n_offs[None, :]
    tl.store(c_ptrs, c, mask=m_mask[:, None] & n_mask[None, :])


def quant_matmul_v1(
    a, b,
    B: int, M: int, N: int, K: int,
    BLOCK_M: int = 128, BLOCK_N: int = 128, BLOCK_K: int = 64,
    NUM_WARPS: int = 4, NUM_STAGES: int = 3,
):
    c = triton.empty((B, M, N), dtype=a.dtype, device=a.device)
    grid = (B, (M + BLOCK_M - 1) // BLOCK_M)
    _quant_matmul_v1_kernel[grid](
        a, b, c,
        B=B, M=M, N=N, K=K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return c
