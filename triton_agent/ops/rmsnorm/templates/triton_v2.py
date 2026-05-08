"""RMSNorm Triton kernel template v2: vectorized row-wise with persistent scheduler."""

import triton
import triton.language as tl


@triton.jit
def _rmsnorm_v2_kernel(
    x_ptr, weight_ptr, y_ptr,
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, NUM_WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = B * T
    row_id = (pid * NUM_WARPS + tl.arange(0, NUM_WARPS))[:, None]
    col_offs = tl.arange(0, BLOCK_SIZE)[None, :]
    mask = col_offs < D

    x_ptrs = x_ptr + row_id * D + col_offs
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=1) / D
    rstd = tl.rsqrt(mean_sq + 1e-6)

    w_ptrs = weight_ptr + col_offs
    w = tl.load(w_ptrs, mask=mask, other=0.0)

    y = x * rstd[:, None] * w
    y_ptrs = y_ptr + row_id * D + col_offs
    tl.store(y_ptrs, y, mask=mask)


def rmsnorm_v2(
    x,
    weight,
    B: int,
    T: int,
    D: int,
    BLOCK_SIZE: int = 256,
    NUM_WARPS: int = 4,
    NUM_STAGES: int = 3,
):
    N = B * T
    y = triton.empty_like(x)
    grid = lambda meta: ((N + NUM_WARPS - 1) // NUM_WARPS,)
    _rmsnorm_v2_kernel[grid](
        x, weight, y,
        B=B, T=T, D=D,
        BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return y
