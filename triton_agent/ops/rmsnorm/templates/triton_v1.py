"""RMSNorm Triton kernel template v1: baseline block-wise implementation."""

import triton
import triton.language as tl


@triton.jit
def _rmsnorm_v1_kernel(
    x_ptr, weight_ptr, y_ptr, rstd_ptr,
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, NUM_WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = B * T
    if pid >= total_rows:
        return

    row_start = pid * D
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < D

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)

    rstd = tl.rsqrt(sum_sq / D + 1e-6)
    tl.store(rstd_ptr + pid, rstd)

    w = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = x * rstd * w
    tl.store(y_ptr + offsets, y, mask=mask)


def rmsnorm_v1(
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
    rstd = triton.empty((N,), dtype=x.dtype, device=x.device)
    grid = lambda meta: (N,)
    _rmsnorm_v1_kernel[grid](
        x, weight, y, rstd,
        B=B, T=T, D=D,
        BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return y
