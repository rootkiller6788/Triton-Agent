"""LayerNorm Triton kernel template v1: baseline."""

import triton
import triton.language as tl


@triton.jit
def _layernorm_v1_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    EPS: tl.constexpr, BLOCK_SIZE: tl.constexpr, NUM_WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = B * T
    if pid >= total_rows:
        return

    row_start = pid * D
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < D

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / D
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / D

    rstd = tl.rsqrt(var + EPS)
    w = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    b = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    y = x_centered * rstd * w + b
    tl.store(y_ptr + offsets, y, mask=mask)


def layernorm_v1(
    x, weight, bias,
    B: int, T: int, D: int, EPS: float = 1e-5,
    BLOCK_SIZE: int = 256, NUM_WARPS: int = 4, NUM_STAGES: int = 3,
):
    N = B * T
    y = triton.empty_like(x)
    grid = lambda meta: (N,)
    _layernorm_v1_kernel[grid](
        x, weight, bias, y,
        B=B, T=T, D=D, EPS=EPS,
        BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return y
