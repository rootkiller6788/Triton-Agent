"""Fused Bias GELU Triton kernel template v1: baseline element-wise fusion."""

import triton
import triton.language as tl


@triton.jit
def _fused_bias_gelu_v1_kernel(
    x_ptr, bias_ptr, y_ptr,
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
    b = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    x_biased = x + b

    c = 0.7978845608028654
    x3 = x_biased * x_biased * x_biased
    inner = c * (x_biased + 0.044715 * x3)
    y = 0.5 * x_biased * (1.0 + tl.tanh(inner))

    tl.store(y_ptr + offsets, y, mask=mask)


def fused_bias_gelu_v1(
    x,
    bias,
    B: int,
    T: int,
    D: int,
    BLOCK_SIZE: int = 256,
    NUM_WARPS: int = 4,
    NUM_STAGES: int = 3,
):
    N = B * T
    y = triton.empty_like(x)
    grid = lambda meta: (N,)
    _fused_bias_gelu_v1_kernel[grid](
        x, bias, y,
        B=B, T=T, D=D,
        BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return y
