"""Fused Bias GELU Triton kernel template v2: tiled with multi-WARP parallelism."""

import triton
import triton.language as tl


@triton.jit
def _fused_bias_gelu_v2_kernel(
    x_ptr, bias_ptr, y_ptr,
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

    b_ptrs = bias_ptr + col_offs
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    x_biased = x + b

    c = 0.7978845608028654
    x3 = x_biased * x_biased * x_biased
    inner = c * (x_biased + 0.044715 * x3)
    y = 0.5 * x_biased * (1.0 + tl.tanh(inner))

    y_ptrs = y_ptr + row_id * D + col_offs
    tl.store(y_ptrs, y, mask=mask)


def fused_bias_gelu_v2(
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
    grid = lambda meta: ((N + NUM_WARPS - 1) // NUM_WARPS,)
    _fused_bias_gelu_v2_kernel[grid](
        x, bias, y,
        B=B, T=T, D=D,
        BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return y
