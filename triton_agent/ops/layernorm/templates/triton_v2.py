"""LayerNorm Triton kernel template v2: tiled multi-WARP."""

import triton
import triton.language as tl


@triton.jit
def _layernorm_v2_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    EPS: tl.constexpr, BLOCK_SIZE: tl.constexpr, NUM_WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = B * T
    row_id = (pid * NUM_WARPS + tl.arange(0, NUM_WARPS))[:, None]
    col_offs = tl.arange(0, BLOCK_SIZE)[None, :]
    mask = col_offs < D

    x_ptrs = x_ptr + row_id * D + col_offs
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=1) / D
    x_centered = x - mean[:, None]
    var = tl.sum(x_centered * x_centered, axis=1) / D

    rstd = tl.rsqrt(var + EPS)
    w = tl.load(weight_ptr + col_offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + col_offs, mask=mask, other=0.0)

    y = x_centered * rstd[:, None] * w + b
    y_ptrs = y_ptr + row_id * D + col_offs
    tl.store(y_ptrs, y, mask=mask)


def layernorm_v2(
    x, weight, bias,
    B: int, T: int, D: int, EPS: float = 1e-5,
    BLOCK_SIZE: int = 256, NUM_WARPS: int = 4, NUM_STAGES: int = 3,
):
    N = B * T
    y = triton.empty_like(x)
    grid = lambda meta: ((N + NUM_WARPS - 1) // NUM_WARPS,)
    _layernorm_v2_kernel[grid](
        x, weight, bias, y,
        B=B, T=T, D=D, EPS=EPS,
        BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return y
