"""Quantize-Dequantize Triton kernel template v2: block-level scale per row."""

import triton
import triton.language as tl


@triton.jit
def _quant_dequant_v2_kernel(
    x_ptr, y_ptr,
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr, BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, NUM_WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = B * T
    row_id = (pid * NUM_WARPS + tl.arange(0, NUM_WARPS))[:, None]
    col_offs = tl.arange(0, BLOCK_SIZE)[None, :]
    mask = col_offs < D

    x_ptrs = x_ptr + row_id * D + col_offs
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    abs_x = tl.abs(x)
    amax = tl.max(abs_x, axis=1)
    max_val = float((1 << (BITS - 1)) - 1)
    scale = tl.where(amax > 0, amax / max_val, 1.0)

    x_q = tl.round(x / scale[:, None])
    clip_val = max_val
    x_q = tl.clamp(x_q, -clip_val, clip_val)

    y = x_q * scale[:, None]
    y_ptrs = y_ptr + row_id * D + col_offs
    tl.store(y_ptrs, y, mask=mask)


def quant_dequant_v2(
    x,
    B: int, T: int, D: int, BITS: int = 8,
    BLOCK_SIZE: int = 256, NUM_WARPS: int = 4, NUM_STAGES: int = 3,
):
    N = B * T
    y = triton.empty_like(x)
    grid = lambda meta: ((N + NUM_WARPS - 1) // NUM_WARPS,)
    _quant_dequant_v2_kernel[grid](
        x, y,
        B=B, T=T, D=D, BITS=BITS,
        BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return y
