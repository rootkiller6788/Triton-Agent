"""Quantize-Dequantize Triton kernel template v1: per-tensor symmetric quant."""

import triton
import triton.language as tl


@triton.jit
def _quant_dequant_v1_kernel(
    x_ptr, y_ptr,
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr, BITS: tl.constexpr,
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

    abs_x = tl.abs(x)
    amax = tl.max(abs_x, axis=0)
    max_val = (1 << (BITS - 1)) - 1
    max_val = max_val.to(tl.float32)
    scale = tl.where(amax > 0, amax / max_val, 1.0)

    x_q = tl.round(x / scale)
    clip_val = max_val
    x_q = tl.clamp(x_q, -clip_val, clip_val)

    y = x_q * scale
    tl.store(y_ptr + offsets, y, mask=mask)


def quant_dequant_v1(
    x,
    B: int, T: int, D: int, BITS: int = 8,
    BLOCK_SIZE: int = 256, NUM_WARPS: int = 4, NUM_STAGES: int = 3,
):
    N = B * T
    y = triton.empty_like(x)
    grid = lambda meta: (N,)
    _quant_dequant_v1_kernel[grid](
        x, y,
        B=B, T=T, D=D, BITS=BITS,
        BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return y
