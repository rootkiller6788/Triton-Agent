"""RoPE Triton kernel template v1: baseline per-element rotation."""

import triton
import triton.language as tl


@triton.jit
def _rope_v1_kernel(
    x_ptr, cos_ptr, sin_ptr, y_ptr,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, NUM_WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tokens = B * H * T
    if pid >= total_tokens:
        return

    D_half = D // 2
    row_start = pid * D
    offsets_full = row_start + tl.arange(0, BLOCK_SIZE)
    mask_full = tl.arange(0, BLOCK_SIZE) < D

    x = tl.load(x_ptr + offsets_full, mask=mask_full, other=0.0)

    t_idx = (pid // H) % T
    cos_offsets = t_idx * D + tl.arange(0, BLOCK_SIZE)
    sin_offsets = t_idx * D + tl.arange(0, BLOCK_SIZE)

    cos = tl.load(cos_ptr + cos_offsets, mask=mask_full, other=0.0)
    sin = tl.load(sin_ptr + sin_offsets, mask=mask_full, other=0.0)

    x1 = tl.where(tl.arange(0, BLOCK_SIZE) < D_half, x, 0.0)
    x2 = tl.where(tl.arange(0, BLOCK_SIZE) >= D_half, x, 0.0)
    x2 = tl.where(tl.arange(0, BLOCK_SIZE) >= D_half, x2, 0.0)

    x2_shifted = tl.arange(0, BLOCK_SIZE)
    x2_shifted = tl.where(x2_shifted >= D_half, x2_shifted - D_half, x2_shifted + D_half)

    cos_half = tl.where(tl.arange(0, BLOCK_SIZE) < D_half, cos, 0.0)
    sin_half = tl.where(tl.arange(0, BLOCK_SIZE) < D_half, sin, 0.0)

    y1 = x1 * cos_half - x2 * sin_half
    y2 = x2 * cos_half + x1 * sin_half

    y = y1 + y2
    tl.store(y_ptr + offsets_full, y, mask=mask_full)


def rope_v1(
    x,
    cos,
    sin,
    B: int,
    H: int,
    T: int,
    D: int,
    BLOCK_SIZE: int = 128,
    NUM_WARPS: int = 4,
    NUM_STAGES: int = 3,
):
    y = triton.empty_like(x)
    N = B * H * T
    grid = lambda meta: (N,)
    _rope_v1_kernel[grid](
        x, cos, sin, y,
        B=B, H=H, T=T, D=D,
        BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return y
