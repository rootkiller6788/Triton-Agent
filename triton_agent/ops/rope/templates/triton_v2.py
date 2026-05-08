"""RoPE Triton kernel template v2: block-vectorized with precomputed lookup."""

import triton
import triton.language as tl


@triton.jit
def _rope_v2_kernel(
    x_ptr, cos_ptr, sin_ptr, y_ptr,
    B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, NUM_WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tokens = B * H * T
    row_id = (pid * NUM_WARPS + tl.arange(0, NUM_WARPS))[:, None]
    col_offs = tl.arange(0, BLOCK_SIZE)[None, :]
    mask = col_offs < D

    D_half = D // 2
    half_mask = col_offs < D_half

    x_ptrs = x_ptr + row_id * D + col_offs
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    t_idx = (row_id // H) % T
    cos_ptrs = cos_ptr + t_idx * D + col_offs
    sin_ptrs = sin_ptr + t_idx * D + col_offs

    cos = tl.load(cos_ptrs, mask=mask, other=0.0)
    sin = tl.load(sin_ptrs, mask=mask, other=0.0)

    x1 = tl.where(half_mask, x, 0.0)
    x2 = tl.where(half_mask, 0.0, x)

    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin

    y = tl.where(half_mask, y1, y2)
    y_ptrs = y_ptr + row_id * D + col_offs
    tl.store(y_ptrs, y, mask=mask)


def rope_v2(
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
    grid = lambda meta: ((N + NUM_WARPS - 1) // NUM_WARPS,)
    _rope_v2_kernel[grid](
        x, cos, sin, y,
        B=B, H=H, T=T, D=D,
        BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return y
