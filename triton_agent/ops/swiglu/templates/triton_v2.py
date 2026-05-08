"""SwiGLU Triton kernel template v2: tiled with multi-WARP."""

import triton
import triton.language as tl


@triton.jit
def _swiglu_v2_kernel(
    x_ptr, gate_ptr, y_ptr,
    B: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, NUM_WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_rows = B * T
    row_id = (pid * NUM_WARPS + tl.arange(0, NUM_WARPS))[:, None]
    col_offs = tl.arange(0, BLOCK_SIZE)[None, :]
    mask = col_offs < D

    x_ptrs = x_ptr + row_id * D + col_offs
    gate_ptrs = gate_ptr + row_id * D + col_offs

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    gate = tl.load(gate_ptrs, mask=mask, other=0.0)

    silu_gate = gate * tl.sigmoid(gate)
    y = silu_gate * x

    y_ptrs = y_ptr + row_id * D + col_offs
    tl.store(y_ptrs, y, mask=mask)


def swiglu_v2(
    x, gate,
    B: int, T: int, D: int,
    BLOCK_SIZE: int = 256, NUM_WARPS: int = 4, NUM_STAGES: int = 3,
):
    N = B * T
    y = triton.empty_like(x)
    grid = lambda meta: ((N + NUM_WARPS - 1) // NUM_WARPS,)
    _swiglu_v2_kernel[grid](
        x, gate, y,
        B=B, T=T, D=D,
        BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return y
