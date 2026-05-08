"""SwiGLU Triton kernel template v1: baseline element-wise."""

import triton
import triton.language as tl


@triton.jit
def _swiglu_v1_kernel(
    x_ptr, gate_ptr, y_ptr,
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
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)

    silu_gate = gate * tl.sigmoid(gate)
    y = silu_gate * x

    tl.store(y_ptr + offsets, y, mask=mask)


def swiglu_v1(
    x, gate,
    B: int, T: int, D: int,
    BLOCK_SIZE: int = 256, NUM_WARPS: int = 4, NUM_STAGES: int = 3,
):
    N = B * T
    y = triton.empty_like(x)
    grid = lambda meta: (N,)
    _swiglu_v1_kernel[grid](
        x, gate, y,
        B=B, T=T, D=D,
        BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return y
