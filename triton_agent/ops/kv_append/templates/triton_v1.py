"""KV Append Triton kernel template v1: per-token scatter write."""

import triton
import triton.language as tl


@triton.jit
def _kv_append_v1_kernel(
    k_cache_ptr, v_cache_ptr, k_new_ptr, v_new_ptr, slot_mapping_ptr,
    B: tl.constexpr, H: tl.constexpr, L: tl.constexpr, T_NEW: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, NUM_WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tokens = B * H * T_NEW
    if pid >= total_tokens:
        return

    batch_idx = pid // (H * T_NEW)
    head_idx = (pid // T_NEW) % H
    t_idx = pid % T_NEW

    slot = tl.load(slot_mapping_ptr + batch_idx * T_NEW + t_idx)
    if slot < 0 or slot >= L:
        return

    d_offs = tl.arange(0, BLOCK_SIZE)
    mask = d_offs < D

    k_new_offs = ((batch_idx * H + head_idx) * T_NEW + t_idx) * D + d_offs
    v_new_offs = ((batch_idx * H + head_idx) * T_NEW + t_idx) * D + d_offs
    cache_offs = ((batch_idx * H + head_idx) * L + slot) * D + d_offs

    k_val = tl.load(k_new_ptr + k_new_offs, mask=mask, other=0.0)
    v_val = tl.load(v_new_ptr + v_new_offs, mask=mask, other=0.0)

    tl.store(k_cache_ptr + cache_offs, k_val, mask=mask)
    tl.store(v_cache_ptr + cache_offs, v_val, mask=mask)


def kv_append_v1(
    k_cache, v_cache, k_new, v_new, slot_mapping,
    B: int, H: int, L: int, T_new: int, D: int,
    BLOCK_SIZE: int = 128, NUM_WARPS: int = 4, NUM_STAGES: int = 3,
):
    N = B * H * T_new
    grid = lambda meta: (N,)
    _kv_append_v1_kernel[grid](
        k_cache, v_cache, k_new, v_new, slot_mapping,
        B=B, H=H, L=L, T_NEW=T_new, D=D,
        BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return k_cache, v_cache
