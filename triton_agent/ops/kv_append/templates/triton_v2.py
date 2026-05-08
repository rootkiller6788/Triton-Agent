"""KV Append Triton kernel template v2: vectorized multi-WARP."""

import triton
import triton.language as tl


@triton.jit
def _kv_append_v2_kernel(
    k_cache_ptr, v_cache_ptr, k_new_ptr, v_new_ptr, slot_mapping_ptr,
    B: tl.constexpr, H: tl.constexpr, L: tl.constexpr, T_NEW: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, NUM_WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tokens = B * H * T_NEW
    token_id = (pid * NUM_WARPS + tl.arange(0, NUM_WARPS))[:, None]
    mask_token = token_id < total_tokens

    batch_idx = token_id // (H * T_NEW)
    head_idx = (token_id // T_NEW) % H
    t_idx = token_id % T_NEW

    slot = tl.load(slot_mapping_ptr + batch_idx * T_NEW + t_idx, mask=mask_token, other=0)
    valid = (slot >= 0) & (slot < L) & mask_token

    d_offs = tl.arange(0, BLOCK_SIZE)[None, :]
    mask_d = d_offs < D

    k_new_offs = token_id * D + d_offs
    v_new_offs = token_id * D + d_offs
    cache_offs = (batch_idx * H * L + head_idx * L + slot) * D + d_offs

    k_val = tl.load(k_new_ptr + k_new_offs, mask=mask_d, other=0.0)
    v_val = tl.load(v_new_ptr + v_new_offs, mask=mask_d, other=0.0)

    tl.store(k_cache_ptr + cache_offs, k_val, mask=mask_d & valid)
    tl.store(v_cache_ptr + cache_offs, v_val, mask=mask_d & valid)


def kv_append_v2(
    k_cache, v_cache, k_new, v_new, slot_mapping,
    B: int, H: int, L: int, T_new: int, D: int,
    BLOCK_SIZE: int = 128, NUM_WARPS: int = 4, NUM_STAGES: int = 3,
):
    N = B * H * T_new
    grid = lambda meta: ((N + NUM_WARPS - 1) // NUM_WARPS,)
    _kv_append_v2_kernel[grid](
        k_cache, v_cache, k_new, v_new, slot_mapping,
        B=B, H=H, L=L, T_NEW=T_new, D=D,
        BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return k_cache, v_cache
