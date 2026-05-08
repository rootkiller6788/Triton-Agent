"""RoPE + KV Append Triton kernel v1: fused rotate-then-write."""

import triton
import triton.language as tl


@triton.jit
def _rope_kv_append_v1_kernel(
    k_new_ptr, cos_ptr, sin_ptr, k_cache_ptr, slot_mapping_ptr,
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
    D_half = D // 2

    slot = tl.load(slot_mapping_ptr + batch_idx * T_NEW + t_idx)
    if slot < 0 or slot >= L:
        return

    d_offs = tl.arange(0, BLOCK_SIZE)
    mask = d_offs < D
    half_mask = d_offs < D_half

    k_offs = ((batch_idx * H + head_idx) * T_NEW + t_idx) * D + d_offs
    cos_offs = t_idx * D + d_offs
    sin_offs = t_idx * D + d_offs

    k = tl.load(k_new_ptr + k_offs, mask=mask, other=0.0)
    cos = tl.load(cos_ptr + cos_offs, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + sin_offs, mask=mask, other=0.0)

    k1 = tl.where(half_mask, k, 0.0)
    k2 = tl.where(half_mask, 0.0, k)

    k_rope1 = k1 * cos - k2 * sin
    k_rope2 = k2 * cos + k1 * sin
    k_rope = tl.where(half_mask, k_rope1, k_rope2)

    cache_offs = ((batch_idx * H + head_idx) * L + slot) * D + d_offs
    tl.store(k_cache_ptr + cache_offs, k_rope, mask=mask)


def rope_kv_append_v1(
    k_new, cos, sin, k_cache, slot_mapping,
    B: int, H: int, L: int, T_new: int, D: int,
    BLOCK_SIZE: int = 128, NUM_WARPS: int = 4, NUM_STAGES: int = 3,
):
    N = B * H * T_new
    grid = lambda meta: (N,)
    _rope_kv_append_v1_kernel[grid](
        k_new, cos, sin, k_cache, slot_mapping,
        B=B, H=H, L=L, T_NEW=T_new, D=D,
        BLOCK_SIZE=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return k_cache
