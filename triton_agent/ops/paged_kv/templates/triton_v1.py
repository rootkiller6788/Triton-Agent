"""Paged KV Triton kernel v1: block-table indexed scatter write."""

import triton
import triton.language as tl


@triton.jit
def _paged_kv_v1_kernel(
    k_cache_ptr, v_cache_ptr, k_new_ptr, v_new_ptr, slot_mapping_ptr,
    NUM_BLOCKS: tl.constexpr, BLOCK_SIZE: tl.constexpr, B: tl.constexpr, H: tl.constexpr,
    T_NEW: tl.constexpr, D: tl.constexpr,
    TILE_D: tl.constexpr, NUM_WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tokens = B * T_NEW
    if pid >= total_tokens:
        return

    batch_idx = pid // T_NEW
    t_idx = pid % T_NEW

    global_slot = tl.load(slot_mapping_ptr + pid)
    block_idx = global_slot // BLOCK_SIZE
    offset = global_slot % BLOCK_SIZE

    if block_idx < 0 or block_idx >= NUM_BLOCKS:
        return

    d_offs = tl.arange(0, TILE_D)
    mask = d_offs < D

    new_offs = ((batch_idx * T_NEW + t_idx) * H * D) + d_offs
    cache_offs = (block_idx * (BLOCK_SIZE * H * D) + offset * (H * D)) + d_offs

    for h in range(H):
        k_val = tl.load(k_new_ptr + new_offs + h * D, mask=mask, other=0.0)
        v_val = tl.load(v_new_ptr + new_offs + h * D, mask=mask, other=0.0)
        tl.store(k_cache_ptr + cache_offs + h * D, k_val, mask=mask)
        tl.store(v_cache_ptr + cache_offs + h * D, v_val, mask=mask)


def paged_kv_v1(
    k_cache, v_cache, k_new, v_new, slot_mapping,
    B: int, H: int, num_blocks: int, block_size: int, T_new: int, D: int,
    BLOCK_SIZE: int = 128, NUM_WARPS: int = 4, NUM_STAGES: int = 3,
):
    N = B * T_new
    grid = lambda meta: (N,)
    _paged_kv_v1_kernel[grid](
        k_cache, v_cache, k_new, v_new, slot_mapping,
        NUM_BLOCKS=num_blocks, BLOCK_SIZE=block_size, B=B, H=H,
        T_NEW=T_new, D=D,
        TILE_D=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return k_cache, v_cache
