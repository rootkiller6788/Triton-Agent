"""PagedAttention Triton kernel v1: block-table indexed flash decode."""

import triton
import triton.language as tl


@triton.jit
def _paged_attention_v1_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, out_ptr, block_table_ptr, context_len_ptr,
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE_CACHE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
    TILE_D: tl.constexpr, NUM_WARPS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_b >= B:
        return

    ctx_len = tl.load(context_len_ptr + pid_b)
    num_blocks = (ctx_len + BLOCK_SIZE_CACHE - 1) // BLOCK_SIZE_CACHE

    d_offs = tl.arange(0, TILE_D)
    mask_d = d_offs < D

    q = tl.load(q_ptr + (pid_b * H + pid_h) * D + d_offs, mask=mask_d, other=0.0)

    acc = tl.zeros((TILE_D,), dtype=tl.float32)
    max_score = tl.full((1,), -float("inf"), dtype=tl.float32)
    softmax_denom = tl.zeros((1,), dtype=tl.float32)

    for block_idx in range(MAX_BLOCKS):
        if block_idx >= num_blocks:
            break

        blk_id = tl.load(block_table_ptr + pid_b * MAX_BLOCKS + block_idx)
        if blk_id < 0:
            break

        for offset in range(BLOCK_SIZE_CACHE):
            token_idx = block_idx * BLOCK_SIZE_CACHE + offset
            if token_idx >= ctx_len:
                break

            k_ptrs = k_cache_ptr + (blk_id * BLOCK_SIZE_CACHE + offset) * (H * D) + pid_h * D + d_offs
            v_ptrs = v_cache_ptr + (blk_id * BLOCK_SIZE_CACHE + offset) * (H * D) + pid_h * D + d_offs

            k = tl.load(k_ptrs, mask=mask_d, other=0.0)
            score = tl.sum(q * k)

            new_max = tl.maximum(max_score, score)
            exp_old = tl.exp(max_score - new_max)
            exp_new = tl.exp(score - new_max)
            softmax_denom = softmax_denom * exp_old + exp_new
            max_score = new_max

            v = tl.load(v_ptrs, mask=mask_d, other=0.0)
            acc = acc * exp_old + v * exp_new

    scale = 1.0 / tl.sqrt(tl.full((1,), float(D), dtype=tl.float32))
    acc = acc / softmax_denom

    out_ptrs = out_ptr + (pid_b * H + pid_h) * D + d_offs
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask_d)


def paged_attention_v1(
    q, k_cache, v_cache, block_table, context_len,
    B: int, H: int, D: int,
    BLOCK_SIZE: int = 128, NUM_WARPS: int = 4, NUM_STAGES: int = 3,
):
    out = triton.empty_like(q)
    grid = (B, H)
    _paged_attention_v1_kernel[grid](
        q, k_cache, v_cache, out, block_table, context_len,
        B=B, H=H, D=D,
        BLOCK_SIZE_CACHE=BLOCK_SIZE, MAX_BLOCKS=block_table.shape[1],
        TILE_D=BLOCK_SIZE, NUM_WARPS=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out
