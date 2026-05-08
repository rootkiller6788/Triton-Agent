"""PagedAttention: PyTorch reference for block-table indexed attention.

This is a simplified version of vLLM's PagedAttention for correctness baseline.
"""

import torch
import math


def paged_attention(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
    block_table: torch.Tensor, context_len: torch.Tensor,
) -> torch.Tensor:
    """Paged attention decode step.

    Args:
        q: [B, H, D] query (single token per batch)
        k_cache: [num_blocks, block_size, H, D]
        v_cache: [num_blocks, block_size, H, D]
        block_table: [B, max_blocks] int indices into k/v_cache
        context_len: [B] number of valid tokens per sequence

    Returns:
        out: [B, H, D]
    """
    B, H, D = q.shape
    num_blocks, block_size, _, _ = k_cache.shape
    scale = 1.0 / math.sqrt(D)

    out = torch.zeros_like(q)

    for b in range(B):
        ctx = context_len[b].item()
        num_valid_blocks = (ctx + block_size - 1) // block_size

        k_parts = []
        v_parts = []
        for blk in range(num_valid_blocks):
            blk_idx = block_table[b, blk].item()
            if blk_idx < 0 or blk_idx >= num_blocks:
                continue
            k_blk = k_cache[blk_idx, :, :, :]  # [block_size, H, D]
            v_blk = v_cache[blk_idx, :, :, :]
            k_parts.append(k_blk.permute(1, 0, 2).reshape(H, -1, D))
            v_parts.append(v_blk.permute(1, 0, 2).reshape(H, -1, D))

        k_all = torch.cat(k_parts, dim=1)[:, :ctx, :]
        v_all = torch.cat(v_parts, dim=1)[:, :ctx, :]

        scores = torch.einsum("hd,hld->hl", q[b:b+1], k_all) * scale
        attn = torch.softmax(scores, dim=-1)
        out[b:b+1] = torch.einsum("hl,hld->hd", attn, v_all)

    return out


def generate_test_inputs(
    B: int = 8, H: int = 32, D: int = 128,
    num_blocks: int = 256, block_size: int = 16, max_blocks: int = 32,
    ctx_len: int = 512,
    dtype: torch.dtype = torch.float16, device: str = "cuda",
) -> tuple:
    q = torch.randn(B, H, D, dtype=dtype, device=device)
    k_cache = torch.randn(num_blocks, block_size, H, D, dtype=dtype, device=device)
    v_cache = torch.randn(num_blocks, block_size, H, D, dtype=dtype, device=device)
    block_table = torch.randint(0, num_blocks, (B, max_blocks), device=device)
    context_len = torch.full((B,), ctx_len, dtype=torch.long, device=device)
    return q, k_cache, v_cache, block_table, context_len


def generate_boundary_inputs(device: str = "cuda") -> list:
    return []
