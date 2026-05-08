"""RoPE + KV Append fusion: PyTorch reference.

Fuses RoPE rotation into the KV append step, avoiding a separate kernel launch.
This is a key optimization in LLM decode for reducing dispatch overhead.
"""

import torch


def rope_kv_append(
    k_new: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    k_cache: torch.Tensor, slot_mapping: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE then write to KV cache in a fused operation.

    Args:
        k_new: [B, H, T_new, D] new key tokens
        cos: [T_new, D] cosine frequencies
        sin: [T_new, D] sine frequencies
        k_cache: [B, H, L, D] pre-allocated key cache
        slot_mapping: [B, T_new] int positions

    Returns:
        k_out: updated key cache
    """
    D = k_new.shape[-1]
    D_half = D // 2
    k1 = k_new[..., :D_half]
    k2 = k_new[..., D_half:]
    cos_half = cos[..., :D_half]
    sin_half = sin[..., :D_half]

    k_rope1 = k1 * cos_half - k2 * sin_half
    k_rope2 = k2 * cos_half + k1 * sin_half
    k_rope = torch.cat([k_rope1, k_rope2], dim=-1)

    B, H, T_new, D_k = k_rope.shape
    L = k_cache.shape[2]
    k_out = k_cache.clone()

    for b in range(B):
        for t in range(T_new):
            slot = slot_mapping[b, t].item()
            if 0 <= slot < L:
                k_out[b, :, slot, :] = k_rope[b, :, t, :]

    return k_out


def generate_test_inputs(
    B: int = 4, H: int = 32, L: int = 4096, T_new: int = 1, D: int = 128,
    dtype: torch.dtype = torch.float16, device: str = "cuda",
) -> tuple:
    k_new = torch.randn(B, H, T_new, D, dtype=dtype, device=device)
    cos = torch.randn(T_new, D, dtype=dtype, device=device)
    sin = torch.randn(T_new, D, dtype=dtype, device=device)
    k_cache = torch.randn(B, H, L, D, dtype=dtype, device=device)
    slot_mapping = torch.randint(0, L, (B, T_new), device=device)
    return k_new, cos, sin, k_cache, slot_mapping


def generate_boundary_inputs(device: str = "cuda") -> list:
    return []
