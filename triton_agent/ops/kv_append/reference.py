"""KV Append: PyTorch reference implementation.

Copies new KV tokens into a pre-allocated cache at specified slot positions.
This is the hot path in LLM prefill/decode.
"""

import torch


def kv_append(
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    k_new: torch.Tensor, v_new: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append new KV tokens to cache.

    Args:
        k_cache: [B, H, L, D] key cache (pre-allocated)
        v_cache: [B, H, L, D] value cache
        k_new: [B, H, T_new, D] new key tokens
        v_new: [B, H, T_new, D] new value tokens
        slot_mapping: [B, T_new] int positions in cache

    Returns:
        k_out, v_out: updated caches
    """
    B, H, L, D = k_cache.shape
    T_new = k_new.shape[2]

    k_out = k_cache.clone()
    v_out = v_cache.clone()

    for b in range(B):
        for t in range(T_new):
            slot = slot_mapping[b, t].item()
            if 0 <= slot < L:
                k_out[b, :, slot, :] = k_new[b, :, t, :]
                v_out[b, :, slot, :] = v_new[b, :, t, :]

    return k_out, v_out


def generate_test_inputs(
    B: int = 4, H: int = 32, L: int = 4096, T_new: int = 1, D: int = 128,
    dtype: torch.dtype = torch.float16, device: str = "cuda",
) -> tuple:
    k_cache = torch.randn(B, H, L, D, dtype=dtype, device=device)
    v_cache = torch.randn(B, H, L, D, dtype=dtype, device=device)
    k_new = torch.randn(B, H, T_new, D, dtype=dtype, device=device)
    v_new = torch.randn(B, H, T_new, D, dtype=dtype, device=device)
    slot_mapping = torch.randint(0, L, (B, T_new), device=device)
    return k_cache, v_cache, k_new, v_new, slot_mapping


def generate_boundary_inputs(device: str = "cuda") -> list:
    cases = []
    for T_new in [1, 8]:
        B, H, L, D = 1, 1, 256, 64
        k_cache = torch.randn(B, H, L, D, device=device)
        v_cache = torch.randn(B, H, L, D, device=device)
        k_new = torch.randn(B, H, T_new, D, device=device)
        v_new = torch.randn(B, H, T_new, D, device=device)
        slot_mapping = torch.randint(0, L, (B, T_new), device=device)
        cases.append((k_cache, v_cache, k_new, v_new, slot_mapping, {}))
    return cases
