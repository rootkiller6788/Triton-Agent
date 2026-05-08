"""Paged KV Cache: reference implementation with block-table addressing."""

import torch


def paged_kv(
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    k_new: torch.Tensor, v_new: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks, block_size, H, D = k_cache.shape
    B, T_new, H2, D2 = k_new.shape
    assert H == H2 and D == D2

    k_out = k_cache.clone()
    v_out = v_cache.clone()

    for b in range(B):
        for t in range(T_new):
            global_slot = slot_mapping[b, t].item()
            block_idx = global_slot // block_size
            offset = global_slot % block_size
            if 0 <= block_idx < num_blocks:
                k_out[block_idx, offset, :, :] = k_new[b, t, :, :]
                v_out[block_idx, offset, :, :] = v_new[b, t, :, :]

    return k_out, v_out


def generate_test_inputs(
    B: int = 4, H: int = 32, num_blocks: int = 256, block_size: int = 16,
    T_new: int = 1, D: int = 128,
    dtype: torch.dtype = torch.float16, device: str = "cuda",
) -> tuple:
    k_cache = torch.randn(num_blocks, block_size, H, D, dtype=dtype, device=device)
    v_cache = torch.randn(num_blocks, block_size, H, D, dtype=dtype, device=device)
    k_new = torch.randn(B, T_new, H, D, dtype=dtype, device=device)
    v_new = torch.randn(B, T_new, H, D, dtype=dtype, device=device)
    slot_mapping = torch.randint(0, num_blocks * block_size, (B, T_new), device=device)
    return k_cache, v_cache, k_new, v_new, slot_mapping


def generate_boundary_inputs(device: str = "cuda") -> list:
    return []
