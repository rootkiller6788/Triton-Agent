"""FlashAttention-like: PyTorch reference (standard scaled dot-product attention)."""

import torch
import math


def flash_attn_like(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Standard scaled dot-product attention.

    Args:
        q: [B, H, T, D]
        k: [B, H, S, D]
        v: [B, H, S, D]
        causal: whether to apply causal mask

    Returns:
        out: [B, H, T, D]
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bhtd,bhsd->bhts", q, k) * scale

    if causal:
        T_len = q.shape[2]
        mask = torch.tril(torch.ones(T_len, T_len, device=q.device), diagonal=0)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bhts,bhsd->bhtd", attn, v)


def generate_test_inputs(
    B: int = 4, H: int = 32, T: int = 2048, S: int = 2048, D: int = 64,
    dtype: torch.dtype = torch.float16, device: str = "cuda",
) -> tuple:
    q = torch.randn(B, H, T, D, dtype=dtype, device=device)
    k = torch.randn(B, H, S, D, dtype=dtype, device=device)
    v = torch.randn(B, H, S, D, dtype=dtype, device=device)
    return q, k, v


def generate_boundary_inputs(device: str = "cuda") -> list:
    cases = []
    for shape in [(1, 1, 1, 1, 64), (2, 8, 128, 128, 64)]:
        B, H, T, S, D = shape
        q = torch.randn(B, H, T, D, device=device)
        k = torch.randn(B, H, S, D, device=device)
        v = torch.randn(B, H, S, D, device=device)
        cases.append((q, k, v, {"causal": False}))
    return cases
