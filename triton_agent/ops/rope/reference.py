"""RoPE: PyTorch reference implementation for correctness baseline."""

import torch


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary Position Embedding.

    Args:
        x: input tensor of shape [B, H, T, D]
        cos: cosine frequencies of shape [T, D]
        sin: sine frequencies of shape [T, D]

    Returns:
        y: rotated tensor of shape [B, H, T, D]
    """
    D = x.shape[-1]
    x_half = D // 2
    x1 = x[..., :x_half]
    x2 = x[..., x_half:]
    cos_half = cos[..., :x_half]
    sin_half = sin[..., :x_half]
    y1 = x1 * cos_half - x2 * sin_half
    y2 = x2 * cos_half + x1 * sin_half
    return torch.cat([y1, y2], dim=-1)


def generate_test_inputs(
    B: int = 4,
    H: int = 32,
    T: int = 2048,
    D: int = 128,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random test inputs for RoPE."""
    x = torch.randn(B, H, T, D, dtype=dtype, device=device)
    cos = torch.randn(T, D, dtype=dtype, device=device)
    sin = torch.randn(T, D, dtype=dtype, device=device)
    return x, cos, sin


def generate_boundary_inputs(
    device: str = "cuda",
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]]:
    """Generate boundary-test cases for RoPE."""
    cases = []
    for dtype in [torch.float16, torch.float32]:
        for shape in [(1, 1, 1, 64), (4, 32, 2048, 128)]:
            B, H, T, D = shape
            x = torch.randn(B, H, T, D, dtype=dtype, device=device)
            cos = torch.randn(T, D, dtype=dtype, device=device)
            sin = torch.randn(T, D, dtype=dtype, device=device)
            cases.append((x, cos, sin, {}))
    return cases
