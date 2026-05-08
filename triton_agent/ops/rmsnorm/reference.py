"""RMSNorm: PyTorch reference implementation for correctness baseline."""

import torch


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Root Mean Square Layer Normalization.

    Args:
        x: input tensor of shape [B, T, D]
        weight: learnable scale of shape [D]
        eps: numerical stability epsilon

    Returns:
        y: normalized tensor of shape [B, T, D]
    """
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rstd * weight


def generate_test_inputs(
    B: int = 8, T: int = 2048, D: int = 4096, dtype: torch.dtype = torch.float16, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random test inputs for RMSNorm."""
    x = torch.randn(B, T, D, dtype=dtype, device=device)
    weight = torch.randn(D, dtype=dtype, device=device)
    return x, weight


def generate_boundary_inputs(
    device: str = "cuda",
) -> list[tuple[torch.Tensor, torch.Tensor, dict]]:
    """Generate boundary-test cases for RMSNorm."""
    cases = []
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        for shape in [(1, 1, 64), (8, 2048, 4096), (32, 1, 128)]:
            B, T, D = shape
            x = torch.randn(B, T, D, dtype=dtype, device=device)
            weight = torch.randn(D, dtype=dtype, device=device)
            cases.append((x, weight, {"eps": 1e-6}))
    x = torch.zeros(1, 1, 64, device=device)
    weight = torch.ones(64, device=device)
    cases.append((x, weight, {"eps": 1e-6}))
    return cases
