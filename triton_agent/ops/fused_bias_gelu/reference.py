"""Fused Bias GELU: PyTorch reference implementation for correctness baseline."""

import torch


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x.pow(3))))


def fused_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused bias addition + GELU activation.

    Args:
        x: input tensor of shape [B, T, D]
        bias: bias vector of shape [D]

    Returns:
        y: output tensor of shape [B, T, D]
    """
    x_biased = x + bias
    return gelu(x_biased)


def generate_test_inputs(
    B: int = 8, T: int = 2048, D: int = 4096, dtype: torch.dtype = torch.float16, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random test inputs for fused_bias_gelu."""
    x = torch.randn(B, T, D, dtype=dtype, device=device)
    bias = torch.randn(D, dtype=dtype, device=device)
    return x, bias


def generate_boundary_inputs(
    device: str = "cuda",
) -> list[tuple[torch.Tensor, torch.Tensor, dict]]:
    """Generate boundary-test cases for fused_bias_gelu."""
    cases = []
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        for shape in [(1, 1, 64), (8, 2048, 4096)]:
            B, T, D = shape
            x = torch.randn(B, T, D, dtype=dtype, device=device)
            bias = torch.randn(D, dtype=dtype, device=device)
            cases.append((x, bias, {}))
    x = torch.zeros(1, 1, 64, device=device)
    bias = torch.zeros(64, device=device)
    cases.append((x, bias, {}))
    return cases
