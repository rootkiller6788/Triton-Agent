"""SwiGLU: PyTorch reference implementation for correctness baseline."""

import torch


def silu(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid Linear Unit."""
    return x * torch.sigmoid(x)


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: SiLU(gate) * x.

    Args:
        x: input tensor of shape [B, T, D]
        gate: gating tensor of shape [B, T, D]

    Returns:
        y: output tensor of shape [B, T, D]
    """
    return silu(gate) * x


def generate_test_inputs(
    B: int = 8, T: int = 2048, D: int = 4096, dtype: torch.dtype = torch.float16, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(B, T, D, dtype=dtype, device=device)
    gate = torch.randn(B, T, D, dtype=dtype, device=device)
    return x, gate


def generate_boundary_inputs(
    device: str = "cuda",
) -> list[tuple[torch.Tensor, torch.Tensor, dict]]:
    cases = []
    for dtype in [torch.float16, torch.float32]:
        for shape in [(1, 1, 64), (8, 2048, 4096)]:
            B, T, D = shape
            x = torch.randn(B, T, D, dtype=dtype, device=device)
            gate = torch.randn(B, T, D, dtype=dtype, device=device)
            cases.append((x, gate, {}))
    x = torch.zeros(1, 1, 64, device=device)
    gate = torch.zeros(1, 1, 64, device=device)
    cases.append((x, gate, {}))
    return cases
