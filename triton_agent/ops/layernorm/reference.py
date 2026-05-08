"""LayerNorm: PyTorch reference implementation."""

import torch


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * weight + bias


def generate_test_inputs(
    B: int = 8, T: int = 2048, D: int = 4096, dtype: torch.dtype = torch.float16, device: str = "cuda"
) -> tuple:
    x = torch.randn(B, T, D, dtype=dtype, device=device)
    weight = torch.randn(D, dtype=dtype, device=device)
    bias = torch.randn(D, dtype=dtype, device=device)
    return x, weight, bias


def generate_boundary_inputs(device: str = "cuda") -> list:
    cases = []
    for dtype in [torch.float16, torch.float32]:
        for shape in [(1, 1, 64), (8, 2048, 4096)]:
            B, T, D = shape
            x = torch.randn(B, T, D, dtype=dtype, device=device)
            weight = torch.randn(D, dtype=dtype, device=device)
            bias = torch.randn(D, dtype=dtype, device=device)
            cases.append((x, weight, bias, {"eps": 1e-5}))
    return cases
