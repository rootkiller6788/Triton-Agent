"""Simple Matmul Epilogue: PyTorch reference (matmul + bias + GELU)."""

import torch


def matmul_epilogue(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused matmul + bias + GELU activation.

    Args:
        a: [B, M, K]
        b: [K, N]
        bias: [N]

    Returns:
        c: [B, M, N]
    """
    c = a @ b + bias
    return 0.5 * c * (1.0 + torch.tanh(0.7978845608028654 * (c + 0.044715 * c.pow(3))))


def generate_test_inputs(
    B: int = 8, M: int = 256, K: int = 4096, N: int = 4096,
    dtype: torch.dtype = torch.float16, device: str = "cuda",
) -> tuple:
    a = torch.randn(B, M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)
    bias = torch.randn(N, dtype=dtype, device=device)
    return a, b, bias


def generate_boundary_inputs(device: str = "cuda") -> list:
    cases = []
    for shape in [(1, 1, 64, 64), (4, 128, 2048, 2048)]:
        B, M, K, N = shape
        a = torch.randn(B, M, K, device=device)
        b = torch.randn(K, N, device=device)
        bias = torch.randn(N, device=device)
        cases.append((a, b, bias, {}))
    return cases
