"""Quantized MatMul: PyTorch reference with INT8 quantization."""

import torch


def quant_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """INT8 quantized matrix multiplication: A @ B with row-wise scales.

    Args:
        a: [B, M, K] float input
        b: [K, N] float weight

    Returns:
        c: [B, M, N] float output
    """
    a_max = a.abs().max(dim=-1, keepdim=True).values
    a_scale = a_max / 127.0
    a_scale[a_scale == 0] = 1e-6
    a_q = (a / a_scale).round().clamp(-127, 127).to(torch.int8)

    b_max = b.abs().max(dim=0, keepdim=True).values
    b_scale = b_max / 127.0
    b_scale[b_scale == 0] = 1e-6
    b_q = (b / b_scale).round().clamp(-127, 127).to(torch.int8)

    c = torch.matmul(a_q.float(), b_q.float())
    c = c * a_scale * b_scale
    return c


def generate_test_inputs(
    B: int = 8, M: int = 256, K: int = 4096, N: int = 4096,
    dtype: torch.dtype = torch.float16, device: str = "cuda",
) -> tuple:
    a = torch.randn(B, M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)
    return a, b


def generate_boundary_inputs(device: str = "cuda") -> list:
    return []
