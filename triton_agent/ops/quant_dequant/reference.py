"""Quantize-Dequantize: PyTorch reference implementation for correctness baseline."""

import torch


def quant_dequant(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Simulated quantize → dequantize using FP8/INT8-style scaling.

    Args:
        x: input tensor of shape [B, T, D]
        bits: quantization bit-width

    Returns:
        y: dequantized tensor of shape [B, T, D]
    """
    amax = x.abs().max()
    if amax == 0:
        return torch.zeros_like(x)
    scale = amax / (2 ** (bits - 1) - 1)
    x_q = torch.round(x / scale).clamp(-(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1)
    return x_q * scale


def generate_test_inputs(
    B: int = 8, T: int = 2048, D: int = 4096, dtype: torch.dtype = torch.float16, device: str = "cuda"
) -> tuple[torch.Tensor]:
    x = torch.randn(B, T, D, dtype=dtype, device=device)
    return (x,)


def generate_boundary_inputs(
    device: str = "cuda",
) -> list[tuple[torch.Tensor, dict]]:
    cases = []
    for dtype in [torch.float16]:
        for shape in [(1, 1, 64), (8, 2048, 4096)]:
            B, T, D = shape
            x = torch.randn(B, T, D, dtype=dtype, device=device)
            cases.append((x, {}))
    x = torch.zeros(1, 1, 64, device=device)
    cases.append((x, {}))
    return cases
