"""torch.compile comparison: benchmark Triton kernel vs torch.compile."""

from typing import Any, Callable, Optional
import time


def compare_torch_compile(
    triton_kernel: Callable,
    ref_fn: Callable,
    inputs: tuple,
    warmup: int = 20,
    repeat: int = 100,
) -> dict[str, Any]:
    """Compare a Triton kernel against torch.compile'd reference.

    Args:
        triton_kernel: compiled Triton kernel
        ref_fn: PyTorch reference function (e.g., from reference.py)
        inputs: tuple of tensors
        warmup: warmup iterations
        repeat: measurement iterations

    Returns:
        dict with latency comparison and speedup.
    """
    import torch

    if not torch.cuda.is_available():
        return {"error": "GPU not available", "triton_speedup": 1.0}

    compiled_ref = torch.compile(ref_fn, dynamic=False)

    for _ in range(warmup):
        triton_kernel(*inputs)
        compiled_ref(*inputs)

    torch.cuda.synchronize()

    triton_latencies = []
    ref_latencies = []

    for _ in range(repeat):
        start = time.perf_counter()
        triton_kernel(*inputs)
        torch.cuda.synchronize()
        triton_latencies.append((time.perf_counter() - start) * 1e6)

    for _ in range(repeat):
        start = time.perf_counter()
        compiled_ref(*inputs)
        torch.cuda.synchronize()
        ref_latencies.append((time.perf_counter() - start) * 1e6)

    triton_t = torch.tensor(triton_latencies)
    ref_t = torch.tensor(ref_latencies)

    triton_p50 = triton_t.quantile(0.5).item()
    ref_p50 = ref_t.quantile(0.5).item()

    return {
        "triton_latency_us_p50": triton_p50,
        "torch_compile_latency_us_p50": ref_p50,
        "triton_speedup": ref_p50 / (triton_p50 + 1e-12),
        "triton_stability": triton_t.std().item() / (triton_t.mean().item() + 1e-12),
        "torch_compile_stability": ref_t.std().item() / (ref_t.mean().item() + 1e-12),
    }
