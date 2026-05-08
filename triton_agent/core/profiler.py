"""Performance profiler: measures latency, throughput, memory of Triton kernels."""

from typing import Any, Callable

from triton_agent.core.contract import OpContract
from triton_agent.core.spec import CandidateResult


def profile_kernel(
    kernel_fn: Callable,
    input_args: tuple,
    kwargs: dict[str, Any] | None = None,
    warmup: int = 20,
    repeat: int = 100,
    baseline_latency_us: float = 0.0,
) -> CandidateResult:
    """Profile a Triton kernel for latency and stability.

    Args:
        kernel_fn: compiled kernel callable
        input_args: positional arguments to the kernel
        kwargs: keyword arguments to the kernel (shape params etc.)
        warmup: number of warmup iterations
        repeat: number of measurement iterations
        baseline_latency_us: reference latency for speedup calculation

    Returns:
        CandidateResult with latency p50/p90/p99, speedup, variance.
    """
    import time
    import torch

    result = CandidateResult(compile_pass=True, verify_pass=True)
    kwargs = kwargs or {}

    try:
        for _ in range(warmup):
            kernel_fn(*input_args, **kwargs)
    except Exception as e:
        result.compile_pass = False
        result.compile_log = str(e)
        return result

    torch.cuda.synchronize()

    latencies = []
    for _ in range(repeat):
        start = time.perf_counter()
        kernel_fn(*input_args, **kwargs)
        torch.cuda.synchronize()
        elapsed_us = (time.perf_counter() - start) * 1e6
        latencies.append(elapsed_us)

    lat_tensor = torch.tensor(latencies)
    result.latency_us_p50 = lat_tensor.quantile(0.5).item()
    result.latency_us_p90 = lat_tensor.quantile(0.9).item()
    result.latency_us_p99 = lat_tensor.quantile(0.99).item()
    result.variance = lat_tensor.std().item() / (lat_tensor.mean().item() + 1e-12)

    if baseline_latency_us > 0:
        result.speedup = baseline_latency_us / (result.latency_us_p50 + 1e-12)
    else:
        result.speedup = 1.0

    try:
        mem = torch.cuda.max_memory_allocated()
        result.memory_peak_mb = mem / (1024 * 1024)
    except Exception:
        result.memory_peak_mb = 0.0

    return result
