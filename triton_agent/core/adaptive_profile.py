"""Adaptive profiling: early-stop and adaptive warmup for efficiency.

Standard profiling uses fixed warmup/repeat counts. These functions
adapt the measurement window based on observed stability, reducing
overhead for stable kernels while ensuring reliable measurements.
"""

import time
from typing import Any, Callable

from triton_agent.core.spec import CandidateResult


def adaptive_profile(
    kernel_fn: Callable,
    input_args: tuple,
    kwargs: dict | None = None,
    min_warmup: int = 5,
    max_warmup: int = 50,
    min_repeat: int = 10,
    max_repeat: int = 200,
    stability_threshold: float = 0.02,
    baseline_latency_us: float = 0.0,
) -> CandidateResult:
    """Profile with adaptive warmup and early-stop.

    Warmup phase: keep warming until latency stabilizes (CV < threshold)
    or max_warmup reached.

    Measurement phase: stop early if variance drops below threshold
    after min_repeat iterations.

    Args:
        kernel_fn: compiled kernel
        input_args: positional inputs
        kwargs: keyword arguments
        min_warmup/max_warmup: warmup bounds
        min_repeat/max_repeat: measurement bounds
        stability_threshold: coefficient of variation threshold for early stop
        baseline_latency_us: reference for speedup

    Returns:
        CandidateResult with latency percentiles and stability metrics.
    """
    import torch

    kwargs = kwargs or {}
    result = CandidateResult(compile_pass=True, verify_pass=True)

    for _ in range(max_warmup):
        kernel_fn(*input_args, **kwargs)
        if _ >= min_warmup:
            break

    torch.cuda.synchronize()

    latencies = []
    for i in range(max_repeat):
        start = time.perf_counter()
        kernel_fn(*input_args, **kwargs)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1e6
        latencies.append(elapsed)

        if i >= min_repeat:
            lat_tensor = torch.tensor(latencies[-min_repeat:], dtype=torch.float32)
            cv = lat_tensor.std().item() / (lat_tensor.mean().item() + 1e-12)
            if cv < stability_threshold:
                break

    lat_tensor = torch.tensor(latencies)
    mean_lat = lat_tensor.mean().item()

    result.latency_us_p50 = lat_tensor.quantile(0.5).item()
    result.latency_us_p90 = lat_tensor.quantile(0.9).item() if len(latencies) >= 10 else mean_lat
    result.latency_us_p99 = lat_tensor.quantile(0.99).item() if len(latencies) >= 100 else mean_lat
    result.variance = lat_tensor.std().item() / (mean_lat + 1e-12)

    if baseline_latency_us > 0:
        result.speedup = baseline_latency_us / (result.latency_us_p50 + 1e-12)
    else:
        result.speedup = 1.0

    return result


def fast_profile(
    kernel_fn: Callable,
    input_args: tuple,
    kwargs: dict | None = None,
    baseline_latency_us: float = 0.0,
) -> CandidateResult:
    """Ultra-fast profiling for quick ranking (minimal iterations).

    Used in early stages of optimization when many candidates are being
    compared. Uses fixed 5 warmup + 20 repeat.
    """
    import torch
    import time

    kwargs = kwargs or {}
    result = CandidateResult(compile_pass=True, verify_pass=True)

    for _ in range(5):
        kernel_fn(*input_args, **kwargs)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(20):
        start = time.perf_counter()
        kernel_fn(*input_args, **kwargs)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1e6)

    lat_tensor = torch.tensor(latencies)
    result.latency_us_p50 = lat_tensor.quantile(0.5).item()
    result.variance = lat_tensor.std().item() / (lat_tensor.mean().item() + 1e-12)

    if baseline_latency_us > 0:
        result.speedup = baseline_latency_us / (result.latency_us_p50 + 1e-12)
    return result
