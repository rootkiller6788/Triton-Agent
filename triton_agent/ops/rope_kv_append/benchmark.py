"""RoPE+KV Append: performance benchmark."""

from typing import Any
from triton_agent.core.spec import CandidateResult


def benchmark_rope_kv_append(
    kernel_fn: Any, inputs: tuple, contract: Any,
    baseline_latency_us: float = 0.0, warmup: int = 20, repeat: int = 100,
) -> CandidateResult:
    import torch, time
    k_new, cos, sin, k_cache, slot_mapping = inputs
    result = CandidateResult(compile_pass=True, verify_pass=True)
    for _ in range(warmup):
        kernel_fn(k_new, cos, sin, k_cache, slot_mapping)
    torch.cuda.synchronize()
    latencies = []
    for _ in range(repeat):
        start = time.perf_counter()
        kernel_fn(k_new, cos, sin, k_cache, slot_mapping)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1e6)
    latencies = torch.tensor(latencies)
    result.latency_us_p50 = latencies.quantile(0.5).item()
    result.variance = latencies.std().item() / (latencies.mean().item() + 1e-12)
    if baseline_latency_us > 0:
        result.speedup = baseline_latency_us / (result.latency_us_p50 + 1e-12)
    return result
