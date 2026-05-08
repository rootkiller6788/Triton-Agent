"""QuantMatMul: verify + benchmark."""

import json
from typing import Any
from triton_agent.core.spec import CandidateResult


def verify_quant_matmul(triton_output: Any, ref_output: Any, contract) -> CandidateResult:
    import torch
    r = CandidateResult(compile_pass=True)
    max_ae = (triton_output.float() - ref_output.float()).abs().max().item()
    r.verify_pass = max_ae <= contract.tolerance.max_abs_error
    r.verify_log = json.dumps({"max_abs_error": max_ae})
    return r


def benchmark_quant_matmul(kernel_fn, inputs, contract, baseline_latency_us=0.0, warmup=20, repeat=100):
    import torch, time
    a, b = inputs
    for _ in range(warmup): kernel_fn(a, b)
    torch.cuda.synchronize()
    lats = []
    for _ in range(repeat):
        t0 = time.perf_counter(); kernel_fn(a, b); torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1e6)
    lats = torch.tensor(lats)
    r = CandidateResult(compile_pass=True, verify_pass=True)
    r.latency_us_p50 = lats.quantile(0.5).item()
    r.variance = lats.std().item() / (lats.mean().item() + 1e-12)
    if baseline_latency_us > 0: r.speedup = baseline_latency_us / (r.latency_us_p50 + 1e-12)
    return r
