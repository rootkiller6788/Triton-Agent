"""FlashAttention-like: verify + benchmark."""

import json
from typing import Any
from triton_agent.core.spec import CandidateResult


def verify_flash_attn_like(triton_output, ref_output, contract) -> CandidateResult:
    import torch
    max_ae = (triton_output.float() - ref_output.float()).abs().max().item()
    r = CandidateResult(compile_pass=True)
    r.verify_pass = max_ae <= contract.tolerance.max_abs_error
    r.verify_log = json.dumps({"max_abs_error": max_ae})
    return r


def benchmark_flash_attn_like(kernel_fn, inputs, contract, baseline_latency_us=0.0, warmup=20, repeat=100):
    import torch, time
    q, k, v = inputs
    for _ in range(warmup): kernel_fn(q, k, v)
    torch.cuda.synchronize()
    lats = []
    for _ in range(repeat):
        t0 = time.perf_counter(); kernel_fn(q, k, v); torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1e6)
    lats = torch.tensor(lats)
    r = CandidateResult(compile_pass=True, verify_pass=True)
    r.latency_us_p50 = lats.quantile(0.5).item()
    r.variance = lats.std().item() / (lats.mean().item() + 1e-12)
    if baseline_latency_us > 0: r.speedup = baseline_latency_us / (r.latency_us_p50 + 1e-12)
    return r
