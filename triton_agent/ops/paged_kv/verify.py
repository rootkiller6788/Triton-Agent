"""Paged KV: verify + benchmark."""

import json
from typing import Any
from triton_agent.core.spec import CandidateResult


def verify_paged_kv(triton_output, ref_output, contract) -> CandidateResult:
    import torch
    tk, tv = triton_output; rk, rv = ref_output
    max_ae = max((tk-rk).abs().max().item(), (tv-rv).abs().max().item())
    r = CandidateResult(compile_pass=True)
    r.verify_pass = max_ae <= contract.tolerance.max_abs_error
    r.verify_log = json.dumps({"max_abs_error": max_ae})
    return r


def benchmark_paged_kv(kernel_fn, inputs, contract, baseline_latency_us=0.0, warmup=20, repeat=100):
    import torch, time
    kc, vc, kn, vn, sm = inputs
    for _ in range(warmup): kernel_fn(kc, vc, kn, vn, sm)
    torch.cuda.synchronize()
    lats = [((t:=time.perf_counter()), kernel_fn(kc, vc, kn, vn, sm), torch.cuda.synchronize(), (time.perf_counter()-t)*1e6)[3] for _ in range(repeat)]
    lats = [l[3] for l in [((t:=time.perf_counter()), kernel_fn(kc, vc, kn, vn, sm), torch.cuda.synchronize(), (time.perf_counter()-t)*1e6) for _ in range(repeat)]]
    lats = torch.tensor(lats)
    r = CandidateResult(compile_pass=True, verify_pass=True)
    r.latency_us_p50 = lats.quantile(0.5).item()
    r.variance = lats.std().item() / (lats.mean().item() + 1e-12)
    if baseline_latency_us > 0: r.speedup = baseline_latency_us / (r.latency_us_p50 + 1e-12)
    return r
