"""KV Append: correctness verification."""

import json
from typing import Any
from triton_agent.core.contract import OpContract
from triton_agent.core.spec import CandidateResult


def verify_kv_append(triton_output: Any, ref_output: Any, contract: OpContract) -> CandidateResult:
    result = CandidateResult()
    import torch
    triton_k, triton_v = triton_output
    ref_k, ref_v = ref_output
    max_ae = max(
        (triton_k - ref_k).abs().max().item(),
        (triton_v - ref_v).abs().max().item(),
    )
    passed = max_ae <= contract.tolerance.max_abs_error
    result.compile_pass = True
    result.verify_pass = passed
    result.verify_log = json.dumps({"max_abs_error": max_ae, "passed": passed})
    return result
