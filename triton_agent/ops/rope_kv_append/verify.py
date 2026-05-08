"""RoPE+KV Append: correctness verification."""

import json
from typing import Any
from triton_agent.core.contract import OpContract
from triton_agent.core.spec import CandidateResult


def verify_rope_kv_append(triton_output: Any, ref_output: Any, contract: OpContract) -> CandidateResult:
    result = CandidateResult()
    import torch
    if isinstance(ref_output, torch.Tensor) and isinstance(triton_output, torch.Tensor):
        max_ae = (triton_output - ref_output).abs().max().item()
    else:
        max_ae = 1.0
    passed = max_ae <= contract.tolerance.max_abs_error
    result.compile_pass = True
    result.verify_pass = passed
    result.verify_log = json.dumps({"max_abs_error": max_ae, "passed": passed})
    return result
