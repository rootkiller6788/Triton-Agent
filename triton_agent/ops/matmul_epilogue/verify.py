"""Matmul Epilogue: correctness verification and benchmark."""

import json
from typing import Any
from triton_agent.core.contract import OpContract
from triton_agent.core.spec import CandidateResult


def verify_matmul_epilogue(triton_output: Any, ref_output: Any, contract: OpContract) -> CandidateResult:
    result = CandidateResult()
    import torch
    max_ae = (triton_output - ref_output).abs().max().item()
    passed = max_ae <= contract.tolerance.max_abs_error
    result.compile_pass = True
    result.verify_pass = passed
    result.verify_log = json.dumps({"max_abs_error": max_ae, "passed": passed})
    return result
