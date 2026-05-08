"""RoPE: correctness verification against PyTorch reference."""

import json
from typing import Any

from triton_agent.core.contract import OpContract
from triton_agent.core.spec import CandidateResult


def verify_rope(
    triton_output: Any,
    ref_output: Any,
    contract: OpContract,
) -> CandidateResult:
    """Verify Triton RoPE output against PyTorch reference."""
    result = CandidateResult()
    import torch

    if not isinstance(triton_output, torch.Tensor):
        result.compile_pass = True
        result.verify_pass = False
        result.verify_log = "Triton output is not a tensor (compile may have failed)"
        return result

    result.compile_pass = True

    if triton_output.shape != ref_output.shape:
        result.verify_pass = False
        result.verify_log = (
            f"Shape mismatch: triton={list(triton_output.shape)} vs ref={list(ref_output.shape)}"
        )
        return result

    if triton_output.dtype != ref_output.dtype:
        result.verify_pass = False
        result.verify_log = (
            f"Dtype mismatch: triton={triton_output.dtype} vs ref={ref_output.dtype}"
        )
        return result

    max_ae = (triton_output - ref_output).abs().max().item()
    mean_ae = (triton_output - ref_output).abs().mean().item()

    has_nan = torch.isnan(triton_output).any().item()
    has_inf = torch.isinf(triton_output).any().item()

    passed = (
        max_ae <= contract.tolerance.max_abs_error
        and mean_ae <= contract.tolerance.mean_abs_error
        and not has_nan
        and not has_inf
    )

    result.verify_pass = passed
    result.verify_log = json.dumps({
        "max_abs_error": max_ae,
        "mean_abs_error": mean_ae,
        "tolerance_max": contract.tolerance.max_abs_error,
        "tolerance_mean": contract.tolerance.mean_abs_error,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "passed": passed,
    })
    return result
