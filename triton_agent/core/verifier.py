"""Correctness verifier: compares Triton output against PyTorch reference."""

import json
from typing import Any

from triton_agent.core.contract import OpContract
from triton_agent.core.spec import CandidateResult


def check_numerical(
    output: Any,
    reference: Any,
    contract: OpContract,
) -> CandidateResult:
    """Verify numerical correctness of Triton output against reference.

    Args:
        output: Triton kernel output tensor
        reference: PyTorch reference output tensor
        contract: operator contract with tolerance specs

    Returns:
        CandidateResult with verify_pass and diagnostic info.
    """
    result = CandidateResult(compile_pass=True)

    try:
        import torch

        if not isinstance(output, torch.Tensor):
            result.verify_pass = False
            result.verify_log = "output is not a torch.Tensor"
            return result

        if output.shape != reference.shape:
            result.verify_pass = False
            result.verify_log = (
                f"shape mismatch: output={list(output.shape)} reference={list(reference.shape)}"
            )
            return result

        if str(output.dtype) != str(reference.dtype):
            result.verify_pass = False
            result.verify_log = (
                f"dtype mismatch: output={output.dtype} reference={reference.dtype}"
            )
            return result

        ref_cpu = reference.float().cpu()
        out_cpu = output.float().cpu()

        max_ae = (out_cpu - ref_cpu).abs().max().item()
        mean_ae = (out_cpu - ref_cpu).abs().mean().item()

        has_nan = bool(torch.isnan(output).any().item())
        has_inf = bool(torch.isinf(output).any().item())

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
            "tolerance_max_abs": contract.tolerance.max_abs_error,
            "tolerance_mean_abs": contract.tolerance.mean_abs_error,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "passed": passed,
        })
    except Exception as e:
        result.verify_pass = False
        result.verify_log = str(e)

    return result
