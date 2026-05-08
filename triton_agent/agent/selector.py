"""Selector: Best-of-N selection from candidate results."""

from typing import Any

from triton_agent.core.contract import OpContract
from triton_agent.core.spec import CandidateResult


def select_best(
    candidates: list[Any],
    results: list[CandidateResult],
) -> tuple[int, CandidateResult]:
    """Select the best candidate by reward (highest score).

    Args:
        candidates: list of (template_id, OpAction) or similar
        results: list of CandidateResult aligned with candidates

    Returns:
        (best_index, best_result) where best_index is the candidate index.
    """
    best_idx = 0
    best_result = results[0]
    for i, r in enumerate(results[1:], start=1):
        if r.reward > best_result.reward:
            best_idx = i
            best_result = r
        elif r.reward == best_result.reward and r.latency_us_p50 < best_result.latency_us_p50:
            best_idx = i
            best_result = r
    return best_idx, best_result
