"""Reward / score computation for candidate evaluation."""

from triton_agent.core.spec import CandidateResult


def compute_score(
    result: CandidateResult,
    baseline_memory_mb: float = 0.0,
) -> float:
    """Compute a scalar reward/score for a candidate result.

    Scoring logic (from contract design):
        +0.2  for compile_pass
        +0.6  for verify_pass
        -1.0  for verify_fail
        +min(0.5, speedup - 1.0) for speedup >= 1.05
        -0.2  for high variance (>0.10)
        -0.2  for memory > 110% of baseline
        -1.0  for NaN/Inf

    Args:
        result: CandidateResult from the pipeline
        baseline_memory_mb: baseline memory for comparison

    Returns:
        float score (higher is better).
    """
    import json

    score = 0.0

    if result.compile_pass:
        score += 0.2

    if result.verify_pass:
        score += 0.6
    else:
        score -= 1.0

    if result.speedup >= 1.05:
        score += min(0.5, result.speedup - 1.0)

    if result.variance > 0.10:
        score -= 0.2

    if baseline_memory_mb > 0 and result.memory_peak_mb > baseline_memory_mb * 1.10:
        score -= 0.2

    try:
        log = json.loads(result.verify_log)
        has_nan_or_inf = log.get("has_nan", False) or log.get("has_inf", False)
        if has_nan_or_inf:
            score -= 1.0
    except (json.JSONDecodeError, TypeError):
        pass

    result.reward = score
    return score
