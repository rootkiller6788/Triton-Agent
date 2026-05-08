"""Promoter: variant promotion / rollback decision."""

from triton_agent.core.contract import OpContract
from triton_agent.core.spec import CandidateResult


def should_promote(result: CandidateResult, contract: OpContract) -> bool:
    """Decide whether a candidate result meets promotion criteria.

    Promotion rules:
        - compile_pass == True
        - verify_pass == True
        - speedup >= min_speedup
        - variance <= max_variance

    Returns True if the variant should be promoted.
    """
    if not result.compile_pass:
        return False
    if not result.verify_pass:
        return False
    if result.speedup < contract.promotion.min_speedup:
        return False
    if result.variance > contract.promotion.max_variance:
        return False
    return True
