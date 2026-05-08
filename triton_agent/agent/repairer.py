"""Repairer: automatically fixes compile/verify/profile failures.

Strategies:
- Compile error → adjust BLOCK_SIZE down / increase num_stages, then retry
- Verify failure → check tolerance, adjust numerical parameters
- Memory overflow → reduce BLOCK_SIZE or use smaller dtype
"""

import re
from copy import deepcopy
from typing import Any, Optional

from triton_agent.core.spec import OpAction, CandidateResult


def repair_action(
    action: OpAction,
    result: CandidateResult,
    max_block_size: int = 1024,
    min_block_size: int = 32,
) -> Optional[OpAction]:
    """Suggest a repaired action based on failure feedback.

    Args:
        action: the original action that failed
        result: the CandidateResult containing error details
        max_block_size: maximum allowed block size (hardware limit)
        min_block_size: minimum block size to try

    Returns:
        A modified OpAction, or None if no repair is feasible.
    """
    if result.compile_pass and result.verify_pass:
        return None

    repaired = deepcopy(action)

    if not result.compile_pass:
        repaired = _repair_compile_failure(repaired, result, max_block_size, min_block_size)

    elif not result.verify_pass:
        repaired = _repair_verify_failure(repaired, result)

    return repaired


def _repair_compile_failure(
    action: OpAction,
    result: CandidateResult,
    max_block_size: int,
    min_block_size: int,
) -> OpAction:
    """Adjust block size and stages for compile errors."""
    error_lower = result.compile_log.lower()

    if any(kw in error_lower for kw in ["out of registers", "register", "spill"]):
        action.block_d = max(min_block_size, action.block_d // 2)
        action.num_stages = min(4, action.num_stages + 1)

    elif any(kw in error_lower for kw in ["shared memory", "out of memory", "too large"]):
        action.block_d = max(min_block_size, action.block_d // 2)
        action.num_stages = max(1, action.num_stages - 1)

    elif any(kw in error_lower for kw in ["not supported", "incompatible"]):
        action.block_d = max(min_block_size, min(action.block_d, 256))
        action.num_warps = min(8, max(1, action.num_warps))

    elif "num_warps" in error_lower or "warp" in error_lower:
        action.num_warps = max(1, action.num_warps - 1)

    else:
        action.block_d = max(min_block_size, action.block_d // 2)

    return action


def _repair_verify_failure(
    action: OpAction,
    result: CandidateResult,
) -> OpAction:
    """Adjust numerical parameters for verification failures.

    Common causes: vectorization causing numerical drift, large block causing
    accumulated rounding errors.
    """
    try:
        log = __import__("json").loads(result.verify_log)
    except Exception:
        return action

    max_ae = log.get("max_abs_error", 0)
    has_nan = log.get("has_nan", False)
    has_inf = log.get("has_inf", False)

    if has_nan or has_inf:
        action.vectorize = False
        action.block_d = max(32, action.block_d // 2)
        action.num_stages = max(1, action.num_stages - 1)

    elif max_ae > 1e-2:
        action.vectorize = False
        action.block_d = max(64, action.block_d // 2)

    elif action.vectorize:
        action.vectorize = False

    return action


def should_retry(action: OpAction, original: OpAction, result: CandidateResult) -> bool:
    """Check whether retrying with a repaired action is worthwhile.

    Returns False if:
    - The action hasn't changed (stuck in a loop)
    - We've already tried all reasonable permutations
    """
    if action is None:
        return False

    if (
        action.block_d == original.block_d
        and action.num_warps == original.num_warps
        and action.num_stages == original.num_stages
        and action.vectorize == original.vectorize
    ):
        return False

    if action.block_d < 32:
        return False

    if action.num_warps < 1:
        return False

    if result.compile_pass and not result.verify_pass:
        return True

    return True
