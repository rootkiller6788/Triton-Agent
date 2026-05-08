"""Candidate generator: grid-search enumeration of Triton variant configs."""

import itertools
from typing import Any

from triton_agent.core.contract import OpContract
from triton_agent.core.spec import OpState, OpAction


def generate_grid(
    contract: OpContract,
    state: OpState,
    templates: dict[str, Any],
) -> list[tuple[str, OpAction]]:
    """Generate all candidate (template_id, OpAction) pairs via grid search.

    Enumerates the cartesian product of search_space parameters defined in the contract.

    Args:
        contract: operator contract with search_space
        state: current OpState (shape/dtype/device)
        templates: dict of template_id -> (kernel_fn, compile_args_fn) or callable

    Returns:
        list of (template_id, OpAction) tuples covering the full grid.
    """
    candidates: list[tuple[str, OpAction]] = []
    ss = contract.search_space

    for template_id in templates:
        block_values = ss.BLOCK_D if hasattr(ss, "BLOCK_D") and ss.BLOCK_D else [128]
        warp_values = ss.num_warps if ss.num_warps else [4]
        stages_values = ss.num_stages if ss.num_stages else [3]
        vec_values = ss.vectorize if ss.vectorize else [False]

        for block_d, num_warps, num_stages, vectorize in itertools.product(
            block_values, warp_values, stages_values, vec_values
        ):
            action = OpAction(
                template_id=template_id,
                block_d=block_d,
                num_warps=num_warps,
                num_stages=num_stages,
                vectorize=vectorize,
                fusion=False,
            )
            candidates.append((template_id, action))

    return candidates


def generate_best_of_n(
    contract: OpContract,
    state: OpState,
    templates: dict[str, Any],
    n: int = 32,
    seed: int | None = None,
) -> list[tuple[str, OpAction]]:
    """Generate N random candidates from the search space (Best-of-N sampling)."""
    import random

    rng = random.Random(seed)
    ss = contract.search_space
    candidates: list[tuple[str, OpAction]] = []

    for _ in range(n):
        template_id = rng.choice(list(templates.keys()))
        block_d = rng.choice(ss.BLOCK_D) if ss.BLOCK_D else 128
        num_warps = rng.choice(ss.num_warps) if ss.num_warps else 4
        num_stages = rng.choice(ss.num_stages) if ss.num_stages else 3
        vectorize = rng.choice(ss.vectorize) if ss.vectorize else False

        action = OpAction(
            template_id=template_id,
            block_d=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
            vectorize=vectorize,
            fusion=False,
        )
        candidates.append((template_id, action))

    return candidates
