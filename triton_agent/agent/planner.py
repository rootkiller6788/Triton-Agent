"""Planner: search strategy dispatch based on op/shape/device."""

from triton_agent.core.contract import OpContract
from triton_agent.core.spec import OpState


def plan(contract: OpContract, state: OpState, history: list | None = None) -> dict:
    """Determine the search strategy for a given operator state.

    Currently supports:
        - "grid": full grid search over contract search_space
        - "best_of_n": random sampling with N candidates

    Returns a dict with strategy name and parameters.
    """
    params = contract.search_space
    total_combinations = 1
    for values in [params.BLOCK_D, params.num_warps, params.num_stages, params.vectorize]:
        if values:
            total_combinations *= len(values)

    if total_combinations <= 64:
        return {"strategy": "grid", "candidates": total_combinations}
    else:
        return {"strategy": "best_of_n", "n": 64}
