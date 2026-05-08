"""Parallel candidate evaluation and regression detection.

Uses multiprocessing (or torch.cuda.Stream) to evaluate multiple candidates
concurrently, reducing total optimization time.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from triton_agent.core.contract import OpContract
from triton_agent.core.spec import OpState, OpAction, CandidateResult


def evaluate_parallel(
    candidates: list[tuple[str, OpAction]],
    evaluate_fn: Callable[[str, OpAction], CandidateResult],
    max_workers: int = 4,
    progress_callback: Optional[Callable[[int, int, Any], None]] = None,
) -> list[CandidateResult]:
    """Evaluate multiple candidates in parallel using a thread pool.

    Each candidate is dispatched to a separate thread. GPU kernels are
    serialized by the CUDA driver, so true parallelism comes from overlapping
    Python overhead with kernel execution.

    Args:
        candidates: list of (template_id, OpAction) to evaluate
        evaluate_fn: function(template_id, action) -> CandidateResult
        max_workers: max concurrent threads
        progress_callback: called with (completed, total, result) per candidate

    Returns:
        list of CandidateResult aligned with candidates.
    """
    results: list[Optional[CandidateResult]] = [None] * len(candidates)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for i, (tid, action) in enumerate(candidates):
            future = executor.submit(evaluate_fn, tid, action)
            future_map[future] = i

        completed = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            completed += 1
            try:
                results[idx] = future.result(timeout=300)
            except Exception as e:
                results[idx] = CandidateResult(
                    compile_pass=False,
                    compile_log=f"parallel eval error: {e}",
                )
            if progress_callback:
                progress_callback(completed, len(candidates), results[idx])

    return [r or CandidateResult() for r in results]


class RegressionDetector:
    """Detects performance regressions by comparing against historical best.

    Maintains a per-(op, shape, dtype) baseline. If a new candidate is
    slower than the historical best by more than the tolerance, it is
    flagged as a regression.
    """

    def __init__(self, tolerance: float = 0.95):
        self.tolerance = tolerance
        self._baselines: dict[str, float] = {}

    def _key(self, op_name: str, shape: str, dtype: str) -> str:
        return f"{op_name}|{shape}|{dtype}"

    def set_baseline(self, op_name: str, shape: str, dtype: str, latency_us: float) -> None:
        self._baselines[self._key(op_name, shape, dtype)] = latency_us

    def get_baseline(self, op_name: str, shape: str, dtype: str) -> float:
        return self._baselines.get(self._key(op_name, shape, dtype), float("inf"))

    def check(
        self, op_name: str, shape: str, dtype: str, latency_us: float
    ) -> tuple[bool, float]:
        """Check if latency_us is a regression.

        Returns:
            (is_regression, ratio) where ratio = latency_us / baseline.
            ratio > 1.0 means slower than baseline.
        """
        baseline = self.get_baseline(op_name, shape, dtype)
        if baseline == float("inf"):
            return False, 1.0
        ratio = latency_us / (baseline + 1e-12)
        return ratio > (1.0 / self.tolerance), ratio

    def check_result(
        self, op_name: str, shape: str, dtype: str, result: CandidateResult
    ) -> bool:
        """Check if a CandidateResult is a regression. Updates baseline if better."""
        is_reg, ratio = self.check(op_name, shape, dtype, result.latency_us_p50)
        if not is_reg and result.latency_us_p50 > 0:
            self.set_baseline(op_name, shape, dtype, result.latency_us_p50)
        return is_reg
