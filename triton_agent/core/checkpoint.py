"""Checkpoint / resume: save and restore optimization state for long runs.

Enables:
- Interrupt and resume optimization without losing progress
- Crash recovery: partial results are preserved
- Incremental exploration across sessions
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

from triton_agent.core.spec import OpState, OpAction, CandidateResult


class Checkpoint:
    """Saves and restores full optimization state.

    Checkpoint file layout:
    {
        "op_name": str,
        "state": OpState.as_dict(),
        "strategy": str,
        "max_candidates": int,
        "evaluated_count": int,
        "candidates": [{"template_id": ..., "action": OpAction.as_dict()}],
        "results": [CandidateResult.as_dict()],
        "best_idx": int,
        "best_result": CandidateResult.as_dict(),
        "timestamp": float,
        "version": "1.0"
    }
    """

    def __init__(self, checkpoint_dir: str | Path = "episodes/checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        op_name: str,
        state: OpState,
        strategy: str,
        max_candidates: int,
        evaluated: list[tuple],
        best_idx: int,
        best_result: CandidateResult,
    ) -> Path:
        """Save current optimization progress."""
        path = self.checkpoint_dir / f"{op_name}_{int(time.time() * 1000)}.json"

        payload = {
            "op_name": op_name,
            "state": state.to_dict(),
            "strategy": strategy,
            "max_candidates": max_candidates,
            "evaluated_count": len(evaluated),
            "candidates": [
                {"template_id": tid, "action": act.to_dict()}
                for tid, act in evaluated
            ],
            "results": [r.to_dict() for _, _, r in evaluated] if evaluated and len(evaluated[0]) > 2 else [],
            "best_idx": best_idx,
            "best_result": best_result.to_dict(),
            "timestamp": time.time(),
            "version": "1.0",
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        return path

    def load(self, op_name: str) -> Optional[dict]:
        """Load the latest checkpoint for an operator."""
        pattern = f"{op_name}_*.json"
        files = sorted(self.checkpoint_dir.glob(pattern), reverse=True)
        if not files:
            return None

        with open(files[0], encoding="utf-8") as f:
            return json.load(f)

    def list_checkpoints(self, op_name: str) -> list[Path]:
        """Return all checkpoint files for an operator, newest first."""
        return sorted(
            self.checkpoint_dir.glob(f"{op_name}_*.json"),
            reverse=True,
        )

    def cleanup(self, op_name: str, keep: int = 5) -> int:
        """Remove old checkpoints, keeping the most recent N."""
        files = self.list_checkpoints(op_name)
        removed = 0
        for f in files[keep:]:
            f.unlink()
            removed += 1
        return removed
