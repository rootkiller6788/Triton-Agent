"""Episode replay, comparison, and rollback."""

import json
from pathlib import Path
from typing import Any


def replay_episode(episode_path: str | Path) -> dict | None:
    """Load and return an episode's data for replay."""
    record_path = Path(episode_path) / "record.json"
    if not record_path.exists():
        return None
    with open(record_path, encoding="utf-8") as f:
        return json.load(f)


def compare_episodes(episode_a: dict, episode_b: dict) -> dict[str, Any]:
    """Compare two episodes and return summary differences."""
    ra = episode_a.get("result", {})
    rb = episode_b.get("result", {})

    return {
        "a_latency_p50": ra.get("latency_us_p50"),
        "b_latency_p50": rb.get("latency_us_p50"),
        "a_speedup": ra.get("speedup"),
        "b_speedup": rb.get("speedup"),
        "a_reward": ra.get("reward"),
        "b_reward": rb.get("reward"),
        "a_promoted": ra.get("promoted"),
        "b_promoted": rb.get("promoted"),
        "winner": "a" if (ra.get("reward") or 0) >= (rb.get("reward") or 0) else "b",
    }


def rollback_to_safe(
    store,
    op_name: str,
    shape: str,
    dtype: str,
    device: str = "cuda",
) -> dict | None:
    """Find the last promoted safe config for rollback.

    Returns the best known promoted config, or None.
    """
    rows = store.query(op_name, shape=shape, dtype=dtype, limit=50)
    for row in rows:
        if row.get("promoted"):
            return row
    return None
