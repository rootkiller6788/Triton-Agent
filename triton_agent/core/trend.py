"""Leaderboard history: tracks score/speedup trends over time.

Queries the leaderboard's timestamp field to compute improvement trends,
detect stagnation, and generate summary reports.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from triton_agent.core.storage import LeaderboardStore


@dataclass
class TrendPoint:
    timestamp: float
    speedup: float
    reward: float
    promoted: bool
    template_id: str


@dataclass
class TrendReport:
    op_name: str
    points: list[TrendPoint] = field(default_factory=list)
    best_speedup: float = 1.0
    best_reward: float = 0.0
    total_episodes: int = 0
    promoted_count: int = 0
    is_improving: bool = True
    stagnation_count: int = 0


def compute_trend(
    leaderboard: LeaderboardStore,
    op_name: str,
    shape: str | None = None,
    dtype: str | None = None,
    window: int = 10,
) -> TrendReport:
    """Compute improvement trend from leaderboard history.

    Args:
        leaderboard: LeaderboardStore instance
        op_name: operator name
        shape: optional shape filter
        dtype: optional dtype filter
        window: number of recent entries to check for stagnation

    Returns:
        TrendReport with trend analysis.
    """
    rows = leaderboard.query(op_name, shape=shape, dtype=dtype, limit=100)
    report = TrendReport(op_name=op_name)

    if not rows:
        return report

    for row in reversed(rows):
        report.points.append(TrendPoint(
            timestamp=row.get("timestamp", 0),
            speedup=row.get("speedup", 1.0),
            reward=row.get("reward", 0.0),
            promoted=bool(row.get("promoted", False)),
            template_id=row.get("template_id", ""),
        ))

    speedups = [p.speedup for p in report.points if p.promoted]
    report.best_speedup = max(speedups) if speedups else max(
        (p.speedup for p in report.points), default=1.0
    )

    rewards = [p.reward for p in report.points]
    report.best_reward = max(rewards) if rewards else 0.0

    report.total_episodes = len(report.points)
    report.promoted_count = sum(1 for p in report.points if p.promoted)

    recent = report.points[-window:] if len(report.points) >= window else report.points
    if len(recent) >= 3:
        speeds = [p.speedup for p in recent if p.promoted]
        if not speeds:
            speeds = [p.speedup for p in recent]
        if speeds:
            report.is_improving = speeds[-1] >= max(speeds[:-1]) * 0.99

    report.stagnation_count = sum(
        1 for i in range(1, len(recent))
        if abs(recent[i].speedup - recent[i - 1].speedup) < 0.01
    )

    return report


def format_trend_report(report: TrendReport) -> str:
    """Format a trend report as a human-readable string."""
    lines = [
        f"Trend Report: {report.op_name}",
        f"  Total Episodes: {report.total_episodes}",
        f"  Promoted: {report.promoted_count}",
        f"  Best Speedup: {report.best_speedup:.2f}x",
        f"  Best Reward: {report.best_reward:.2f}",
        f"  Improving: {'yes' if report.is_improving else 'STAGNANT'}",
        f"  Stagnation Count: {report.stagnation_count}",
    ]
    if report.points:
        latest = report.points[-1]
        lines.append(
            f"  Latest: {latest.template_id} "
            f"speedup={latest.speedup:.2f}x reward={latest.reward:.2f}"
        )
    return "\n".join(lines)
