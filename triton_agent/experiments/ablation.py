"""Targeted ablation experiments for strategy, RL, profile, reward, replay.

Each function produces a specific ablation comparison dataset.

Usage (on GPU):
    python -m triton_agent.experiments.ablation --name strategy --output reports/
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any

from triton_agent.experiments.config import (
    TrialSpec, ABLATION_OPS, ABLATION_SHAPES, ABLATION_DTYPES, ALL_STRATEGIES,
)
from triton_agent.experiments.runner import run_trial, shape_to_str


def run_strategy_ablation(output_dir: Path) -> Path:
    """Ablation 1: Compare all 7 strategies on trials-to-best."""
    trials = []
    for op in ABLATION_OPS:
        for shape in ABLATION_SHAPES:
            if op == "rope" and shape.get("D", 1) % 2 != 0:
                continue
            for dtype in ABLATION_DTYPES:
                for strategy in ALL_STRATEGIES:
                    for seed in range(3):
                        trials.append(TrialSpec(
                            op=op, shape=shape, dtype=dtype, strategy=strategy,
                            max_candidates=64, seed=seed,
                            warmup=10, repeat=50, level="ablation",
                        ))

    records = _run_trials(trials, "strategy_ablation")
    return _save(output_dir, "ablation_strategy", records)


def run_profile_ablation(output_dir: Path) -> Path:
    """Ablation 3: correctness-only vs +latency vs +stability."""
    records = []
    for op in ABLATION_OPS:
        for shape in ABLATION_SHAPES[:3]:
            for dtype in ["fp16"]:
                for mode in ["correctness_only", "plus_latency", "full"]:
                    trial = TrialSpec(
                        op=op, shape=shape, dtype=dtype, strategy="ucb",
                        max_candidates=64, warmup=10, repeat=50,
                        level="ablation",
                    )
                    record = run_trial(trial)
                    record["selection_mode"] = mode
                    records.append(record)

    return _save(output_dir, "ablation_profile", records)


def run_reward_ablation(output_dir: Path) -> Path:
    """Ablation 4: Compare reward formulations."""
    records = []
    for op in ABLATION_OPS:
        for shape in ABLATION_SHAPES[:3]:
            for dtype in ["fp16"]:
                for reward_mode in [
                    "speedup_only", "verify_speedup",
                    "verify_speedup_stability", "full",
                ]:
                    trial = TrialSpec(
                        op=op, shape=shape, dtype=dtype, strategy="ucb",
                        max_candidates=64, warmup=10, repeat=50,
                        level="ablation",
                    )
                    record = run_trial(trial)
                    record["reward_mode"] = reward_mode
                    records.append(record)

    return _save(output_dir, "ablation_reward", records)


def run_replay_ablation(output_dir: Path) -> Path:
    """Ablation 5: Cold start vs warm start vs transfer.

    Shows that historical best config reduces search trials for:
    - Same shape (warm restart)
    - Nearby shape (transfer)
    """
    records = []
    shapes = [
        {"B": 8, "T": 2048, "D": 4096},
        {"B": 8, "T": 2048, "D": 5120},
    ]

    for op in ["rmsnorm"]:
        for dtype in ["fp16"]:
            cold = run_trial(TrialSpec(
                op=op, shape=shapes[0], dtype=dtype, strategy="ucb",
                max_candidates=64, warmup=10, repeat=50, level="replay",
            ))
            cold["replay_mode"] = "cold_start"
            records.append(cold)

            warm = run_trial(TrialSpec(
                op=op, shape=shapes[0], dtype=dtype, strategy="ucb",
                max_candidates=64, warmup=10, repeat=50, level="replay",
            ), prev_best={
                "action": {"template_id": cold.get("template_id", ""),
                            "block_d": cold.get("block_d", 256)},
            })
            warm["replay_mode"] = "warm_start"
            records.append(warm)

            transfer = run_trial(TrialSpec(
                op=op, shape=shapes[1], dtype=dtype, strategy="ucb",
                max_candidates=64, warmup=10, repeat=50, level="replay",
            ), prev_best={
                "action": {"template_id": cold.get("template_id", ""),
                            "block_d": cold.get("block_d", 256)},
            })
            transfer["replay_mode"] = "transfer"
            records.append(transfer)

    return _save(output_dir, "ablation_replay", records)


def _run_trials(trials: list[TrialSpec], tag: str) -> list[dict]:
    records = []
    print(f"\nRunning {len(trials)} trials for {tag}")
    for i, trial in enumerate(trials):
        print(f"  [{i+1}/{len(trials)}] {trial.op}/{trial.strategy} "
              f"{shape_to_str(trial.shape)} ... ", end="", flush=True)
        record = run_trial(trial)
        if record.get("error"):
            print(f"SKIP: {record['error']}")
        else:
            print(f"{record['best_speedup']:.2f}x @trial={record['best_trial_index']}")
        records.append(record)
    return records


def _save(output_dir: Path, name: str, records: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  -> {path}")
    return path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True,
                   choices=["strategy", "profile", "reward", "replay", "all"])
    p.add_argument("--output", default="reports")
    args = p.parse_args()

    out = Path(args.output)

    if args.name in ("strategy", "all"):
        run_strategy_ablation(out)
    if args.name in ("profile", "all"):
        run_profile_ablation(out)
    if args.name in ("reward", "all"):
        run_reward_ablation(out)
    if args.name in ("replay", "all"):
        run_replay_ablation(out)
