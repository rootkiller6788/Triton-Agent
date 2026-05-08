"""Experiment runner: executes trial matrices, collects results, saves JSON.

Usage (on GPU machine):
    python -m triton_agent.experiments.runner --level smoke --output reports/
    python -m triton_agent.experiments.runner --level core
    python -m triton_agent.experiments.runner --level ablation
    python -m triton_agent.experiments.runner --level replay
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

from triton_agent.experiments.config import (
    TrialSpec,
    gen_smoke_matrix,
    gen_core_matrix,
    gen_ablation_matrix,
    gen_replay_matrix,
)


OUTPUT_BASE = Path("reports")


def shape_to_str(shape: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(shape.items()))


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _load_templates_and_ref(op: str) -> dict:
    from pathlib import Path as P
    from triton_agent.cli import _load_templates, _load_reference

    ops_dir = P(__file__).parent.parent / "ops"
    templates = _load_templates(op, ops_dir)
    ref_fn, gen_fn = _load_reference(op, ops_dir)
    return {"templates": templates, "ref_fn": ref_fn, "gen_fn": gen_fn, "ops_dir": ops_dir}


def run_trial(trial: TrialSpec, prev_best: dict | None = None) -> dict[str, Any]:
    """Run a single trial. Returns a result record dict."""
    start_time = time.time()

    assets = _load_templates_and_ref(trial.op)
    templates = assets["templates"]
    ref_fn = assets["ref_fn"]
    gen_fn = assets["gen_fn"]

    if not _gpu_available():
        return {
            "trial": trial.__dict__,
            "error": "GPU not available",
            "duration_s": time.time() - start_time,
        }

    import torch
    from triton_agent.core.contract import OpContract
    from triton_agent.core.spec import OpState, OpAction, CandidateResult
    from triton_agent.core.registry import get_registry
    from triton_agent.core.compiler import compile_kernel
    from triton_agent.core.verifier import check_numerical
    from triton_agent.core.profiler import profile_kernel
    from triton_agent.core.reward import compute_score
    from triton_agent.agent.generator import generate_grid, generate_best_of_n
    from triton_agent.agent.selector import select_best
    from triton_agent.agent.promoter import should_promote
    from triton_agent.agent.repairer import repair_action, should_retry
    from triton_agent.microrl.trainer import Trainer

    ops_dir = assets["ops_dir"]
    contract_file = ops_dir / trial.op / "contract.yaml"
    contract = OpContract.from_yaml(contract_file)

    shape = trial.shape
    state = OpState(
        op_name=trial.op,
        B=shape.get("B", 8),
        T=shape.get("T", 2048),
        D=shape.get("D", 4096),
        dtype=trial.dtype,
        device=trial.device,
    )

    if prev_best and prev_best.get("action"):
        state.historical_best_config = prev_best["action"]

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(trial.dtype, torch.float16)

    inputs = gen_fn(**{**shape, "dtype": torch_dtype, "device": trial.device})
    ref_output = ref_fn(*inputs)

    from triton_agent.cli import _measure_baseline
    baseline_us = _measure_baseline(ref_fn, inputs)

    torch.cuda.reset_peak_memory_stats()

    is_bandit = trial.strategy in ("ucb", "thompson", "epsilon",
                                    "reinforce", "grpo", "epsilon_greedy")

    if trial.strategy == "random":
        candidates = generate_best_of_n(
            contract, state, templates,
            n=trial.max_candidates, seed=trial.seed,
        )
    elif trial.strategy == "grid":
        candidates = generate_grid(contract, state, templates)
    else:
        candidates = generate_grid(contract, state, templates)
        if len(candidates) > trial.max_candidates:
            candidates = candidates[:trial.max_candidates]

    trainer = Trainer(strategy=trial.strategy) if is_bandit else None

    results: list[CandidateResult] = []
    compile_count = 0
    best_trial_idx = 0
    best_speedup = 0.0

    for i, (tid, action) in enumerate(candidates):
        if is_bandit and trainer:
            idx = trainer.select(candidates, state)
            tid, action = candidates[idx]
        elif is_bandit:
            pass

        compile_args = _build_compile_args(state, action, trial.op)
        comp = compile_kernel(templates.get(tid), compile_args)
        compile_count += 1

        if not comp.success:
            for attempt in range(3):
                repaired = repair_action(action, CandidateResult(
                    compile_pass=False, compile_log=comp.error_log,
                ))
                if repaired and should_retry(repaired, action,
                    CandidateResult(compile_pass=False, verify_pass=False)):
                    action = repaired
                    comp = compile_kernel(templates.get(tid), compile_args)
                    compile_count += 1
                    if comp.success:
                        break
                else:
                    break

        if not comp.success:
            r = CandidateResult(compile_pass=False, compile_log=comp.error_log)
            results.append(r)
            continue

        try:
            triton_out = comp.kernel_fn(*inputs)
        except Exception as e:
            r = CandidateResult(compile_pass=True, verify_pass=False, compile_log=str(e))
            results.append(r)
            continue

        r = check_numerical(triton_out, ref_output, contract)
        if not r.verify_pass:
            results.append(r)
            continue

        r = profile_kernel(
            comp.kernel_fn, inputs,
            warmup=trial.warmup, repeat=trial.repeat,
            baseline_latency_us=baseline_us,
        )
        compute_score(r)

        if is_bandit and trainer:
            trainer.update(action, state, r, candidates)

        results.append(r)

        if r.speedup > best_speedup:
            best_speedup = r.speedup
            best_trial_idx = i

    best_result = results[best_trial_idx] if results else CandidateResult()
    duration = time.time() - start_time

    record = {
        "trial": trial.__dict__,
        "baseline_latency_us": baseline_us,
        "best_speedup": best_speedup,
        "best_latency_p50": best_result.latency_us_p50,
        "best_latency_p90": best_result.latency_us_p90,
        "best_latency_p99": best_result.latency_us_p99,
        "best_trial_index": best_trial_idx,
        "num_trials": len(results),
        "compile_count": compile_count,
        "verify_pass": best_result.verify_pass,
        "memory_peak_mb": best_result.memory_peak_mb,
        "best_reward": best_result.reward,
        "best_promoted": best_result.promoted,
        "duration_s": duration,
        "gpu_name": torch.cuda.get_device_name(0) if _gpu_available() else "N/A",
        "timestamp": datetime.now().isoformat(),
    }
    return record


def _build_compile_args(state, action, op_name: str) -> dict:
    args = {
        "B": state.B, "T": state.T, "D": state.D,
        "BLOCK_SIZE": action.block_d, "NUM_WARPS": action.num_warps,
        "NUM_STAGES": action.num_stages,
    }
    if "rope" in op_name.lower():
        args["H"] = getattr(state, "H", 32)
    return args


def run_level(level: str, output_dir: str | Path = "reports") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_map = {
        "smoke": gen_smoke_matrix,
        "core": gen_core_matrix,
        "ablation": gen_ablation_matrix,
        "replay": gen_replay_matrix,
    }

    gen_fn = matrix_map.get(level)
    if gen_fn is None:
        print(f"Unknown level: {level}. Options: {list(matrix_map.keys())}")
        return output_dir

    trials = gen_fn()
    records = []
    prev_best: dict | None = None

    print(f"Running {len(trials)} trials at level '{level}'")
    for i, trial in enumerate(trials):
        tag = f"[{i + 1}/{len(trials)}] {trial.op} {trial.strategy} {shape_to_str(trial.shape)}"
        print(f"  {tag} ... ", end="", flush=True)
        record = run_trial(trial, prev_best=prev_best if level == "replay" else None)

        if record.get("error"):
            print(f"SKIP: {record['error']}")
        else:
            print(f"speedup={record['best_speedup']:.2f}x trials={record['num_trials']}")

        records.append(record)

        if level == "replay" and record.get("best_speedup", 0) > 0:
            prev_best = {
                "action": {
                    "template_id": records[-1].get("template_id", ""),
                    "block_d": records[-1].get("block_d", 0),
                },
            }

    out_path = output_dir / f"{level}_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump({"level": level, "records": records, "gpu": _gpu_name()}, f, indent=2)

    print(f"Saved {len(records)} records to {out_path}")
    return out_path


def _gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return "N/A"


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="smoke", choices=["smoke", "core", "ablation", "replay"])
    p.add_argument("--output", default="reports")
    args = p.parse_args()
    run_level(args.level, args.output)
