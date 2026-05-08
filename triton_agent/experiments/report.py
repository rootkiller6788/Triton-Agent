"""Experiment report generator: reads benchmark JSON, produces markdown tables.

Generates:
    reports/smoke_benchmark.md
    reports/core_benchmark.md
    reports/ablation_study.md
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Any


def _load_records(json_path: Path) -> list[dict]:
    with open(json_path) as f:
        return json.load(f).get("records", [])


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    hdr = "| " + " | ".join(headers) + " |\n"
    sep = "|" + "|".join(["---" for _ in headers]) + "|\n"
    body = "".join("| " + " | ".join(str(c) for c in row) + " |\n" for row in rows)
    return hdr + sep + body


def _fmt(x: Any, prec: int = 2) -> str:
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def gen_smoke_report(records: list[dict]) -> str:
    lines = [
        "# Smoke Benchmark Report",
        "",
        f"**Generated**: {records[0].get('timestamp', 'N/A') if records else 'N/A'}",
        f"**GPU**: {records[0].get('gpu_name', 'N/A') if records else 'N/A'}",
        "",
        "## Correctness & Latency Summary",
        "",
        "All 13 operators, single shape, grid strategy, fp16.",
        "",
    ]

    headers = ["Operator", "Shape", "Verify", "Latency(p50)", "Speedup", "Memory(MB)", "Status"]
    rows = []
    for r in records:
        t = r.get("trial", {})
        shape_str = ",".join(f"{k}={v}" for k, v in sorted(t.get("shape", {}).items()))
        status = "EXP" if t.get("is_experimental") else "OK"
        rows.append([
            t.get("op", ""),
            shape_str,
            "PASS" if r.get("verify_pass") else "FAIL",
            f"{r.get('best_latency_p50', 0):.1f} us",
            f"{r.get('best_speedup', 0):.2f}x",
            f"{r.get('memory_peak_mb', 0):.0f}",
            status,
        ])

    lines.append(_md_table(headers, rows))

    exp_ops = [r for r in records if r.get("trial", {}).get("is_experimental")]
    if exp_ops:
        lines.extend([
            "",
            "## Experimental Operators",
            "",
            "PagedAttention and FlashAttention-like are marked experimental. "
            "They are not expected to outperform industrial implementations "
            "(vLLM, FlashAttention-2) but serve as validation targets.",
        ])

    return "\n".join(lines)


def gen_core_report(records: list[dict]) -> str:
    lines = [
        "# Core Benchmark Report",
        "",
        f"**GPU**: {records[0].get('gpu_name', 'N/A') if records else 'N/A'}",
        "",
        "## Performance Summary (6 ops x 3 shapes x 4 strategies)",
        "",
    ]

    headers = ["Op", "Shape", "Strategy", "Speedup", "Latency(p50)", "p90", "Trials", "Best@Trial"]
    rows = []
    for r in records:
        t = r.get("trial", {})
        shape_str = ",".join(f"{k}={v}" for k, v in sorted(t.get("shape", {}).items()))
        rows.append([
            t.get("op", ""),
            shape_str,
            t.get("strategy", ""),
            f"{r.get('best_speedup', 0):.2f}x",
            f"{r.get('best_latency_p50', 0):.1f} us",
            f"{r.get('best_latency_p90', 0):.1f} us",
            str(r.get("num_trials", 0)),
            str(r.get("best_trial_index", 0)),
        ])

    lines.append(_md_table(headers, rows))

    lines.extend([
        "",
        "## Baseline Comparison",
        "",
        "Each operator is compared against:",
        "1. PyTorch eager (baseline = 1.0x)",
        "2. torch.compile (reported when stable)",
        "3. Naive Triton (fixed config, BLOCK_SIZE=128, num_warps=4)",
        "4. Triton-agent best (this benchmark)",
        "",
    ])

    by_op = defaultdict(list)
    for r in records:
        t = r.get("trial", {})
        by_op[t.get("op", "")].append(r)

    for op, recs in sorted(by_op.items()):
        best = max(recs, key=lambda x: x.get("best_speedup", 0))
        lines.extend([
            f"### {op}",
            f"- Best speedup: **{best['best_speedup']:.2f}x** "
            f"(strategy={best['trial']['strategy']}, "
            f"latency_p50={best['best_latency_p50']:.1f}us)",
        ])

    return "\n".join(lines)


def gen_ablation_report(records: list[dict]) -> str:
    lines = [
        "# Ablation Study Report",
        "",
        f"**GPU**: {records[0].get('gpu_name', 'N/A') if records else 'N/A'}",
        "",
        "## 1. Strategy Ablation: Trials to Best Speedup",
        "",
        "7 strategies across 3 ops x 5 shapes x 2 dtypes x 3 seeds.",
        "",
    ]

    headers = ["Op", "Strategy", "Best Speedup", "Trials→Best", "Tuning Time(s)", "Stability"]
    rows = []
    by_key = defaultdict(list)
    for r in records:
        t = r.get("trial", {})
        key = (t.get("op", ""), t.get("strategy", ""))
        by_key[key].append(r)

    for (op, strat), recs in sorted(by_key.items()):
        avg_speedup = sum(r.get("best_speedup", 0) for r in recs) / len(recs)
        avg_trials = sum(r.get("best_trial_index", 0) for r in recs) / len(recs)
        avg_dur = sum(r.get("duration_s", 0) for r in recs) / len(recs)
        variances = [r.get("best_speedup", 0) for r in recs if r.get("best_speedup", 0) > 0]
        stability = "high" if len(variances) > 1 and max(variances) - min(variances) < 0.05 else "medium"
        rows.append([
            op, strat,
            f"{avg_speedup:.2f}x",
            f"{avg_trials:.0f}",
            f"{avg_dur:.0f}",
            stability,
        ])

    lines.append(_md_table(headers, rows))

    lines.extend([
        "",
        "## 2. MicroRL Benefit (vs Grid)",
        "",
        "Comparing bandit/RL strategies against grid search.",
    ])

    lines.extend([
        "",
        "## 3. Profile-Guided Selection",
        "",
        "Correctness-only vs Correctness+Latency vs Full (latency+stability+memory).",
        "",
        "| Op | Correctness Only | +Latency | +Stability+Memory |",
        "|----|-----------------|----------|-------------------|",
    ])

    lines.extend([
        "",
        "## 4. Reward Component Ablation",
        "",
        "Comparing reward formulations:",
        "- speedup only",
        "- verify + speedup",
        "- verify + speedup + stability",
        "- verify + speedup + stability + memory",
    ])

    lines.extend([
        "",
        "## 5. Replay / Warm Start Benefit",
        "",
        "Cold start vs warm start using historical best config.",
        "",
        "| Op | Cold Trials | Warm Trials | Reduction |",
        "|----|-------------|-------------|-----------|",
    ])

    return "\n".join(lines)


def generate_report(json_path: str | Path, output_dir: str | Path = "reports") -> Path:
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(json_path)
    if not records:
        raise ValueError("No records found in benchmark JSON")

    level = records[0].get("trial", {}).get("level", "core")

    generators = {
        "smoke": gen_smoke_report,
        "core": gen_core_report,
        "ablation": gen_ablation_report,
        "replay": gen_core_report,
    }

    gen = generators.get(level, gen_core_report)
    md_content = gen(records)

    out_path = output_dir / f"{level}_benchmark.md"
    out_path.write_text(md_content, encoding="utf-8")
    print(f"Report written to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("json_path", help="Path to benchmark JSON file")
    p.add_argument("--output", default="reports")
    args = p.parse_args()
    generate_report(args.json_path, args.output)
