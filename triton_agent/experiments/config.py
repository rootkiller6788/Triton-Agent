"""Experiment configuration: defines benchmark matrices for all experiment levels.

Each experiment is a list of TrialSpec dicts. The runner iterates them,
runs the optimization pipeline, and collects results.
"""

from dataclasses import dataclass, field
from typing import Any


# ── Operator shape profiles ──────────────────────────────────────────

SMOKE_SHAPES = {
    "rmsnorm":          [{"B": 8,  "T": 2048, "D": 4096}],
    "rope":             [{"B": 4,  "H": 32,   "T": 2048, "D": 128}],
    "fused_bias_gelu":  [{"B": 8,  "T": 2048, "D": 4096}],
    "swiglu":           [{"B": 8,  "T": 2048, "D": 4096}],
    "layernorm":        [{"B": 8,  "T": 2048, "D": 4096}],
    "quant_dequant":    [{"B": 8,  "T": 2048, "D": 4096}],
    "kv_append":        [{"B": 4,  "H": 32,   "L": 4096, "T_new": 1, "D": 128}],
    "rope_kv_append":   [{"B": 4,  "H": 32,   "L": 4096, "T_new": 1, "D": 128}],
    "matmul_epilogue":  [{"B": 8,  "M": 256,  "K": 4096, "N": 4096}],
    "quant_matmul":     [{"B": 8,  "M": 256,  "K": 4096, "N": 4096}],
    "paged_kv":         [{"B": 4,  "H": 32,   "num_blocks": 256, "block_size": 16, "T_new": 1, "D": 128}],
    "paged_attention":  [{"B": 8,  "H": 32,   "D": 128,   "ctx_len": 512}],
    "flash_attn_like":  [{"B": 4,  "H": 32,   "T": 2048,  "S": 2048, "D": 64}],
}

CORE_SHAPES = {
    "rmsnorm": [
        {"B": 1,  "T": 1,    "D": 64},
        {"B": 8,  "T": 2048, "D": 4096},
        {"B": 32, "T": 2048, "D": 4096},
    ],
    "rope": [
        {"B": 1,  "H": 32, "T": 1024, "D": 128},
        {"B": 4,  "H": 32, "T": 2048, "D": 128},
        {"B": 32, "H": 32, "T": 4096, "D": 128},
    ],
    "fused_bias_gelu": [
        {"B": 1, "T": 1,    "D": 64},
        {"B": 8, "T": 2048, "D": 4096},
        {"B": 64,"T": 256,  "D": 4096},
    ],
    "swiglu": [
        {"B": 1, "T": 1,    "D": 64},
        {"B": 8, "T": 2048, "D": 4096},
        {"B": 32,"T": 2048, "D": 4096},
    ],
    "kv_append": [
        {"B": 1, "H": 32, "L": 2048, "T_new": 1,  "D": 128},
        {"B": 4, "H": 32, "L": 4096, "T_new": 8,  "D": 128},
        {"B": 8, "H": 32, "L": 8192, "T_new": 1,  "D": 128},
    ],
    "matmul_epilogue": [
        {"B": 1, "M": 64,   "K": 2048, "N": 2048},
        {"B": 4, "M": 128,  "K": 4096, "N": 4096},
        {"B": 8, "M": 256,  "K": 4096, "N": 4096},
    ],
}

ABLATION_OPS = ["rmsnorm", "rope", "fused_bias_gelu"]
ABLATION_SHAPES = [
    {"B": 1,  "T": 128,   "D": 512},
    {"B": 1,  "T": 1024,  "D": 2048},
    {"B": 8,  "T": 2048,  "D": 4096},
    {"B": 32, "T": 2048,  "D": 4096},
    {"B": 64, "T": 128,   "D": 4096},
]
ABLATION_DTYPES = ["fp16", "bf16"]

ALL_STRATEGIES = [
    "random",
    "grid",
    "epsilon",
    "ucb",
    "thompson",
    "reinforce",
    "grpo",
]

CORE_STRATEGIES = ["grid", "ucb", "thompson", "grpo"]
SMOKE_STRATEGIES = ["grid"]

BENCHMARK_ITERATIONS = {"smoke": (5, 20), "core": (10, 50), "ablation": (20, 100)}


@dataclass
class TrialSpec:
    op: str
    shape: dict[str, int]
    dtype: str
    device: str = "cuda"
    strategy: str = "grid"
    max_candidates: int = 64
    seed: int = 0
    warmup: int = 20
    repeat: int = 100
    level: str = "core"
    is_experimental: bool = False


def gen_smoke_matrix() -> list[TrialSpec]:
    trials = []
    for op, shapes in SMOKE_SHAPES.items():
        is_exp = op in ("paged_attention", "flash_attn_like")
        for shape in shapes:
            for dtype in ["fp16"]:
                for strategy in SMOKE_STRATEGIES:
                    w, r = BENCHMARK_ITERATIONS["smoke"]
                    trials.append(TrialSpec(
                        op=op, shape=shape, dtype=dtype, strategy=strategy,
                        max_candidates=32, warmup=w, repeat=r,
                        level="smoke", is_experimental=is_exp,
                    ))
    return trials


def gen_core_matrix() -> list[TrialSpec]:
    trials = []
    for op, shapes in CORE_SHAPES.items():
        for shape in shapes:
            for dtype in ["fp16"]:
                for strategy in CORE_STRATEGIES:
                    w, r = BENCHMARK_ITERATIONS["core"]
                    trials.append(TrialSpec(
                        op=op, shape=shape, dtype=dtype, strategy=strategy,
                        max_candidates=64, warmup=w, repeat=r,
                        level="core",
                    ))
    return trials


def gen_ablation_matrix() -> list[TrialSpec]:
    trials = []
    for op in ABLATION_OPS:
        for shape in ABLATION_SHAPES:
            if op == "rope" and "D" in shape and shape["D"] % 2 != 0:
                continue
            for dtype in ABLATION_DTYPES:
                for strategy in ALL_STRATEGIES:
                    for seed in range(3):
                        w, r = BENCHMARK_ITERATIONS["ablation"]
                        trials.append(TrialSpec(
                            op=op, shape=shape, dtype=dtype, strategy=strategy,
                            max_candidates=128, seed=seed, warmup=w, repeat=r,
                            level="ablation",
                        ))
    return trials


def gen_replay_matrix() -> list[TrialSpec]:
    """Generate trial specs for reproducibility: cold vs warm start."""
    trials = []
    shapes = [
        {"B": 8,  "T": 2048, "D": 4096},
        {"B": 8,  "T": 2048, "D": 5120},
    ]
    for op in ["rmsnorm"]:
        for dtype in ["fp16"]:
            for i, shape in enumerate(shapes):
                tag = "cold_start" if i == 0 else "warm_start"
                w, r = BENCHMARK_ITERATIONS["core"]
                trials.append(TrialSpec(
                    op=op, shape=shape, dtype=dtype, strategy="ucb",
                    max_candidates=64, warmup=w, repeat=r,
                    level="replay",
                ))
    return trials
