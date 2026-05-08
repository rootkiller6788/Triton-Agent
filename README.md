# Triton-agent / Apeinx OpEvolver

A lightweight operator self-optimization agent for low-cost, high-quality Triton kernel development.

```text
Triton-agent = 算子注册表 + 算子契约 + PyTorch 正确性基线
             + Triton 候选变体生成 + 编译 + 验证 + 性能 Profile
             + MicroRL/Bandit 调优 + 晋升/回滚 + Evidence Replay
```

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux / WSL
# venv\Scripts\activate       # Windows

# 2. Install core (no GPU needed for development)
pip install -e ".[dev]"

# 3. Install with GPU support (Linux/WSL with NVIDIA GPU only)
pip install -e ".[gpu]"

# 4. Run unit tests to verify install
pytest test/ -q
```

---

## CLI Commands

### Operator workflow

| Command | Description |
|---------|-------------|
| `triton-agent init <op>` | Register an operator from a contract.yaml directory |
| `triton-agent optimize <op> --shape B=8,T=2048,D=4096 --dtype fp16` | Run full optimization pipeline |
| `triton-agent verify-correctness <op> --shape B=8,T=2048,D=4096` | Check kernel output against PyTorch reference |

### Inspection

| Command | Description |
|---------|-------------|
| `triton-agent leaderboard <op>` | View best configs for an operator |
| `triton-agent replay <episode_path>` | Replay a past optimization episode |
| `triton-agent compare <op> --baseline torch --variant best` | Compare best variant vs baseline |
| `triton-agent cross-compare [ops...]` | Compare best configs across multiple operators |

### Experiments

| Command | Description |
|---------|-------------|
| `triton-agent experiment run --level smoke` | 13 ops, 1 shape each, correctness + latency |
| `triton-agent experiment run --level core` | 6 ops x 3 shapes x 4 strategies |
| `triton-agent experiment run --level ablation` | 3 ops x 5 shapes x 7 strategies x 3 seeds |
| `triton-agent experiment run --level replay` | Cold start vs warm start reproducibility |
| `triton-agent experiment ablation --name all` | 5 targeted ablations (strategy / profile / reward / replay) |
| `triton-agent experiment report reports/<file>.json` | Generate Markdown report from benchmark JSON |

### Options

| Flag | Values | Default |
|------|--------|---------|
| `--strategy` | grid, best_of_n, ucb, thompson, epsilon, reinforce, grpo | auto (based on search space size) |
| `--max-candidates` | 1–1024 | 64 |
| `--retry` | 0–10 | 3 |
| `--shape` | e.g. `B=8,T=2048,D=4096` | B=8,T=2048,D=4096 |
| `--dtype` | fp16, bf16, fp32 | fp16 |
| `--device` | cuda | cuda |

---

## Experiment Workflow

Run these **on a Linux machine with NVIDIA GPU**.

### Level 1 — Smoke (correctness gate)

```bash
triton-agent experiment run --level smoke
triton-agent experiment report reports/smoke_benchmark_*.json
```

Verifies all 13 operators pass correctness checks. Output: `reports/smoke_benchmark.md`.

### Level 2 — Core (performance)

```bash
triton-agent experiment run --level core
triton-agent experiment report reports/core_benchmark_*.json
```

6 core operators x 3 shapes x 4 strategies. Output: `reports/core_benchmark.md`.

### Level 3 — Ablation (evidence)

```bash
triton-agent experiment run --level ablation
triton-agent experiment report reports/ablation_benchmark_*.json

# Or targeted ablations:
triton-agent experiment ablation --name strategy    # Which strategy needs fewest trials?
triton-agent experiment ablation --name profile     # Correctness-only vs +latency vs full
triton-agent experiment ablation --name reward      # Reward function component ablation
triton-agent experiment ablation --name replay      # Cold start vs warm start vs transfer
```

Proves: strategy efficiency, MicroRL value, profile-guided selection, reward design, and reproducibility.

---

## Operators

### P0 — Core closed loop
| Operator | Description |
|----------|-------------|
| `rmsnorm` | Root Mean Square Layer Normalization |
| `rope` | Rotary Position Embedding |
| `fused_bias_gelu` | Fused bias + GELU activation |

### P1 — Extended operators
| Operator | Description |
|----------|-------------|
| `swiglu` | SiLU-gated linear unit |
| `quant_dequant` | Quantize-dequantize (INT8 simulation) |
| `layernorm` | Standard Layer Normalization |

### P2 — LLM decode hot path
| Operator | Description |
|----------|-------------|
| `kv_append` | KV cache token append |
| `rope_kv_append` | Fused RoPE + KV cache write |
| `matmul_epilogue` | Matmul + bias + GELU fusion |

### P3 — High-difficulty kernels
| Operator | Description | Note |
|----------|-------------|------|
| `quant_matmul` | INT8 quantized matrix multiplication | |
| `paged_kv` | Paged KV cache (vLLM-style block tables) | |
| `paged_attention` | PagedAttention decode kernel | experimental |
| `flash_attn_like` | FlashAttention-like tiled attention | experimental |

---

## Search Strategies

| Strategy | Type | When to use |
|----------|------|-------------|
| `grid` | Exhaustive | Search space < 64 combos |
| `best_of_n` | Random sampling | Quick exploration |
| `ucb` | Bandit | Balanced explore/exploit |
| `thompson` | Bandit | Noisy reward environments |
| `epsilon` | Bandit | Simple adaptive baseline |
| `reinforce` | Policy gradient | Medium/large search spaces |
| `grpo` | Policy gradient | Large search spaces, group-relative advantage |

---

## Architecture

```text
triton_agent/
├── cli.py                 # CLI entry point (click)
├── agent/                 # Agent loop
│   ├── planner.py         # Strategy dispatch (grid / bandit / RL)
│   ├── generator.py       # Candidate generation (grid / best-of-N / bandit)
│   ├── repairer.py        # Auto-repair compile/verify failures
│   ├── selector.py        # Best-of-N selection by reward
│   └── promoter.py        # Promotion gate (min_speedup + max_variance)
├── core/                  # Infrastructure
│   ├── contract.py        # YAML contract parser + 8 dataclasses
│   ├── spec.py            # OpState / OpAction / CandidateResult
│   ├── registry.py        # Operator registry singleton
│   ├── compiler.py        # Triton JIT compile wrapper
│   ├── cuda_ext.py        # CUDA Extension fallback compiler
│   ├── compile_cache.py   # SQLite compile result cache
│   ├── verifier.py        # Numerical correctness checker
│   ├── profiler.py        # Latency p50/p90/p99 profiler
│   ├── adaptive_profile.py # Adaptive warmup + early-stop profiler
│   ├── reward.py          # 6-factor reward scoring
│   ├── storage.py         # EpisodeStore (JSONL) + LeaderboardStore (SQLite)
│   ├── replay.py          # Replay / compare / rollback
│   ├── eval_engine.py     # Parallel eval + regression detection
│   ├── checkpoint.py      # Save/resume optimization state
│   └── trend.py           # Leaderboard trend analysis
├── microrl/               # Lightweight RL layer
│   ├── bandit.py           # UCB / Thompson / Epsilon-Greedy
│   ├── reinforce_lite.py   # REINFORCE policy gradient
│   ├── grpo_lite.py        # Group Relative Policy Optimization
│   └── trainer.py          # Multi-strategy trainer + ShapeConfigStore
├── ops/                   # 13 operator families (contract + ref + templates + verify + bench)
├── experiments/            # Benchmark framework
│   ├── config.py           # Experiment matrix definitions
│   ├── runner.py           # Trial execution engine
│   ├── ablation.py         # Targeted ablation scripts
│   └── report.py           # Markdown report generator
├── integrations/           # PyTorch / torch.compile / vLLM / SGLang adapters
├── benchmarks/             # Multi-shape benchmark suites
└── test/                   # 95 unit + integration tests
```

---

## Adding a New Operator

1. Create directory `triton_agent/ops/<name>/templates/`
2. Write `contract.yaml` — define inputs/outputs/tolerance/search_space
3. Write `reference.py` — PyTorch reference implementation + `generate_test_inputs()`
4. Write at least one Triton kernel template in `templates/triton_v1.py`
5. Write `verify.py` — compare against reference
6. Write `benchmark.py` — measure latency

The agent auto-discovers operators under `ops/`. No registration code needed.

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest test/ -v

# Run a single test file
pytest test/test_microrl.py -v

# Coverage
pytest test/ --cov=triton_agent --cov-report=html
```

All Python libraries use the project's `venv`. GPU dependencies (`torch`, `triton`) are optional extras and only install on Linux.

---

## Requirements

| Environment | Dependencies |
|-------------|-------------|
| Development (any OS) | Python >= 3.10, click, pyyaml, numpy, rich, pytest |
| GPU (Linux only) | torch >= 2.0, triton >= 2.1 |
| CUDA fallback | CUDA Toolkit, nvcc |

---

## License

MIT
