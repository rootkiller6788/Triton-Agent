"""CLI entry point for triton-agent."""

import json
import time
from pathlib import Path
from typing import Any

import click

from triton_agent.core.registry import get_registry
from triton_agent.core.contract import OpContract
from triton_agent.core.spec import OpState
from triton_agent.core.storage import EpisodeStore, LeaderboardStore
from triton_agent.core.replay import replay_episode, compare_episodes


@click.group()
@click.version_option()
def main():
    """Triton-agent: lightweight operator self-optimization agent.

    Low-cost, high-quality Triton kernel development through
    automated variant generation, verification, profiling, and tuning.
    """


@main.command()
@click.argument("op")
def init(op: str):
    """Initialize a new operator from a contract.yaml file.

    OP: Path to the operator directory containing contract.yaml.
    """
    path = Path(op)
    contract_file = path / "contract.yaml" if path.is_dir() else path

    if not contract_file.exists():
        click.echo(
            f"Contract file not found: {contract_file}\n"
            f"Run 'triton-agent init --help' for usage."
        )
        return

    try:
        contract = OpContract.from_yaml(contract_file)
        registry = get_registry()
        registry.register(contract)
        click.echo(f"Registered operator: {contract.op}")
        click.echo(f"  Inputs: {[(i.name, i.shape.dims) for i in contract.inputs]}")
        click.echo(f"  Outputs: {[(o.name, o.shape.dims) for o in contract.outputs]}")
        click.echo(f"  Dtypes: {contract.dtype}")
        click.echo(f"  Devices: {contract.device}")
    except Exception as e:
        click.echo(f"Failed to load contract: {e}", err=True)


@main.command()
@click.argument("op")
@click.option("--shape", default="B=8,T=2048,D=4096", help="Shape specification, e.g. B=8,T=2048,D=4096")
@click.option("--dtype", default="fp16", help="Data type (fp16, bf16, fp32)")
@click.option("--device", default="cuda", help="Target device")
@click.option("--strategy", default=None, help="Search strategy: grid, best_of_n, ucb, thompson, epsilon, reinforce, grpo")
@click.option("--max-candidates", default=64, help="Maximum candidates to evaluate")
@click.option("--retry", default=3, help="Max repair retries per candidate")
def optimize(op: str, shape: str, dtype: str, device: str, strategy: str, max_candidates: int, retry: int):
    """Optimize an operator for a specific shape.

    OP: Registered operator name or path to operator directory.
    """
    import importlib
    import sys

    ops_dir = Path(__file__).parent / "ops"
    registry = get_registry()
    _ensure_op_registered(op, ops_dir)
    if not registry.has(op):
        click.echo(f"Operator '{op}' not found. Check ops/ directory or run 'triton-agent init'.", err=True)
        return

    contract = registry.get(op)

    try:
        parsed = {}
        for kv in shape.split(","):
            k, v = kv.strip().split("=")
            parsed[k.strip()] = int(v.strip())
    except Exception:
        click.echo(f"Invalid shape format: {shape}. Use B=8,T=2048,D=4096", err=True)
        return

    click.echo(f"Optimizing {op} for shape {parsed} dtype={dtype} device={device}")

    B = parsed.get("B", 8)
    T = parsed.get("T", 2048)
    D = parsed.get("D", 4096)
    H = parsed.get("H", 32)

    state = OpState(
        op_name=op,
        B=B, T=T, D=D,
        dtype=dtype,
        device=device,
    )

    click.echo("  [1/6] Loading templates and reference...")
    templates = _load_templates(op, ops_dir)
    if not templates:
        click.echo(f"  No Triton templates found for {op}.", err=True)
        return

    ref_fn, gen_inputs_fn = _load_reference(op, ops_dir)
    is_simulated = (
        isinstance(ref_fn, dict)
        or isinstance(gen_inputs_fn, dict)
    )
    if ref_fn is None and not is_simulated:
        click.echo(f"  No reference implementation found for {op}.", err=True)
        return

    click.echo(f"  Templates: {list(templates.keys())}")
    if is_simulated:
        click.echo("  Running in simulation mode (no GPU/torch available).")

    from triton_agent.agent.generator import generate_grid, generate_best_of_n
    from triton_agent.agent.planner import plan

    plan_strategy = plan(contract, state)
    effective_strategy = strategy or plan_strategy.get("strategy", "grid")

    bandit_strategies = {"ucb", "thompson", "epsilon", "epsilon_greedy", "reinforce", "reinforce_lite", "grpo", "grpo_lite"}

    if effective_strategy in bandit_strategies:
        candidates = _run_bandit_optimize(
            op, contract, state, templates, ref_fn, gen_inputs_fn, ops_dir,
            effective_strategy, max_candidates, retry, is_simulated,
        )
        if isinstance(candidates, dict) and candidates.get("done"):
            return

    if effective_strategy == "grid":
        candidates = generate_grid(contract, state, templates)
    else:
        candidates = generate_best_of_n(contract, state, templates, n=max_candidates)

    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]

    click.echo(f"  [2/6] Generated {len(candidates)} candidates ({effective_strategy})")

    click.echo("  [3/6] Compiling and evaluating candidates...")
    results = _evaluate_candidates(
        op, contract, state, candidates, templates,
        ref_fn, gen_inputs_fn, ops_dir,
    )

    click.echo("  [4/6] Computing rewards and selecting best...")
    from triton_agent.core.reward import compute_score
    from triton_agent.agent.selector import select_best
    from triton_agent.agent.promoter import should_promote

    for r in results:
        compute_score(r)

    best_idx, best_result = select_best(candidates, results)
    best_template, best_action = candidates[best_idx]
    result = best_result

    result.promoted = should_promote(result, contract)

    click.echo("  [5/6] Saving episode and updating leaderboard...")
    episode_store = EpisodeStore()
    leaderboard_store = LeaderboardStore()

    episode_id = f"{int(time.time())}_{best_template}"
    episode_store.save(op, episode_id, state, best_action, result)

    shape_key = f"B={B},T={T},D={D}"
    leaderboard_store.upsert(
        op, shape_key, dtype, device,
        best_action, result, episode_id,
    )

    click.echo("  [6/6] Results:")
    click.echo(f"    Best Variant: {best_template}")
    click.echo(f"    Verify: {'passed' if result.verify_pass else 'FAILED'}")
    click.echo(f"    Latency p50: {result.latency_us_p50:.1f} us")
    click.echo(f"    Latency p90: {result.latency_us_p90:.1f} us")
    click.echo(f"    Speedup: {result.speedup:.2f}x")
    click.echo(f"    Promoted: {result.promoted}")
    click.echo(f"    Episode: episodes/{op}/{episode_id}")


def _evaluate_candidates(
    op: str,
    contract: OpContract,
    state: OpState,
    candidates: list,
    templates: dict,
    ref_fn: Any,
    gen_inputs_fn: Any,
    ops_dir: Path,
) -> list:
    """Compile, verify, and profile each candidate. Returns list of CandidateResult."""
    from triton_agent.core.spec import CandidateResult
    from triton_agent.core.compiler import compile_kernel
    from triton_agent.core.verifier import check_numerical
    from triton_agent.core.profiler import profile_kernel

    is_gpu_available = _gpu_available()

    if is_gpu_available:
        import torch
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(state.dtype, torch.float16)
        inputs = gen_inputs_fn(B=state.B, T=state.T, D=state.D, dtype=torch_dtype, device=state.device)
        ref_output = ref_fn(*inputs)
        baseline_latency_us = _measure_baseline(ref_fn, inputs)

        torch.cuda.reset_peak_memory_stats()
    else:
        inputs = ()
        ref_output = None
        baseline_latency_us = 0.0

    results = []
    for i, (template_id, action) in enumerate(candidates):
        template_fn = templates.get(template_id)
        if template_fn is None:
            r = CandidateResult()
            r.compile_pass = False
            r.compile_log = f"template '{template_id}' not found"
            results.append(r)
            continue

        is_simulated = isinstance(template_fn, dict) and template_fn.get("_simulated")
        compile_args = _build_compile_args(state, action, op)

        if is_gpu_available and not is_simulated:
            import torch
            dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
            torch_dtype = dtype_map.get(state.dtype, torch.float16)

            compile_result = compile_kernel(template_fn, compile_args)
            if not compile_result.success:
                r = CandidateResult()
                r.compile_pass = False
                r.compile_log = compile_result.error_log
                results.append(r)
                continue

            try:
                triton_output = compile_result.kernel_fn(*inputs)
            except Exception as e:
                r = CandidateResult()
                r.compile_pass = True
                r.verify_pass = False
                r.compile_log = str(e)
                results.append(r)
                continue

            r = check_numerical(triton_output, ref_output, contract)
            if not r.verify_pass:
                results.append(r)
                continue

            r = profile_kernel(
                compile_result.kernel_fn, inputs,
                warmup=contract.benchmark.warmup,
                repeat=contract.benchmark.repeat,
                baseline_latency_us=baseline_latency_us,
            )
            results.append(r)
        else:
            r = CandidateResult()
            click.echo(f"    [{i+1}/{len(candidates)}] {template_id}: simulated (no GPU)")
            results.append(r)

    return results


def _build_compile_args(state: OpState, action: Any, op_name: str) -> dict:
    """Build the keyword arguments for the kernel template."""
    args: dict[str, Any] = {
        "B": state.B,
        "T": state.T,
        "D": state.D,
        "BLOCK_SIZE": action.block_d,
        "NUM_WARPS": action.num_warps,
        "NUM_STAGES": action.num_stages,
    }
    if "rope" in op_name.lower():
        args["H"] = getattr(state, "H", 32)
    return args


def _measure_baseline(ref_fn: Any, inputs: tuple) -> float:
    """Measure the baseline PyTorch reference latency."""
    import torch
    import time

    warmup, repeat = 20, 100
    for _ in range(warmup):
        ref_fn(*inputs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        ref_fn(*inputs)
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) * 1e6 / repeat
    return elapsed_us


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _load_templates(op: str, ops_dir: Path) -> dict[str, Any]:
    """Load Triton kernel templates for an operator.

    Returns a dict of template_id -> callable. On non-GPU systems, registers
    template_ids as dicts with dummy metadata so the pipeline can still run
    in simulation mode.
    """
    import importlib
    import sys

    templates: dict[str, Any] = {}
    templates_dir = ops_dir / op / "templates"
    if not templates_dir.exists():
        return templates

    for py_file in sorted(templates_dir.glob("triton_v*.py")):
        module_name = py_file.stem
        try:
            spec = importlib.util.spec_from_file_location(
                f"triton_agent.ops.{op}.templates.{module_name}", str(py_file)
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                for attr_name in dir(mod):
                    if attr_name.startswith(op) or attr_name.endswith("_kernel"):
                        pass
                    elif callable(getattr(mod, attr_name, None)) and not attr_name.startswith("_"):
                        templates[attr_name] = getattr(mod, attr_name)
        except (ImportError, ModuleNotFoundError):
            templates[module_name] = {"_simulated": True}
        except Exception as e:
            click.echo(f"  Warning: failed to load {py_file}: {e}", err=True)

    if not templates:
        for py_file in sorted(templates_dir.glob("triton_v*.py")):
            templates[py_file.stem] = {"_simulated": True}

    return templates


def _load_reference(op: str, ops_dir: Path):
    """Load PyTorch reference implementation for an operator.

    Returns (ref_fn, gen_inputs_fn) or (None, None) on failure.
    """
    import importlib
    import sys

    ref_path = ops_dir / op / "reference.py"
    if not ref_path.exists():
        return None, None

    try:
        spec = importlib.util.spec_from_file_location(
            f"triton_agent.ops.{op}.reference", str(ref_path)
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            ref_fn = getattr(mod, op, None) or getattr(mod, f"{op}_reference", None)
            gen_fn = getattr(mod, "generate_test_inputs", None)
            return ref_fn, gen_fn
    except (ImportError, ModuleNotFoundError):
        return {"_simulated": True}, {"_simulated": True}
    except Exception as e:
        click.echo(f"  Warning: failed to load reference: {e}", err=True)
    return None, None

    try:
        spec = importlib.util.spec_from_file_location(
            f"triton_agent.ops.{op}.reference", str(ref_path)
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            ref_fn = getattr(mod, op, None) or getattr(mod, f"{op}_reference", None)
            gen_fn = getattr(mod, "generate_test_inputs", None)
            return ref_fn, gen_fn
    except Exception as e:
        click.echo(f"  Warning: failed to load reference: {e}", err=True)
    return None, None


def _ensure_op_registered(op: str, ops_dir: Path) -> None:
    """Ensure the operator is registered from the ops/ directory if not already."""
    registry = get_registry()
    if registry.has(op):
        return
    contract_path = ops_dir / op / "contract.yaml"
    if contract_path.exists():
        contract = OpContract.from_yaml(contract_path)
        registry.register(contract)


@main.command()
@click.argument("op")
@click.option("--shape", default=None, help="Filter by shape")
@click.option("--dtype", default=None, help="Filter by dtype")
def leaderboard(op: str, shape: str, dtype: str):
    """View the leaderboard for an operator.

    OP: Registered operator name.
    """
    registry = get_registry()
    if not registry.has(op):
        ops_dir = Path(__file__).parent / "ops"
        _ensure_op_registered(op, ops_dir)

    if not get_registry().has(op):
        click.echo(f"Operator '{op}' not registered.", err=True)
        return

    store = LeaderboardStore()
    rows = store.query(op, shape=shape, dtype=dtype, limit=30)

    if not rows:
        click.echo(f"Leaderboard for {op}: no entries yet.")
        return

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Leaderboard: {op}")
    table.add_column("Template", style="cyan")
    table.add_column("Shape", style="green")
    table.add_column("Dtype")
    table.add_column("Latency(p50)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Reward", justify="right")
    table.add_column("Promoted")

    for row in rows:
        table.add_row(
            row.get("template_id", ""),
            row.get("shape", ""),
            row.get("dtype", ""),
            f"{row.get('latency_us_p50', 0):.1f} us",
            f"{row.get('speedup', 1.0):.2f}x",
            f"{row.get('reward', 0):.2f}",
            "yes" if row.get("promoted") else "no",
        )
    console.print(table)


@main.command()
@click.argument("episode_path")
def replay(episode_path: str):
    """Replay a previous optimization episode.

    EPISODE_PATH: Path to the episode directory or ID (e.g., episodes/rmsnorm/000042).
    """
    path = Path(episode_path)
    if not path.exists():
        click.echo(f"Episode not found: {episode_path}", err=True)
        return

    data = replay_episode(path)
    if data is None:
        click.echo(f"No record.json found in {episode_path}", err=True)
        return

    click.echo(f"Episode: {data.get('episode_id')}")
    click.echo(f"State: {json.dumps(data.get('state', {}), indent=2)}")
    click.echo(f"Action: {json.dumps(data.get('action', {}), indent=2)}")
    click.echo(f"Result: {json.dumps(data.get('result', {}), indent=2)}")


@main.command()
@click.argument("op")
@click.option("--baseline", default="torch", help="Baseline to compare against (torch)")
@click.option("--variant", default="best", help="Variant to compare: best, latest, or episode_id")
def compare(op: str, baseline: str, variant: str):
    """Compare best variant against baseline.

    OP: Registered operator name.
    """
    registry = get_registry()
    if not registry.has(op):
        ops_dir = Path(__file__).parent / "ops"
        _ensure_op_registered(op, ops_dir)

    if not get_registry().has(op):
        click.echo(f"Operator '{op}' not registered.", err=True)
        return

    store = LeaderboardStore()
    episode_store = EpisodeStore()

    if variant == "best":
        rows = store.query(op, limit=1)
        if not rows:
            click.echo("No leaderboard entries found.")
            return
        entry = rows[0]
        episode_id = entry.get("episode_id")
    elif variant == "latest":
        ids = episode_store.list_episodes(op)
        if not ids:
            click.echo("No episodes found.")
            return
        episode_id = ids[-1]
    else:
        episode_id = variant

    episode = episode_store.load(op, episode_id)
    if not episode:
        click.echo(f"Episode {episode_id} not found.")
        return

    click.echo(f"Episode: {episode_id}")
    click.echo(f"Template: {episode['action']['template_id']}")
    click.echo(f"Speedup: {episode['result']['speedup']:.2f}x")
    click.echo(f"Latency p50: {episode['result']['latency_us_p50']:.1f} us")
    click.echo(f"Reward: {episode['result']['reward']:.2f}")
    click.echo(f"Promoted: {episode['result']['promoted']}")


@main.command()
@click.argument("op")
@click.option("--shape", default="B=8,T=2048,D=4096", help="Shape specification")
@click.option("--out", default="dist/", help="Output directory")
def export(op: str, shape: str, out: str):
    """Export the best kernel variant.

    OP: Registered operator name.
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = LeaderboardStore()
    rows = store.query(op, shape=shape, limit=1)
    if not rows:
        click.echo(f"No leaderboard entries found for {op} shape={shape}")
        return

    entry = rows[0]
    export_path = out_dir / f"{op}_best.py"
    entry_json = json.dumps(entry, indent=2)

    ops_dir = Path(__file__).parent / "ops"
    template_path = ops_dir / op / "templates" / f"{entry['template_id']}.py"

    if template_path.exists():
        kernel_code = template_path.read_text(encoding="utf-8")
        export_path.write_text(kernel_code, encoding="utf-8")
        click.echo(f"Exported kernel to {export_path}")
    else:
        export_path.write_text(entry_json, encoding="utf-8")
        click.echo(f"Exported config to {export_path} (template source not found)")

    click.echo(f"  Template: {entry['template_id']}")
    click.echo(f"  Speedup: {entry['speedup']:.2f}x")


@main.command()
@click.argument("ops", nargs=-1)
@click.option("--shape", default="B=8,T=2048,D=4096", help="Shape filter")
@click.option("--dtype", default="fp16", help="Dtype filter")
def cross_compare(ops: tuple, shape: str, dtype: str):
    """Compare best configs across multiple operators.

    OPS: One or more operator names. If empty, shows all registered operators.
    """
    registry = get_registry()
    ops_dir = Path(__file__).parent / "ops"

    if not ops:
        ops_list = [d.name for d in ops_dir.iterdir() if (d / "contract.yaml").exists()]
    else:
        ops_list = list(ops)

    store = LeaderboardStore()

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Cross-Operator Comparison")
    table.add_column("Operator", style="cyan")
    table.add_column("Best Template")
    table.add_column("Latency(p50)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Reward", justify="right")
    table.add_column("Promoted")

    for op_name in ops_list:
        _ensure_op_registered(op_name, ops_dir)
        rows = store.query(op_name, shape=shape, dtype=dtype, limit=1)
        if rows:
            r = rows[0]
            table.add_row(
                op_name,
                r.get("template_id", ""),
                f"{r.get('latency_us_p50', 0):.1f} us",
                f"{r.get('speedup', 1.0):.2f}x",
                f"{r.get('reward', 0):.2f}",
                "yes" if r.get("promoted") else "no",
            )
        else:
            table.add_row(op_name, "—", "—", "—", "—", "—")

    console.print(table)


def _run_bandit_optimize(
    op: str,
    contract: Any,
    state: Any,
    templates: dict,
    ref_fn: Any,
    gen_inputs_fn: Any,
    ops_dir: Path,
    strategy: str,
    max_candidates: int,
    retry: int,
    is_simulated: bool,
) -> list:
    """Run bandit-based iterative optimization.

    Selects one candidate at a time using the bandit policy, evaluates it,
    and feeds reward back to update policy beliefs. Returns the final list
    of evaluated (candidate, result) pairs.
    """
    from triton_agent.microrl.trainer import Trainer
    from triton_agent.agent.generator import generate_grid
    from triton_agent.agent.repairer import repair_action, should_retry
    from triton_agent.core.reward import compute_score
    from triton_agent.core.spec import CandidateResult

    if is_simulated:
        click.echo(f"  [Bandit/{strategy}] Simulation mode: evaluating {max_candidates} samples")
        candidates = generate_grid(contract, state, templates)
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]
        return candidates

    all_candidates = generate_grid(contract, state, templates)

    trainer = Trainer(strategy=strategy)

    evaluated_candidates: list = []
    evaluated_results: list = []

    for step in range(max_candidates):
        idx = trainer.select(all_candidates, state)
        template_id, action = all_candidates[idx]

        result = _evaluate_single(
            op, contract, state, template_id, action, templates,
            ref_fn, gen_inputs_fn,
        )

        for attempt in range(retry):
            if result.compile_pass and result.verify_pass:
                break
            repaired = repair_action(action, result)
            if repaired and should_retry(repaired, action, result):
                action = repaired
                result = _evaluate_single(
                    op, contract, state, template_id, action, templates,
                    ref_fn, gen_inputs_fn,
                )
            else:
                break

        compute_score(result)
        trainer.update(action, state, result)

        evaluated_candidates.append((template_id, action))
        evaluated_results.append(result)

        click.echo(
            f"    [{step + 1}/{max_candidates}] {template_id} "
            f"reward={result.reward:.2f} speedup={result.speedup:.2f}x"
        )

    return evaluated_candidates


def _evaluate_single(
    op: str,
    contract: Any,
    state: Any,
    template_id: str,
    action: Any,
    templates: dict,
    ref_fn: Any,
    gen_inputs_fn: Any,
) -> Any:
    """Evaluate a single candidate through compile → verify → profile."""
    from triton_agent.core.spec import CandidateResult
    from triton_agent.core.compiler import compile_kernel
    from triton_agent.core.verifier import check_numerical
    from triton_agent.core.profiler import profile_kernel

    if not _gpu_available():
        return CandidateResult()

    import torch
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(state.dtype, torch.float16)

    inputs = gen_inputs_fn(B=state.B, T=state.T, D=state.D, dtype=torch_dtype, device=state.device)
    ref_output = ref_fn(*inputs)
    baseline_latency_us = _measure_baseline(ref_fn, inputs)

    torch.cuda.reset_peak_memory_stats()

    template_fn = templates.get(template_id)
    if template_fn is None or isinstance(template_fn, dict):
        r = CandidateResult()
        r.compile_pass = False
        r.compile_log = f"template '{template_id}' not available"
        return r

    compile_args = _build_compile_args(state, action, op)
    compile_result = compile_kernel(template_fn, compile_args)

    if not compile_result.success:
        r = CandidateResult()
        r.compile_pass = False
        r.compile_log = compile_result.error_log
        return r

    try:
        triton_output = compile_result.kernel_fn(*inputs)
    except Exception as e:
        r = CandidateResult()
        r.compile_pass = True
        r.verify_pass = False
        r.compile_log = str(e)
        return r

    r = check_numerical(triton_output, ref_output, contract)
    if not r.verify_pass:
        return r

    r = profile_kernel(
        compile_result.kernel_fn, inputs,
        warmup=contract.benchmark.warmup,
        repeat=contract.benchmark.repeat,
        baseline_latency_us=baseline_latency_us,
    )
    return r


@main.group()
def experiment():
    """Run benchmark experiments and ablation studies."""


@experiment.command()
@click.option("--level", default="smoke", type=click.Choice(["smoke", "core", "ablation", "replay"]))
@click.option("--output", default="reports", help="Output directory")
def run(level: str, output: str):
    """Run a benchmark experiment at the given level.

    Levels:
      smoke   - 13 ops x 1 shape, grid only, verify + latency
      core    - 6 ops x 3 shapes x 4 strategies
      ablation - 3 ops x 5 shapes x 7 strategies x 3 seeds
      replay  - cold start vs warm start reproducibility
    """
    from triton_agent.experiments.runner import run_level
    path = run_level(level, output)
    click.echo(f"Benchmark saved to {path}")


@experiment.command()
@click.argument("json_path")
@click.option("--output", default="reports", help="Output directory")
def report(json_path: str, output: str):
    """Generate a markdown report from a benchmark JSON file."""
    from triton_agent.experiments.report import generate_report
    path = generate_report(json_path, output)
    click.echo(f"Report written to {path}")


@experiment.command()
@click.option("--name", required=True,
              type=click.Choice(["strategy", "profile", "reward", "replay", "all"]))
@click.option("--output", default="reports", help="Output directory")
def ablation(name: str, output: str):
    """Run a targeted ablation study.

    Ablations:
      strategy - Compare all 7 strategies on trials-to-best
      profile  - Correctness-only vs +latency vs full selection
      reward   - Compare reward formulations
      replay   - Cold start vs warm start vs transfer
    """
    from triton_agent.experiments.ablation import (
        run_strategy_ablation, run_profile_ablation,
        run_reward_ablation, run_replay_ablation,
    )
    from pathlib import Path as P
    out = P(output)
    if name in ("strategy", "all"):
        run_strategy_ablation(out)
    if name in ("profile", "all"):
        run_profile_ablation(out)
    if name in ("reward", "all"):
        run_reward_ablation(out)
    if name in ("replay", "all"):
        run_replay_ablation(out)
    click.echo("Ablation complete.")


@main.command()
@click.argument("op")
@click.option("--shape", default="B=8,T=2048,D=4096")
@click.option("--dtype", default="fp16")
def verify_correctness(op: str, shape: str, dtype: str):
    """Verify operator correctness against PyTorch reference.

    Runs the PyTorch reference, compiles each Triton template, and checks
    max_abs_error / mean_abs_error / NaN / Inf against contract tolerance.
    """
    ops_dir = Path(__file__).parent / "ops"
    _ensure_op_registered(op, ops_dir)
    if not get_registry().has(op):
        click.echo(f"Operator '{op}' not found.")
        return

    contract = get_registry().get(op)
    parsed = {}
    for kv in shape.split(","):
        k, v = kv.strip().split("=")
        parsed[k.strip()] = int(v.strip())

    templates = _load_templates(op, ops_dir)
    ref_fn, gen_fn = _load_reference(op, ops_dir)

    if isinstance(ref_fn, dict) or isinstance(gen_fn, dict):
        click.echo("Running in simulation mode — GPU required for correctness verification.")
        return

    import torch
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.float16)
    inputs = gen_fn(**{**parsed, "dtype": torch_dtype, "device": "cuda"})
    ref_output = ref_fn(*inputs)

    from triton_agent.core.compiler import compile_kernel
    from triton_agent.core.verifier import check_numerical
    from rich.table import Table
    from rich.console import Console

    console = Console()
    table = Table(title=f"Correctness: {op} ({shape}, {dtype})")
    table.add_column("Template", style="cyan")
    table.add_column("Compile")
    table.add_column("Max AE", justify="right")
    table.add_column("Mean AE", justify="right")
    table.add_column("NaN/Inf")
    table.add_column("Verdict")

    for tid in sorted(templates.keys()):
        tf = templates.get(tid)
        if isinstance(tf, dict):
            continue

        class _S:
            pass
        state_spec = _S()
        state_spec.B = parsed.get("B", 8)
        state_spec.T = parsed.get("T", 2048)
        state_spec.D = parsed.get("D", 4096)

        class _A:
            pass
        action_spec = _A()
        action_spec.block_d = 256
        action_spec.num_warps = 4
        action_spec.num_stages = 3
        action_spec.vectorize = False

        compile_args = _build_compile_args(state_spec, action_spec, op)
        comp = compile_kernel(tf, compile_args)

        if not comp.success:
            table.add_row(tid, "FAIL", "—", "—", "—", "FAIL")
            continue

        try:
            triton_out = comp.kernel_fn(*inputs)
        except Exception as e:
            table.add_row(tid, "OK", "—", "—", "—", f"ERROR: {str(e)[:40]}")
            continue

        r = check_numerical(triton_out, ref_output, contract)
        log = json.loads(r.verify_log) if r.verify_log else {}
        table.add_row(
            tid, "OK",
            f"{log.get('max_abs_error', 0):.1e}",
            f"{log.get('mean_abs_error', 0):.1e}",
            "YES" if (log.get("has_nan") or log.get("has_inf")) else "no",
            "PASS" if r.verify_pass else "FAIL",
        )

    console.print(table)
