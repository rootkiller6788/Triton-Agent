"""Integration tests: end-to-end pipeline validation in simulation mode."""

import pytest
import tempfile
import os
from pathlib import Path

from triton_agent.core.contract import OpContract
from triton_agent.core.spec import OpState, OpAction, CandidateResult
from triton_agent.core.registry import get_registry, OpRegistry
from triton_agent.core.storage import EpisodeStore, LeaderboardStore
from triton_agent.core.reward import compute_score
from triton_agent.core.checkpoint import Checkpoint
from triton_agent.core.compile_cache import CompileCache
from triton_agent.core.eval_engine import RegressionDetector
from triton_agent.core.trend import compute_trend, format_trend_report
from triton_agent.agent.generator import generate_grid, generate_best_of_n
from triton_agent.agent.selector import select_best
from triton_agent.agent.promoter import should_promote
from triton_agent.agent.planner import plan
from triton_agent.agent.repairer import repair_action
from triton_agent.microrl.bandit import UCBPolicy, ThompsonSamplingPolicy, EpsilonGreedyPolicy
from triton_agent.microrl.reinforce_lite import REINFORCELite
from triton_agent.microrl.grpo_lite import GRPOLite
from triton_agent.microrl.trainer import Trainer


CONTRACT_DICT = {
    "op": "test_op",
    "inputs": {"x": ["B", "T", "D"]},
    "outputs": {"y": ["B", "T", "D"]},
    "tolerance": {"max_abs_error": 1e-3, "mean_abs_error": 1e-4},
    "search_space": {
        "BLOCK_D": [128, 256],
        "num_warps": [2, 4],
        "num_stages": [3],
        "vectorize": [True, False],
    },
    "benchmark": {"warmup": 20, "repeat": 100},
    "promotion": {"min_speedup": 1.05, "max_variance": 0.10, "regression_required": True},
}


class TestEndToEndPipeline:
    """Simulates the full optimization pipeline without GPU."""

    def test_full_pipeline_grid(self):
        contract = OpContract.from_dict(CONTRACT_DICT)
        state = OpState(op_name="test_op", B=8, T=2048, D=4096, dtype="fp16", device="cuda")
        templates = {"triton_v1": None, "triton_v2": None}

        strategy = plan(contract, state)
        assert strategy["strategy"] in ("grid", "best_of_n")

        candidates = generate_grid(contract, state, templates)
        assert len(candidates) > 0

        results = [CandidateResult() for _ in candidates]
        for i, (tid, action) in enumerate(candidates):
            results[i].compile_pass = True
            results[i].verify_pass = True
            results[i].speedup = 1.0 + 0.1 * (i + 1) / len(candidates)
            compute_score(results[i])

        best_idx, best_result = select_best(candidates, results)
        assert best_idx >= 0
        assert best_result.reward >= 0

        best_result.promoted = should_promote(best_result, contract)

    def test_full_pipeline_bandit(self):
        contract = OpContract.from_dict(CONTRACT_DICT)
        state = OpState(op_name="test_op", B=8, T=2048, D=4096, dtype="fp16", device="cuda")

        trainer = Trainer(strategy="ucb")
        candidates = generate_grid(contract, state, {"v1": None, "v2": None})

        for i in range(8):
            idx = trainer.select(candidates, state)
            tid, action = candidates[idx]
            result = CandidateResult(speedup=1.0 + i * 0.05, reward=i * 0.1)
            result.compile_pass = True
            result.verify_pass = True
            compute_score(result)
            trainer.update(action, state, result, candidates)


class TestCheckpoint:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Checkpoint(tmpdir)
            state = OpState(op_name="test", B=8, T=2048, D=4096, dtype="fp16", device="cuda")
            action = OpAction(template_id="v1")
            best = CandidateResult(speedup=1.15)

            path = ckpt.save(
                "test", state, "grid", 64,
                [("v1", action)], 0, best,
            )
            assert path.exists()

            loaded = ckpt.load("test")
            assert loaded is not None
            assert loaded["strategy"] == "grid"

    def test_cleanup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Checkpoint(tmpdir)
            state = OpState(op_name="test", B=1, T=1, D=1, dtype="fp32", device="cpu")
            action = OpAction(template_id="v1")
            best = CandidateResult()
            import time
            for _ in range(10):
                ckpt.save("test", state, "grid", 64, [("v1", action)], 0, best)
                time.sleep(0.01)
            assert len(ckpt.list_checkpoints("test")) == 10
            removed = ckpt.cleanup("test", keep=3)
            assert removed == 7
            assert len(ckpt.list_checkpoints("test")) == 3


class TestCompileCache:
    def test_put_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CompileCache(Path(tmpdir) / "cache.db")
            action = OpAction(template_id="v1", block_d=256, num_warps=4)
            cache.put("v1", action, "rmsnorm", "fp16", 8, 2048, 4096, 0.05, True)

            entry = cache.get("v1", action, "rmsnorm", "fp16", 8, 2048, 4096)
            assert entry is not None
            assert entry["success"] == 1

    def test_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CompileCache(Path(tmpdir) / "cache.db")
            action = OpAction(template_id="v1")
            assert cache.get("v1", action, "nope", "fp16", 1, 1, 1) is None

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CompileCache(Path(tmpdir) / "cache.db")
            action = OpAction(template_id="v1")
            cache.put("v1", action, "op", "fp16", 1, 1, 1, 0.1, True)
            cache.put("v2", action, "op", "fp16", 1, 1, 1, 0.2, False)
            stats = cache.stats()
            assert stats["total_entries"] == 2


class TestRegressionDetector:
    def test_baseline_and_check(self):
        detector = RegressionDetector(tolerance=0.95)
        detector.set_baseline("rmsnorm", "B=8,T=2048,D=4096", "fp16", 100.0)

        is_reg, ratio = detector.check("rmsnorm", "B=8,T=2048,D=4096", "fp16", 120.0)
        assert is_reg, f"Expected regression at 120us vs 100us baseline (ratio={ratio:.2f})"

        is_reg2, _ = detector.check("rmsnorm", "B=8,T=2048,D=4096", "fp16", 95.0)
        assert not is_reg2

    def test_no_baseline(self):
        detector = RegressionDetector()
        is_reg, ratio = detector.check("new_op", "shape", "fp16", 50.0)
        assert not is_reg
        assert ratio == 1.0


class TestTrend:
    def test_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LeaderboardStore(Path(tmpdir) / "empty.db")
            report = compute_trend(store, "nonexistent")
            assert report.total_episodes == 0

    def test_improving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LeaderboardStore(Path(tmpdir) / "trend.db")
            action = OpAction(template_id="v1")
            for i in range(5):
                result = CandidateResult(speedup=1.0 + i * 0.1, reward=i * 0.15, promoted=True)
                store.upsert("rmsnorm", "shape", "fp16", "cuda", action, result, f"ep{i}")
                action.template_id = f"v{i}"

            report = compute_trend(store, "rmsnorm")
            assert report.total_episodes > 0
            assert report.best_speedup >= 1.0
            formatted = format_trend_report(report)
            assert "rmsnorm" in formatted


class TestCLISimulation:
    """End-to-end simulation of CLI commands."""

    def test_pipeline_full_cycle(self):
        state = OpState(op_name="test", B=8, T=2048, D=4096, dtype="fp16", device="cuda")
        contract = OpContract.from_dict(CONTRACT_DICT)

        templates = {"v1": {"_simulated": True}, "v2": {"_simulated": True}}
        candidates = generate_grid(contract, state, templates)

        results = [CandidateResult() for _ in candidates]
        for r in results:
            r.compile_pass = True
            r.verify_pass = True
            r.speedup = 1.1
            compute_score(r)

        best_idx, best = select_best(candidates, results)
        best.promoted = should_promote(best, contract)

        assert best_idx is not None
        assert best.promoted or best.speedup < 1.05

    def test_all_strategies_work(self):
        state = OpState(op_name="test", B=8, T=2048, D=4096, dtype="fp16", device="cuda")
        contract = OpContract.from_dict(CONTRACT_DICT)
        templates = {"v1": None, "v2": None}
        candidates = generate_grid(contract, state, templates)

        for strat_name in ["ucb", "thompson", "epsilon", "reinforce", "grpo"]:
            trainer = Trainer(strategy=strat_name)
            idx = trainer.select(candidates, state)
            assert 0 <= idx < len(candidates)

            _, action = candidates[idx]
            result = CandidateResult(reward=0.5)
            trainer.update(action, state, result, candidates)

    def test_cross_operator_compare(self):
        registry = get_registry()
        registry.register(OpContract.from_dict({
            "op": "op_a", "inputs": {"x": ["N"]}, "outputs": {"y": ["N"]},
            "tolerance": {"max_abs_error": 0.01, "mean_abs_error": 0.001},
        }))
        assert registry.has("op_a")
        assert "op_a" in registry.list()
