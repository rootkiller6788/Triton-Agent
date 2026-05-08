import pytest

from triton_agent.core.contract import OpContract
from triton_agent.core.spec import OpState, CandidateResult
from triton_agent.agent.generator import generate_grid, generate_best_of_n
from triton_agent.agent.selector import select_best
from triton_agent.agent.promoter import should_promote
from triton_agent.agent.planner import plan


RMSNORM_CONTRACT = OpContract.from_dict({
    "op": "rmsnorm",
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
})


class TestGenerator:
    def test_generate_grid(self):
        state = OpState(op_name="rmsnorm", B=8, T=2048, D=4096, dtype="fp16", device="cuda")
        templates = {"triton_v1": None, "triton_v2": None}

        candidates = generate_grid(RMSNORM_CONTRACT, state, templates)
        # 2 templates * 2 BLOCK_D * 2 warps * 1 stages * 2 vectorize = 16
        assert len(candidates) == 16

    def test_generate_best_of_n(self):
        state = OpState(op_name="rmsnorm", B=8, T=2048, D=4096, dtype="fp16", device="cuda")
        templates = {"triton_v1": None}

        candidates = generate_best_of_n(RMSNORM_CONTRACT, state, templates, n=8, seed=42)
        assert len(candidates) == 8

    def test_generate_best_of_n_reproducible(self):
        state = OpState(op_name="rmsnorm", B=8, T=2048, D=4096, dtype="fp16", device="cuda")
        templates = {"triton_v1": None}

        c1 = generate_best_of_n(RMSNORM_CONTRACT, state, templates, n=4, seed=42)
        c2 = generate_best_of_n(RMSNORM_CONTRACT, state, templates, n=4, seed=42)
        for a, b in zip(c1, c2):
            assert a[0] == b[0]
            assert a[1].block_d == b[1].block_d


class TestSelector:
    def test_select_highest_reward(self):
        r0 = CandidateResult(reward=0.0, latency_us_p50=100)
        r1 = CandidateResult(reward=0.5, latency_us_p50=80)
        r2 = CandidateResult(reward=0.8, latency_us_p50=90)

        candidates = [("v1", None), ("v2", None), ("v3", None)]
        results = [r0, r1, r2]

        idx, best = select_best(candidates, results)
        assert idx == 2
        assert best.reward == 0.8

    def test_select_tiebreaker_by_latency(self):
        r0 = CandidateResult(reward=0.5, latency_us_p50=100)
        r1 = CandidateResult(reward=0.5, latency_us_p50=80)

        idx, best = select_best([("a", None), ("b", None)], [r0, r1])
        assert idx == 1
        assert best.latency_us_p50 == 80


class TestPromoter:
    def test_should_promote_pass(self):
        r = CandidateResult(compile_pass=True, verify_pass=True, speedup=1.10, variance=0.05)
        assert should_promote(r, RMSNORM_CONTRACT) is True

    def test_should_promote_low_speedup(self):
        r = CandidateResult(compile_pass=True, verify_pass=True, speedup=1.02, variance=0.05)
        assert should_promote(r, RMSNORM_CONTRACT) is False

    def test_should_promote_high_variance(self):
        r = CandidateResult(compile_pass=True, verify_pass=True, speedup=1.20, variance=0.15)
        assert should_promote(r, RMSNORM_CONTRACT) is False

    def test_should_promote_compile_fail(self):
        r = CandidateResult(compile_pass=False, verify_pass=False)
        assert should_promote(r, RMSNORM_CONTRACT) is False

    def test_should_promote_verify_fail(self):
        r = CandidateResult(compile_pass=True, verify_pass=False, speedup=1.10)
        assert should_promote(r, RMSNORM_CONTRACT) is False


class TestPlanner:
    def test_grid_for_small_space(self):
        state = OpState(op_name="test", B=1, T=1, D=1, dtype="fp32", device="cpu")
        strategy = plan(RMSNORM_CONTRACT, state)
        assert strategy["strategy"] == "grid"

    def test_best_of_n_for_large_space(self):
        contract = OpContract.from_dict({
            "op": "big_op",
            "inputs": {"x": ["N"]},
            "outputs": {"y": ["N"]},
            "tolerance": {"max_abs_error": 0.01, "mean_abs_error": 0.001},
            "search_space": {
                "BLOCK_D": [32, 64, 128, 256, 512, 1024],
                "num_warps": [1, 2, 4, 8],
                "num_stages": [1, 2, 3, 4],
                "vectorize": [True, False],
            },
        })
        state = OpState(op_name="big_op", B=1, T=1, D=1, dtype="fp32", device="cpu")
        strategy = plan(contract, state)
        assert strategy["strategy"] == "best_of_n"
