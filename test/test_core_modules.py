import pytest
import tempfile
import os
from pathlib import Path

from triton_agent.core.spec import OpState, OpAction, CandidateResult
from triton_agent.core.storage import EpisodeStore, LeaderboardStore
from triton_agent.core.reward import compute_score
from triton_agent.core.replay import replay_episode, compare_episodes


class TestReward:
    def test_compile_pass_only(self):
        r = CandidateResult(compile_pass=True, verify_pass=True)
        score = compute_score(r)
        assert score == 0.8  # 0.2 (compile) + 0.6 (verify)

    def test_compile_fail(self):
        r = CandidateResult(compile_pass=False, verify_pass=False)
        score = compute_score(r)
        assert score == -1.0  # verify fail penalty

    def test_speedup_bonus(self):
        r = CandidateResult(compile_pass=True, verify_pass=True, speedup=1.20)
        score = compute_score(r)
        assert score >= 1.0

    def test_variance_penalty(self):
        r = CandidateResult(compile_pass=True, verify_pass=True, variance=0.15)
        score = compute_score(r)
        assert score < 1.0

    def test_memory_penalty(self):
        r = CandidateResult(compile_pass=True, verify_pass=True, memory_peak_mb=120.0)
        score = compute_score(r, baseline_memory_mb=100.0)
        assert score < 1.0


class TestEpisodeStore:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EpisodeStore(tmpdir)
            state = OpState(op_name="test", B=1, T=1, D=1, dtype="fp32", device="cpu")
            action = OpAction(template_id="v1")
            result = CandidateResult(compile_pass=True, verify_pass=True, speedup=1.1, reward=0.9)

            ep_dir = store.save("test_op", "ep001", state, action, result)
            assert ep_dir.exists()

            loaded = store.load("test_op", "ep001")
            assert loaded is not None
            assert loaded["action"]["template_id"] == "v1"
            assert loaded["result"]["speedup"] == 1.1

    def test_list_episodes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EpisodeStore(tmpdir)
            state = OpState(op_name="test", B=1, T=1, D=1, dtype="fp32", device="cpu")
            action = OpAction(template_id="v1")
            result = CandidateResult()
            store.save("op1", "ep001", state, action, result)
            store.save("op1", "ep002", state, action, result)
            ids = store.list_episodes("op1")
            assert ids == ["ep001", "ep002"]

    def test_load_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EpisodeStore(tmpdir)
            assert store.load("missing", "nope") is None


class TestLeaderboardStore:
    def test_upsert_and_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = LeaderboardStore(db_path)
            action = OpAction(template_id="v1", block_d=256, num_warps=4)
            result = CandidateResult(
                compile_pass=True, verify_pass=True,
                latency_us_p50=23.4, speedup=1.18, reward=0.88, promoted=True,
            )

            store.upsert("rmsnorm", "B=8,T=2048,D=4096", "fp16", "cuda", action, result, "ep_001")
            store.upsert("rmsnorm", "B=8,T=2048,D=4096", "fp16", "cuda", action, result, "ep_002")

            rows = store.query("rmsnorm")
            assert len(rows) == 1  # UPSERT on same key

            row = rows[0]
            assert row["template_id"] == "v1"
            assert row["speedup"] == 1.18

    def test_best_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = LeaderboardStore(db_path)
            action1 = OpAction(template_id="v1")
            result1 = CandidateResult(speedup=1.10, reward=0.5, promoted=True)
            action2 = OpAction(template_id="v2")
            result2 = CandidateResult(speedup=1.18, reward=0.7, promoted=True)

            store.upsert("op", "shape", "fp16", "cuda", action1, result1, "ep1")
            store.upsert("op", "shape", "fp16", "cuda", action2, result2, "ep2")

            best = store.best_config("op", "shape", "fp16")
            assert best is not None
            assert best["template_id"] == "v2"

    def test_empty_leaderboard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = LeaderboardStore(db_path)
            assert store.query("nonexistent") == []
            assert store.best_config("nonexistent", "s", "fp16") is None


class TestReplay:
    def test_replay_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EpisodeStore(tmpdir)
            state = OpState(op_name="test", B=1, T=1, D=1, dtype="fp32", device="cpu")
            action = OpAction(template_id="v1")
            result = CandidateResult(speedup=1.2)
            ep_dir = store.save("op", "ep1", state, action, result)

            data = replay_episode(ep_dir)
            assert data is not None
            assert data["result"]["speedup"] == 1.2

    def test_replay_missing(self):
        assert replay_episode("/nonexistent/path") is None

    def test_compare_episodes(self):
        ep_a = {"result": {"latency_us_p50": 20.0, "speedup": 1.2, "reward": 0.8, "promoted": True}}
        ep_b = {"result": {"latency_us_p50": 25.0, "speedup": 1.1, "reward": 0.6, "promoted": False}}
        cmp = compare_episodes(ep_a, ep_b)
        assert cmp["winner"] == "a"
        assert cmp["a_speedup"] == 1.2
