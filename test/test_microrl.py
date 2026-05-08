import pytest
import tempfile
import json
from pathlib import Path

from triton_agent.core.spec import OpState, OpAction, CandidateResult
from triton_agent.microrl.bandit import UCBPolicy, ThompsonSamplingPolicy, EpsilonGreedyPolicy
from triton_agent.microrl.trainer import Trainer, ShapeConfigStore


def _make_state():
    return OpState(op_name="test", B=8, T=2048, D=4096, dtype="fp16", device="cuda")


def _make_actions():
    return [
        OpAction(template_id="v1", block_d=128, num_warps=4, num_stages=3, vectorize=False),
        OpAction(template_id="v1", block_d=256, num_warps=4, num_stages=3, vectorize=False),
        OpAction(template_id="v2", block_d=128, num_warps=2, num_stages=4, vectorize=True),
        OpAction(template_id="v2", block_d=512, num_warps=8, num_stages=2, vectorize=False),
    ]


class TestUCBPolicy:
    def test_initial_exploration(self):
        policy = UCBPolicy(c=2.0)
        actions = _make_actions()
        state = _make_state()

        idx = policy.select(actions, state)
        assert 0 <= idx < len(actions)

    def test_exploit_best(self):
        policy = UCBPolicy(c=0.1)
        actions = _make_actions()
        state = _make_state()

        policy.update(actions[1], state, 0.9)
        policy.update(actions[1], state, 0.95)

        for action in actions:
            if action is not actions[1]:
                policy.update(action, state, 0.1)

        selections = []
        for _ in range(100):
            selections.append(policy.select(actions, state))

        assert selections.count(1) > len(selections) // 2

    def test_best_action(self):
        policy = UCBPolicy()
        actions = _make_actions()
        state = _make_state()
        policy.update(actions[0], state, 0.1)
        policy.update(actions[2], state, 0.9)
        idx, mean = policy.best_action(actions, state)
        assert idx == 2


class TestThompsonPolicy:
    def test_select(self):
        policy = ThompsonSamplingPolicy()
        actions = _make_actions()
        state = _make_state()
        idx = policy.select(actions, state)
        assert 0 <= idx < len(actions)

    def test_update_converges(self):
        policy = ThompsonSamplingPolicy()
        actions = _make_actions()
        state = _make_state()

        for _ in range(50):
            policy.update(actions[0], state, 0.9)
        for _ in range(50):
            policy.update(actions[1], state, 0.1)

        idx, _ = policy.best_action(actions, state)
        assert idx == 0

    def test_arm_count(self):
        policy = ThompsonSamplingPolicy()
        actions = _make_actions()
        state = _make_state()
        for a in actions:
            policy.update(a, state, 0.5)
        assert policy.arm_count == 4


class TestEpsilonGreedy:
    def test_select(self):
        policy = EpsilonGreedyPolicy(epsilon_init=0.3)
        actions = _make_actions()
        state = _make_state()
        idx = policy.select(actions, state)
        assert 0 <= idx < len(actions)

    def test_epsilon_decay(self):
        policy = EpsilonGreedyPolicy(epsilon_init=0.5, epsilon_min=0.01, decay_rate=0.1)
        actions = _make_actions()
        state = _make_state()

        policy.update(actions[0], state, 0.9)
        for _ in range(200):
            idx = policy.select(actions, state)
            policy.update(actions[idx], state, 0.5 * (1 if idx == 0 else 0.1))

        best, _ = policy.best_action(actions, state)
        assert best == 0


class TestShapeConfigStore:
    def test_set_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ShapeConfigStore(Path(tmpdir) / "configs.json")
            state = _make_state()
            action = _make_actions()[0]
            result = CandidateResult(speedup=1.2, latency_us_p50=20.0, reward=0.8)

            assert store.get_best(state) is None
            store.set_best(state, action, result)
            entry = store.get_best(state)
            assert entry is not None
            assert entry["speedup"] == 1.2

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "configs.json"
            s1 = ShapeConfigStore(path)
            state = _make_state()
            action = _make_actions()[2]
            result = CandidateResult(speedup=1.5)
            s1.set_best(state, action, result)

            s2 = ShapeConfigStore(path)
            entry = s2.get_best(state)
            assert entry is not None
            assert entry["speedup"] == 1.5

    def test_historical_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ShapeConfigStore(Path(tmpdir) / "configs.json")
            state = _make_state()
            action = _make_actions()[1]
            result = CandidateResult(speedup=1.3)
            store.set_best(state, action, result)

            new_state = _make_state()
            assert new_state.historical_best_config is None
            cfg = store.get_historical_best(new_state)
            assert cfg is not None
            assert cfg["template_id"] == "v1"
            assert new_state.historical_best_config is not None


class TestTrainer:
    def test_ucb_trainer(self):
        trainer = Trainer(strategy="ucb")
        actions = _make_actions()
        state = _make_state()

        for i, a in enumerate(actions):
            result = CandidateResult(reward=(i + 1) * 0.2, promoted=(i >= 2))
            trainer.update(a, state, result)

        assert trainer.policy.arm_count > 0

    def test_thompson_trainer(self):
        trainer = Trainer(strategy="thompson")
        actions = _make_actions()
        state = _make_state()
        result = CandidateResult(reward=0.5, promoted=True)
        trainer.update(actions[0], state, result)
        assert trainer.get_best_config(state) is not None
