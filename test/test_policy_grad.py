import pytest

from triton_agent.core.spec import OpState, OpAction
from triton_agent.microrl.reinforce_lite import REINFORCELite, _arm_key, _softmax, _sample
from triton_agent.microrl.grpo_lite import GRPOLite
from triton_agent.microrl.trainer import Trainer, ALL_STRATEGIES


def _make_state():
    return OpState(op_name="test", B=8, T=2048, D=4096, dtype="fp16", device="cuda")


def _make_candidates():
    return [
        ("v1", OpAction(template_id="v1", block_d=128, num_warps=4)),
        ("v1", OpAction(template_id="v1", block_d=256, num_warps=4)),
        ("v2", OpAction(template_id="v2", block_d=128, num_warps=2)),
        ("v2", OpAction(template_id="v2", block_d=512, num_warps=8)),
    ]


class TestSoftmax:
    def test_sum_to_one(self):
        probs = _softmax([1.0, 2.0, 3.0])
        total = sum(probs)
        assert abs(total - 1.0) < 1e-6

    def test_higher_logit_higher_prob(self):
        probs = _softmax([1.0, 5.0, 2.0])
        assert probs[1] > probs[0]
        assert probs[1] > probs[2]

    def test_temperature(self):
        probs_high = _softmax([1.0, 3.0], temperature=10.0)
        probs_low = _softmax([1.0, 3.0], temperature=0.1)
        diff_high = abs(probs_high[0] - probs_high[1])
        diff_low = abs(probs_low[0] - probs_low[1])
        assert diff_high < diff_low


class TestREINFORCELite:
    def test_select(self):
        policy = REINFORCELite()
        candidates = _make_candidates()
        state = _make_state()
        idx = policy.select(candidates, state)
        assert 0 <= idx < len(candidates)

    def test_update_and_converge(self):
        policy = REINFORCELite(learning_rate=0.5)
        candidates = _make_candidates()
        state = _make_state()

        for _ in range(200):
            idx = policy.select(candidates, state)
            _, action = candidates[idx]
            reward = 0.9 if action.block_d == 256 else 0.1
            policy.update(action, state, reward, candidates)

        best = policy.best_action(candidates, state)
        assert candidates[best][1].block_d == 256

    def test_best_action(self):
        policy = REINFORCELite()
        candidates = _make_candidates()
        state = _make_state()
        for _ in range(100):
            _, action = candidates[2]
            policy.update(action, state, 0.9, candidates)
            _, action = candidates[0]
            policy.update(action, state, 0.1, candidates)
        best = policy.best_action(candidates, state)
        assert best == 2


class TestGRPOLite:
    def test_select(self):
        policy = GRPOLite(group_size=4)
        candidates = _make_candidates()
        state = _make_state()
        idx = policy.select(candidates, state)
        assert 0 <= idx < len(candidates)

    def test_select_group(self):
        policy = GRPOLite(group_size=3)
        candidates = _make_candidates()
        state = _make_state()
        indices = policy.select_group(candidates, state)
        assert len(indices) == 3
        assert len(set(indices)) == 3

    def test_group_update_converges(self):
        policy = GRPOLite(learning_rate=0.5, group_size=4)
        candidates = _make_candidates()
        state = _make_state()

        for _ in range(100):
            indices = policy.select_group(candidates, state)
            actions = [candidates[i][1] for i in indices]
            rewards = [0.9 if act.block_d == 256 else 0.1 for act in actions]
            policy.update_group(actions, rewards, state, candidates)

        best = policy.best_action(candidates, state)
        assert candidates[best][1].block_d == 256


class TestAllStrategies:
    def test_all_strategies_registered(self):
        assert "ucb" in ALL_STRATEGIES
        assert "thompson" in ALL_STRATEGIES
        assert "epsilon" in ALL_STRATEGIES
        assert "reinforce" in ALL_STRATEGIES
        assert "grpo" in ALL_STRATEGIES

    def test_trainer_all_strategies(self):
        for strat in ["ucb", "thompson", "epsilon", "reinforce", "grpo"]:
            trainer = Trainer(strategy=strat)
            candidates = _make_candidates()
            state = _make_state()
            idx = trainer.select(candidates, state)
            assert 0 <= idx < len(candidates)
