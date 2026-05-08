"""REINFORCE-lite: lightweight policy gradient for config optimization.

Maintains a softmax policy over candidate actions. Each action's log-probability
is weighted by the reward (speedup + correctness) to compute a policy gradient
update step.

Unlike full REINFORCE, this uses a non-differentiable reward signal and only
needs action sampling + log-prob storage, no neural network required.
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from triton_agent.core.spec import OpState, OpAction


@dataclass
class PolicyWeights:
    """Learnable log-preference weights per arm (action fingerprint)."""
    weights: dict[str, float] = field(default_factory=dict)

    def get_logit(self, key: str) -> float:
        return self.weights.get(key, 0.0)

    def update(self, key: str, logit_delta: float) -> None:
        self.weights[key] = self.weights.get(key, 0.0) + logit_delta


def _arm_key(action: OpAction) -> str:
    return f"{action.template_id}|{action.block_d}|{action.num_warps}|{action.num_stages}|{int(action.vectorize)}"


def _context_key(state: OpState) -> str:
    return f"{state.op_name}|{state.B}|{state.T}|{state.D}|{state.dtype}|{state.device}"


def _softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        temperature = 1e-6
    scaled = [l / temperature for l in logits]
    max_l = max(scaled)
    exp_sum = sum(math.exp(l - max_l) for l in scaled)
    return [math.exp(l - max_l) / exp_sum for l in scaled]


def _sample(probs: list[float]) -> int:
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


class REINFORCELite:
    """Lightweight REINFORCE: maintains logit weights per arm, samples from
    softmax, updates via reward-weighted log-prob gradient.

    delta_weight_i = learning_rate * (reward - baseline) * (1 - prob_i)   for selected arm
    delta_weight_j = learning_rate * (reward - baseline) * (0 - prob_j)   for non-selected arms
    """

    def __init__(self, learning_rate: float = 0.1, temperature: float = 1.0, baseline_smoothing: float = 0.9):
        self.lr = learning_rate
        self.temperature = temperature
        self.baseline_smoothing = baseline_smoothing
        self._policy = PolicyWeights()
        self._baseline: dict[str, float] = defaultdict(float)
        self._step_count: int = 0

    def select(self, candidate_actions: list[tuple[str, OpAction]], state: OpState) -> int:
        """Sample an action index from the softmax policy."""
        ctx = _context_key(state)
        keys = [_arm_key(act) for _, act in candidate_actions]
        logits = [self._policy.get_logit(f"{ctx}|{k}") for k in keys]
        probs = _softmax(logits, self.temperature)

        if random.random() < 0.05:
            return random.randrange(len(candidate_actions))

        return _sample(probs)

    def update(
        self,
        action: OpAction,
        state: OpState,
        reward: float,
        candidate_actions: list[tuple[str, OpAction]],
    ) -> None:
        """REINFORCE update: reward-weighted gradient on softmax logits."""
        self._step_count += 1
        ctx = _context_key(state)
        keys = [_arm_key(act) for _, act in candidate_actions]
        logits = [self._policy.get_logit(f"{ctx}|{k}") for k in keys]
        probs = _softmax(logits, self.temperature)

        baseline_key = f"{ctx}|baseline"
        old_baseline = self._baseline[baseline_key]
        new_baseline = self.baseline_smoothing * old_baseline + (1 - self.baseline_smoothing) * reward
        self._baseline[baseline_key] = new_baseline
        advantage = reward - new_baseline

        selected_idx = next(
            i for i, (_, act) in enumerate(candidate_actions)
            if _arm_key(act) == _arm_key(action)
        )

        for i, key in enumerate(keys):
            full_key = f"{ctx}|{key}"
            grad = advantage * ((1.0 if i == selected_idx else 0.0) - probs[i])
            self._policy.update(full_key, self.lr * grad)

    def best_action(self, candidate_actions: list[tuple[str, OpAction]], state: OpState) -> int:
        """Return the index of the action with the highest logit weight."""
        ctx = _context_key(state)
        best_idx = 0
        best_logit = -float("inf")
        for i, (_, act) in enumerate(candidate_actions):
            key = f"{ctx}|{_arm_key(act)}"
            logit = self._policy.get_logit(key)
            if logit > best_logit:
                best_logit = logit
                best_idx = i
        return best_idx

    @property
    def arm_count(self) -> int:
        return len(self._policy.weights)


def reinforce_step(policy: REINFORCELite, episode: dict) -> None:
    """Legacy compatibility wrapper for a single REINFORCE update."""
    pass
