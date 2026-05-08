"""GRPO-lite: Group Relative Policy Optimization for config search.

Unlike raw REINFORCE which uses an absolute baseline, GRPO compares each
candidate within a group (batch) against the group mean reward. This reduces
variance and gives more stable updates for operator tuning where reward
distributions may be skewed.

Implementation: simplified GRPO without KL penalty.
  - Sample G actions per step
  - Evaluate all G actions, compute group mean mu and std sigma
  - For each action: advantage_i = (reward_i - mu) / (sigma + eps)
  - Update logits: delta_i = lr * advantage_i * (1 - prob_i)
"""

import math
import random
from collections import defaultdict
from typing import Any

from triton_agent.core.spec import OpState, OpAction
from triton_agent.microrl.reinforce_lite import (
    PolicyWeights,
    _arm_key,
    _context_key,
    _softmax,
    _sample,
)


class GRPOLite:
    """Group Relative Policy Optimization for config tuning.

    Group size G controls how many actions are sampled and compared per step.
    Higher G gives more stable advantage estimates but costs more evaluations.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        temperature: float = 1.0,
        group_size: int = 4,
    ):
        self.lr = learning_rate
        self.temperature = temperature
        self.group_size = group_size
        self._policy = PolicyWeights()
        self._step_count: int = 0

    def select(self, candidate_actions: list[tuple[str, OpAction]], state: OpState) -> int:
        """Sample one action index from the softmax policy."""
        ctx = _context_key(state)
        keys = [_arm_key(act) for _, act in candidate_actions]
        logits = [self._policy.get_logit(f"{ctx}|{k}") for k in keys]
        probs = _softmax(logits, self.temperature)

        if random.random() < 0.05 or self._step_count < self.group_size:
            return random.randrange(len(candidate_actions))

        return _sample(probs)

    def select_group(
        self, candidate_actions: list[tuple[str, OpAction]], state: OpState
    ) -> list[int]:
        """Sample a group of G distinct action indices for batch evaluation."""
        ctx = _context_key(state)
        keys = [_arm_key(act) for _, act in candidate_actions]
        logits = [self._policy.get_logit(f"{ctx}|{k}") for k in keys]
        probs = _softmax(logits, self.temperature)

        indices: list[int] = []
        population = list(range(len(candidate_actions)))
        weights = probs
        batch_size = min(self.group_size, len(candidate_actions))

        chosen = random.choices(population, weights=weights, k=batch_size)
        seen: set[int] = set()
        for idx in chosen:
            if idx not in seen:
                indices.append(idx)
                seen.add(idx)
        while len(indices) < batch_size:
            idx = random.randrange(len(candidate_actions))
            if idx not in seen:
                indices.append(idx)
                seen.add(idx)

        return indices

    def update_group(
        self,
        actions: list[OpAction],
        rewards: list[float],
        state: OpState,
        candidate_actions: list[tuple[str, OpAction]],
    ) -> None:
        """GRPO group update: normalize rewards within group, apply policy gradient."""
        self._step_count += 1
        if len(rewards) == 0:
            return

        mu = sum(rewards) / len(rewards)
        var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
        sigma = math.sqrt(var) + 1e-6

        ctx = _context_key(state)
        keys = [_arm_key(act) for _, act in candidate_actions]
        logits = [self._policy.get_logit(f"{ctx}|{k}") for k in keys]
        probs = _softmax(logits, self.temperature)

        for action, reward in zip(actions, rewards):
            advantage = (reward - mu) / sigma

            selected_idx = next(
                i for i, (_, act) in enumerate(candidate_actions)
                if _arm_key(act) == _arm_key(action)
            )

            for i, key in enumerate(keys):
                full_key = f"{ctx}|{key}"
                grad = advantage * ((1.0 if i == selected_idx else 0.0) - probs[i])
                self._policy.update(full_key, self.lr * grad)

    def update(
        self,
        action: OpAction,
        state: OpState,
        reward: float,
        candidate_actions: list[tuple[str, OpAction]],
    ) -> None:
        """Single-action update (stored for group batch later)."""
        pass

    def best_action(self, candidate_actions: list[tuple[str, OpAction]], state: OpState) -> int:
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


def grpo_step(policy: GRPOLite, group_episodes: list[dict]) -> None:
    """Legacy compatibility wrapper for group update."""
    pass
