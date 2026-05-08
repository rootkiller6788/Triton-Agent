"""Bandit-based optimization: UCB and Thompson Sampling for config selection.

Core idea: each (template_id, BLOCK_SIZE, num_warps, num_stages, vectorize)
combination is an arm. The bandit learns which arms perform best for a given
(op, shape, dtype, device) context.
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from triton_agent.core.spec import OpState, OpAction, CandidateResult


@dataclass
class ArmStats:
    """Running statistics for one bandit arm."""
    count: int = 0
    mean_reward: float = 0.0
    variance: float = 0.0
    sum_rewards: float = 0.0
    sum_sq_rewards: float = 0.0

    def update(self, reward: float) -> None:
        self.count += 1
        self.sum_rewards += reward
        self.sum_sq_rewards += reward * reward
        self.mean_reward = self.sum_rewards / self.count
        if self.count > 1:
            self.variance = (self.sum_sq_rewards - self.sum_rewards ** 2 / self.count) / (self.count - 1)
            self.variance = max(0.0, self.variance)


def _arm_key(action: OpAction) -> str:
    """Stable string key for an action/arm."""
    return f"{action.template_id}|{action.block_d}|{action.num_warps}|{action.num_stages}|{int(action.vectorize)}"


def _context_key(state: OpState) -> str:
    """Context key grouping similar (op, shape, dtype, device) configurations."""
    return f"{state.op_name}|{state.B}|{state.T}|{state.D}|{state.dtype}|{state.device}"


class UCBPolicy:
    """Upper Confidence Bound policy for arm selection.

    UCB_i = mean_i + c * sqrt( log(total_count) / count_i )
    """

    def __init__(self, c: float = 2.0):
        self.c = c
        self._arms: dict[str, ArmStats] = defaultdict(ArmStats)
        self._total_count: int = 0

    def select(self, candidate_actions: list[OpAction], state: OpState) -> int:
        """Select the arm with highest UCB. Returns index into candidate_actions."""
        self._total_count += 1
        ctx = _context_key(state)

        best_idx = 0
        best_ucb = -float("inf")

        for i, action in enumerate(candidate_actions):
            key = f"{ctx}|{_arm_key(action)}"
            stats = self._arms[key]

            if stats.count == 0:
                ucb = float("inf")
            else:
                exploration = self.c * math.sqrt(math.log(self._total_count) / stats.count)
                ucb = stats.mean_reward + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = i

        return best_idx

    def update(self, action: OpAction, state: OpState, reward: float) -> None:
        """Update the statistics for the selected arm."""
        ctx = _context_key(state)
        key = f"{ctx}|{_arm_key(action)}"
        self._arms[key].update(reward)

    def best_action(self, candidate_actions: list[OpAction], state: OpState) -> tuple[int, float]:
        """Return the index and mean reward of the empirically best arm."""
        ctx = _context_key(state)
        best_idx = 0
        best_mean = -float("inf")
        for i, action in enumerate(candidate_actions):
            key = f"{ctx}|{_arm_key(action)}"
            mean = self._arms[key].mean_reward
            if mean > best_mean:
                best_mean = mean
                best_idx = i
        return best_idx, best_mean

    @property
    def arm_count(self) -> int:
        return len(self._arms)


class ThompsonSamplingPolicy:
    """Thompson Sampling using Beta-distribution approximation for reward.

    Reward is clamped/transformed to [0, 1] range. Each arm maintains
    (alpha, beta) parameters of a Beta distribution.

    Simplified: we treat each arm's reward history as successes/failures
    based on whether reward > 0.5 threshold.
    """

    def __init__(self):
        self._arms: dict[str, tuple[float, float]] = defaultdict(lambda: (1.0, 1.0))

    def select(self, candidate_actions: list[OpAction], state: OpState) -> int:
        """Select the arm with highest Thompson sample."""
        ctx = _context_key(state)
        best_idx = 0
        best_sample = -float("inf")
        rng = random.Random()

        for i, action in enumerate(candidate_actions):
            key = f"{ctx}|{_arm_key(action)}"
            alpha, beta = self._arms[key]
            sample = rng.betavariate(alpha, beta)
            if sample > best_sample:
                best_sample = sample
                best_idx = i

        return best_idx

    def update(self, action: OpAction, state: OpState, reward: float) -> None:
        """Update Beta parameters with a soft reward update."""
        ctx = _context_key(state)
        key = f"{ctx}|{_arm_key(action)}"
        alpha, beta = self._arms[key]

        clipped = max(0.0, min(1.0, reward))
        weight = 1.0
        alpha += clipped * weight
        beta += (1.0 - clipped) * weight

        self._arms[key] = (alpha, beta)

    def best_action(self, candidate_actions: list[OpAction], state: OpState) -> tuple[int, float]:
        """Return index and mean of the empirically best arm."""
        ctx = _context_key(state)
        best_idx = 0
        best_mean = -float("inf")
        for i, action in enumerate(candidate_actions):
            key = f"{ctx}|{_arm_key(action)}"
            alpha, beta = self._arms[key]
            mean = alpha / (alpha + beta)
            if mean > best_mean:
                best_mean = mean
                best_idx = i
        return best_idx, best_mean

    @property
    def arm_count(self) -> int:
        return len(self._arms)


class EpsilonGreedyPolicy:
    """Epsilon-greedy with adaptive epsilon decay.

    epsilon = max(epsilon_min, epsilon_init / sqrt(1 + total_count * decay_rate))
    """

    def __init__(self, epsilon_init: float = 0.3, epsilon_min: float = 0.05, decay_rate: float = 0.01):
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self._arms: dict[str, ArmStats] = defaultdict(ArmStats)
        self._total_count: int = 0

    def _epsilon(self) -> float:
        return max(self.epsilon_min, self.epsilon_init / math.sqrt(1 + self._total_count * self.decay_rate))

    def select(self, candidate_actions: list[OpAction], state: OpState) -> int:
        """Select an arm using epsilon-greedy."""
        self._total_count += 1
        ctx = _context_key(state)

        if random.random() < self._epsilon():
            return random.randrange(len(candidate_actions))

        best_idx = 0
        best_mean = -float("inf")
        for i, action in enumerate(candidate_actions):
            key = f"{ctx}|{_arm_key(action)}"
            mean = self._arms[key].mean_reward
            if self._arms[key].count == 0:
                mean = 0.5
            if mean > best_mean:
                best_mean = mean
                best_idx = i
        return best_idx

    def update(self, action: OpAction, state: OpState, reward: float) -> None:
        ctx = _context_key(state)
        key = f"{ctx}|{_arm_key(action)}"
        self._arms[key].update(reward)

    def best_action(self, candidate_actions: list[OpAction], state: OpState) -> tuple[int, float]:
        ctx = _context_key(state)
        best_idx = 0
        best_mean = -float("inf")
        for i, action in enumerate(candidate_actions):
            key = f"{ctx}|{_arm_key(action)}"
            mean = self._arms[key].mean_reward
            if mean > best_mean:
                best_mean = mean
                best_idx = i
        return best_idx, best_mean

    @property
    def arm_count(self) -> int:
        return len(self._arms)


# Legacy aliases for CLI dispatch
def select_action_ucb(state: OpState, history: list) -> Any:
    """Legacy wrapper: returns a bandit policy instance."""
    return UCBPolicy()


def select_action_thompson(state: OpState, history: list) -> Any:
    """Legacy wrapper: returns a bandit policy instance."""
    return ThompsonSamplingPolicy()


def update_bandit(policy: Any, action: OpAction, state: OpState, reward: float) -> None:
    """Legacy wrapper: update bandit policy with feedback."""
    policy.update(action, state, reward)
