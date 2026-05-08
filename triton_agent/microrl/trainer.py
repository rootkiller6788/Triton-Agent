"""MicroRL trainer: policy update orchestration and shape-to-config mapping.

Supports three strategy families:
  Bandit:   UCB / Thompson / Epsilon-Greedy
  PG:       REINFORCE-lite
  GroupPG:  GRPO-lite (group relative policy optimization)
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

from triton_agent.core.spec import OpState, OpAction, CandidateResult
from triton_agent.microrl.bandit import UCBPolicy, ThompsonSamplingPolicy, EpsilonGreedyPolicy
from triton_agent.microrl.reinforce_lite import REINFORCELite
from triton_agent.microrl.grpo_lite import GRPOLite


PolicyType = Union[UCBPolicy, ThompsonSamplingPolicy, EpsilonGreedyPolicy, REINFORCELite, GRPOLite]

BANDIT_MAP = {
    "ucb": UCBPolicy,
    "thompson": ThompsonSamplingPolicy,
    "epsilon_greedy": EpsilonGreedyPolicy,
    "epsilon": EpsilonGreedyPolicy,
}

PG_MAP = {
    "reinforce": REINFORCELite,
    "reinforce_lite": REINFORCELite,
}

GRPO_MAP = {
    "grpo": GRPOLite,
    "grpo_lite": GRPOLite,
}

ALL_STRATEGIES = {**BANDIT_MAP, **PG_MAP, **GRPO_MAP}


class ShapeConfigStore:
    """Persistent mapping from (op, shape, dtype) -> best config."""

    def __init__(self, path: str | Path = "leaderboard/best_configs.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def _shape_key(self, state: OpState) -> str:
        return f"{state.op_name}|{state.B}|{state.T}|{state.D}|{state.dtype}|{state.device}"

    def get_best(self, state: OpState) -> Optional[dict[str, Any]]:
        return self._data.get(self._shape_key(state))

    def set_best(self, state: OpState, action: OpAction, result: CandidateResult) -> None:
        key = self._shape_key(state)
        self._data[key] = {
            "action": action.to_dict(),
            "speedup": result.speedup,
            "latency_us_p50": result.latency_us_p50,
            "reward": result.reward,
        }
        self.save()

    def get_historical_best(self, state: OpState) -> Optional[dict]:
        entry = self.get_best(state)
        if entry:
            state.historical_best_config = entry["action"]
            return entry["action"]
        return None


class Trainer:
    """Orchestrates RL policy training across episodes.

    Supports bandit, REINFORCE-lite, and GRPO-lite strategies with
    a unified select/update interface.
    """

    def __init__(self, strategy: str = "ucb", **policy_kwargs) -> None:
        self.strategy = strategy
        self._policy_type = self._classify(strategy)
        policy_cls = ALL_STRATEGIES.get(strategy, UCBPolicy)
        self.policy: PolicyType = policy_cls(**policy_kwargs)
        self.config_store = ShapeConfigStore()
        self._group_buffer: list[tuple[OpAction, float]] = []

    def _classify(self, strategy: str) -> str:
        if strategy in BANDIT_MAP:
            return "bandit"
        elif strategy in PG_MAP:
            return "pg"
        elif strategy in GRPO_MAP:
            return "grpo"
        return "bandit"

    def select(
        self, candidates: list[tuple[str, OpAction]], state: OpState
    ) -> int:
        """Select next arm to try using the current policy."""
        if self._policy_type in ("pg", "grpo"):
            return self.policy.select(candidates, state)
        actions = [action for _, action in candidates]
        return self.policy.select(actions, state)

    def select_group(
        self, candidates: list[tuple[str, OpAction]], state: OpState
    ) -> list[int]:
        """For GRPO: sample a group of action indices."""
        if hasattr(self.policy, "select_group"):
            return self.policy.select_group(candidates, state)
        return [self.select(candidates, state)]

    def update(
        self,
        action: OpAction,
        state: OpState,
        result: CandidateResult,
        candidates: list[tuple[str, OpAction]] | None = None,
    ) -> None:
        """Feed a candidate evaluation result back into the policy."""
        reward = result.reward

        if self._policy_type == "bandit":
            self.policy.update(action, state, reward)
        elif self._policy_type in ("pg", "grpo"):
            if candidates is not None and hasattr(self.policy, "update"):
                self.policy.update(action, state, reward, candidates)
            else:
                self.policy.update(action, state, reward)

        if result.promoted:
            self.config_store.set_best(state, action, result)

    def update_group(
        self,
        actions: list[OpAction],
        rewards: list[float],
        state: OpState,
        candidates: list[tuple[str, OpAction]],
    ) -> None:
        """GRPO group update."""
        if self._policy_type == "grpo" and hasattr(self.policy, "update_group"):
            self.policy.update_group(actions, rewards, state, candidates)

    def get_best_config(self, state: OpState) -> Optional[dict]:
        return self.config_store.get_best(state)

    def flush_group(self) -> None:
        self._group_buffer.clear()


def update(policy: Any, action: OpAction, state: OpState, reward: float) -> None:
    """Legacy wrapper."""
    policy.update(action, state, reward)
