"""Random baseline agent."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces


class RandomPolicyAgent:
    """Simple random-action baseline used for comparison against Q-learning."""

    def __init__(self, action_space: spaces.Space, seed: int | None = None) -> None:
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: Any, training: bool = False) -> int:
        return int(self.rng.integers(0, int(self.action_space.n)))

    def update(self, observation: Any, action: int, reward: float, next_observation: Any, terminated: bool) -> None:
        return None

    def decay_epsilon(self) -> float:
        return 0.0
