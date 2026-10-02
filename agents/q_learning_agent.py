"""Tabular Q-learning agent for discrete delivery tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces


@dataclass
class QLearningConfig:
    learning_rate: float = 0.3
    discount_factor: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05
    seed: int | None = None


class QLearningAgent:
    """Simple tabular Q-learning agent for discrete state/action tasks."""

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, config: QLearningConfig | None = None) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config or QLearningConfig()
        self.q_table: dict[tuple[int, ...], np.ndarray] = {}
        self.epsilon = self.config.epsilon
        self.rng = np.random.default_rng(self.config.seed)

    def _state_key(self, observation: Any) -> tuple[int, ...]:
        arr = np.asarray(observation, dtype=np.int32)
        return tuple(int(value) for value in arr.tolist())

    def initialize_q_table(self) -> None:
        self.q_table = {}

    def _ensure_state(self, observation: Any) -> tuple[int, ...]:
        state_key = self._state_key(observation)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(int(self.action_space.n), dtype=float)
        return state_key

    def select_action(self, observation: Any, training: bool = True) -> int:
        state_key = self._ensure_state(observation)
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, int(self.action_space.n)))
        return int(np.argmax(self.q_table[state_key]))

    def update(self, observation: Any, action: int, reward: float, next_observation: Any, terminated: bool) -> float:
        current_key = self._ensure_state(observation)
        next_key = self._ensure_state(next_observation)
        current_q = self.q_table[current_key]

        if terminated:
            target = reward
        else:
            target = reward + self.config.discount_factor * float(np.max(self.q_table[next_key]))

        old_value = current_q[action]
        current_q[action] = (1.0 - self.config.learning_rate) * old_value + self.config.learning_rate * target
        self.q_table[current_key] = current_q
        return float(current_q[action])

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.epsilon_decay)
        return self.epsilon

    def save(self, file_path: str) -> None:
        import pickle

        with open(file_path, "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, file_path: str, observation_space: spaces.Space | None = None, action_space: spaces.Space | None = None) -> "QLearningAgent":
        import pickle

        with open(file_path, "rb") as handle:
            loaded = pickle.load(handle)
        if observation_space is not None:
            loaded.observation_space = observation_space
        if action_space is not None:
            loaded.action_space = action_space
        loaded.rng = np.random.default_rng(loaded.config.seed)
        return loaded
