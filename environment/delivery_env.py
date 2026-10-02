"""Gymnasium environment for a simple food-delivery task."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import Env, spaces

from .actions import ACTION_NAMES, MOVE_DELTAS


class FoodDeliveryEnv(Env):
    """A minimal delivery grid environment for a rider to pick up and deliver an order."""

    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = 6,
        rider_start: tuple[int, int] = (0, 0),
        restaurant_position: tuple[int, int] = (2, 0),
        customer_position: tuple[int, int] = (4, 5),
        max_steps: int = 60,
        reward_config: dict[str, float] | None = None,
    ) -> None:
        self.grid_size = int(grid_size)
        self.rider_start = tuple(rider_start)
        self.restaurant_position = tuple(restaurant_position)
        self.customer_position = tuple(customer_position)
        self.max_steps = int(max_steps)

        self.reward_config = {
            "move_penalty": -0.05,
            "restaurant_bonus": 0.25,
            "pickup_bonus": 1.0,
            "customer_bonus": 0.5,
            "delivery_bonus": 10.0,
            "invalid_penalty": -1.0,
        }

        if reward_config:
            self.reward_config.update(reward_config)

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
            high=np.array(
                [self.grid_size - 1, self.grid_size - 1, self.grid_size - 1, self.grid_size - 1, self.grid_size - 1, self.grid_size - 1, 1],
                dtype=np.int32,
            ),
            dtype=np.int32,
        )

        self.rider_position = np.zeros(2, dtype=np.int32)
        self.picked_up = False
        self.delivered = False
        self.steps_taken = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.rider_position = np.array(self.rider_start, dtype=np.int32)
        self.picked_up = False
        self.delivered = False
        self.steps_taken = 0

        return self._get_observation(), {"status": "ready"}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not isinstance(action, (int, np.integer)):
            try:
                action = int(action)
            except (TypeError, ValueError):
                action = -1

        reward = 0.0
        info: dict[str, Any] = {"status": "in_progress"}
        terminated = False
        truncated = False

        if action < 0 or action >= self.action_space.n:
            reward = self.reward_config["invalid_penalty"]
            self.steps_taken += 1
            truncated = self.steps_taken >= self.max_steps
            info["status"] = "invalid_action"
            return self._get_observation(), reward, terminated, truncated, info

        action_name = ACTION_NAMES[action]

        if action_name in MOVE_DELTAS:
            delta_x, delta_y = MOVE_DELTAS[action_name]
            candidate = self.rider_position.copy()
            candidate[0] += delta_x
            candidate[1] += delta_y

            if 0 <= candidate[0] < self.grid_size and 0 <= candidate[1] < self.grid_size:
                self.rider_position = candidate
                reward += self.reward_config["move_penalty"]
                if tuple(self.rider_position) == self.restaurant_position and not self.picked_up:
                    reward += self.reward_config["restaurant_bonus"]
                if self.picked_up and tuple(self.rider_position) == self.customer_position:
                    reward += self.reward_config["customer_bonus"]
                info["status"] = "moved"
            else:
                reward += self.reward_config["invalid_penalty"]
                info["status"] = "wall_hit"

        elif action_name == "PICKUP":
            if tuple(self.rider_position) == self.restaurant_position and not self.picked_up and not self.delivered:
                self.picked_up = True
                reward += self.reward_config["pickup_bonus"]
                info["status"] = "picked_up"
            else:
                reward += self.reward_config["invalid_penalty"]
                info["status"] = "pickup_invalid"

        elif action_name == "DELIVER":
            if self.picked_up and tuple(self.rider_position) == self.customer_position and not self.delivered:
                self.delivered = True
                reward += self.reward_config["delivery_bonus"]
                terminated = True
                info["status"] = "delivered"
            else:
                reward += self.reward_config["invalid_penalty"]
                info["status"] = "delivery_invalid"

        self.steps_taken += 1
        truncated = self.steps_taken >= self.max_steps

        if truncated and not terminated:
            info["status"] = "max_steps_reached"

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        rider_position = tuple(int(value) for value in self.rider_position)
        return np.array(
            [
                rider_position[0],
                rider_position[1],
                self.restaurant_position[0],
                self.restaurant_position[1],
                self.customer_position[0],
                self.customer_position[1],
                int(self.picked_up),
            ],
            dtype=np.int32,
        )

    def render(self):
        """Simple debug rendering for console output."""
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        restaurant_x, restaurant_y = self.restaurant_position
        customer_x, customer_y = self.customer_position

        grid[restaurant_y][restaurant_x] = "R"
        grid[customer_y][customer_x] = "C"
        rider_x, rider_y = self.rider_position
        grid[rider_y][rider_x] = "A"

        if self.delivered:
            grid[rider_y][rider_x] = "D"

        return "\n".join(" ".join(row) for row in grid)
