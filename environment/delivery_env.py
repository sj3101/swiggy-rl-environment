"""Gymnasium environment for simple and extended food-delivery tasks."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from gymnasium import Env, spaces

from .actions import ACTION_NAMES, ACTION_TO_INDEX, MOVE_DELTAS


class FoodDeliveryEnv(Env):
    """Base delivery grid with a rider, restaurant, and customer."""

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
            "deadline_bonus": 2.0,
            "late_penalty": -5.0,
            "assignment_bonus": 1.5,
        }

        if reward_config:
            self.reward_config.update(reward_config)

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(
            low=np.zeros(7, dtype=np.int32),
            high=np.array(
                [
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    1,
                ],
                dtype=np.int32,
            ),
            dtype=np.int32,
        )

        self.rider_position = np.zeros(2, dtype=np.int32)
        self.picked_up = False
        self.delivered = False
        self.steps_taken = 0
        self.reset()

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
        else:
            reward += self.reward_config["invalid_penalty"]
            info["status"] = "invalid_action"

        self.steps_taken += 1
        truncated = self.steps_taken >= self.max_steps and not terminated

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

    def render(self) -> str:
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


class DeadlineDeliveryEnv(FoodDeliveryEnv):
    """A delivery task with a strict delivery deadline."""

    def __init__(
        self,
        grid_size: int = 6,
        rider_start: tuple[int, int] = (0, 0),
        restaurant_position: tuple[int, int] = (2, 0),
        customer_position: tuple[int, int] = (4, 5),
        deadline_steps: int = 12,
        reward_config: dict[str, float] | None = None,
    ) -> None:
        self.deadline_steps = int(deadline_steps)
        self.time_remaining = self.deadline_steps
        super().__init__(
            grid_size=grid_size,
            rider_start=rider_start,
            restaurant_position=restaurant_position,
            customer_position=customer_position,
            max_steps=self.deadline_steps,
            reward_config=reward_config,
        )
        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.int32),
            high=np.array(
                [
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    1,
                    self.deadline_steps,
                ],
                dtype=np.int32,
            ),
            dtype=np.int32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = super().reset(seed=seed, options=options)
        self.time_remaining = self.deadline_steps
        return self._get_observation(), {"status": "ready"}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        self.time_remaining = max(0, self.deadline_steps - self.steps_taken)
        info["time_remaining"] = self.time_remaining

        if self.delivered and self.time_remaining > 0:
            reward += self.reward_config["deadline_bonus"]
            terminated = True
            info["status"] = "delivered_on_time"
        elif self.steps_taken >= self.deadline_steps and not self.delivered:
            reward += self.reward_config["late_penalty"]
            truncated = True
            info["status"] = "deadline_exceeded"

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        base = super()._get_observation()
        return np.concatenate([base, np.array([self.time_remaining], dtype=np.int32)])


class MultiOrderDeliveryEnv(FoodDeliveryEnv):
    """A simplified multi-order delivery task with two selectable orders."""

    def __init__(
        self,
        grid_size: int = 6,
        rider_start: tuple[int, int] = (0, 0),
        orders: Sequence[tuple[tuple[int, int], tuple[int, int]]] | None = None,
        max_steps: int = 40,
        reward_config: dict[str, float] | None = None,
    ) -> None:
        self.orders = [
            {"restaurant": tuple(order[0]), "customer": tuple(order[1]), "status": 0}
            for order in (orders or (( (1, 0), (4, 5) ), ( (0, 2), (5, 5) )))
        ]
        self.selected_order_index = 0
        self.completed_orders = 0
        super().__init__(
            grid_size=grid_size,
            rider_start=rider_start,
            restaurant_position=self.orders[0]["restaurant"],
            customer_position=self.orders[0]["customer"],
            max_steps=max_steps,
            reward_config=reward_config,
        )
        self.observation_space = spaces.Box(
            low=np.zeros(14, dtype=np.int32),
            high=np.array(
                [
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    2,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    1,
                ],
                dtype=np.int32,
            ),
            dtype=np.int32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        for order in self.orders:
            order["status"] = 0
        self.selected_order_index = 0
        self.completed_orders = 0
        self.picked_up = False
        self.delivered = False
        self.restaurant_position = tuple(self.orders[0]["restaurant"])
        self.customer_position = tuple(self.orders[0]["customer"])
        self.steps_taken = 0
        return self._get_observation(), {"status": "ready"}

    def _selected_order(self) -> dict[str, Any]:
        return self.orders[self.selected_order_index]

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

        if action_name.startswith("SELECT_ORDER_"):
            order_index = int(action_name.split("_")[-1])
            if self.picked_up and self.selected_order_index != order_index:
                reward = self.reward_config["invalid_penalty"]
                info["status"] = "switch_order_invalid"
                self.steps_taken += 1
                truncated = self.steps_taken >= self.max_steps
                return self._get_observation(), reward, terminated, truncated, info
            if 0 <= order_index < len(self.orders):
                self.selected_order_index = order_index
                self.restaurant_position = self.orders[order_index]["restaurant"]
                self.customer_position = self.orders[order_index]["customer"]
                info["status"] = f"selected_order_{order_index}"
            else:
                reward = self.reward_config["invalid_penalty"]
                info["status"] = "order_selection_invalid"
            return self._get_observation(), reward, terminated, truncated, info

        if action_name in MOVE_DELTAS:
            delta_x, delta_y = MOVE_DELTAS[action_name]
            candidate = self.rider_position.copy()
            candidate[0] += delta_x
            candidate[1] += delta_y

            if 0 <= candidate[0] < self.grid_size and 0 <= candidate[1] < self.grid_size:
                self.rider_position = candidate
                reward += self.reward_config["move_penalty"]
                current_order = self._selected_order()
                if tuple(self.rider_position) == current_order["restaurant"] and current_order["status"] == 0:
                    reward += self.reward_config["restaurant_bonus"]
                if self.picked_up and tuple(self.rider_position) == current_order["customer"]:
                    reward += self.reward_config["customer_bonus"]
                info["status"] = "moved"
            else:
                reward += self.reward_config["invalid_penalty"]
                info["status"] = "wall_hit"
        elif action_name == "PICKUP":
            current_order = self._selected_order()
            if tuple(self.rider_position) == current_order["restaurant"] and current_order["status"] == 0 and not self.picked_up:
                current_order["status"] = 1
                self.picked_up = True
                reward += self.reward_config["pickup_bonus"]
                info["status"] = f"picked_up_order_{self.selected_order_index}"
            else:
                reward += self.reward_config["invalid_penalty"]
                info["status"] = "pickup_invalid"
        elif action_name == "DELIVER":
            current_order = self._selected_order()
            if self.picked_up and tuple(self.rider_position) == current_order["customer"] and current_order["status"] == 1:
                current_order["status"] = 2
                self.completed_orders += 1
                self.picked_up = False
                reward += self.reward_config["delivery_bonus"]
                info["status"] = f"delivered_order_{self.selected_order_index}"
                if self.completed_orders >= len(self.orders):
                    terminated = True
                    info["status"] = "all_orders_completed"
            else:
                reward += self.reward_config["invalid_penalty"]
                info["status"] = "delivery_invalid"
        else:
            reward += self.reward_config["invalid_penalty"]
            info["status"] = "invalid_action"

        self.steps_taken += 1
        truncated = self.steps_taken >= self.max_steps and not terminated
        if truncated and not terminated:
            info["status"] = "max_steps_reached"

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        obs = [
            int(self.rider_position[0]),
            int(self.rider_position[1]),
        ]
        for order in self.orders:
            obs.extend(
                [
                    order["restaurant"][0],
                    order["restaurant"][1],
                    order["customer"][0],
                    order["customer"][1],
                    int(order["status"]),
                ]
            )
        obs.extend([self.selected_order_index, int(self.picked_up)])
        return np.array(obs, dtype=np.int32)


class MultiRiderDeliveryEnv(FoodDeliveryEnv):
    """A simplified order-assignment task with multiple riders."""

    def __init__(
        self,
        grid_size: int = 6,
        rider_positions: Sequence[tuple[int, int]] = ((0, 0), (5, 5)),
        orders: Sequence[tuple[tuple[int, int], tuple[int, int]]] | None = None,
        max_steps: int = 30,
        max_workload: int = 2,
        reward_config: dict[str, float] | None = None,
    ) -> None:
        self.initial_rider_positions = [tuple(position) for position in rider_positions]
        self.rider_positions = [tuple(position) for position in self.initial_rider_positions]
        self.rider_workloads = [0 for _ in self.rider_positions]
        self.orders = [
            {"restaurant": tuple(order[0]), "customer": tuple(order[1]), "status": 0}
            for order in (orders or (((1, 0), (3, 4)), ((4, 5), (1, 1))))
        ]
        self.max_workload = int(max_workload)
        self.assigned_orders = 0
        super().__init__(
            grid_size=grid_size,
            rider_start=self.rider_positions[0],
            restaurant_position=self.orders[0]["restaurant"],
            customer_position=self.orders[0]["customer"],
            max_steps=max_steps,
            reward_config=reward_config,
        )
        self.observation_space = spaces.Box(
            low=np.zeros(17, dtype=np.int32),
            high=np.array(
                [
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.max_workload,
                    self.max_workload,
                ],
                dtype=np.int32,
            ),
            dtype=np.int32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.rider_positions = [tuple(position) for position in self.initial_rider_positions]
        self.rider_workloads = [0 for _ in self.rider_positions]
        for order in self.orders:
            order["status"] = 0
        self.assigned_orders = 0
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

        if action_name.startswith("ASSIGN_ORDER_") and "_TO_RIDER_" in action_name:
            left, right = action_name.split("_TO_")
            order_index = int(left.split("_")[-1])
            rider_index = int(right.split("_")[-1])

            if 0 <= order_index < len(self.orders) and 0 <= rider_index < len(self.rider_positions):
                order = self.orders[order_index]
                if order["status"] == 0 and self.rider_workloads[rider_index] < self.max_workload:
                    order["status"] = 1
                    self.rider_workloads[rider_index] += 1
                    self.assigned_orders += 1
                    reward += self.reward_config["assignment_bonus"]
                    info["status"] = f"assigned_order_{order_index}_to_rider_{rider_index}"
                    if self.assigned_orders >= len(self.orders):
                        terminated = True
                        info["status"] = "all_orders_assigned"
                else:
                    reward += self.reward_config["invalid_penalty"]
                    info["status"] = "assignment_invalid"
            else:
                reward += self.reward_config["invalid_penalty"]
                info["status"] = "assignment_invalid"
        else:
            reward += self.reward_config["invalid_penalty"]
            info["status"] = "invalid_action"

        self.steps_taken += 1
        truncated = self.steps_taken >= self.max_steps and not terminated
        if truncated and not terminated:
            info["status"] = "max_steps_reached"

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        obs = []
        for rider_position in self.rider_positions:
            obs.extend([int(rider_position[0]), int(rider_position[1])])
        for order in self.orders:
            obs.extend(
                [
                    order["restaurant"][0],
                    order["restaurant"][1],
                    order["customer"][0],
                    order["customer"][1],
                    int(order["status"]),
                ]
            )
        obs.extend(self.rider_workloads)
        obs.append(self.assigned_orders)
        return np.array(obs, dtype=np.int32)
