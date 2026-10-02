"""Simplified multi-rider assignment task configuration and demo script."""

from __future__ import annotations

from dataclasses import dataclass, field

from environment.actions import ACTION_TO_INDEX
from environment.delivery_env import MultiRiderDeliveryEnv


@dataclass(frozen=True)
class OrderConfig:
    restaurant_position: tuple[int, int]
    customer_position: tuple[int, int]


@dataclass(frozen=True)
class RewardConfig:
    assignment_bonus: float = 1.5
    invalid_penalty: float = -1.0
    move_penalty: float = -0.05


@dataclass(frozen=True)
class MultiRiderTaskConfig:
    grid_size: int = 6
    rider_positions: tuple[tuple[int, int], ...] = ((0, 0), (5, 5))
    orders: tuple[OrderConfig, ...] = (
        OrderConfig(restaurant_position=(1, 0), customer_position=(3, 4)),
        OrderConfig(restaurant_position=(4, 5), customer_position=(1, 1)),
    )
    max_steps: int = 30
    max_workload: int = 2
    reward_config: RewardConfig = field(default_factory=RewardConfig)


def create_environment(task_config: MultiRiderTaskConfig | None = None) -> MultiRiderDeliveryEnv:
    config = task_config or MultiRiderTaskConfig()
    return MultiRiderDeliveryEnv(
        grid_size=config.grid_size,
        rider_positions=config.rider_positions,
        orders=[(order.restaurant_position, order.customer_position) for order in config.orders],
        max_steps=config.max_steps,
        max_workload=config.max_workload,
        reward_config={
            "assignment_bonus": config.reward_config.assignment_bonus,
            "invalid_penalty": config.reward_config.invalid_penalty,
            "move_penalty": config.reward_config.move_penalty,
        },
    )


def run_demo() -> None:
    env = create_environment()
    observation, info = env.reset()

    print("=== Demo: Multi-Rider Assignment Task ===")
    print(f"Initial observation={observation.tolist()}")
    print(f"info={info}")

    actions = ["ASSIGN_ORDER_0_TO_RIDER_0", "ASSIGN_ORDER_1_TO_RIDER_1", "ASSIGN_ORDER_0_TO_RIDER_1"]
    for action_name in actions:
        next_obs, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action_name])
        print(
            f"Action {action_name:22} -> obs={next_obs.tolist()} reward={reward:.2f} "
            f"terminated={terminated} truncated={truncated} info={info}"
        )
        if terminated or truncated:
            break

    print("Reset check:")
    reset_obs, _ = env.reset()
    print(f"reset_obs={reset_obs.tolist()}")


if __name__ == "__main__":
    run_demo()
