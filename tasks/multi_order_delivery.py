"""Multi-order delivery task configuration and demo script."""

from __future__ import annotations

from dataclasses import dataclass, field

from environment.actions import ACTION_TO_INDEX
from environment.delivery_env import MultiOrderDeliveryEnv


@dataclass(frozen=True)
class OrderConfig:
    restaurant_position: tuple[int, int]
    customer_position: tuple[int, int]


@dataclass(frozen=True)
class RewardConfig:
    move_penalty: float = -0.05
    restaurant_bonus: float = 0.25
    pickup_bonus: float = 1.0
    customer_bonus: float = 0.5
    delivery_bonus: float = 10.0
    invalid_penalty: float = -1.0


@dataclass(frozen=True)
class MultiOrderTaskConfig:
    grid_size: int = 6
    rider_start: tuple[int, int] = (0, 0)
    orders: tuple[OrderConfig, ...] = (
        OrderConfig(restaurant_position=(1, 0), customer_position=(4, 5)),
        OrderConfig(restaurant_position=(0, 2), customer_position=(5, 5)),
    )
    max_steps: int = 40
    reward_config: RewardConfig = field(default_factory=RewardConfig)


def create_environment(task_config: MultiOrderTaskConfig | None = None) -> MultiOrderDeliveryEnv:
    config = task_config or MultiOrderTaskConfig()
    return MultiOrderDeliveryEnv(
        grid_size=config.grid_size,
        rider_start=config.rider_start,
        orders=[(order.restaurant_position, order.customer_position) for order in config.orders],
        max_steps=config.max_steps,
        reward_config={
            "move_penalty": config.reward_config.move_penalty,
            "restaurant_bonus": config.reward_config.restaurant_bonus,
            "pickup_bonus": config.reward_config.pickup_bonus,
            "customer_bonus": config.reward_config.customer_bonus,
            "delivery_bonus": config.reward_config.delivery_bonus,
            "invalid_penalty": config.reward_config.invalid_penalty,
        },
    )


def run_demo() -> None:
    env = create_environment()
    observation, info = env.reset()

    print("=== Demo: Multi-Order Delivery Task ===")
    print(f"Initial observation={observation.tolist()}")
    print(f"info={info}")

    actions = [
        "SELECT_ORDER_0",
        "RIGHT",
        "PICKUP",
        "DOWN",
        "DOWN",
        "DOWN",
        "DOWN",
        "DOWN",
        "RIGHT",
        "RIGHT",
        "RIGHT",
        "DELIVER",
        "SELECT_ORDER_1",
        "LEFT",
        "LEFT",
        "LEFT",
        "LEFT",
        "UP",
        "UP",
        "UP",
        "PICKUP",
        "RIGHT",
        "RIGHT",
        "RIGHT",
        "RIGHT",
        "RIGHT",
        "DOWN",
        "DOWN",
        "DOWN",
        "DELIVER",
    ]

    for action_name in actions:
        next_obs, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action_name])
        print(
            f"Action {action_name:12} -> obs={next_obs.tolist()} reward={reward:.2f} "
            f"terminated={terminated} truncated={truncated} info={info}"
        )
        if terminated or truncated:
            break

    print("Reset check:")
    reset_obs, _ = env.reset()
    print(f"reset_obs={reset_obs.tolist()}")


if __name__ == "__main__":
    run_demo()
