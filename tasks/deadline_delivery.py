"""Deadline-based delivery task configuration and demo script."""

from __future__ import annotations

from dataclasses import dataclass, field

from environment.actions import ACTION_TO_INDEX
from environment.delivery_env import DeadlineDeliveryEnv


@dataclass(frozen=True)
class RewardConfig:
    move_penalty: float = -0.05
    restaurant_bonus: float = 0.25
    pickup_bonus: float = 1.0
    customer_bonus: float = 0.5
    delivery_bonus: float = 10.0
    invalid_penalty: float = -1.0
    deadline_bonus: float = 2.0
    late_penalty: float = -5.0


@dataclass(frozen=True)
class DeadlineTaskConfig:
    grid_size: int = 6
    rider_start: tuple[int, int] = (0, 0)
    restaurant_position: tuple[int, int] = (2, 0)
    customer_position: tuple[int, int] = (4, 5)
    deadline_steps: int = 12
    reward_config: RewardConfig = field(default_factory=RewardConfig)


def create_environment(task_config: DeadlineTaskConfig | None = None) -> DeadlineDeliveryEnv:
    config = task_config or DeadlineTaskConfig()
    return DeadlineDeliveryEnv(
        grid_size=config.grid_size,
        rider_start=config.rider_start,
        restaurant_position=config.restaurant_position,
        customer_position=config.customer_position,
        deadline_steps=config.deadline_steps,
        reward_config={
            "move_penalty": config.reward_config.move_penalty,
            "restaurant_bonus": config.reward_config.restaurant_bonus,
            "pickup_bonus": config.reward_config.pickup_bonus,
            "customer_bonus": config.reward_config.customer_bonus,
            "delivery_bonus": config.reward_config.delivery_bonus,
            "invalid_penalty": config.reward_config.invalid_penalty,
            "deadline_bonus": config.reward_config.deadline_bonus,
            "late_penalty": config.reward_config.late_penalty,
        },
    )


def run_demo() -> None:
    env = create_environment()
    observation, info = env.reset()

    print("=== Demo: Delivery Deadline Task ===")
    print(f"Initial observation={observation.tolist()}")
    print(f"info={info}")

    actions = ["RIGHT", "RIGHT", "PICKUP", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "RIGHT", "RIGHT", "DELIVER"]
    for action_name in actions:
        next_obs, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action_name])
        print(
            f"Action {action_name:7} -> obs={next_obs.tolist()} reward={reward:.2f} "
            f"terminated={terminated} truncated={truncated} info={info}"
        )
        if terminated or truncated:
            break

    print("Final render:")
    print(env.render())
    print("Reset check:")
    reset_obs, _ = env.reset()
    print(f"reset_obs={reset_obs.tolist()}")


if __name__ == "__main__":
    run_demo()
