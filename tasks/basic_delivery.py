"""Configuration and demo flow for the basic delivery environment."""

from __future__ import annotations

from dataclasses import dataclass, field

from environment.actions import ACTION_TO_INDEX
from environment.delivery_env import FoodDeliveryEnv


@dataclass(frozen=True)
class RewardConfig:
    move_penalty: float = -0.05
    restaurant_bonus: float = 0.25
    pickup_bonus: float = 1.0
    customer_bonus: float = 0.5
    delivery_bonus: float = 10.0
    invalid_penalty: float = -1.0


@dataclass(frozen=True)
class TaskConfig:
    grid_size: int = 6
    rider_start: tuple[int, int] = (0, 0)
    restaurant_position: tuple[int, int] = (2, 0)
    customer_position: tuple[int, int] = (4, 5)
    max_steps: int = 60
    reward_config: RewardConfig = field(default_factory=RewardConfig)


def create_environment(task_config: TaskConfig | None = None) -> FoodDeliveryEnv:
    config = task_config or TaskConfig()
    return FoodDeliveryEnv(
        grid_size=config.grid_size,
        rider_start=config.rider_start,
        restaurant_position=config.restaurant_position,
        customer_position=config.customer_position,
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

    print("=== Demo: Basic Delivery Environment ===")
    print("Initial state:")
    print(f"  observation={observation.tolist()}")
    print(f"  info={info}")
    print()

    demo_actions = [
        "RIGHT",
        "RIGHT",
        "PICKUP",
        "DOWN",
        "DOWN",
        "DOWN",
        "DOWN",
        "DOWN",
        "RIGHT",
        "RIGHT",
        "DELIVER",
    ]

    for action_name in demo_actions:
        action = ACTION_TO_INDEX[action_name]
        next_observation, reward, terminated, truncated, info = env.step(action)

        print(
            f"Action: {action_name:7} | "
            f"obs={next_observation.tolist()} | "
            f"reward={reward:>6.2f} | "
            f"terminated={terminated} | "
            f"truncated={truncated} | "
            f"info={info}"
        )

        if terminated:
            print("Episode ended successfully after delivery.")
            break

    print("\nFinal render:")
    print(env.render())

    print("\nReset check:")
    obs_2, _ = env.reset()
    print(f"  next_reset_obs={obs_2.tolist()}")


def verify_multiple_episodes() -> None:
    print("\n=== Multiple Episode Reset Check ===")
    for episode_no in range(2):
        env = create_environment()
        observation, _ = env.reset()
        print(f"Episode {episode_no + 1} initial observation: {observation.tolist()}")

        actions = ["RIGHT", "RIGHT", "PICKUP", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "RIGHT", "RIGHT", "DELIVER"]
        for action_name in actions:
            action = ACTION_TO_INDEX[action_name]
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Episode {episode_no + 1} ended with reward {reward}. Status: {info['status']}")
                break

        _ = env.reset()
        print(f"Episode {episode_no + 1} reset successfully without crash.")


if __name__ == "__main__":
    run_demo()
    verify_multiple_episodes()
