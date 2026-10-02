"""Training utilities for tabular Q-learning agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agents.q_learning_agent import QLearningAgent


@dataclass
class TrainingSummary:
    average_reward: float
    average_steps: float
    success_rate: float
    deadline_success_rate: float | None = None
    total_episodes: int = 0


def _episode_success(info: dict[str, Any]) -> bool:
    success_statuses = {"delivered", "delivered_on_time", "all_orders_completed", "all_orders_assigned"}
    return bool(info.get("status") in success_statuses)


def _episode_deadline_success(info: dict[str, Any]) -> bool:
    return bool(info.get("status") in {"delivered_on_time", "delivered"})


def run_episode(env: Any, agent: QLearningAgent, training: bool = True, seed: int | None = None) -> dict[str, Any]:
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0

    while True:
        action = agent.select_action(observation, training=training)
        next_observation, reward, terminated, truncated, info = env.step(action)
        if training:
            agent.update(observation, action, reward, next_observation, terminated)
        total_reward += float(reward)
        steps += 1
        observation = next_observation

        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "steps": steps,
        "success": _episode_success(info),
        "deadline_success": _episode_deadline_success(info),
        "status": info.get("status", "unknown"),
    }


def train_agent(env: Any, agent: QLearningAgent, episodes: int = 200, verbose: bool = True, seed: int | None = None, print_every: int = 50) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    base_seed = seed if seed is not None else 0

    for episode_index in range(episodes):
        episode_seed = None if seed is None else base_seed + episode_index
        entry = run_episode(env, agent, training=True, seed=episode_seed)
        agent.decay_epsilon()
        history.append(entry)

        if verbose and (episode_index + 1) % print_every == 0:
            avg_reward = float(np.mean([item["total_reward"] for item in history[-print_every:]]))
            success_rate = float(np.mean([1.0 if item["success"] else 0.0 for item in history[-print_every:]]))
            print(f"Episode {episode_index + 1} | Avg reward: {avg_reward:.3f} | Success rate: {success_rate:.2f}")

    return history


def summarize_history(history: list[dict[str, Any]]) -> TrainingSummary:
    if not history:
        return TrainingSummary(average_reward=0.0, average_steps=0.0, success_rate=0.0, deadline_success_rate=0.0, total_episodes=0)

    rewards = [float(item["total_reward"]) for item in history]
    steps = [int(item["steps"]) for item in history]
    successes = [1.0 if item["success"] else 0.0 for item in history]
    deadline_successes = [1.0 if item["deadline_success"] else 0.0 for item in history]

    return TrainingSummary(
        average_reward=float(np.mean(rewards)),
        average_steps=float(np.mean(steps)),
        success_rate=float(np.mean(successes)),
        deadline_success_rate=float(np.mean(deadline_successes)) if deadline_successes and any(item["status"] in {"delivered_on_time", "delivered"} for item in history) else None,
        total_episodes=len(history),
    )
