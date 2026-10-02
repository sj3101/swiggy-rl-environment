"""Evaluation utilities for comparing trained agents against baselines."""

from __future__ import annotations

from typing import Any

from agents.random_agent import RandomPolicyAgent
from training.train import TrainingSummary, summarize_history, run_episode


def evaluate_agent(env: Any, agent: Any, episodes: int = 20, seed: int | None = None) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        episode_seed = None if seed is None else seed + episode_index
        metrics.append(run_episode(env, agent, training=False, seed=episode_seed))

    summary = summarize_history(metrics)
    return {
        "average_reward": summary.average_reward,
        "average_steps": summary.average_steps,
        "success_rate": summary.success_rate,
        "deadline_success_rate": summary.deadline_success_rate,
        "episodes": summary.total_episodes,
        "history": metrics,
    }


def compare_agents(env: Any, trained_agent: Any, baseline_agent: Any, episodes: int = 20, seed: int | None = None) -> dict[str, dict[str, Any]]:
    trained_metrics = evaluate_agent(env, trained_agent, episodes=episodes, seed=seed)
    baseline_metrics = evaluate_agent(env, baseline_agent, episodes=episodes, seed=seed)

    return {
        "trained": {
            "average_reward": trained_metrics["average_reward"],
            "average_steps": trained_metrics["average_steps"],
            "success_rate": trained_metrics["success_rate"],
            "trained_success_rate": trained_metrics["success_rate"],
            "deadline_success_rate": trained_metrics["deadline_success_rate"],
        },
        "baseline": {
            "average_reward": baseline_metrics["average_reward"],
            "average_steps": baseline_metrics["average_steps"],
            "success_rate": baseline_metrics["success_rate"],
            "baseline_success_rate": baseline_metrics["success_rate"],
            "deadline_success_rate": baseline_metrics["deadline_success_rate"],
        },
    }


def run_random_baseline(env: Any, episodes: int = 20, seed: int | None = None) -> dict[str, Any]:
    baseline = RandomPolicyAgent(env.action_space, seed=seed)
    return evaluate_agent(env, baseline, episodes=episodes, seed=seed)
