"""Evaluate a trained Q-learning agent for a chosen task."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomPolicyAgent
from evaluation.evaluate import compare_agents, evaluate_agent
from tasks.basic_delivery import create_environment as create_basic_env
from tasks.deadline_delivery import create_environment as create_deadline_env
from tasks.multi_order_delivery import create_environment as create_multi_order_env
from tasks.multi_rider_delivery import create_environment as create_multi_rider_env


TASK_FACTORIES = {
    "basic": create_basic_env,
    "deadline": create_deadline_env,
    "multi-order": create_multi_order_env,
    "multi-rider": create_multi_rider_env,
}


def load_agent(task_name: str) -> QLearningAgent:
    checkpoint = Path("training/checkpoints") / f"{task_name}_agent.pkl"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No trained checkpoint found for task '{task_name}' at {checkpoint}")
    with checkpoint.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained delivery task agent.")
    parser.add_argument("--task", choices=list(TASK_FACTORIES.keys()), required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    env = TASK_FACTORIES[args.task]()
    trained_agent = load_agent(args.task)
    baseline_agent = RandomPolicyAgent(env.action_space, seed=args.seed)
    comparison = compare_agents(env, trained_agent, baseline_agent, episodes=args.episodes, seed=args.seed)

    print(f"Task: {args.task}")
    for label, summary in comparison.items():
        print(f"{label.title()} -> avg_reward={summary['average_reward']:.3f}, avg_steps={summary['average_steps']:.2f}, success_rate={summary['success_rate']:.2f}")
        if summary.get("deadline_success_rate") is not None:
            print(f"  deadline_success_rate={summary['deadline_success_rate']:.2f}")


if __name__ == "__main__":
    main()
