"""Train a tabular Q-learning agent for a chosen task."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from agents.q_learning_agent import QLearningAgent, QLearningConfig
from tasks.basic_delivery import create_environment as create_basic_env
from tasks.deadline_delivery import create_environment as create_deadline_env
from tasks.multi_order_delivery import create_environment as create_multi_order_env
from tasks.multi_rider_delivery import create_environment as create_multi_rider_env
from training.train import summarize_history, train_agent


TASK_FACTORIES = {
    "basic": create_basic_env,
    "deadline": create_deadline_env,
    "multi-order": create_multi_order_env,
    "multi-rider": create_multi_rider_env,
}


def save_agent(agent: QLearningAgent, task_name: str) -> Path:
    output_dir = Path("training/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task_name}_agent.pkl"
    with output_path.open("wb") as handle:
        pickle.dump(agent, handle)
    return output_path


def save_history(history: list[dict], task_name: str) -> Path:
    output_dir = Path("training/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task_name}_history.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning agent for a delivery task.")
    parser.add_argument("--task", choices=list(TASK_FACTORIES.keys()), required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--min-epsilon", type=float, default=0.05)
    args = parser.parse_args()

    env = TASK_FACTORIES[args.task]()
    config = QLearningConfig(
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
        seed=args.seed,
    )
    agent = QLearningAgent(env.observation_space, env.action_space, config=config)
    history = train_agent(env, agent, episodes=args.episodes, verbose=True, seed=args.seed)
    summary = summarize_history(history)

    print(f"\nTraining complete for task: {args.task}")
    print(f"Average reward: {summary.average_reward:.3f}")
    print(f"Average steps: {summary.average_steps:.2f}")
    print(f"Success rate: {summary.success_rate:.2f}")
    if summary.deadline_success_rate is not None:
        print(f"Deadline success rate: {summary.deadline_success_rate:.2f}")

    save_agent(agent, args.task)
    save_history(history, args.task)


if __name__ == "__main__":
    main()
