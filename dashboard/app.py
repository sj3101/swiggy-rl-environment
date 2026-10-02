"""Lightweight Streamlit dashboard for the delivery RL project."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.q_learning_agent import QLearningAgent, QLearningConfig
from agents.random_agent import RandomPolicyAgent
from evaluation.evaluate import compare_agents
from environment.actions import ACTION_NAMES
from tasks.basic_delivery import create_environment as create_basic_environment
from tasks.deadline_delivery import create_environment as create_deadline_environment
from tasks.multi_order_delivery import create_environment as create_multi_order_environment
from tasks.multi_rider_delivery import create_environment as create_multi_rider_environment
from training.train import run_episode, summarize_history, train_agent

TASK_FACTORIES = {
    "basic": create_basic_environment,
    "deadline": create_deadline_environment,
    "multi-order": create_multi_order_environment,
    "multi-rider": create_multi_rider_environment,
}

TASK_LABELS = {
    "basic": "Basic Delivery",
    "deadline": "Deadline Delivery",
    "multi-order": "Multi-Order Delivery",
    "multi-rider": "Multi-Rider Delivery",
}


def create_agent_for_task(task_name: str, seed: int = 42) -> QLearningAgent:
    env = TASK_FACTORIES[task_name]()
    config = QLearningConfig(seed=seed)
    return QLearningAgent(env.observation_space, env.action_space, config=config)


def reset_session_for_task(task_name: str) -> None:
    st.session_state.task_name = task_name
    st.session_state.env = TASK_FACTORIES[task_name]()
    st.session_state.observation, st.session_state.info = st.session_state.env.reset(seed=42)
    st.session_state.current_action = None
    st.session_state.current_reward = 0.0
    st.session_state.cumulative_reward = 0.0
    st.session_state.step_count = 0
    st.session_state.episode_status = "ready"
    st.session_state.agent = None
    st.session_state.training_history = []
    st.session_state.training_summary = None
    st.session_state.evaluation_summary = None


def ensure_task_state(task_name: str) -> None:
    if "task_name" not in st.session_state or st.session_state.task_name != task_name:
        reset_session_for_task(task_name)
    elif "env" not in st.session_state:
        reset_session_for_task(task_name)


def get_task_state_summary(task_name: str) -> dict[str, Any]:
    env = st.session_state.env
    summary: dict[str, Any] = {"rider_positions": [], "restaurant_positions": [], "customer_positions": [], "order_status": [], "task_name": task_name}

    if task_name in {"basic", "deadline"}:
        summary["rider_positions"] = [tuple(int(value) for value in env.rider_position)]
        summary["restaurant_positions"] = [tuple(env.restaurant_position)]
        summary["customer_positions"] = [tuple(env.customer_position)]
        summary["order_status"] = {"picked_up": bool(env.picked_up), "delivered": bool(env.delivered)}
    elif task_name == "multi-order":
        summary["rider_positions"] = [tuple(int(value) for value in env.rider_position)]
        summary["restaurant_positions"] = [tuple(order["restaurant"]) for order in env.orders]
        summary["customer_positions"] = [tuple(order["customer"]) for order in env.orders]
        summary["order_status"] = {f"order_{index}": order["status"] for index, order in enumerate(env.orders)}
    elif task_name == "multi-rider":
        summary["rider_positions"] = [tuple(rider) for rider in env.rider_positions]
        summary["restaurant_positions"] = [tuple(order["restaurant"]) for order in env.orders]
        summary["customer_positions"] = [tuple(order["customer"]) for order in env.orders]
        summary["order_status"] = {f"order_{index}": order["status"] for index, order in enumerate(env.orders)}
        summary["workloads"] = list(env.rider_workloads)

    return summary


def render_grid(task_name: str, env: Any) -> str:
    grid_size = env.grid_size
    grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

    if task_name in {"basic", "deadline"}:
        restaurant_x, restaurant_y = env.restaurant_position
        customer_x, customer_y = env.customer_position
        rider_x, rider_y = tuple(int(value) for value in env.rider_position)
        grid[restaurant_y][restaurant_x] = "R"
        grid[customer_y][customer_x] = "C"
        grid[rider_y][rider_x] = "A"
        if env.delivered:
            grid[rider_y][rider_x] = "D"
        return "\n".join(" ".join(row) for row in grid)

    if task_name == "multi-order":
        for index, order in enumerate(env.orders):
            restaurant_x, restaurant_y = order["restaurant"]
            customer_x, customer_y = order["customer"]
            grid[restaurant_y][restaurant_x] = f"R{index}"
            grid[customer_y][customer_x] = f"C{index}"
        rider_x, rider_y = tuple(int(value) for value in env.rider_position)
        grid[rider_y][rider_x] = "A"
        return "\n".join(" ".join(row) for row in grid)

    for index, rider_position in enumerate(env.rider_positions):
        rider_x, rider_y = rider_position
        grid[rider_y][rider_x] = chr(65 + index)
    for index, order in enumerate(env.orders):
        restaurant_x, restaurant_y = order["restaurant"]
        customer_x, customer_y = order["customer"]
        grid[restaurant_y][restaurant_x] = f"R{index}"
        grid[customer_y][customer_x] = f"C{index}"
    return "\n".join(" ".join(row) for row in grid)


def execute_single_episode() -> None:
    env = st.session_state.env
    agent = st.session_state.get("agent")
    if agent is None:
        agent = RandomPolicyAgent(env.action_space, seed=42)

    observation, info = env.reset(seed=42)
    total_reward = 0.0
    current_action = None
    current_reward = 0.0
    step_count = 0
    episode_status = "started"

    while True:
        action = agent.select_action(observation, training=False)
        next_observation, reward, terminated, truncated, info = env.step(action)
        current_action = action
        current_reward = float(reward)
        total_reward += float(reward)
        step_count += 1
        observation = next_observation
        episode_status = info.get("status", "unknown")
        if terminated or truncated:
            break

    st.session_state.current_action = ACTION_NAMES[current_action] if current_action is not None else None
    st.session_state.current_reward = current_reward
    st.session_state.cumulative_reward = total_reward
    st.session_state.step_count = step_count
    st.session_state.episode_status = episode_status
    st.session_state.observation = observation
    st.session_state.info = info


def train_selected_agent(episodes: int) -> None:
    env = st.session_state.env
    agent = QLearningAgent(env.observation_space, env.action_space, QLearningConfig(seed=42))
    history = train_agent(env, agent, episodes=episodes, verbose=False, seed=42)
    st.session_state.agent = agent
    st.session_state.training_history = history
    st.session_state.training_summary = summarize_history(history)


def evaluate_selected_agent(episodes: int) -> None:
    env = st.session_state.env
    if "agent" not in st.session_state or st.session_state.agent is None:
        st.warning("Train an agent before evaluating it.")
        return
    baseline = RandomPolicyAgent(env.action_space, seed=42)
    comparison = compare_agents(env, st.session_state.agent, baseline, episodes=episodes, seed=42)
    st.session_state.evaluation_summary = comparison


def main() -> None:
    st.set_page_config(page_title="Swiggy Delivery RL Dashboard", layout="wide")
    st.title("Swiggy Delivery RL Dashboard")

    task_name = st.selectbox("Select a task", list(TASK_LABELS.keys()), format_func=lambda key: TASK_LABELS[key])
    ensure_task_state(task_name)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Reset environment"):
            reset_session_for_task(task_name)
    with col2:
        if st.button("Run one episode"):
            execute_single_episode()
    with col3:
        if st.button("Train agent"):
            with st.spinner("Training agent..."):
                train_selected_agent(int(st.session_state.get("training_episodes", 200)))
    with col4:
        if st.button("Evaluate agent"):
            with st.spinner("Evaluating agent..."):
                evaluate_selected_agent(int(st.session_state.get("evaluation_episodes", 20)))

    training_episodes = st.number_input("Training episode count", min_value=1, max_value=5000, value=200, step=50, key="training_episodes")
    evaluation_episodes = st.number_input("Evaluation episode count", min_value=1, max_value=500, value=20, step=5, key="evaluation_episodes")

    env = st.session_state.env
    observation = st.session_state.get("observation", st.session_state.env.reset()[0])
    info = st.session_state.get("info", {"status": "ready"})

    st.subheader("Environment simulation")
    left, right = st.columns([1.5, 1])
    with left:
        st.code(render_grid(task_name, env))
    with right:
        summary = get_task_state_summary(task_name)
        st.write("Rider positions:", summary["rider_positions"])
        st.write("Restaurant positions:", summary["restaurant_positions"])
        st.write("Customer positions:", summary["customer_positions"])
        st.write("Order status:", summary["order_status"])
        if "workloads" in summary:
            st.write("Rider workloads:", summary["workloads"])

    st.subheader("Current action and reward")
    action_col, reward_col, cumulative_col, steps_col, status_col = st.columns(5)
    action_col.metric("Current action", st.session_state.get("current_action", "-"))
    reward_col.metric("Current reward", f"{st.session_state.get('current_reward', 0.0):.2f}")
    cumulative_col.metric("Cumulative reward", f"{st.session_state.get('cumulative_reward', 0.0):.2f}")
    steps_col.metric("Steps", int(st.session_state.get("step_count", 0)))
    status_col.metric("Episode status", st.session_state.get("episode_status", info.get("status", "ready")))

    st.write("Observation:", observation.tolist())
    st.write("Info:", info)

    st.subheader("Agent training")
    if st.session_state.get("training_summary") is not None:
        summary = st.session_state.training_summary
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Average reward", f"{summary.average_reward:.2f}")
        col_b.metric("Average steps", f"{summary.average_steps:.2f}")
        col_c.metric("Success rate", f"{summary.success_rate:.2f}")
        if summary.deadline_success_rate is not None:
            st.metric("Deadline success rate", f"{summary.deadline_success_rate:.2f}")

        history = st.session_state.training_history
        if history:
            reward_points = [{"Episode": idx + 1, "Reward": float(item["total_reward"])} for idx, item in enumerate(history)]
            st.line_chart(reward_points, x="Episode", y="Reward")
    else:
        st.info("No training has been run yet.")

    st.subheader("Evaluation")
    evaluation_summary = st.session_state.get("evaluation_summary")
    if evaluation_summary is not None:
        trained = evaluation_summary["trained"]
        baseline = evaluation_summary["baseline"]
        t1, t2, t3 = st.columns(3)
        t1.metric("Trained average reward", f"{trained['average_reward']:.2f}")
        t2.metric("Trained success rate", f"{trained['trained_success_rate']:.2f}")
        t3.metric("Baseline success rate", f"{baseline['baseline_success_rate']:.2f}")
        st.write("Trained vs baseline:")
        st.json({
            "trained": trained,
            "baseline": baseline,
        })
    else:
        st.info("No evaluation has been run yet.")

    st.caption("This dashboard visualizes the existing environment and RL stack without replacing the underlying task logic, rewards, or Q-learning implementation.")


if __name__ == "__main__":
    main()
