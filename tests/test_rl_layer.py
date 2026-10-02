"""Tests for the tabular Q-learning, training, evaluation, and baseline pipeline."""

import numpy as np

from agents.q_learning_agent import QLearningAgent, QLearningConfig
from agents.random_agent import RandomPolicyAgent
from evaluation.evaluate import compare_agents, evaluate_agent
from tasks.basic_delivery import create_environment
from training.train import train_agent


def test_q_table_initialization() -> None:
    env = create_environment()
    agent = QLearningAgent(env.observation_space, env.action_space, QLearningConfig(seed=7))
    assert hasattr(agent, "q_table")
    assert isinstance(agent.q_table, dict)


def test_action_selection() -> None:
    env = create_environment()
    agent = QLearningAgent(env.observation_space, env.action_space, QLearningConfig(seed=11, epsilon=0.0))
    obs = env.reset()[0]
    action = agent.select_action(obs, training=False)
    assert 0 <= action < env.action_space.n

    agent.epsilon = 1.0
    random_action = agent.select_action(obs, training=True)
    assert 0 <= random_action < env.action_space.n


def test_q_value_update() -> None:
    env = create_environment()
    agent = QLearningAgent(env.observation_space, env.action_space, QLearningConfig(seed=4, learning_rate=0.5, discount_factor=0.9))
    obs = np.array([0, 0, 2, 0, 4, 5, 0], dtype=np.int32)
    next_obs = np.array([1, 0, 2, 0, 4, 5, 0], dtype=np.int32)
    action = 1

    agent.q_table[tuple(obs.tolist())] = np.zeros(env.action_space.n, dtype=float)
    old_value = agent.q_table[tuple(obs.tolist())][action]
    agent.update(obs, action, 1.0, next_obs, terminated=False)
    new_value = agent.q_table[tuple(obs.tolist())][action]
    assert new_value != old_value
    assert new_value > old_value


def test_training_loop() -> None:
    env = create_environment()
    agent = QLearningAgent(env.observation_space, env.action_space, QLearningConfig(seed=13, epsilon=0.8, epsilon_decay=0.995, min_epsilon=0.05))
    history = train_agent(env, agent, episodes=20, verbose=False)
    assert len(history) == 20
    assert all("total_reward" in item for item in history)
    assert all("steps" in item for item in history)


def test_evaluation() -> None:
    env = create_environment()
    agent = QLearningAgent(env.observation_space, env.action_space, QLearningConfig(seed=22, epsilon=0.0))
    for _ in range(5):
        obs, _ = env.reset()
        while True:
            action = agent.select_action(obs, training=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            obs = next_obs

    metrics = evaluate_agent(env, agent, episodes=5)
    assert "average_reward" in metrics
    assert "success_rate" in metrics


def test_baseline_comparison() -> None:
    env = create_environment()
    agent = QLearningAgent(env.observation_space, env.action_space, QLearningConfig(seed=99, epsilon=0.0))
    baseline = RandomPolicyAgent(env.action_space)
    comparison = compare_agents(env, agent, baseline, episodes=10)
    assert "trained" in comparison
    assert "baseline" in comparison
    assert "trained_success_rate" in comparison["trained"]
