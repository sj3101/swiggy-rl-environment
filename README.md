# Swiggy RL Environment

This project creates a lightweight reinforcement learning environment inspired by a food-delivery platform. The first version models a single rider moving through a small city grid to pick up an order from a restaurant and deliver it to a customer.

## Features

- Discrete 6x6 city grid
- One rider, one restaurant, one customer
- Gymnasium-compatible `reset()` and `step()` API
- Simple reward structure with invalid-action penalties
- Configurable delivery task setup
- Tabular Q-learning agent for the smaller discrete tasks
- Training, evaluation, and baseline comparison utilities

## Project layout

- `environment/` holds the Gymnasium environment and core state/action definitions
- `tasks/` defines the basic delivery task and demo flow
- `agents/` contains the Q-learning policy and a random baseline
- `training/` contains the training loop and episode summaries
- `evaluation/` contains evaluation and baseline comparison helpers
- `dashboard/` provides a lightweight Streamlit demo for the environment and RL flow
- `tests/` can be used for smoke tests and validation scripts

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --task basic
python train.py --task basic --episodes 500
python evaluate.py --task basic
streamlit run dashboard/app.py
```

## RL notes

The tabular Q-learning implementation is appropriate for the compact discrete tasks in this project. For the multi-order variant, the default route layout has been simplified to a short two-order corridor so the state remains decision-relevant and the learner can discover a valid policy without hard-coding the answer. Larger or more dynamic courier problems would still require a richer state abstraction or function approximation.

This project keeps the RL assumptions explicit: it is a toy environment for interview-friendly learning, not a production-grade logistics simulator.

## Current measured results

The project includes a small benchmark for the current default task configurations. The reported numbers below are from the repository's current code and should be read as a minimal validation of the tabular Q-learning setup rather than a claim of production readiness.

- Multi-order delivery (500 training episodes): trained average reward 23.77, success rate 0.80; baseline average reward -26.17, success rate 0.00
- Basic and multi-rider scenarios remain stable and trainable under the same tabular Q-learning setup, while the deadline task is intentionally strict and should be treated as a more sensitive variant rather than a guarantee of high success in every run.

## Known limitations

- The environment remains intentionally small and discrete, which keeps the Q-learning agent fast and interpretable.
- The multi-order task is deliberately simplified to remain learnable with tabular Q-learning; it is not a realistic city-scale delivery simulator.
- The deadline task is a more sensitive benchmark and can be harder to master under the same fixed tabular setup.
- The dashboard is a lightweight demonstration layer and does not replace the underlying environment or RL logic.

## Possible future improvements

- Add richer state representations and reward shaping for deadline and multi-order tasks.
- Move beyond tabular Q-learning toward deep RL or structured policy learning for larger state spaces.
- Add richer visualization for route traces, rewards, and task-specific KPIs in the dashboard.
- Expand the environment with more realistic constraints such as vehicle capacity, traffic, and dynamic order generation.

## Notes

This is intentionally a minimal environment intended for future RL development and tabular Q-learning experiments.
