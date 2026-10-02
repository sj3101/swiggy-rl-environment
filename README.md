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
- `tests/` can be used for smoke tests and validation scripts

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --task basic
python train.py --task basic --episodes 500
python evaluate.py --task basic
```

## RL notes

The tabular Q-learning implementation is appropriate for the basic and deadline tasks because their state/action spaces remain compact and discrete. The multi-order and multi-rider scenarios are intentionally kept simplified and are not treated as large-scale tabular learning problems; their states grow combinatorially and would require function approximation or a more structured state abstraction to train efficiently beyond small toy examples.

This project makes that limitation explicit rather than silently pretending a naive full tabular Q-table is a valid solution for every task.

## Notes

This is intentionally a minimal environment intended for future RL development and tabular Q-learning experiments.
