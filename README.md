# Swiggy RL Environment

This project creates a lightweight reinforcement learning environment inspired by a food-delivery platform. The first version models a single rider moving through a small city grid to pick up an order from a restaurant and deliver it to a customer.

## Features

- Discrete 6x6 city grid
- One rider, one restaurant, one customer
- Gymnasium-compatible `reset()` and `step()` API
- Simple reward structure with invalid-action penalties
- Configurable delivery task setup

## Project layout

- `environment/` holds the Gymnasium environment and core state/action definitions
- `tasks/` defines the basic delivery task and demo flow
- `agents/`, `training/`, and `evaluation/` are reserved for future RL work
- `tests/` can be used for smoke tests and validation scripts

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Notes

This is intentionally a minimal environment intended for future RL development and tabular Q-learning experiments.
