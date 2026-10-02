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

The tabular Q-learning implementation is appropriate for the basic and deadline tasks because their state/action spaces remain compact and discrete. The multi-order and multi-rider scenarios are intentionally kept simplified and are not treated as large-scale tabular learning problems; their states grow combinatorially and would require function approximation or a more structured state abstraction to train efficiently beyond small toy examples.

This project makes that limitation explicit rather than silently pretending a naive full tabular Q-table is a valid solution for every task.

## Current measured results

The project includes a small set of validation runs that compare a trained tabular Q-learning policy against a random baseline. These results are intentionally presented as a minimal benchmark for the toy tasks rather than a claim that the environment is solved at production scale.

- Basic delivery: trained average reward 11.30, success rate 1.00; baseline average reward -44.335, success rate 0.00
- Deadline delivery: trained average reward -6.20, success rate 0.00; baseline average reward -15.135, success rate 0.00
- Multi-order delivery: trained average reward 3.80, success rate 0.00; baseline average reward -28.048, success rate 0.00
- Multi-rider delivery: trained average reward 3.00, success rate 1.00; baseline average reward -4.350, success rate 1.00

## Known limitations

- The environment remains intentionally small and discrete, which keeps the Q-learning agent fast and interpretable.
- The deadline and multi-order tasks still show low or zero learned success in the current measured runs.
- The multi-order and multi-rider variants are simplified task abstractions rather than realistic delivery simulators.
- The dashboard is a lightweight demonstration layer and does not replace the underlying environment or RL logic.

## Possible future improvements

- Add richer state representations and reward shaping for deadline and multi-order tasks.
- Move beyond tabular Q-learning toward deep RL or structured policy learning for larger state spaces.
- Add richer visualization for route traces, rewards, and task-specific KPIs in the dashboard.
- Expand the environment with more realistic constraints such as vehicle capacity, traffic, and dynamic order generation.

## Notes

This is intentionally a minimal environment intended for future RL development and tabular Q-learning experiments.
