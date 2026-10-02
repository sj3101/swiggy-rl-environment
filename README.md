# Swiggy-Inspired Reinforcement Learning Environment

A simplified reinforcement learning environment inspired by food-delivery logistics. The project contains four delivery/assignment scenarios, a tabular Q-learning agent, a random baseline, training and evaluation scripts, and a Streamlit dashboard.

This is an educational simulation and does not attempt to reproduce Swiggy's production logistics system.

---

## What the Project Does

The project models delivery-related decisions as a reinforcement learning problem.

The basic interaction is:

```text
State
  ↓
Agent chooses Action
  ↓
Environment returns Reward
  ↓
New State
  ↓
Agent updates its Q-value
  ↓
Repeat
```

The environments use small, discrete state and action spaces so that tabular Q-learning can be applied.

---

## Implemented Tasks

### 1. Basic Delivery

A rider starts on a grid and must:

```text
Rider → Restaurant → Pickup → Customer → Delivery
```

The environment supports movement, pickup, and delivery actions.

### 2. Deadline Delivery

This extends the delivery task by adding a time limit.

The observation includes the remaining time, and each action consumes one timestep. The agent must complete the delivery within the deadline.

### 3. Multi-Order Delivery

The environment contains two orders.

The agent must select an order, pick it up, deliver it, and then handle the remaining order.

The state tracks the selected order and the status of both orders.

### 4. Multi-Rider Assignment

This task contains multiple riders and orders.

The agent's main decision is to assign orders to available riders.

The episode finishes when all orders have been assigned.

---

## Project Structure

```text
swiggy-rl-environment/
│
├── environment/
│   ├── __init__.py
│   ├── delivery_env.py
│   ├── state.py
│   └── actions.py
│
├── tasks/
│   ├── basic_delivery.py
│   ├── deadline_delivery.py
│   ├── multi_order_delivery.py
│   └── multi_rider_delivery.py
│
├── agents/
│   ├── q_learning_agent.py
│   └── random_agent.py
│
├── training/
│   └── checkpoints/
│
├── evaluation/
│   └── evaluate.py
│
├── dashboard/
│   └── app.py
│
├── tests/
│   ├── test_delivery_env.py
│   └── test_task_variants.py
│
├── main.py
├── train.py
├── evaluate.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Main Components

**`environment/`**
Contains the shared environment, state representation, and action definitions.

**`tasks/`**
Contains the four task-specific environments.

**`agents/`**
Contains the Q-learning agent and random baseline agent.

**`training/`**
Contains training-related files and saved checkpoints.

**`evaluation/`**
Evaluates the trained agent and compares it with the random baseline.

**`dashboard/`**
Contains the Streamlit interface for interacting with the project.

**`tests/`**
Contains automated tests for the environment and task variants.

---

## Reinforcement Learning

### State

The state depends on the task and can contain information such as:

* Rider position
* Restaurant position
* Customer position
* Pickup/order status
* Selected order
* Status of individual orders
* Remaining deadline time
* Rider/order assignment information

The states are represented in a discrete form suitable for a Q-table.

### Actions

For delivery tasks, actions include:

```text
UP
DOWN
LEFT
RIGHT
PICKUP
DELIVER
```

Multi-order delivery also includes order-selection actions.

Multi-rider assignment includes assignment actions such as:

```text
ASSIGN_ORDER_0_TO_RIDER_0
ASSIGN_ORDER_1_TO_RIDER_1
```

### Rewards

The environment provides rewards and penalties based on the agent's actions.

Examples include:

* Movement rewards/penalties
* Pickup rewards
* Delivery rewards
* Order-selection rewards
* Assignment rewards
* Completion rewards
* Invalid-action penalties
* Deadline-related rewards/penalties

The reward system gives the agent feedback during an episode instead of only providing feedback at the very end.

---

## Q-Learning

The project uses **tabular Q-learning**.

The Q-table stores an estimated value for taking an action in a particular state:

```text
Q(state, action)
```

The update follows the standard Q-learning formulation:

```text
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]
```

where:

* `s` = current state
* `a` = selected action
* `r` = received reward
* `s'` = next state
* `α` = learning rate
* `γ` = discount factor

The agent uses epsilon-greedy action selection to balance exploration and exploitation.

Tabular Q-learning was used because the environments were intentionally kept small and discrete.

---

## Training

Training is performed over multiple episodes.

For example:

```powershell
python train.py --task basic --episodes 1000
```

During training, the agent initially explores the environment and gradually updates its Q-table.

Therefore, the training summary represents performance across the training process and should not be interpreted as the performance of only the final learned policy.

For example, the Deadline task had:

```text
Training success rate: 0.08
```

while its separately evaluated final trained policy achieved:

```text
Evaluation success rate: 1.00
```

The training log shows the success rate increasing substantially toward the later episodes.

---

## Evaluation

After training, the saved Q-table is evaluated separately.

Example:

```powershell
python evaluate.py --task basic
```

The evaluation also runs a random baseline for comparison.

### Evaluation Results

These are the results from the evaluation runs performed for this project:

| Task        | Trained Avg Reward | Trained Avg Steps | Trained Success Rate | Baseline Avg Reward | Baseline Avg Steps | Baseline Success Rate |
| ----------- | -----------------: | ----------------: | -------------------: | ------------------: | -----------------: | --------------------: |
| Basic       |             11.300 |             11.00 |                 1.00 |             -44.335 |              60.00 |                  0.00 |
| Deadline    |              9.700 |             15.00 |                 1.00 |             -17.565 |              15.00 |                  0.00 |
| Multi-Order |             36.200 |             11.00 |                 1.00 |             -26.170 |              47.65 |                  0.00 |
| Multi-Rider |              3.000 |              2.00 |                 1.00 |              -4.350 |               9.35 |                  1.00 |

### Training Summary

The corresponding 1,000-episode training runs produced:

| Task        | Average Training Reward | Average Training Steps | Training Success Rate |
| ----------- | ----------------------: | ---------------------: | --------------------: |
| Basic       |                   5.196 |                  18.37 |                  0.90 |
| Deadline    |                  -7.471 |                  15.00 |                  0.08 |
| Multi-Order |                  29.782 |                  17.51 |                  0.90 |
| Multi-Rider |                   2.395 |                   2.60 |                  1.00 |

These values are different from the evaluation results because training includes the agent's exploration throughout all episodes.

### Notes on the Results

* **Basic:** The trained policy successfully completed the evaluated episodes and required 11 steps on average.
* **Deadline:** The trained policy successfully completed the evaluated episodes on time. The training process was slower to learn this task, which is reflected in the 8% overall training success rate.
* **Multi-Order:** The trained policy successfully completed the evaluated episodes and required 11 steps on average.
* **Multi-Rider:** Both the trained and random policies achieved 100% success in evaluation. The trained policy used fewer steps on average and received a higher average reward.

The results above are recorded from the project's actual training and evaluation runs.

---

## Running the Project

### 1. Create and activate the virtual environment

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

### 2. Run the task demonstrations

Basic:

```powershell
python main.py --task basic
```

Deadline:

```powershell
python main.py --task deadline
```

Multi-Order:

```powershell
python main.py --task multi-order
```

Multi-Rider:

```powershell
python main.py --task multi-rider
```

### 3. Train an agent

```powershell
python train.py --task basic --episodes 1000
```

The same command can be used for the other tasks:

```powershell
python train.py --task deadline --episodes 1000
python train.py --task multi-order --episodes 1000
python train.py --task multi-rider --episodes 1000
```

### 4. Evaluate the trained agent

```powershell
python evaluate.py --task basic
python evaluate.py --task deadline
python evaluate.py --task multi-order
python evaluate.py --task multi-rider
```

---

## Streamlit Dashboard

Run:

```powershell
streamlit run dashboard/app.py
```

If the default port is unavailable:

```powershell
streamlit run dashboard/app.py --server.port 8502
```

The dashboard provides an interactive interface for:

* Selecting a task
* Running an episode
* Training the agent
* Evaluating the agent
* Viewing rewards and steps
* Viewing task status
* Comparing trained and baseline results
* Viewing reward progression

The dashboard acts as the presentation layer and uses the underlying environment and agent implementation.

---

## Testing

Run the tests with:

```powershell
python -m pytest -q
```

The current test suite was verified with:

```text
12 passed
```

Python compilation was also checked using:

```powershell
python -m compileall .
```

The four task demonstrations were also run individually and completed successfully.

---

## Design Choices

### Why tabular Q-learning?

The environment is intentionally small and discrete. This makes a Q-table practical and keeps the learning process relatively easy to inspect.

Using a more complex deep reinforcement learning method would add complexity without being necessary for the current environment size.

### Why a random baseline?

The random baseline provides a simple reference for evaluation. It helps show how the trained policy behaves compared with an agent that does not learn from previous interactions.

### Why separate tasks?

The shared environment handles common delivery mechanics while individual task classes implement the additional logic required by each scenario.

This keeps the task-specific behavior separate from the common environment functionality.

---

## Limitations

This project is a simplified simulation.

It does not model the full complexity of a real food-delivery platform, including:

* Real road networks
* Live traffic
* Dynamic travel times
* Real-time order arrivals
* Restaurant preparation times
* Customer cancellation
* Large-scale rider fleets
* Production dispatching systems
* Real-world geographic data

The current environment is intentionally small so that the RL behavior can be demonstrated and evaluated clearly.

---

## Possible Extensions

The environment could be extended with:

* Dynamic orders
* Traffic and variable travel times
* Rider capacity constraints
* Restaurant preparation times
* More riders and simultaneous orders
* More complex assignment decisions
* Larger state spaces
* Deep reinforcement learning
* Multi-agent reinforcement learning

---

## Project Goal

The goal of the project is to demonstrate a complete, understandable reinforcement learning workflow in a simplified delivery setting:

```text
Environment
    ↓
State
    ↓
Actions
    ↓
Rewards
    ↓
Q-Learning Agent
    ↓
Training
    ↓
Evaluation
    ↓
Random Baseline Comparison
    ↓
Streamlit Visualization
```

The project focuses on implementing and evaluating the core RL workflow rather than attempting to reproduce a production-scale delivery platform.
