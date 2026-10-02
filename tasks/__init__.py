"""Task definitions for the RL project."""

from .basic_delivery import TaskConfig, create_environment, run_demo
from .deadline_delivery import DeadlineTaskConfig, create_environment as create_deadline_environment, run_demo as run_deadline_demo
from .multi_order_delivery import MultiOrderTaskConfig, create_environment as create_multi_order_environment, run_demo as run_multi_order_demo
from .multi_rider_delivery import MultiRiderTaskConfig, create_environment as create_multi_rider_environment, run_demo as run_multi_rider_demo

__all__ = [
    "TaskConfig",
    "DeadlineTaskConfig",
    "MultiOrderTaskConfig",
    "MultiRiderTaskConfig",
    "create_environment",
    "create_deadline_environment",
    "create_multi_order_environment",
    "create_multi_rider_environment",
    "run_demo",
    "run_deadline_demo",
    "run_multi_order_demo",
    "run_multi_rider_demo",
]
