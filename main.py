import argparse

from tasks.basic_delivery import run_demo as run_basic_demo, verify_multiple_episodes
from tasks.deadline_delivery import run_demo as run_deadline_demo
from tasks.multi_order_delivery import run_demo as run_multi_order_demo
from tasks.multi_rider_delivery import run_demo as run_multi_rider_demo


def run_task(task_name: str) -> None:
    if task_name == "basic":
        run_basic_demo()
        verify_multiple_episodes()
    elif task_name == "deadline":
        run_deadline_demo()
    elif task_name == "multi-order":
        run_multi_order_demo()
    elif task_name == "multi-rider":
        run_multi_rider_demo()
    else:
        raise ValueError(f"Unknown task: {task_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one delivery task demo episode.")
    parser.add_argument("--task", choices=["basic", "deadline", "multi-order", "multi-rider"], default="basic", help="Which delivery scenario to run.")
    args = parser.parse_args()
    run_task(args.task)
