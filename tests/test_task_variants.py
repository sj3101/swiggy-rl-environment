"""Regression tests for the four delivery scenarios."""

from environment.actions import ACTION_TO_INDEX
from tasks.basic_delivery import create_environment as create_basic_env
from tasks.deadline_delivery import create_environment as create_deadline_env
from tasks.multi_order_delivery import create_environment as create_multi_order_env
from tasks.multi_rider_delivery import create_environment as create_multi_rider_env


def _assert_invalid_action_is_safe(env) -> None:
    initial_obs, _ = env.reset()
    invalid_action = env.action_space.n
    next_obs, reward, terminated, truncated, info = env.step(invalid_action)
    assert isinstance(next_obs, object)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "status" in info


def test_basic_delivery_task() -> None:
    env = create_basic_env()
    obs, info = env.reset()
    assert obs.shape == (7,)
    assert info["status"] == "ready"

    _assert_invalid_action_is_safe(env)

    for action_name in ["RIGHT", "RIGHT", "PICKUP", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "RIGHT", "RIGHT", "DELIVER"]:
        obs, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action_name])
        assert isinstance(reward, float)
        if terminated:
            assert info["status"] == "delivered"
            break
    else:
        raise AssertionError("Basic delivery scenario did not terminate successfully.")


def test_deadline_delivery_task() -> None:
    env = create_deadline_env()
    obs, info = env.reset()
    assert obs.shape == (8,)
    assert info["status"] == "ready"

    _assert_invalid_action_is_safe(env)

    actions = ["RIGHT", "RIGHT", "PICKUP", "DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "RIGHT", "RIGHT", "DELIVER"]
    for action_name in actions:
        obs, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action_name])
        assert isinstance(reward, float)
        if terminated or truncated:
            assert "status" in info
            break
    else:
        raise AssertionError("Deadline scenario should end in success or failure.")


def test_multi_order_delivery_task() -> None:
    env = create_multi_order_env()
    obs, info = env.reset()
    assert obs.shape[0] >= 12
    assert info["status"] == "ready"

    _assert_invalid_action_is_safe(env)

    actions = [
        "SELECT_ORDER_0",
        "RIGHT",
        "PICKUP",
        "DOWN",
        "DOWN",
        "DOWN",
        "DOWN",
        "DOWN",
        "RIGHT",
        "RIGHT",
        "RIGHT",
        "DELIVER",
        "SELECT_ORDER_1",
        "LEFT",
        "LEFT",
        "LEFT",
        "LEFT",
        "UP",
        "UP",
        "UP",
        "PICKUP",
        "RIGHT",
        "RIGHT",
        "RIGHT",
        "RIGHT",
        "RIGHT",
        "DOWN",
        "DOWN",
        "DOWN",
        "DELIVER",
    ]

    final_terminated = False
    for action_name in actions:
        obs, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action_name])
        assert isinstance(reward, float)
        final_terminated = terminated or final_terminated
        if terminated or truncated:
            break

    assert final_terminated is True or env.completed_orders >= 1


def test_multi_rider_assignment_task() -> None:
    env = create_multi_rider_env()
    obs, info = env.reset()
    assert obs.shape[0] >= 12
    assert info["status"] == "ready"

    _assert_invalid_action_is_safe(env)

    actions = ["ASSIGN_ORDER_0_TO_RIDER_0", "ASSIGN_ORDER_1_TO_RIDER_1", "ASSIGN_ORDER_0_TO_RIDER_1"]
    for action_name in actions:
        obs, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action_name])
        assert isinstance(reward, float)
        if terminated or truncated:
            break

    assert "status" in info
