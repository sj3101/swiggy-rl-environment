"""Smoke test for the basic delivery environment."""

from tasks.basic_delivery import create_environment
from environment.actions import ACTION_TO_INDEX


def test_basic_delivery_smoke() -> None:
    env = create_environment()
    observation, _ = env.reset()
    assert observation.shape == (7,)
    assert observation.tolist() == [0, 0, 2, 0, 4, 5, 0]

    actions = [
        "RIGHT",
        "RIGHT",
        "PICKUP",
        "DOWN",
        "DOWN",
        "DOWN",
        "DOWN",
        "DOWN",
        "RIGHT",
        "RIGHT",
        "DELIVER",
    ]

    for action_name in actions:
        obs, reward, terminated, truncated, info = env.step(ACTION_TO_INDEX[action_name])
        if terminated:
            assert truncated is False
            assert info["status"] == "delivered"
            assert reward > 0
            break
    else:
        raise AssertionError("The environment did not terminate after successful delivery.")

    second_obs, _ = env.reset()
    assert second_obs.tolist() == [0, 0, 2, 0, 4, 5, 0]

    print("Smoke test passed.")


if __name__ == "__main__":
    test_basic_delivery_smoke()
