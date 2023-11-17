import numpy as np
import pytest
from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.env_checker import check_env

from src.environments.awake_e_steering import AwakeESteering


def test_check_env_cheetah():
    """Test SB3's `check_env` on all environments using their Cheetah backends."""
    env = AwakeESteering(backend="cheetah")
    env = RescaleAction(env, -1, 1)  # Prevents SB3 action space scale warning
    check_env(env)


def test_passing_backend_args():
    """
    Test that backend_args are passed through the environment to the backend correctly.
    """
    env = AwakeESteering(
        backend="cheetah",
        backend_args={
            "quad_drift_frequency": np.pi * 0.002,
            "quad_drift_amplitude": 0.3142,
            "quad_random_scale": 0.42,
        },
    )

    # Test that config is passed through to backend
    assert env.unwrapped.backend.quad_drift_frequency == np.pi * 0.002
    assert env.unwrapped.backend.quad_drift_amplitude == 0.3142
    assert env.unwrapped.backend.quad_random_scale == 0.42


def test_public_members():
    """
    Make sure that all and only intended members are exposed to the user (named withouth
    leading underscore).
    """
    gymnasium_public_members = [
        "reset",
        "step",
        "render",
        "close",
        "action_space",
        "observation_space",
        "metadata",
        "np_random",
        "render_mode",
        "reward_range",
        "spec",
        "unwrapped",
        "get_wrapper_attr",
    ]
    custom_public_members = ["backend"]
    allowed_public_members = gymnasium_public_members + custom_public_members

    env = AwakeESteering(backend="cheetah")
    _, _ = env.reset()
    _, _, _, _, _ = env.step(env.action_space.sample())

    members = dir(env)
    public_members = [m for m in members if not m.startswith("_")]

    for member in public_members:
        assert member in allowed_public_members

        # Remove member from list of allowed members
        allowed_public_members.remove(member)


@pytest.mark.skip(reason="Random seeds are not fixed yet")
def test_seed():
    """
    Test that using a fixed seed produces reproducible initial magnet settings and
    target beams, while different seeds produce different values.
    """
    env = AwakeESteering(
        backend="cheetah", magnet_init_mode="random", target_beam_mode="random"
    )
    observation_ref, _ = env.reset(seed=42)
    observation_same, _ = env.reset(seed=42)
    observation_diff, _ = env.reset(seed=24)

    # Magnet settings
    assert all(observation_ref["magnets"] == observation_same["magnets"])
    assert all(observation_ref["magnets"] != observation_diff["magnets"])

    # Target beams
    assert all(observation_ref["target"] == observation_same["target"])
    assert all(observation_ref["target"] != observation_diff["target"])
