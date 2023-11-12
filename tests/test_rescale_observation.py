import numpy as np
from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.env_checker import check_env

from src.environments import ea
from src.wrappers import RescaleObservation


def test_check_env():
    """Test that the `RecordEpisode` wrapper throws no exceptions under `check_env`."""
    env = ea.TransverseTuning(backend="cheetah")
    env = RescaleObservation(env, -1, 1)
    env = RescaleAction(env, -1, 1)

    check_env(env)


def test_observation_space():
    """
    Check that the observation space is correctly rescaled, but otherwise matches the
    wrapped environment.
    """
    env = ea.TransverseTuning(backend="cheetah")
    wrapped_env = RescaleObservation(env, -1, 1)

    assert wrapped_env.observation_space.shape == env.observation_space.shape
    assert wrapped_env.observation_space.dtype == env.observation_space.dtype

    for key in wrapped_env.observation_space.spaces.keys():
        assert (np.isin(wrapped_env.observation_space[key].low, [-1, -np.inf])).all()
        assert (np.isin(wrapped_env.observation_space[key].high, [1, np.inf])).all()
