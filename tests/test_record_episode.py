from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.env_checker import check_env

from src.environments.awake_e_steering import AwakeESteering
from src.wrappers import RecordEpisode


def test_check_env():
    """Test that the `RecordEpisode` wrapper throws no exceptions under `check_env`."""
    env = AwakeESteering(backend="cheetah")
    env = RecordEpisode(env)
    env = RescaleAction(env, -1, 1)

    check_env(env)


def test_reset_info():
    """Test that `info` from reset is saved."""
    env = AwakeESteering(backend="cheetah")
    env = RecordEpisode(env)

    _, _ = env.reset()

    assert len(env.infos) == 1
