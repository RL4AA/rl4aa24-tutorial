import numpy as np
import pytest

from src.environments.awake_e_steering import AwakeESteering


@pytest.mark.skip(
    reason="Random seeds are not fixed yet and this test is not yet adapted to Awake."
)
def test_seed():
    """
    Test that using a fixed seed produces reproducible initial magnet settings and
    target beams, while different seeds produce different values.
    """
    env = AwakeESteering(
        backend="cheetah",
        backend_args={"incoming_mode": "random", "misalignment_mode": "random"},
    )
    _, info_ref = env.reset(seed=42)
    _, info_same = env.reset(seed=42)
    _, info_diff = env.reset(seed=24)

    # Incoming beam
    assert all(info_ref["incoming"] == info_same["incoming"])
    assert all(info_ref["incoming"] != info_diff["incoming"])

    # Misalignments
    assert all(info_ref["misalignments"] == info_same["misalignments"])
    assert all(info_ref["misalignments"] != info_diff["misalignments"])


def test_quad_drift_off():
    """
    Test that when the quadrupole drifts are turned off (drift amplitude is set to 0.0),
    the BPM readings are stable when zero actions are performed.
    """
    env = AwakeESteering(backend="cheetah", backend_args={"quad_drift_amplitude": 0.0})

    # Reset the environment
    observation_on_reset, _ = env.reset()

    # Perform 10 steps without any actions
    step_observations = [env.step(np.zeros(10))[0] for _ in range(10)]

    # Perform 10 resets
    reset_observations = [env.reset()[0] for _ in range(10)]

    # Check that the BPM readings are stable
    assert all(
        np.allclose(obs, observation_on_reset)
        for obs in step_observations + reset_observations
    )


def test_quad_drift_on():
    """
    Test that when the quadrupole drifts are turned on (drift amplitude is set to a
    non-zero value), the BPM readings are not stable when zero actions are performed.
    """
    env = AwakeESteering(backend="cheetah", backend_args={"quad_drift_amplitude": 1.0})

    # Reset the environment
    observation_on_reset, _ = env.reset()

    # Perform 10 steps without any actions
    step_observations = [env.step(np.zeros(10))[0] for _ in range(10)]

    # Perform 10 resets
    reset_observations = [env.reset()[0] for _ in range(10)]

    # Check that the BPM readings are not stable
    assert not any(
        np.allclose(obs, observation_on_reset)
        for obs in step_observations + reset_observations
    )


def test_random_quads_off():
    """
    Test that when random quads are off, we get the same BPM readings on each reset.
    """
    env = AwakeESteering(backend="cheetah", backend_args={"quad_random_scale": 0.0})

    observation_1, _ = env.reset()
    observation_2, _ = env.reset()

    assert np.allclose(observation_1, observation_2)


def test_random_quads_on():
    """
    Test that when random quads are on, we get different BPM readings on each reset.
    """
    env = AwakeESteering(backend="cheetah", backend_args={"quad_random_scale": 1.0})

    observation_1, _ = env.reset()
    observation_2, _ = env.reset()

    assert not np.allclose(observation_1, observation_2)
