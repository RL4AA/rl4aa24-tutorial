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


def test_mandatory_backend_argument(section):
    """Test that the `backend` argument is mandatory."""
    with pytest.raises(TypeError):
        AwakeESteering(
            # backend="cheetah"
        )


@pytest.mark.skip(reason="Not yet adapted to Awake e-steering")
def test_passing_backend_args(section):
    """
    Test that backend_args are passed through the environment to the backend correctly.
    """
    incoming_mode = np.array(
        [
            160e6,
            1e-3,
            1e-4,
            1e-3,
            1e-4,
            5e-4,
            5e-5,
            5e-4,
            5e-5,
            5e-5,
            1e-3,
        ]
    )
    if section in [ea, bc]:  # EA and BC with 3 quadrupoles + screen
        misalignment_mode = np.array([-1e-4, 1e-4, -1e-5, 1e-5, 3e-4, 0, -3e-4, 9e-5])
    else:  # DL and SH with 2 quadrupoles + screen
        misalignment_mode = np.array([-1e-4, 1e-4, -1e-5, 1e-5, -3e-4, 9e-5])

    simulate_finite_screen = True

    env = section.TransverseTuning(
        backend="cheetah",
        backend_args={
            "incoming_mode": incoming_mode,
            "misalignment_mode": misalignment_mode,
            "simulate_finite_screen": simulate_finite_screen,
        },
    )

    # Test that config is passed through to backend
    assert all(env.unwrapped.backend.incoming_mode == incoming_mode)
    assert all(env.unwrapped.backend.misalignment_mode == misalignment_mode)
    assert env.unwrapped.backend.simulate_finite_screen == simulate_finite_screen

    # Test that configs are used correctly
    _, _ = env.reset()
    incoming_parameters = np.array(
        [
            env.unwrapped.backend.incoming.parameters["energy"],
            env.unwrapped.backend.incoming.parameters["mu_x"],
            env.unwrapped.backend.incoming.parameters["mu_xp"],
            env.unwrapped.backend.incoming.parameters["mu_y"],
            env.unwrapped.backend.incoming.parameters["mu_yp"],
            env.unwrapped.backend.incoming.parameters["sigma_x"],
            env.unwrapped.backend.incoming.parameters["sigma_xp"],
            env.unwrapped.backend.incoming.parameters["sigma_y"],
            env.unwrapped.backend.incoming.parameters["sigma_yp"],
            env.unwrapped.backend.incoming.parameters["sigma_s"],
            env.unwrapped.backend.incoming.parameters["sigma_p"],
        ]
    )

    assert np.allclose(incoming_parameters, incoming_mode)
    assert np.allclose(env.unwrapped.backend.get_misalignments(), misalignment_mode)


@pytest.mark.skip(reason="Not yet adapted to Awake e-steering")
def test_public_members(section):
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
    custom_public_members = [
        "backend",
        "action_mode",
        "magnet_init_mode",
        "max_quad_delta",
        "max_steerer_delta",
        "target_beam_mode",
        "target_threshold",
        "threshold_hold",
        "unidirectional_quads",
        "clip_magnets",
    ]
    allowed_public_members = gymnasium_public_members + custom_public_members

    env = section.TransverseTuning(backend="cheetah")
    _, _ = env.reset()
    _, _, _, _, _ = env.step(env.action_space.sample())

    members = dir(env)
    public_members = [m for m in members if not m.startswith("_")]

    for member in public_members:
        assert member in allowed_public_members

        # Remove member from list of allowed members
        allowed_public_members.remove(member)


@pytest.mark.skip(reason="Random seeds are not fixed yet")
def test_seed(section):
    """
    Test that using a fixed seed produces reproducible initial magnet settings and
    target beams, while different seeds produce different values.
    """
    env = section.TransverseTuning(
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


@pytest.mark.skip(reason="Not yet adapted to Awake e-steering")
def test_magnet_clipping_direct(section):
    """
    Test that magnet settings are clipped to the allowed range when the action mode is
    set to "direct".
    """
    env = section.TransverseTuning(
        backend="cheetah",
        action_mode="direct",
        clip_magnets=True,
    )
    min_magnet_settings = env.observation_space["magnets"].low
    max_magnet_settings = env.observation_space["magnets"].high

    _, _ = env.reset()
    observation, _, _, _, _ = env.step(max_magnet_settings * 2)

    assert all(observation["magnets"] >= min_magnet_settings)
    assert all(observation["magnets"] <= max_magnet_settings)


@pytest.mark.skip(reason="Not yet adapted to Awake e-steering")
def test_magnet_clipping_delta(section):
    """
    Test that magnet settings are clipped to the allowed range when the action mode is
    set to "delta".
    """
    env = section.TransverseTuning(
        backend="cheetah",
        action_mode="direct",
        clip_magnets=True,
    )
    min_magnet_settings = env.observation_space["magnets"].low
    max_magnet_settings = env.observation_space["magnets"].high

    env.reset(options={"magnet_init": max_magnet_settings * 0.5})
    observation, _, _, _, _ = env.step(max_magnet_settings)

    assert all(observation["magnets"] >= min_magnet_settings)
    assert all(observation["magnets"] <= max_magnet_settings)


@pytest.mark.skip(reason="Not yet adapted to Awake e-steering")
def test_fixed_magnet_init_mode_array(section, settings):
    """
    Test that if fixed values are set for `magnet_init_mode`, the magnets are in fact
    set to these values. This tests checks two consecutive resets. It considers the
    initials values to be set as a NumPy array.
    """
    env = section.TransverseTuning(
        backend="cheetah", magnet_init_mode=np.array(settings)
    )
    observation_first_reset, _ = env.reset()
    _, _, _, _, _ = env.step(env.action_space.sample())
    observation_second_reset, _ = env.reset()

    assert np.allclose(observation_first_reset["magnets"], np.array(settings))
    assert np.allclose(observation_second_reset["magnets"], np.array(settings))


@pytest.mark.skip(reason="Not yet adapted to Awake e-steering")
def test_fixed_magnet_init_mode_list(section, settings):
    """
    Test that if fixed values are set for `magnet_init_mode`, the magnets are in fact
    set to these values. This tests checks two consecutive resets. It considers the
    initials values to be set as a Python list.
    """
    env = section.TransverseTuning(backend="cheetah", magnet_init_mode=settings)
    observation_first_reset, _ = env.reset()
    _, _, _, _, _ = env.step(env.action_space.sample())
    observation_second_reset, _ = env.reset()

    assert np.allclose(observation_first_reset["magnets"], np.array(settings))
    assert np.allclose(observation_second_reset["magnets"], np.array(settings))
