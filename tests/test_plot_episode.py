import pytest
from gymnasium.wrappers import RecordVideo, RescaleAction, TimeLimit
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import unwrap_wrapper

from src.environments.awake_e_steering import AwakeESteering

# from src.wrappers import PlotEpisode

# TODO Test that episode trigger behaves like RecordVideo


@pytest.mark.skip("Not yet implemented for Awake.")
def test_check_env(tmp_path):
    """Test that the `PlotEpisode` wrapper throws no exceptions under `check_env`."""
    env = AwakeESteering(backend="cheetah")
    env = PlotEpisode(env, save_dir=tmp_path)
    env = RescaleAction(env, -1, 1)

    check_env(env)


@pytest.mark.skip("Not yet implemented for Awake.")
def test_trigger_like_record_video(tmp_path):
    """
    Test that, given the same trigger function, the `PlotEpisode` wrapper records the
    same episodes as the `RecordVideo` wrapper from Gymnasium.
    """
    env = AwakeESteering(
        backend="cheetah",
        backend_args={"generate_screen_images": True},
        render_mode="rgb_array",
    )
    env = TimeLimit(env, 10)
    env = PlotEpisode(
        env, save_dir=str(tmp_path / "plots"), episode_trigger=lambda x: x % 5 == 0
    )
    env = RecordVideo(
        env,
        video_folder=str(tmp_path / "recordings"),
        episode_trigger=lambda x: x % 5 == 0,
    )

    plot_episode = unwrap_wrapper(env, PlotEpisode)
    record_video = unwrap_wrapper(env, RecordVideo)

    for i in range(10):
        _, _ = env.reset()
        assert plot_episode.episode_id == i
        assert record_video.episode_id == i

        assert plot_episode.is_recording == (i % 5 == 0)
        assert plot_episode.is_recording == record_video.recording

        done = False
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()


@pytest.mark.skip("Not yet implemented for Awake.")
def test_file_written(tmp_path):
    """
    Test that the `PlotEpisode` wrapper writes a file when the episode trigger is
    active.
    """
    env = AwakeESteering(
        backend="cheetah", backend_args={"generate_screen_images": True}
    )
    env = TimeLimit(env, 10)
    env = PlotEpisode(
        env,
        save_dir=str(tmp_path / "plots"),
        episode_trigger=lambda _: True,
    )

    _, _ = env.reset()
    assert env.is_recording is True

    done = False
    while not done:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()

    assert (tmp_path / "plots" / "rl-plot-episode-0.png").exists()


@pytest.mark.skip("Not yet implemented for Awake.")
def test_no_file_written(tmp_path):
    """
    Test that the `PlotEpisode` wrapper writes a file when the episode trigger is
    active.
    """
    env = AwakeESteering(
        backend="cheetah", backend_args={"generate_screen_images": True}
    )
    env = TimeLimit(env, 10)
    env = PlotEpisode(
        env,
        save_dir=str(tmp_path / "plots"),
        episode_trigger=lambda _: False,
    )

    _, _ = env.reset()
    assert env.is_recording is False

    done = False
    while not done:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()

    assert not (tmp_path / "plots" / "rl-plot-episode-0.png").exists()


@pytest.mark.skip("Not yet implemented for Awake.")
def test_episode_id_advanced(tmp_path):
    """
    Test that the episode ID advances in the same way as it does in the `RecordVideo`
    wrapper from Gymnasium.
    """
    env = AwakeESteering(
        backend="cheetah",
        backend_args={"generate_screen_images": True},
        render_mode="rgb_array",
    )
    env = TimeLimit(env, 10)
    env = PlotEpisode(
        env, save_dir=str(tmp_path / "plots"), episode_trigger=lambda x: x % 5 == 0
    )
    env = RecordVideo(
        env,
        video_folder=str(tmp_path / "recordings"),
        episode_trigger=lambda x: x % 5 == 0,
    )

    plot_episode = unwrap_wrapper(env, PlotEpisode)
    record_video = unwrap_wrapper(env, RecordVideo)

    # Test normal case where episode was terminated or truncated
    for i in range(10):
        _, _ = env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
        assert plot_episode.episode_id == record_video.episode_id

    # Test abnormal case where episode is just run for some steps and then reset
    for i in range(10):
        _, _ = env.reset()
        for _ in range(5):
            _, _, _, _, _ = env.step(env.action_space.sample())
        assert plot_episode.episode_id == record_video.episode_id

    # To supress unnecessary warnings
    env.close()
