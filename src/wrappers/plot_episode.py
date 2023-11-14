import os
from pathlib import Path
from typing import Callable

import gymnasium as gym
import wandb
from gymnasium import logger

# from src.eval import Episode

# TODO Make record episode, plot episode and area ea elog wrapper based on one wrapper
#     base class.


def capped_cubic_video_schedule(episode_id: int) -> bool:
    """
    The default episode trigger.

    This function will trigger recordings at the episode indices
    0, 1, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Taken from Gymnasium package:
    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/record_video.py

    :param episode_id: The episode number
    :return: If to apply a plot schedule number.
    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


class PlotEpisode(gym.Wrapper):
    """
    Records a plot of the episode when the episode is done.

    NOTE: This wrapper is only compatible with ARES transverse beam parameter tuning
    environments.

    NOTE: A plot will only be saved if at least two steps have been taken in the
    episode.

    :param env: The environment that will be wrapped.
    :param plot_dir: The directory where the plots are be saved.
    :param episode_trigger: A function that takes the episode number as input and
        returns a boolean indicating whether the episode should be recorded.
    """

    def __init__(
        self,
        env: gym.Env,
        save_dir: Path,
        episode_trigger: Callable[[int], bool] = None,
        name_prefix: str = "rl-plot-episode",
        log_to_wandb: bool = False,
    ):
        super().__init__(env)

        self.save_dir = os.path.abspath(save_dir)
        if os.path.isdir(self.save_dir):
            print(
                f"Overwriting existing episode plots at {self.save_dir} folder. Specify"
                " a different `save_dir` for the `RecordEpisode` wrapper if this is"
                " not desired."
            )
        os.makedirs(self.save_dir, exist_ok=True)

        self.name_prefix = name_prefix

        self.episode_id = 0
        self.is_recording = False

        if episode_trigger is None:
            episode_trigger = capped_cubic_video_schedule
        self.episode_trigger = episode_trigger

        self.log_to_wandb = log_to_wandb

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)

        if self.episode_trigger(self.episode_id):
            self.is_recording = True

            self.observations = [observation]
            self.rewards = []
            self.terminateds = []
            self.truncateds = []
            self.infos = []
            self.actions = []

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if self.is_recording:
            self.observations.append(observation)
            self.rewards.append(reward)
            self.terminateds.append(terminated)
            self.truncateds.append(truncated)
            self.infos.append(info)
            self.actions.append(action)

            if terminated or truncated:
                self._save_episode_plot()
                self.is_recording = False

        if terminated or truncated:
            self.episode_id += 1

        return observation, reward, terminated, truncated, info

    def close(self):
        super().close()

        if self.is_recording:
            self._save_episode_plot()

    def _save_episode_plot(self):
        """
        Creates a matplotlib plot of the episode and saves it to the plot directory.
        """
        if len(self.observations) < 2:  # No data to plot
            logger.warn(
                f"Unable to save episode plot for {self.episode_id = } because the"
                " episode was too short."
            )
            return

        episode = Episode(
            observations=self.observations,
            rewards=self.rewards,
            terminateds=self.terminateds,
            truncateds=self.truncateds,
            infos=self.infos,
            actions=self.actions,
        )

        file_path = os.path.join(self.save_dir, f"{self.name_prefix}-{self.episode_id}")
        fig = episode.plot_summary(save_path=file_path)

        if self.log_to_wandb and wandb.run:
            wandb.log({"plots": fig})
