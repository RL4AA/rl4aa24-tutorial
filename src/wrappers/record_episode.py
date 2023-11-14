import os
import pickle
from datetime import datetime

import gymnasium as gym


class RecordEpisode(gym.Wrapper):
    """
    Wrapper for recording epsiode data such as observations, rewards, infos and actions.
    Pass a `save_dir` other than `None` to save the recorded data to pickle files.
    """

    def __init__(self, env, save_dir=None, name_prefix="recorded_episode"):
        super().__init__(env)

        self.save_dir = save_dir
        if self.save_dir is not None:
            self.save_dir = os.path.abspath(save_dir)
            if os.path.isdir(self.save_dir):
                print(
                    f"Overwriting existing data recordings at {self.save_dir} folder."
                    " Specify a different `save_dir` for the `RecordEpisode` wrapper"
                    " if this is not desired."
                )
            os.makedirs(self.save_dir, exist_ok=True)

        self.name_prefix = name_prefix

        self.n_episodes_recorded = 0

    def reset(self, seed=None, options=None):
        self.t_end = datetime.now()

        if self.save_dir is not None and self.n_episodes_recorded > 0:
            self.save_to_file()

        if self.n_episodes_recorded > 0:
            self.previous_observations = self.observations
            self.previous_rewards = self.rewards
            self.previous_terminateds = self.terminateds
            self.previous_truncateds = self.truncateds
            self.previous_infos = self.infos
            self.previous_actions = self.actions
            self.previous_t_start = self.t_start
            self.previous_t_end = self.t_end
            self.previous_steps_taken = self.steps_taken

        self.n_episodes_recorded += 1

        observation, info = self.env.reset(seed=seed, options=options)

        self.observations = [observation]
        self.rewards = []
        self.terminateds = []
        self.truncateds = []
        self.infos = [info]
        self.actions = []
        self.t_start = datetime.now()
        self.t_end = None
        self.steps_taken = 0
        self.step_start_times = []
        self.step_end_times = []

        self.has_previously_run = True

        return observation, info

    def step(self, action):
        self.step_start_times.append(datetime.now())

        observation, reward, terminated, truncated, info = self.env.step(action)

        self.observations.append(observation)
        self.rewards.append(reward)
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)
        self.infos.append(info)
        self.actions.append(action)
        self.steps_taken += 1
        self.step_end_times.append(datetime.now())

        return observation, reward, terminated, truncated, info

    def close(self):
        super().close()

        self.t_end = datetime.now()

        if self.save_dir is not None and self.n_episodes_recorded > 0:
            self.save_to_file()

    def save_to_file(self):
        """Save the data from the current episodes to a `.pkl` file."""
        filename = f"{self.name_prefix}_{self.n_episodes_recorded}.pkl"
        path = os.path.join(self.save_dir, filename)

        d = {
            "observations": self.observations,
            "rewards": self.rewards,
            "terminateds": self.terminateds,
            "truncateds": self.truncateds,
            "infos": self.infos,
            "actions": self.actions,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "steps_taken": self.steps_taken,
            "step_start_times": self.step_start_times,
            "step_end_times": self.step_end_times,
        }

        with open(path, "wb") as f:
            pickle.dump(d, f)
