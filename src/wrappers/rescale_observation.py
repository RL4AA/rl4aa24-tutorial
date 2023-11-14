from typing import Union

import gymnasium as gym
import numpy as np

# TODO: Clean this up to be more general


class RescaleObservation(gym.ObservationWrapper):
    """
    Rescales the observation of transverse tuning environments to `scaled_range`. This
    is intended as a fixed form of observation normalisation.
    """

    def __init__(
        self, env: gym.Env, min_observation: float = -1, max_observation: float = 1
    ):
        super().__init__(env)

        self.min_observation = min_observation
        self.max_observation = max_observation

        if isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict(
                {
                    key: gym.spaces.Box(
                        low=min_observation if key != "beam" else -np.inf,
                        high=max_observation if key != "beam" else np.inf,
                        shape=space.shape,
                        dtype=space.dtype,
                    )
                    for key, space in env.observation_space.spaces.items()
                }
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=min_observation,
                high=max_observation,
                shape=env.observation_space.shape,
                dtype=env.observation_space.dtype,
            )

    def observation(self, observation: Union[np.ndarray, dict]) -> dict:
        if isinstance(observation, np.ndarray):
            return self._rescale_array(observation)
        elif isinstance(observation, dict):
            return {
                key: self._rescale_dict_entry(key, value)
                for key, value in observation.items()
            }

    def _rescale_array(self, observation: np.ndarray) -> np.ndarray:
        return self.min_observation + (observation - self.env.observation_space.low) * (
            self.max_observation - self.min_observation
        ) / (self.env.observation_space.high - self.env.observation_space.low)

    def _rescale_dict_entry(self, key: str, value: np.ndarray) -> np.ndarray:
        if key == "beam":  # Exception for "beam" which has infinite range
            key = "target"  # Scale beam just like target

        return self.min_observation + (value - self.env.observation_space[key].low) * (
            self.max_observation - self.min_observation
        ) / (self.env.observation_space[key].high - self.env.observation_space[key].low)
