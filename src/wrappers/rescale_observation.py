from typing import Optional, Union

import gymnasium as gym
import numpy as np

# TODO: Clean this up to be more general


class RescaleObservation(gym.ObservationWrapper):
    """
    Rescales the observation of transverse tuning environments to `scaled_range`. This
    is intended as a fixed form of observation normalisation.
    """

    def __init__(
        self,
        env: gym.Env,
        min_observation: float = -1,
        max_observation: float = 1,
        assumed_space: Optional[gym.Space] = None,
    ):
        super().__init__(env)

        self.min_observation = min_observation
        self.max_observation = max_observation

        self.assumed_space = assumed_space

        if isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict(
                {
                    key: gym.spaces.Box(
                        low=np.where(space.low != -np.inf, min_observation, -np.inf),
                        high=np.where(space.high != np.inf, max_observation, np.inf),
                        dtype=space.dtype,
                    )
                    for key, space in env.observation_space.spaces.items()
                }
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=np.where(
                    env.observation_space.low != -np.inf, min_observation, -np.inf
                ),
                high=np.where(
                    env.observation_space.high != np.inf, max_observation, np.inf
                ),
                dtype=env.observation_space.dtype,
            )

        assert (
            self.assumed_space is None
            or self.assumed_space.__class__ == self.env.observation_space.__class__
        ), (
            f"Assumed space {self.assumed_space} does not match environment "
            f"observation space {self.env.observation_space}"
        )

    def observation(self, observation: Union[np.ndarray, dict]) -> dict:
        if isinstance(observation, np.ndarray):
            return self._rescale(observation)
        elif isinstance(observation, dict):
            return {
                key: self._rescale(value, key=key) for key, value in observation.items()
            }

    def _rescale(
        self, observation: np.ndarray, key: Optional[str] = None
    ) -> np.ndarray:
        if isinstance(self.env.observation_space, gym.spaces.Box):
            env_low = (
                self.assumed_space.low
                if self.assumed_space is not None
                else self.env.observation_space.low
            )
            env_high = (
                self.assumed_space.high
                if self.assumed_space is not None
                else self.env.observation_space.high
            )
        else:
            assert key is not None, (
                "Observation space is a dict, but no key was provided. "
                "Please provide a key."
            )
            env_low = (
                self.assumed_space[key].low
                if self.assumed_space is not None and key in self.assumed_space
                else self.env.observation_space[key].low
            )
            env_high = (
                self.assumed_space[key].high
                if self.assumed_space is not None and key in self.assumed_space
                else self.env.observation_space[key].high
            )

        return self.min_observation + (observation - env_low) * (
            self.max_observation - self.min_observation
        ) / (env_high - env_low)
