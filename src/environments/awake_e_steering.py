from typing import Literal, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environments.base_backend import ESteeringBaseBackend


class AwakeESteering(gym.Env):
    """
    Environment for electron steering on the AWAKE beam line.

    Magnets: TODO

    :param backend: Backend for communication with either a simulation or the control
        system.
    :param backend_args: Arguments for the backend. NOTE that these may be different
        for different backends.
    :param render_mode: Defines how the environment is rendered according to the
        Gymnasium documentation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        backend: Literal["cheetah"] = "cheetah",
        backend_args: dict = {},
        render_mode: Optional[Literal["human", "rgb_array"]] = None,
    ) -> None:
        self.observation_space = None  # TODO: Define observation space
        self.action_space = None  # TODO: Define action space

        # Setup particle simulation or control system backend
        if backend == "cheetah":
            self.backend = CheetahBackend(**backend_args)
        else:
            raise ValueError(f'Invalid value "{backend}" for backend')

        # Utility variables
        # TODO: Define utility variables

        # Setup rendering according to Gymnasium manual
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        env_options, backend_options = self._preprocess_reset_options(options)

        self.backend.reset(options=backend_options)

        # TODO: Initialise magnets

        # TODO: Initialise incoming beam?

        # Update anything in the accelerator (mainly for running simulations)
        self.backend.update()

        # Set reward variables to None, so that _get_reward works properly
        self._some_reward = None  # TODO: Replace with actual reward variables

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self._take_action(action)

        self.backend.update()  # Run the simulation

        terminated = self._get_terminated()
        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _preprocess_reset_options(self, options: dict) -> tuple[dict, dict]:
        """
        Check that only valid options are passed and split the options into environment
        and backend options.

        NOTE: Backend options are not validated and should be validated by the backend
        itself.
        """
        if options is None:
            return {}, None

        valid_options = ["backend_options"]
        for option in options:
            assert option in valid_options

        env_options = {k: v for k, v in options.items() if k != "backend_options"}
        backend_options = options.get("backend_options", None)

        return env_options, backend_options

    def _get_terminated(self):
        return False  # TODO: Replace with actual termination condition

    def _get_obs(self):
        return {}  # TODO: Replace with actual observation

    def _get_info(self):
        return {
            # TODO: Replace with actual info
            "backend_info": self.backend.get_info(),  # Info specific to the backend
        }

    def _take_action(self, action: np.ndarray) -> None:
        """Take `action` according to the environment's configuration."""

        # TODO: Replace with actual taking of action

        self._previous_magnet_settings = self.backend.get_magnets()

        if self.action_mode == "direct":
            new_settings = action
            if self.clip_magnets:
                new_settings = self._clip_magnets_to_power_supply_limits(new_settings)
            self.backend.set_magnets(new_settings)
        elif self.action_mode == "delta":
            new_settings = self._previous_magnet_settings + action
            if self.clip_magnets:
                new_settings = self._clip_magnets_to_power_supply_limits(new_settings)
            self.backend.set_magnets(new_settings)
        else:
            raise ValueError(f'Invalid value "{self.action_mode}" for action_mode')

    def _clip_magnets_to_power_supply_limits(self, magnets: np.ndarray) -> np.ndarray:
        """Clip `magnets` to limits imposed by the magnets's power supplies."""

        # TODO: Check if this is actually needed

        return np.clip(
            magnets,
            self.observation_space["magnets"].low,
            self.observation_space["magnets"].high,
        )

    def _get_reward(self) -> float:
        return 0.0  # TODO: Replace with actual reward

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        raise NotImplementedError

    def close(self):
        pass


class CheetahBackend(ESteeringBaseBackend):
    """
    Cheetah simulation backend.

    :param incoming_mode: Setting for incoming beam parameters on reset. Can be
        `"random"` to generate random parameters or an array of 11 values to set them to
        a constant value.
    """

    def __init__(
        self, incoming_mode: Union[Literal["random"], np.ndarray] = "random"
    ) -> None:
        # Dynamic import for module only required by this backend
        global cheetah
        import cheetah

        assert isinstance(incoming_mode, (str, np.ndarray))

        self.incoming_mode = incoming_mode

        # Simulation setup
        self.segment = cheetah.Segment.from_ocelot(
            ocelot_lattice.cell, warnings=False, device="cpu"
        ).subcell("AREASOLA1", "AREABSCR1")

        self.segment.AREABSCR1.resolution = (2448, 2040)
        self.segment.AREABSCR1.pixel_size = (3.3198e-6, 2.4469e-6)
        self.segment.AREABSCR1.binning = 1
        self.segment.AREABSCR1.is_active = True

        # Spaces for domain randomisation
        self.incoming_beam_space = spaces.Box(
            low=np.array(
                [
                    80e6,
                    -1e-3,
                    -1e-4,
                    -1e-3,
                    -1e-4,
                    1e-5,
                    1e-6,
                    1e-5,
                    1e-6,
                    1e-6,
                    1e-4,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [160e6, 1e-3, 1e-4, 1e-3, 1e-4, 5e-4, 5e-5, 5e-4, 5e-5, 5e-5, 1e-3],
                dtype=np.float32,
            ),
        )

    def get_magnets(self) -> np.ndarray:
        return np.array(
            [
                self.segment.AREAMQZM1.k1,
                self.segment.AREAMQZM2.k1,
                self.segment.AREAMCVM1.angle,
                self.segment.AREAMQZM3.k1,
                self.segment.AREAMCHM1.angle,
            ]
        )

    def set_magnets(self, values: np.ndarray) -> None:
        self.segment.AREAMQZM1.k1 = values[0]
        self.segment.AREAMQZM2.k1 = values[1]
        self.segment.AREAMCVM1.angle = values[2]
        self.segment.AREAMQZM3.k1 = values[3]
        self.segment.AREAMCHM1.angle = values[4]

    def reset(self, options=None) -> None:
        preprocessed_options = self._preprocess_reset_options(options)

        # Set up incoming beam
        if "incoming" in preprocessed_options:
            incoming_parameters = preprocessed_options["incoming"]
        elif isinstance(self.incoming_mode, np.ndarray):
            incoming_parameters = self.incoming_mode
        elif self.incoming_mode == "random":
            incoming_parameters = self.incoming_beam_space.sample()

        self.incoming = cheetah.ParameterBeam.from_parameters(
            energy=incoming_parameters[0],
            mu_x=incoming_parameters[1],
            mu_xp=incoming_parameters[2],
            mu_y=incoming_parameters[3],
            mu_yp=incoming_parameters[4],
            sigma_x=incoming_parameters[5],
            sigma_xp=incoming_parameters[6],
            sigma_y=incoming_parameters[7],
            sigma_yp=incoming_parameters[8],
            sigma_s=incoming_parameters[9],
            sigma_p=incoming_parameters[10],
        )

    def _preprocess_reset_options(self, options: dict) -> dict:
        """
        Check that only valid options are passed and make it a dict if None was passed.
        """
        if options is None:
            return {}

        valid_options = ["incoming"]
        for option in options:
            assert option in valid_options

        return options

    def update(self) -> None:
        self.segment.track(self.incoming)

    def get_beam_parameters(self) -> np.ndarray:
        if self.simulate_finite_screen and not self.is_beam_on_screen():
            return np.array([0, 3.5, 0, 2.2])  # Estimates from real bo_sim data
        else:
            return np.array(
                [
                    self.segment.AREABSCR1.read_beam.mu_x,
                    self.segment.AREABSCR1.read_beam.sigma_x,
                    self.segment.AREABSCR1.read_beam.mu_y,
                    self.segment.AREABSCR1.read_beam.sigma_y,
                ]
            )

    def get_incoming_parameters(self) -> np.ndarray:
        # Parameters of incoming are typed out to guarantee their order, as the
        # order would not be guaranteed creating np.array from dict.
        return np.array(
            [
                self.incoming.energy,
                self.incoming.mu_x,
                self.incoming.mu_xp,
                self.incoming.mu_y,
                self.incoming.mu_yp,
                self.incoming.sigma_x,
                self.incoming.sigma_xp,
                self.incoming.sigma_y,
                self.incoming.sigma_yp,
                self.incoming.sigma_s,
                self.incoming.sigma_p,
            ]
        )

    def get_info(self) -> dict:
        info = {"incoming_beam": self.get_incoming_parameters()}

        return info
