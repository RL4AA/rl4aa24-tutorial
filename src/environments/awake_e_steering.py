from pathlib import Path
from typing import Literal, Optional, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from src.environments.base_backend import ESteeringBaseBackend

# TODO: Beam pipe wall


class AwakeESteering(gym.Env):
    """
    Environment for electron steering on the AWAKE beam line.

    Magnets: TODO

    :param max_steerer_delta: Maximum change in steerer settings per step. Determines
        the action space limits.
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
        max_steerer_delta: float = 3e-4,
        backend: Literal["cheetah"] = "cheetah",
        backend_args: Optional[dict] = None,
        render_mode: Optional[Literal["human", "rgb_array"]] = None,
    ) -> None:
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,)
        )  # 1e-2
        self.action_space = spaces.Box(
            low=-max_steerer_delta, high=max_steerer_delta, shape=(10,)
        )

        # Setup particle simulation or control system backend
        backend_args = backend_args or {}
        if backend == "cheetah":
            self.backend = CheetahBackend(**backend_args)
        else:
            raise ValueError(f'Invalid value "{backend}" for backend')

        # Utility variables
        # -> Put utility variables here once they are needed

        # Setup rendering according to Gymnasium manual
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        env_options, backend_options = self._preprocess_reset_options(options)

        self.backend.reset(options=backend_options)

        # Initialise magnets
        if "magnet_init" in env_options:
            self.backend.set_magnets(env_options["magnet_init"])
        else:
            self.backend.set_magnets(self.action_space.sample())

        # Update anything in the accelerator (mainly for running simulations)
        self.backend.update()

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

        valid_options = ["backend_options", "magnet_init"]
        for option in options:
            assert option in valid_options

        env_options = {k: v for k, v in options.items() if k != "backend_options"}
        backend_options = options.get("backend_options", None)

        return env_options, backend_options

    def _get_terminated(self) -> bool:
        rms = np.sqrt(np.mean(np.square(self.backend.get_bpms())))
        return bool(rms < 0.0016)  # 1.6 mm

    def _get_obs(self) -> Union[np.ndarray, dict]:
        return self.backend.get_bpms()

    def _get_info(self) -> dict:
        return {
            "steerer_settings": self.backend.get_magnets(),
            "backend_info": self.backend.get_info(),  # Info specific to the backend
        }

    def _take_action(self, action: np.ndarray) -> None:
        """Take `action` according to the environment's configuration."""
        self._previous_magnet_settings = self.backend.get_magnets()
        new_settings = self._previous_magnet_settings + action
        self.backend.set_magnets(new_settings)

    def _get_reward(self) -> float:
        rms = np.sqrt(np.mean(np.square(self.backend.get_bpms())))
        return -100 * rms

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

    :param quad_drift_frequency: Frequency of quadrupole drifts.
    :param quad_drift_amplitude: Amplitude of quadrupole drifts.
    :param quad_random_scale: Scale of random disturbance to quadrupole settings on
        reset.
    """

    def __init__(
        self,
        quad_drift_frequency: float = np.pi * 0.001,
        quad_drift_amplitude: float = 0.0,
        quad_random_scale: float = 0.0,
    ) -> None:
        # Dynamic import for module only required by this backend
        global cheetah
        import cheetah

        self.quad_drift_frequency = quad_drift_frequency
        self.quad_drift_amplitude = quad_drift_amplitude
        self.quad_random_scale = quad_random_scale

        # Load MADX .out file as DataFrame
        # TODO: Switch to LatticeJSON at some point
        outfile_path = Path(__file__).parent / "electron_tt43.out"
        df = pd.read_csv(outfile_path, delim_whitespace=True, skiprows=45)

        # Shift column names to the left
        df.columns = df.columns[1:].tolist() + [""]
        # Remove first row
        df = df.iloc[1:]
        # Drop last column
        df = df.iloc[:, :-1]
        # Convert all columns except for NAME and KEYWORD to float
        df[df.columns[2:]] = df[df.columns[2:]].astype(float)

        # Convert DataFrame to Cheetah segment
        self.segment = cheetah.Segment(
            elements=[self._convert_row_to_element(row) for row in df.itertuples()]
        ).subcell("TT43_START", "PLASMA_E")
        self.segment.BPM_430039.is_active = True
        self.segment.BPM_430103.is_active = True
        self.segment.BPM_430129.is_active = True
        self.segment.BPM_430203.is_active = True
        self.segment.BPM_430308.is_active = True
        self.segment.BPM_412343.is_active = True
        self.segment.BPM_412345.is_active = True
        self.segment.BPM_412347.is_active = True
        self.segment.BPM_412349.is_active = True
        self.segment.BPM_412351.is_active = True

        # Set up incoming beam (currently only constant as in all original environments)
        self.incoming = cheetah.ParameterBeam.from_twiss(
            beta_x=torch.tensor(5.0),
            beta_y=torch.tensor(5.0),
            emittance_x=torch.tensor(0.000000269),
            emittance_y=torch.tensor(0.000000269),
            energy=torch.tensor(0.019006870e9),  # 0.019006870 GeV
        )

        # Utility variables
        self._persistent_step_count = 0  # Step count that is not reset
        self._quads = [
            element
            for element in self.segment.elements
            if isinstance(element, cheetah.Quadrupole)
        ]
        self._original_quad_settings = np.array([quad.k1 for quad in self._quads])

    def _convert_row_to_element(self, row) -> "cheetah.Element":
        """
        Takes a row from the MADX output file and converts it into a Cheetah element.
        """
        sanitized_name = row.NAME.replace(".", "_").replace("$", "_")

        if row.KEYWORD == "MARKER":
            assert row.L == 0.0
            return cheetah.Marker(name=sanitized_name)
        elif row.KEYWORD == "DRIFT":
            return cheetah.Drift(name=sanitized_name, length=torch.as_tensor(row.L))
        elif row.KEYWORD == "MONITOR":
            assert row.L == 0.0
            assert sanitized_name.startswith("BPM")
            return cheetah.BPM(name=sanitized_name)
        elif row.KEYWORD == "KICKER":  # Hardcoded to be horizontal
            return cheetah.HorizontalCorrector(
                name=sanitized_name, length=torch.as_tensor(row.L)
            )
        elif row.KEYWORD == "QUADRUPOLE":
            return cheetah.Quadrupole(
                name=sanitized_name,
                length=torch.as_tensor(row.L),
                k1=torch.as_tensor(row.K1L / row.L),  # K1L / L matches MADX in values
            )
        elif row.KEYWORD == "INSTRUMENT":
            return cheetah.Drift(name=sanitized_name, length=torch.as_tensor(row.L))
        elif row.KEYWORD == "RBEND":
            return cheetah.RBend(name=sanitized_name, length=torch.as_tensor(row.L))
        else:
            raise NotImplementedError(f"Unknown element type: {row.KEYWORD}")

    def get_magnets(self) -> np.ndarray:
        return np.array(
            [
                self.segment.MCAWA_430029.angle,
                self.segment.MCAWA_430040.angle,
                self.segment.MCAWA_430104.angle,
                self.segment.MCAWA_430130.angle,
                self.segment.MCAWA_430204.angle,
                self.segment.MCAWA_430309.angle,
                self.segment.MCAWA_412344.angle,
                self.segment.MCAWA_412345.angle,
                self.segment.MCAWA_412347.angle,
                self.segment.MCAWA_412349.angle,
            ]
        )

    def set_magnets(self, values: np.ndarray) -> None:
        self.segment.MCAWA_430029.angle = torch.as_tensor(values[0])
        self.segment.MCAWA_430040.angle = torch.as_tensor(values[1])
        self.segment.MCAWA_430104.angle = torch.as_tensor(values[2])
        self.segment.MCAWA_430130.angle = torch.as_tensor(values[3])
        self.segment.MCAWA_430204.angle = torch.as_tensor(values[4])
        self.segment.MCAWA_430309.angle = torch.as_tensor(values[5])
        self.segment.MCAWA_412344.angle = torch.as_tensor(values[6])
        self.segment.MCAWA_412345.angle = torch.as_tensor(values[7])
        self.segment.MCAWA_412347.angle = torch.as_tensor(values[8])
        self.segment.MCAWA_412349.angle = torch.as_tensor(values[9])

    def get_bpms(self) -> np.ndarray:
        return np.array(
            [
                self.segment.BPM_430039.reading[0],
                self.segment.BPM_430103.reading[0],
                self.segment.BPM_430129.reading[0],
                self.segment.BPM_430203.reading[0],
                self.segment.BPM_430308.reading[0],
                self.segment.BPM_412343.reading[0],
                self.segment.BPM_412345.reading[0],
                self.segment.BPM_412347.reading[0],
                self.segment.BPM_412349.reading[0],
                self.segment.BPM_412351.reading[0],
            ]
        )

    def _set_quadrupoles(self, values: np.ndarray) -> None:
        for quad, setting in zip(self._quads, values):
            quad.k1 = torch.as_tensor(setting.astype(np.float32))

    def reset(self, options=None) -> None:
        preprocessed_options = self._preprocess_reset_options(options)  # noqa: F841

        # Set quads to random values (if random scale is > 0)
        self._random_quad_disturbance_factors = np.random.uniform(
            1.0 - self.quad_random_scale / 2.0,
            1.0 + self.quad_random_scale / 2.0,
            size=11,
        )

    def _preprocess_reset_options(self, options: dict) -> dict:
        """
        Check that only valid options are passed and make it a dict if None was passed.
        """
        if options is None:
            return {}

        valid_options = []
        for option in options:
            assert option in valid_options

        return options

    def update(self) -> None:
        quad_shift = 1 + self.quad_drift_amplitude * np.sin(
            self.quad_drift_frequency * self._persistent_step_count
        )
        drifted_quad_settings = self._original_quad_settings * quad_shift
        drifted_and_disturbed_quad_settings = (
            drifted_quad_settings * self._random_quad_disturbance_factors
        )
        self._set_quadrupoles(drifted_and_disturbed_quad_settings)

        _ = self.segment.track(self.incoming)

        self._persistent_step_count += 1  # Advance "time"

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
        info = {
            "incoming_beam": self.get_incoming_parameters(),
            "quadrupole_settings": np.array([quad.k1 for quad in self._quads]),
        }

        return info
