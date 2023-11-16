from pathlib import Path
from typing import Literal, Optional, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
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
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,)
        )  # 1e-2
        self.action_space = spaces.Box(low=-3e-4, high=3e-4, shape=(10,))

        # Setup particle simulation or control system backend
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
        self.backend.set_magnets(np.zeros(10))

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

        valid_options = ["backend_options"]
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
            # Other info could go here
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

    :param incoming_mode: Setting for incoming beam parameters on reset. Can be
        `"random"` to generate random parameters or an array of 11 values to set them to
        a constant value.
    """

    def __init__(
        self,
        incoming_mode: Union[Literal["random"], np.ndarray] = "random",
        quad_drift_frequency: float = np.pi * 0.001,
        quad_drift_amplitude: float = 0.0,
    ) -> None:
        # Dynamic import for module only required by this backend
        global cheetah
        import cheetah

        # TODO: Constant incoming

        assert isinstance(incoming_mode, (str, np.ndarray))

        self.incoming_mode = incoming_mode
        self.quad_drift_frequency = quad_drift_frequency
        self.quad_drift_amplitude = quad_drift_amplitude

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
                # k1=torch.as_tensor(row.K1L),
                k1=torch.as_tensor(row.K1L / row.L),  # TODO: Correct?
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
            energy=torch.tensor(incoming_parameters[0]),
            mu_x=torch.tensor(incoming_parameters[1]),
            mu_xp=torch.tensor(incoming_parameters[2]),
            mu_y=torch.tensor(incoming_parameters[3]),
            mu_yp=torch.tensor(incoming_parameters[4]),
            sigma_x=torch.tensor(incoming_parameters[5]),
            sigma_xp=torch.tensor(incoming_parameters[6]),
            sigma_y=torch.tensor(incoming_parameters[7]),
            sigma_yp=torch.tensor(incoming_parameters[8]),
            sigma_s=torch.tensor(incoming_parameters[9]),
            sigma_p=torch.tensor(incoming_parameters[10]),
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
        quad_shift = self.quad_drift_amplitude * np.sin(
            self.quad_drift_frequency * self._persistent_step_count
        )  # TODO: Why did Simon use +1 here?
        drifted_quad_settings = self._original_quad_settings + quad_shift
        for quad, setting in zip(self._quads, drifted_quad_settings):
            quad.k1 = torch.as_tensor(setting)

        self.segment.track(self.incoming)

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
            "quadrupole_settings": [quad.k1 for quad in self._quads],
        }

        return info
