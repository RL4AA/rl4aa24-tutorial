from abc import ABC, abstractmethod

import numpy as np


class ESteeringBaseBackend(ABC):
    """Abstract class for a backend imlementation."""

    def setup(self) -> None:  # noqa: B027
        """
        Prepare the accelerator for use with the environment. Should mostly be used for
        setting up simulations.

        Override with backend-specific imlementation. Optional.
        """
        pass

    @abstractmethod
    def get_magnets(self) -> np.ndarray:
        """
        Return the magnet values as a NumPy array in order as the magnets appear in the
        accelerator.

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    @abstractmethod
    def set_magnets(self, values: np.ndarray) -> None:
        """
        Set the magnets to the given values.

        The argument `magnets` will be passed as a NumPy array in the order the magnets
        appear in the accelerator.

        When applicable, this method should block until the magnet values are acutally
        set!

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    @abstractmethod
    def get_bpms(self) -> np.ndarray:
        """
        Get the readings from the BPMs as a NumPy array in order as the BPMs appear in
        the beamline.

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def reset(self) -> None:  # noqa: B027
        """
        Code that should set the accelerator up for a new episode. Run when the `reset`
        is called.

        Mostly meant for simulations to switch to a new incoming beam / misalignments or
        simular things.

        Override with backend-specific imlementation. Optional.
        """
        pass

    def update(self) -> None:  # noqa: B027
        """
        Update accelerator metrics for later use. Use this to run the simulation or
        cache sensor readings.

        Override with backend-specific imlementation. Optional.
        """
        pass

    def get_incoming_parameters(self) -> np.ndarray:
        """
        Get all physical beam parameters of the incoming beam as NumPy array in order
        energy, mu_x, mu_xp, mu_y, mu_yp, sigma_x, sigma_xp, sigma_y, sigma_yp, sigma_s,
        sigma_p.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_info(self) -> dict:
        """
        Return a dictionary of aditional info from the accelerator backend, e.g.
        incoming beam and misalignments in simulation.

        Override with backend-specific imlementation. Optional.
        """
        return {}
