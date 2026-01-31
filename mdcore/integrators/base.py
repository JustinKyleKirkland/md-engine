"""Base interface for integrators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..system import MDState


class Integrator(ABC):
    """
    Abstract base class for time integration algorithms.

    Integrators advance the system state forward in time given the
    current forces on all atoms.
    """

    @abstractmethod
    def step(self, state: MDState, forces: NDArray[np.floating]) -> MDState:
        """
        Advance the system by one time step.

        Args:
            state: Current MD state.
            forces: Current forces on all atoms, shape (N, 3).

        Returns:
            New MDState after integration step.
        """
        ...

    @property
    @abstractmethod
    def timestep(self) -> float:
        """Return the integration timestep."""
        ...


class ThermostatModifier(ABC):
    """
    Abstract base class for thermostat modifiers.

    Thermostats are composable modifiers that can be applied
    to velocities during or after integration to control temperature.
    """

    @abstractmethod
    def apply(self, state: MDState) -> MDState:
        """
        Apply thermostat modification to state.

        Args:
            state: Current MD state.

        Returns:
            Modified MDState with thermostatted velocities.
        """
        ...

    @property
    @abstractmethod
    def target_temperature(self) -> float:
        """Return target temperature."""
        ...


class BarostatModifier(ABC):
    """
    Abstract base class for barostat modifiers.

    Barostats control system pressure by scaling the box and positions.
    """

    @abstractmethod
    def apply(self, state: MDState) -> MDState:
        """
        Apply barostat modification to state.

        Args:
            state: Current MD state.

        Returns:
            Modified MDState with scaled box/positions.
        """
        ...

    @property
    @abstractmethod
    def target_pressure(self) -> float:
        """Return target pressure."""
        ...
