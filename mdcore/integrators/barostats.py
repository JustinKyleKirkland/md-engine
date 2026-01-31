"""Barostat implementations as composable modifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..system.box import Box
from .base import BarostatModifier

if TYPE_CHECKING:
    from ..system import MDState


# Boltzmann constant in MD units (kJ/mol/K)
K_BOLTZMANN = 8.314462618e-3


class BerendsenBarostat(BarostatModifier):
    """
    Berendsen weak-coupling barostat.

    Scales the box and positions toward target pressure with a
    characteristic relaxation time. Does not produce correct NPT
    ensemble but is useful for equilibration.

    dP/dt = (P_target - P) / tau_p

    Attributes:
        pressure: Target pressure.
        tau: Coupling time constant.
        compressibility: Isothermal compressibility of the system.
        dt: Integration timestep.
    """

    def __init__(
        self,
        pressure: float,
        tau: float,
        dt: float,
        compressibility: float = 4.5e-5,  # Water at 300K, 1/bar
        isotropic: bool = True,
    ) -> None:
        """
        Initialize Berendsen barostat.

        Args:
            pressure: Target pressure (in simulation units).
            tau: Coupling time constant.
            dt: Integration timestep.
            compressibility: Isothermal compressibility (1/pressure units).
            isotropic: If True, scale uniformly in all dimensions.
        """
        self._pressure = pressure
        self._tau = tau
        self._dt = dt
        self._compressibility = compressibility
        self._isotropic = isotropic

        # Current pressure needs to be set externally
        self._current_pressure: float | None = None

    @property
    def target_pressure(self) -> float:
        return self._pressure

    @target_pressure.setter
    def target_pressure(self, value: float) -> None:
        self._pressure = value

    def set_current_pressure(self, pressure: float) -> None:
        """Set current pressure (computed externally from virial)."""
        self._current_pressure = pressure

    def apply(self, state: MDState) -> MDState:
        """
        Apply Berendsen barostat scaling.

        Args:
            state: Current MD state.

        Returns:
            Modified MDState with scaled box and positions.
        """
        if self._current_pressure is None:
            # Can't apply without knowing current pressure
            return state

        # Berendsen scaling factor
        # mu = 1 - (beta * dt / (3 * tau)) * (P_target - P)
        scale_factor = 1.0 - (self._compressibility * self._dt / (3.0 * self._tau)) * (
            self._pressure - self._current_pressure
        )

        # Cube root for isotropic scaling
        if self._isotropic:
            mu = scale_factor ** (1.0 / 3.0)
            mu_vec = np.array([mu, mu, mu])
        else:
            # Anisotropic would require separate pressure components
            mu = scale_factor ** (1.0 / 3.0)
            mu_vec = np.array([mu, mu, mu])

        # Scale box vectors
        new_vectors = state.box.vectors * mu_vec[:, np.newaxis]
        new_box = Box.triclinic(new_vectors)

        # Scale positions
        new_positions = state.positions * mu_vec

        # Wrap positions
        new_positions = new_box.wrap_positions(new_positions)

        # Create new state
        new_state = state.copy()
        new_state.positions = new_positions
        # Need to recreate state with new box
        from ..system.state import MDState as MDStateClass

        new_state = MDStateClass(
            positions=new_positions,
            velocities=state.velocities.copy(),
            forces=state.forces.copy(),
            masses=state.masses.copy(),
            box=new_box,
            time=state.time,
            step=state.step,
        )

        return new_state


class MonteCarloBarostat(BarostatModifier):
    """
    Monte Carlo barostat.

    Performs random volume changes that are accepted or rejected
    based on the Metropolis criterion. Produces correct NPT ensemble.

    The acceptance probability is:
    P_accept = min(1, exp(-beta * (dU + P*dV - N*kT*ln(V_new/V_old))))

    Attributes:
        pressure: Target pressure.
        temperature: Temperature for acceptance criterion.
        frequency: Attempt frequency (every N steps).
        max_volume_change: Maximum fractional volume change per attempt.
    """

    def __init__(
        self,
        pressure: float,
        temperature: float,
        frequency: int = 25,
        max_volume_change: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """
        Initialize Monte Carlo barostat.

        Args:
            pressure: Target pressure (in simulation units).
            temperature: System temperature in K.
            frequency: Attempt volume change every N steps.
            max_volume_change: Maximum fractional volume change.
            seed: Random seed for reproducibility.
        """
        self._pressure = pressure
        self._temperature = temperature
        self._frequency = frequency
        self._max_dv = max_volume_change
        self._rng = np.random.default_rng(seed)

        # Counters for acceptance statistics
        self._n_attempts = 0
        self._n_accepted = 0

        # Energy callback (must be set externally)
        self._energy_callback = None

        # Step counter
        self._step_counter = 0

    @property
    def target_pressure(self) -> float:
        return self._pressure

    @target_pressure.setter
    def target_pressure(self, value: float) -> None:
        self._pressure = value

    @property
    def acceptance_rate(self) -> float:
        """Return acceptance rate."""
        if self._n_attempts == 0:
            return 0.0
        return self._n_accepted / self._n_attempts

    def set_energy_callback(self, callback) -> None:
        """
        Set callback for computing potential energy.

        Args:
            callback: Function that takes MDState and returns float energy.
        """
        self._energy_callback = callback

    def apply(self, state: MDState) -> MDState:
        """
        Attempt Monte Carlo volume move.

        Only attempts a move every 'frequency' calls.

        Args:
            state: Current MD state.

        Returns:
            Modified MDState (if accepted) or original state.
        """
        self._step_counter += 1

        if self._step_counter % self._frequency != 0:
            return state

        if self._energy_callback is None:
            # Can't do MC without energy computation
            return state

        self._n_attempts += 1

        n_atoms = state.n_atoms
        old_volume = state.box.volume
        old_energy = self._energy_callback(state)

        # Random volume change
        dv = (2.0 * self._rng.random() - 1.0) * self._max_dv
        scale = (1.0 + dv) ** (1.0 / 3.0)

        new_volume = old_volume * (1.0 + dv)

        # Scale box and positions
        new_vectors = state.box.vectors * scale
        new_box = Box.triclinic(new_vectors)
        new_positions = state.positions * scale
        new_positions = new_box.wrap_positions(new_positions)

        # Create trial state
        from ..system.state import MDState as MDStateClass

        trial_state = MDStateClass(
            positions=new_positions,
            velocities=state.velocities.copy(),
            forces=state.forces.copy(),
            masses=state.masses.copy(),
            box=new_box,
            time=state.time,
            step=state.step,
        )

        # Compute new energy
        new_energy = self._energy_callback(trial_state)

        # Metropolis criterion for NPT
        # dW = dU + P*dV - N*kT*ln(V_new/V_old)
        beta = 1.0 / (K_BOLTZMANN * self._temperature)
        dW = (
            (new_energy - old_energy)
            + self._pressure * (new_volume - old_volume)
            - (
                n_atoms
                * K_BOLTZMANN
                * self._temperature
                * np.log(new_volume / old_volume)
            )
        )

        # Accept or reject
        if dW < 0 or self._rng.random() < np.exp(-beta * dW):
            self._n_accepted += 1
            return trial_state
        else:
            return state

    def reset_statistics(self) -> None:
        """Reset acceptance statistics."""
        self._n_attempts = 0
        self._n_accepted = 0
