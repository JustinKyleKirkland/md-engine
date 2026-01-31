"""Thermostat implementations as composable modifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .base import ThermostatModifier

if TYPE_CHECKING:
    from ..system import MDState


# Boltzmann constant in MD units (kJ/mol/K)
K_BOLTZMANN = 8.314462618e-3


class VelocityRescaleThermostat(ThermostatModifier):
    """
    Simple velocity rescaling thermostat.

    Rescales all velocities to achieve exactly the target temperature.
    This gives the correct average kinetic energy but incorrect
    velocity distribution (microcanonical rather than canonical).

    Useful for equilibration but not for production NVT simulations.
    """

    def __init__(self, temperature: float) -> None:
        """
        Initialize velocity rescaling thermostat.

        Args:
            temperature: Target temperature in K.
        """
        self._temperature = temperature

    @property
    def target_temperature(self) -> float:
        return self._temperature

    @target_temperature.setter
    def target_temperature(self, value: float) -> None:
        self._temperature = value

    def apply(self, state: MDState) -> MDState:
        """
        Apply velocity rescaling to achieve target temperature.

        Args:
            state: Current MD state.

        Returns:
            Modified MDState with rescaled velocities.
        """
        current_temp = state.temperature

        if current_temp < 1e-10:
            # Can't rescale from zero temperature
            return state

        # Compute scaling factor
        scale = np.sqrt(self._temperature / current_temp)

        new_state = state.copy()
        new_state.velocities = state.velocities * scale

        return new_state


class BerendsenThermostat(ThermostatModifier):
    """
    Berendsen weak-coupling thermostat.

    Scales velocities toward target temperature with a characteristic
    relaxation time. Does not produce correct canonical ensemble but
    is useful for equilibration due to gentle temperature control.

    dT/dt = (T_target - T) / tau

    Attributes:
        temperature: Target temperature in K.
        tau: Coupling time constant.
        dt: Integration timestep.
    """

    def __init__(
        self,
        temperature: float,
        tau: float,
        dt: float,
    ) -> None:
        """
        Initialize Berendsen thermostat.

        Args:
            temperature: Target temperature in K.
            tau: Coupling time constant (same units as dt).
            dt: Integration timestep.
        """
        self._temperature = temperature
        self._tau = tau
        self._dt = dt

    @property
    def target_temperature(self) -> float:
        return self._temperature

    @target_temperature.setter
    def target_temperature(self, value: float) -> None:
        self._temperature = value

    def apply(self, state: MDState) -> MDState:
        """
        Apply Berendsen thermostat coupling.

        Args:
            state: Current MD state.

        Returns:
            Modified MDState with temperature-adjusted velocities.
        """
        current_temp = state.temperature

        if current_temp < 1e-10:
            return state

        # Berendsen scaling factor
        # lambda = sqrt(1 + dt/tau * (T_target/T - 1))
        scale_sq = 1.0 + (self._dt / self._tau) * (
            self._temperature / current_temp - 1.0
        )

        if scale_sq < 0:
            scale_sq = 0.0

        scale = np.sqrt(scale_sq)

        new_state = state.copy()
        new_state.velocities = state.velocities * scale

        return new_state


class AndersenThermostat(ThermostatModifier):
    """
    Andersen stochastic collision thermostat.

    Randomly reassigns velocities of selected atoms from the
    Maxwell-Boltzmann distribution. Produces correct canonical
    ensemble but disrupts dynamics (not suitable for computing
    transport properties).

    Attributes:
        temperature: Target temperature in K.
        collision_frequency: Average collision rate (1/time).
        dt: Integration timestep.
    """

    def __init__(
        self,
        temperature: float,
        collision_frequency: float,
        dt: float,
        seed: int | None = None,
    ) -> None:
        """
        Initialize Andersen thermostat.

        Args:
            temperature: Target temperature in K.
            collision_frequency: Average collision rate (1/time units).
            dt: Integration timestep.
            seed: Random seed for reproducibility.
        """
        self._temperature = temperature
        self._collision_freq = collision_frequency
        self._dt = dt
        self._rng = np.random.default_rng(seed)

        # Probability of collision per timestep
        self._collision_prob = collision_frequency * dt

    @property
    def target_temperature(self) -> float:
        return self._temperature

    @target_temperature.setter
    def target_temperature(self, value: float) -> None:
        self._temperature = value

    def apply(self, state: MDState) -> MDState:
        """
        Apply Andersen thermostat collisions.

        Args:
            state: Current MD state.

        Returns:
            Modified MDState with some velocities resampled.
        """
        n_atoms = state.n_atoms
        masses = state.masses

        # Determine which atoms collide
        collide = self._rng.random(n_atoms) < self._collision_prob

        if not np.any(collide):
            return state

        # Sample new velocities from Maxwell-Boltzmann
        # sigma_v = sqrt(kT/m) for each component
        sigma = np.sqrt(K_BOLTZMANN * self._temperature / masses[collide])
        new_velocities = (
            self._rng.standard_normal((np.sum(collide), 3)) * sigma[:, np.newaxis]
        )

        # Apply new velocities
        new_state = state.copy()
        new_state.velocities[collide] = new_velocities

        return new_state


class NoseHooverThermostat(ThermostatModifier):
    """
    Nosé-Hoover chain thermostat.

    Extended Lagrangian thermostat that produces correct canonical
    ensemble and preserves dynamics. Uses a chain of thermostats
    for better ergodicity.

    This is a simplified single-thermostat version. For production
    use, a chain of 3-5 thermostats is recommended.

    Attributes:
        temperature: Target temperature in K.
        tau: Characteristic time (related to thermostat mass).
        dt: Integration timestep.
    """

    def __init__(
        self,
        temperature: float,
        tau: float,
        dt: float,
        n_chain: int = 1,
    ) -> None:
        """
        Initialize Nosé-Hoover thermostat.

        Args:
            temperature: Target temperature in K.
            tau: Characteristic relaxation time.
            dt: Integration timestep.
            n_chain: Number of thermostats in chain (default 1).
        """
        self._temperature = temperature
        self._tau = tau
        self._dt = dt
        self._n_chain = n_chain

        # Thermostat "velocities" (one per chain element)
        self._xi = np.zeros(n_chain)

        # Thermostat masses Q = kT * tau^2
        self._Q: NDArray[np.floating] | None = None
        self._n_dof: int | None = None

    @property
    def target_temperature(self) -> float:
        return self._temperature

    @target_temperature.setter
    def target_temperature(self, value: float) -> None:
        self._temperature = value
        self._Q = None  # Force recomputation

    def _initialize_masses(self, n_atoms: int) -> None:
        """Initialize thermostat masses based on system size."""
        self._n_dof = 3 * n_atoms - 3  # Assuming no constraints

        # Thermostat mass: Q = N_dof * kT * tau^2
        Q0 = self._n_dof * K_BOLTZMANN * self._temperature * self._tau**2

        # Chain masses (first is largest)
        self._Q = np.array(
            [Q0]
            + [K_BOLTZMANN * self._temperature * self._tau**2] * (self._n_chain - 1)
        )

    def apply(self, state: MDState) -> MDState:
        """
        Apply Nosé-Hoover thermostat.

        Uses a simple velocity Verlet-like integration of the
        extended system.

        Args:
            state: Current MD state.

        Returns:
            Modified MDState with thermostatted velocities.
        """
        if self._Q is None or self._n_dof is None:
            self._initialize_masses(state.n_atoms)

        dt = self._dt
        half_dt = 0.5 * dt

        # Current kinetic energy
        KE = state.kinetic_energy

        # Target kinetic energy
        KE_target = 0.5 * self._n_dof * K_BOLTZMANN * self._temperature

        # Update thermostat (simplified single-thermostat version)
        # xi_dot = (2*KE - 2*KE_target) / Q
        G = (2.0 * KE - 2.0 * KE_target) / self._Q[0]

        # Half step update of xi
        self._xi[0] = self._xi[0] + half_dt * G

        # Scale velocities
        # v_new = v * exp(-xi * dt)
        scale = np.exp(-self._xi[0] * dt)

        new_state = state.copy()
        new_state.velocities = state.velocities * scale

        # Recalculate KE and update xi for second half step
        KE_new = new_state.kinetic_energy
        G_new = (2.0 * KE_new - 2.0 * KE_target) / self._Q[0]
        self._xi[0] = self._xi[0] + half_dt * G_new

        return new_state

    def reset(self) -> None:
        """Reset thermostat state."""
        self._xi = np.zeros(self._n_chain)
        self._Q = None
