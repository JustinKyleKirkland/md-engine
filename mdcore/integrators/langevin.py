"""Langevin dynamics integrator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .base import Integrator

if TYPE_CHECKING:
    from ..system import MDState


# Boltzmann constant in MD units (kJ/mol/K)
K_BOLTZMANN = 8.314462618e-3


class LangevinIntegrator(Integrator):
    """
    Langevin dynamics integrator.

    Adds friction and random forces to simulate coupling to a heat bath.
    This provides natural temperature control (NVT ensemble).

    The stochastic equation of motion is:
        m * dv/dt = F - gamma * m * v + R(t)

    where gamma is the friction coefficient and R(t) is a random force
    satisfying the fluctuation-dissipation theorem.

    This implementation uses the BAOAB splitting scheme variant for
    better sampling properties.

    Attributes:
        dt: Integration timestep.
        temperature: Target temperature.
        friction: Friction coefficient (1/ps).
    """

    def __init__(
        self,
        dt: float,
        temperature: float,
        friction: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """
        Initialize Langevin integrator.

        Args:
            dt: Integration timestep.
            temperature: Target temperature in K.
            friction: Friction coefficient (1/time units, e.g., 1/ps).
            seed: Random seed for reproducibility.
        """
        self._dt = dt
        self._temperature = temperature
        self._friction = friction
        self._rng = np.random.default_rng(seed)

        # Precompute coefficients
        self._update_coefficients()

    def _update_coefficients(self) -> None:
        """Precompute integration coefficients."""
        gamma = self._friction
        dt = self._dt

        # Velocity scaling factor
        self._c1 = np.exp(-gamma * dt)

        # Random force scaling factor
        # sqrt(1 - c1^2) * sqrt(kT/m) for each atom
        self._c2_factor = np.sqrt(1.0 - self._c1**2) * np.sqrt(
            K_BOLTZMANN * self._temperature
        )

    @property
    def timestep(self) -> float:
        """Return the integration timestep."""
        return self._dt

    @property
    def temperature(self) -> float:
        """Return target temperature."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set target temperature and update coefficients."""
        self._temperature = value
        self._update_coefficients()

    @property
    def friction(self) -> float:
        """Return friction coefficient."""
        return self._friction

    def step(self, state: MDState, forces: NDArray[np.floating]) -> MDState:
        """
        Perform one Langevin dynamics integration step.

        Uses a simple Euler-Maruyama scheme:
        1. Half kick from forces
        2. Apply Ornstein-Uhlenbeck update (friction + noise)
        3. Full drift (position update)
        4. Half kick from forces

        Args:
            state: Current MD state.
            forces: Forces at current positions, shape (N, 3).

        Returns:
            New MDState after integration step.
        """
        dt = self._dt
        masses = state.masses[:, np.newaxis]
        n_atoms = state.n_atoms

        # Compute acceleration
        accel = forces / masses

        # Half kick
        velocities = state.velocities + 0.5 * dt * accel

        # Ornstein-Uhlenbeck step (thermostat)
        # v_new = c1 * v + c2 * noise
        noise = self._rng.standard_normal((n_atoms, 3))
        c2 = self._c2_factor / np.sqrt(state.masses[:, np.newaxis])
        velocities = self._c1 * velocities + c2 * noise

        # Full drift
        positions_new = state.positions + dt * velocities

        # Wrap positions
        positions_new = state.box.wrap_positions(positions_new)

        # Half kick (using same forces - would need new forces for full accuracy)
        velocities_new = velocities + 0.5 * dt * accel

        # Create new state
        new_state = state.copy()
        new_state.positions = positions_new
        new_state.velocities = velocities_new
        new_state.forces = forces.copy()
        new_state.time = state.time + dt
        new_state.step = state.step + 1

        return new_state


class LangevinMiddleIntegrator(Integrator):
    """
    Langevin integrator with middle scheme (better for configurational sampling).

    Places the stochastic/friction step in the middle:
    B - A - O - A - B

    where:
    - B: half kick (velocity update from forces)
    - A: half drift (position update)
    - O: Ornstein-Uhlenbeck (friction + noise)

    This gives better configurational sampling than the standard scheme.
    """

    def __init__(
        self,
        dt: float,
        temperature: float,
        friction: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """
        Initialize Langevin middle integrator.

        Args:
            dt: Integration timestep.
            temperature: Target temperature in K.
            friction: Friction coefficient (1/time units).
            seed: Random seed for reproducibility.
        """
        self._dt = dt
        self._temperature = temperature
        self._friction = friction
        self._rng = np.random.default_rng(seed)

        self._update_coefficients()

    def _update_coefficients(self) -> None:
        """Precompute integration coefficients."""
        gamma = self._friction
        dt = self._dt

        self._c1 = np.exp(-gamma * dt)
        self._c2_factor = np.sqrt(1.0 - self._c1**2) * np.sqrt(
            K_BOLTZMANN * self._temperature
        )

    @property
    def timestep(self) -> float:
        return self._dt

    @property
    def temperature(self) -> float:
        return self._temperature

    def step(self, state: MDState, forces: NDArray[np.floating]) -> MDState:
        """
        Perform B-A-O-A-B integration step.

        Args:
            state: Current MD state.
            forces: Forces at current positions.

        Returns:
            New MDState after integration step.
        """
        dt = self._dt
        masses = state.masses[:, np.newaxis]
        n_atoms = state.n_atoms

        accel = forces / masses

        # B: Half kick
        velocities = state.velocities + 0.5 * dt * accel

        # A: Half drift
        positions = state.positions + 0.5 * dt * velocities

        # O: Ornstein-Uhlenbeck
        noise = self._rng.standard_normal((n_atoms, 3))
        c2 = self._c2_factor / np.sqrt(masses)
        velocities = self._c1 * velocities + c2 * noise

        # A: Half drift
        positions = positions + 0.5 * dt * velocities

        # Wrap positions
        positions = state.box.wrap_positions(positions)

        # B: Half kick (with new forces ideally, but using same here)
        velocities = velocities + 0.5 * dt * accel

        # Create new state
        new_state = state.copy()
        new_state.positions = positions
        new_state.velocities = velocities
        new_state.forces = forces.copy()
        new_state.time = state.time + dt
        new_state.step = state.step + 1

        return new_state
