"""BAOAB Langevin integrator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .base import Integrator

if TYPE_CHECKING:
    from ..system import MDState


# Boltzmann constant in MD units (kJ/mol/K)
K_BOLTZMANN = 8.314462618e-3


class BAOABIntegrator(Integrator):
    """
    BAOAB Langevin integrator.

    A geodesic Langevin integrator that provides excellent sampling
    of configurational properties. The name comes from the splitting:

    B - A - O - A - B

    where:
    - B: Half kick (velocity update from forces)
    - A: Half drift (position update from velocities)
    - O: Ornstein-Uhlenbeck process (exact solution for friction + noise)

    This integrator:
    - Is second-order accurate for positions
    - Exactly samples the configurational Boltzmann distribution
    - Has excellent stability properties
    - Is time-reversible (up to stochastic terms)

    Reference:
    Leimkuhler & Matthews, "Molecular Dynamics" (2015)

    Attributes:
        dt: Integration timestep.
        temperature: Target temperature in K.
        friction: Friction coefficient gamma (1/time units).
    """

    def __init__(
        self,
        dt: float,
        temperature: float,
        friction: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """
        Initialize BAOAB integrator.

        Args:
            dt: Integration timestep.
            temperature: Target temperature in K.
            friction: Friction coefficient gamma (1/time units, e.g., 1/ps).
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

        # Exact OU solution coefficients
        # v(t+dt) = c1 * v(t) + c2 * sqrt(kT/m) * noise
        self._c1 = np.exp(-gamma * dt)
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

    @friction.setter
    def friction(self, value: float) -> None:
        """Set friction coefficient and update coefficients."""
        self._friction = value
        self._update_coefficients()

    def step(self, state: MDState, forces: NDArray[np.floating]) -> MDState:
        """
        Perform one BAOAB integration step.

        The splitting is:
        1. B: v += (dt/2) * F/m
        2. A: x += (dt/2) * v
        3. O: v = c1*v + c2*sqrt(kT/m)*R  (Ornstein-Uhlenbeck)
        4. A: x += (dt/2) * v
        5. B: v += (dt/2) * F/m  (with new forces)

        Note: This implementation uses the OLD forces for both B steps.
        For maximum accuracy, the engine should recompute forces after
        the position updates and call a separate velocity update.

        Args:
            state: Current MD state.
            forces: Forces at current positions, shape (N, 3).

        Returns:
            New MDState after integration step.
        """
        dt = self._dt
        half_dt = 0.5 * dt
        masses = state.masses[:, np.newaxis]
        n_atoms = state.n_atoms

        # Compute acceleration
        accel = forces / masses

        # B: Half kick
        velocities = state.velocities + half_dt * accel

        # A: Half drift
        positions = state.positions + half_dt * velocities

        # O: Ornstein-Uhlenbeck (exact thermostat)
        noise = self._rng.standard_normal((n_atoms, 3))
        c2 = self._c2_factor / np.sqrt(masses)
        velocities = self._c1 * velocities + c2 * noise

        # A: Half drift
        positions = positions + half_dt * velocities

        # Wrap positions into box
        positions = state.box.wrap_positions(positions)

        # B: Half kick (using same forces - engine should update)
        velocities = velocities + half_dt * accel

        # Create new state
        new_state = state.copy()
        new_state.positions = positions
        new_state.velocities = velocities
        new_state.forces = forces.copy()
        new_state.time = state.time + dt
        new_state.step = state.step + 1

        return new_state

    def step_with_new_forces(
        self,
        state: MDState,
        old_forces: NDArray[np.floating],
        new_forces: NDArray[np.floating],
    ) -> MDState:
        """
        Perform BAOAB step with force recomputation.

        This is the most accurate version where forces are recomputed
        after the position update and used for the final B step.

        Args:
            state: Current MD state.
            old_forces: Forces at initial positions.
            new_forces: Forces at updated positions.

        Returns:
            New MDState after integration step.
        """
        dt = self._dt
        half_dt = 0.5 * dt
        masses = state.masses[:, np.newaxis]
        n_atoms = state.n_atoms

        # B: Half kick with old forces
        accel_old = old_forces / masses
        velocities = state.velocities + half_dt * accel_old

        # A: Half drift
        positions = state.positions + half_dt * velocities

        # O: Ornstein-Uhlenbeck
        noise = self._rng.standard_normal((n_atoms, 3))
        c2 = self._c2_factor / np.sqrt(masses)
        velocities = self._c1 * velocities + c2 * noise

        # A: Half drift
        positions = positions + half_dt * velocities

        # Wrap positions
        positions = state.box.wrap_positions(positions)

        # B: Half kick with new forces
        accel_new = new_forces / masses
        velocities = velocities + half_dt * accel_new

        # Create new state
        new_state = state.copy()
        new_state.positions = positions
        new_state.velocities = velocities
        new_state.forces = new_forces.copy()
        new_state.time = state.time + dt
        new_state.step = state.step + 1

        return new_state
