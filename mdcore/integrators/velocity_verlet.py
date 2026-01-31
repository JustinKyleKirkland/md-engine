"""Velocity Verlet integrator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .base import Integrator

if TYPE_CHECKING:
    from ..system import MDState


class VelocityVerletIntegrator(Integrator):
    """
    Velocity Verlet integrator.

    The standard symplectic integrator for molecular dynamics.
    Time-reversible and preserves phase space volume.

    Algorithm (split into two half-steps for force recomputation):

    Full step (when forces are recomputed between calls):
        v(t + dt/2) = v(t) + 0.5 * dt * a(t)
        r(t + dt) = r(t) + dt * v(t + dt/2)
        [compute new forces -> a(t + dt)]
        v(t + dt) = v(t + dt/2) + 0.5 * dt * a(t + dt)

    This implementation assumes forces passed in are the NEW forces
    at the updated positions, following the standard MD engine loop.

    Attributes:
        dt: Integration timestep.
        _half_stepped: Whether we're in the middle of a step.
    """

    def __init__(self, dt: float) -> None:
        """
        Initialize Velocity Verlet integrator.

        Args:
            dt: Integration timestep.
        """
        self._dt = dt
        self._prev_forces: NDArray[np.floating] | None = None

    @property
    def timestep(self) -> float:
        """Return the integration timestep."""
        return self._dt

    def step(self, state: MDState, forces: NDArray[np.floating]) -> MDState:
        """
        Perform one Velocity Verlet integration step.

        This performs the full VV algorithm:
        1. Update velocities half step with current forces
        2. Update positions full step
        3. Update velocities half step with new forces (passed in)

        Args:
            state: Current MD state.
            forces: NEW forces at updated positions, shape (N, 3).

        Returns:
            New MDState after integration step.
        """
        dt = self._dt
        masses = state.masses[:, np.newaxis]  # Shape (N, 1) for broadcasting

        # Use previous forces if available, otherwise use current
        # (first step uses same forces for both half-steps)
        prev_forces = forces if self._prev_forces is None else self._prev_forces

        # Compute accelerations
        accel_old = prev_forces / masses
        accel_new = forces / masses

        # Half step velocity update with old forces
        velocities_half = state.velocities + 0.5 * dt * accel_old

        # Full step position update
        positions_new = state.positions + dt * velocities_half

        # Wrap positions into box
        positions_new = state.box.wrap_positions(positions_new)

        # Half step velocity update with new forces
        velocities_new = velocities_half + 0.5 * dt * accel_new

        # Store forces for next step
        self._prev_forces = forces.copy()

        # Create new state
        new_state = state.copy()
        new_state.positions = positions_new
        new_state.velocities = velocities_new
        new_state.forces = forces.copy()
        new_state.time = state.time + dt
        new_state.step = state.step + 1

        return new_state

    def reset(self) -> None:
        """Reset integrator state (e.g., for new simulation)."""
        self._prev_forces = None


class LeapfrogIntegrator(Integrator):
    """
    Leapfrog integrator (equivalent to Velocity Verlet).

    Positions and velocities are offset by half a timestep.
    This is mathematically equivalent to Velocity Verlet but
    with a different interpretation.

    Algorithm:
        v(t + dt/2) = v(t - dt/2) + dt * a(t)
        r(t + dt) = r(t) + dt * v(t + dt/2)
    """

    def __init__(self, dt: float) -> None:
        """
        Initialize Leapfrog integrator.

        Args:
            dt: Integration timestep.
        """
        self._dt = dt

    @property
    def timestep(self) -> float:
        """Return the integration timestep."""
        return self._dt

    def step(self, state: MDState, forces: NDArray[np.floating]) -> MDState:
        """
        Perform one Leapfrog integration step.

        Args:
            state: Current MD state (velocities at t - dt/2).
            forces: Forces at current positions, shape (N, 3).

        Returns:
            New MDState after integration step.
        """
        dt = self._dt
        masses = state.masses[:, np.newaxis]

        # Compute acceleration
        accel = forces / masses

        # Update velocities (full step)
        velocities_new = state.velocities + dt * accel

        # Update positions (using new velocities)
        positions_new = state.positions + dt * velocities_new

        # Wrap positions
        positions_new = state.box.wrap_positions(positions_new)

        # Create new state
        new_state = state.copy()
        new_state.positions = positions_new
        new_state.velocities = velocities_new
        new_state.forces = forces.copy()
        new_state.time = state.time + dt
        new_state.step = state.step + 1

        return new_state
