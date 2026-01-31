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
    Velocity Verlet integrator (Störmer-Verlet formulation).

    A symplectic, time-reversible integrator for molecular dynamics.

    Algorithm:
        r(t + dt) = r(t) + dt * v(t) + 0.5 * dt² * a(t)
        v(t + dt) = v(t) + dt * a(t)

    Properties:
    - Symplectic: preserves phase space volume
    - Time-reversible: negate v, step, negate v returns to start
    - Second-order accurate in positions
    - Deterministic: no internal state, same inputs = same outputs

    For best energy conservation, use small timesteps (dt ≈ 0.001 τ
    for LJ systems, or dt such that dt*ω_max < 0.1 where ω_max is
    the highest frequency in the system).

    Attributes:
        dt: Integration timestep.
    """

    def __init__(self, dt: float) -> None:
        """
        Initialize Velocity Verlet integrator.

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
        Perform one Velocity Verlet integration step.

        Args:
            state: Current MD state.
            forces: Forces at CURRENT positions, shape (N, 3).

        Returns:
            New MDState after integration step.
        """
        dt = self._dt
        masses = state.masses[:, np.newaxis]  # Shape (N, 1) for broadcasting
        accel = forces / masses

        # Störmer-Verlet update (position-first, symplectic, time-reversible):
        # r(t + dt) = r(t) + dt * v(t) + 0.5 * dt² * a(t)
        # v(t + dt) = v(t) + dt * a(t)
        positions_new = state.positions + dt * state.velocities + 0.5 * dt * dt * accel
        velocities_new = state.velocities + dt * accel

        # Wrap positions into box
        if state.box is not None:
            positions_new = state.box.wrap_positions(positions_new)

        # Create new state
        new_state = state.copy()
        new_state.positions = positions_new
        new_state.velocities = velocities_new
        new_state.forces = forces.copy()
        new_state.time = state.time + dt
        new_state.step = state.step + 1

        return new_state

    def reset(self) -> None:
        """Reset integrator state (no-op, integrator is stateless)."""
        pass


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
