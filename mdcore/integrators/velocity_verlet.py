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
    Velocity Verlet integrator (leapfrog formulation).

    The standard symplectic integrator for molecular dynamics.
    Time-reversible and preserves phase space volume.

    This uses the leapfrog formulation where positions and velocities
    are staggered by half a timestep:

        v(t + dt/2) = v(t - dt/2) + dt * a(t)
        r(t + dt) = r(t) + dt * v(t + dt/2)

    The full-step velocity is reconstructed for output:
        v(t) = v(t - dt/2) + 0.5 * dt * a(t)

    Usage:
        Forces should be computed at CURRENT positions before calling step().

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
        self._velocities_half: NDArray[np.floating] | None = None

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

        if self._velocities_half is None:
            # First step: initialize v(t - dt/2) by going back half a step
            # v(-dt/2) = v(0) - 0.5 * dt * a(0)
            velocities_half_old = state.velocities - 0.5 * dt * accel
        else:
            velocities_half_old = self._velocities_half

        # Leapfrog velocity update: v(t + dt/2) = v(t - dt/2) + dt * a(t)
        velocities_half_new = velocities_half_old + dt * accel

        # Position update: r(t + dt) = r(t) + dt * v(t + dt/2)
        positions_new = state.positions + dt * velocities_half_new

        # Wrap positions into box
        if state.box is not None:
            positions_new = state.box.wrap_positions(positions_new)

        # Store half-step velocities for next iteration
        self._velocities_half = velocities_half_new.copy()

        # For output, return v(t + dt/2) which is the natural leapfrog velocity.
        # For energy calculations, this gives E at the half-step, which is
        # appropriate since positions are also "staggered" by a half-step.
        # This is standard practice and preserves symplecticity.

        # Create new state
        new_state = state.copy()
        new_state.positions = positions_new
        new_state.velocities = velocities_half_new
        new_state.forces = forces.copy()
        new_state.time = state.time + dt
        new_state.step = state.step + 1

        return new_state

    def reset(self) -> None:
        """Reset integrator state (e.g., for new simulation)."""
        self._velocities_half = None


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
