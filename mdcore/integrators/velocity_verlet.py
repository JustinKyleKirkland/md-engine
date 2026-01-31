"""Velocity Verlet integrator implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .base import Integrator

if TYPE_CHECKING:
    from ..system import MDState


class VelocityVerletIntegrator(Integrator):
    """
    Velocity Verlet integrator (kick-drift-kick formulation).

    The standard symplectic integrator for molecular dynamics with
    excellent energy conservation and perfect time-reversibility.

    Algorithm:
        v(t + dt/2) = v(t) + 0.5 * dt * a(t)        # First kick
        r(t + dt) = r(t) + dt * v(t + dt/2)         # Drift
        v(t + dt) = v(t + dt/2) + 0.5 * dt * a(t+dt) # Second kick

    Properties:
    - Symplectic: preserves phase space volume
    - Time-reversible: exactly reversible when using full_step()
    - Second-order accurate in positions AND velocities
    - Excellent energy conservation (drift ~ dt⁴)

    Usage:
        For best results (perfect reversibility, energy conservation),
        use full_step() which evaluates forces twice per step.

        For compatibility with simple loops, step() can be used but
        provides slightly degraded reversibility.

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
        self._prev_accel: NDArray[np.floating] | None = None

    @property
    def timestep(self) -> float:
        """Return the integration timestep."""
        return self._dt

    def full_step(
        self,
        state: MDState,
        forces: NDArray[np.floating],
        force_fn: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    ) -> MDState:
        """
        Perform a complete Velocity Verlet step with two force evaluations.

        This is the proper VV implementation with:
        - Perfect time-reversibility
        - Second-order accuracy in both positions and velocities
        - Excellent energy conservation

        Args:
            state: Current MD state with synchronized velocities v(t).
            forces: Forces at CURRENT positions r(t), shape (N, 3).
            force_fn: Function that computes forces given positions.
                      Signature: force_fn(positions) -> forces

        Returns:
            New MDState with r(t+dt) and synchronized v(t+dt).
        """
        dt = self._dt
        masses = state.masses[:, np.newaxis]
        accel = forces / masses

        # First kick: v(t + dt/2) = v(t) + 0.5 * dt * a(t)
        velocities_half = state.velocities + 0.5 * dt * accel

        # Drift: r(t + dt) = r(t) + dt * v(t + dt/2)
        positions_new = state.positions + dt * velocities_half

        # Wrap positions
        if state.box is not None:
            positions_new = state.box.wrap_positions(positions_new)

        # Compute NEW forces at new positions
        forces_new = force_fn(positions_new)
        accel_new = forces_new / masses

        # Second kick: v(t + dt) = v(t + dt/2) + 0.5 * dt * a(t + dt)
        velocities_new = velocities_half + 0.5 * dt * accel_new

        # Create new state with synchronized velocities
        new_state = state.copy()
        new_state.positions = positions_new
        new_state.velocities = velocities_new
        new_state.forces = forces_new.copy()
        new_state.time = state.time + dt
        new_state.step = state.step + 1

        return new_state

    def step(self, state: MDState, forces: NDArray[np.floating]) -> MDState:
        """
        Perform one Velocity Verlet step with single force evaluation.

        This method stores the previous acceleration and completes the
        velocity update at the start of the next step. It provides:
        - Second-order accuracy after the first step
        - Good energy conservation
        - Approximate time-reversibility (error ~ dt²)

        For perfect time-reversibility, use full_step() instead.

        Args:
            state: Current MD state.
            forces: Forces at CURRENT positions, shape (N, 3).

        Returns:
            New MDState after integration step.
        """
        dt = self._dt
        masses = state.masses[:, np.newaxis]
        accel = forces / masses

        if self._prev_accel is not None:
            # Complete velocity from previous step: v(t) = v_half + 0.5*dt*a(t)
            velocities = state.velocities + 0.5 * dt * accel
        else:
            # First step: velocities are already synchronized
            velocities = state.velocities

        # First kick: v(t + dt/2) = v(t) + 0.5 * dt * a(t)
        velocities_half = velocities + 0.5 * dt * accel

        # Drift: r(t + dt) = r(t) + dt * v(t + dt/2)
        positions_new = state.positions + dt * velocities_half

        # Wrap positions
        if state.box is not None:
            positions_new = state.box.wrap_positions(positions_new)

        # Store acceleration for next step
        self._prev_accel = accel.copy()

        # Create new state with half-step velocities
        # (will be completed at start of next step)
        new_state = state.copy()
        new_state.positions = positions_new
        new_state.velocities = velocities_half
        new_state.forces = forces.copy()
        new_state.time = state.time + dt
        new_state.step = state.step + 1

        return new_state

    def reset(self) -> None:
        """Reset integrator state for a new simulation."""
        self._prev_accel = None


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
