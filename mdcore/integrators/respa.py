"""RESPA (Reference System Propagator Algorithm) integrator implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .base import Integrator

if TYPE_CHECKING:
    from ..forcefields import ForceProvider
    from ..neighborlists import NeighborList
    from ..system import MDState


class RESPAIntegrator(Integrator):
    """
    RESPA (Reference System Propagator Algorithm) multiple timestep integrator.

    RESPA allows different force components to be evaluated at different
    frequencies, greatly improving efficiency when there are fast and slow
    force components.

    Typical splitting:
    - Fast forces (bonded): small timestep dt_fast
    - Slow forces (nonbonded): large timestep dt_slow = n_inner * dt_fast

    Algorithm (for 2-level RESPA):
    For each outer step:
        v += (dt_slow/2) * F_slow/m
        For each inner step:
            v += (dt_fast/2) * F_fast/m
            x += dt_fast * v
            Recompute F_fast
            v += (dt_fast/2) * F_fast/m
        Recompute F_slow
        v += (dt_slow/2) * F_slow/m

    Attributes:
        dt_outer: Outer (slow) timestep.
        n_inner: Number of inner steps per outer step.
        fast_force_provider: Force provider for fast forces (e.g., bonded).
        slow_force_provider: Force provider for slow forces (e.g., nonbonded).
    """

    def __init__(
        self,
        dt_inner: float,
        n_inner: int,
        fast_force_provider: ForceProvider,
        slow_force_provider: ForceProvider,
    ) -> None:
        """
        Initialize RESPA integrator.

        Args:
            dt_inner: Inner (fast) timestep.
            n_inner: Number of inner steps per outer step.
            fast_force_provider: Force provider for fast (bonded) forces.
            slow_force_provider: Force provider for slow (nonbonded) forces.
        """
        self._dt_inner = dt_inner
        self._n_inner = n_inner
        self._dt_outer = dt_inner * n_inner
        self._fast_force_provider = fast_force_provider
        self._slow_force_provider = slow_force_provider
        self._neighbor_list: NeighborList | None = None

    @property
    def timestep(self) -> float:
        """Return the outer timestep."""
        return self._dt_outer

    @property
    def dt_inner(self) -> float:
        """Return the inner timestep."""
        return self._dt_inner

    @property
    def n_inner(self) -> int:
        """Return the number of inner steps."""
        return self._n_inner

    def set_neighbor_list(self, neighbor_list: NeighborList) -> None:
        """Set neighbor list for slow force computation."""
        self._neighbor_list = neighbor_list

    def step(self, state: MDState, forces: NDArray[np.floating]) -> MDState:
        """
        Perform one RESPA outer step.

        Note: The forces argument is ignored as RESPA computes forces internally.
        Use step_with_forces() if you want to provide external force computation.

        Args:
            state: Current MD state.
            forces: Ignored (forces computed internally).

        Returns:
            New MDState after integration step.
        """
        dt_fast = self._dt_inner
        dt_slow = self._dt_outer
        half_dt_fast = 0.5 * dt_fast
        half_dt_slow = 0.5 * dt_slow

        masses = state.masses[:, np.newaxis]

        # Work with a mutable copy
        positions = state.positions.copy()
        velocities = state.velocities.copy()

        # Compute slow forces (nonbonded)
        # Create temporary state for force computation
        temp_state = state.copy()
        temp_state.positions = positions

        slow_forces = self._slow_force_provider.compute(temp_state, self._neighbor_list)
        accel_slow = slow_forces / masses

        # Outer half kick with slow forces
        velocities = velocities + half_dt_slow * accel_slow

        # Inner loop
        for _ in range(self._n_inner):
            # Compute fast forces (bonded)
            temp_state.positions = positions
            fast_forces = self._fast_force_provider.compute(temp_state, None)
            accel_fast = fast_forces / masses

            # Inner half kick with fast forces
            velocities = velocities + half_dt_fast * accel_fast

            # Inner full drift
            positions = positions + dt_fast * velocities

            # Wrap positions
            positions = state.box.wrap_positions(positions)

            # Recompute fast forces at new positions
            temp_state.positions = positions
            fast_forces = self._fast_force_provider.compute(temp_state, None)
            accel_fast = fast_forces / masses

            # Inner half kick with new fast forces
            velocities = velocities + half_dt_fast * accel_fast

        # Recompute slow forces at final positions
        temp_state.positions = positions
        if self._neighbor_list is not None:
            self._neighbor_list.update_if_needed(positions)
        slow_forces = self._slow_force_provider.compute(temp_state, self._neighbor_list)
        accel_slow = slow_forces / masses

        # Outer half kick with new slow forces
        velocities = velocities + half_dt_slow * accel_slow

        # Compute total forces for output
        total_forces = fast_forces + slow_forces

        # Create new state
        new_state = state.copy()
        new_state.positions = positions
        new_state.velocities = velocities
        new_state.forces = total_forces
        new_state.time = state.time + dt_slow
        new_state.step = state.step + 1

        return new_state


class SimpleRESPAIntegrator(Integrator):
    """
    Simplified RESPA for use with external force computation.

    This version expects the user to provide separate fast and slow forces
    rather than computing them internally. Useful when force providers are
    managed by the engine.
    """

    def __init__(
        self,
        dt_inner: float,
        n_inner: int,
    ) -> None:
        """
        Initialize simplified RESPA integrator.

        Args:
            dt_inner: Inner (fast) timestep.
            n_inner: Number of inner steps per outer step.
        """
        self._dt_inner = dt_inner
        self._n_inner = n_inner
        self._dt_outer = dt_inner * n_inner

        # Callbacks for force computation
        self._fast_force_callback: Callable | None = None
        self._slow_force_callback: Callable | None = None

    @property
    def timestep(self) -> float:
        return self._dt_outer

    def set_force_callbacks(
        self,
        fast_callback: Callable[[MDState], NDArray[np.floating]],
        slow_callback: Callable[[MDState], NDArray[np.floating]],
    ) -> None:
        """
        Set callbacks for force computation.

        Args:
            fast_callback: Function to compute fast forces given state.
            slow_callback: Function to compute slow forces given state.
        """
        self._fast_force_callback = fast_callback
        self._slow_force_callback = slow_callback

    def step(self, state: MDState, forces: NDArray[np.floating]) -> MDState:
        """
        Perform RESPA step using force callbacks.

        Args:
            state: Current MD state.
            forces: Ignored (computed via callbacks).

        Returns:
            New MDState after integration.
        """
        if self._fast_force_callback is None or self._slow_force_callback is None:
            raise RuntimeError(
                "Force callbacks not set. Call set_force_callbacks() first."
            )

        dt_fast = self._dt_inner
        dt_slow = self._dt_outer
        half_dt_fast = 0.5 * dt_fast
        half_dt_slow = 0.5 * dt_slow

        masses = state.masses[:, np.newaxis]

        # Work with mutable state
        current_state = state.copy()

        # Compute slow forces
        slow_forces = self._slow_force_callback(current_state)
        accel_slow = slow_forces / masses

        # Outer half kick
        current_state.velocities = current_state.velocities + half_dt_slow * accel_slow

        # Inner loop
        for _ in range(self._n_inner):
            # Fast forces
            fast_forces = self._fast_force_callback(current_state)
            accel_fast = fast_forces / masses

            # Inner half kick
            current_state.velocities = (
                current_state.velocities + half_dt_fast * accel_fast
            )

            # Inner drift
            current_state.positions = (
                current_state.positions + dt_fast * current_state.velocities
            )
            current_state.positions = state.box.wrap_positions(current_state.positions)

            # Recompute fast forces
            fast_forces = self._fast_force_callback(current_state)
            accel_fast = fast_forces / masses

            # Inner half kick
            current_state.velocities = (
                current_state.velocities + half_dt_fast * accel_fast
            )

        # Recompute slow forces
        slow_forces = self._slow_force_callback(current_state)
        accel_slow = slow_forces / masses

        # Outer half kick
        current_state.velocities = current_state.velocities + half_dt_slow * accel_slow

        # Update state metadata
        current_state.forces = fast_forces + slow_forces
        current_state.time = state.time + dt_slow
        current_state.step = state.step + 1

        return current_state
