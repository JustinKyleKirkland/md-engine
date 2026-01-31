"""MD simulation engine implementation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..forcefields import ForceProvider
    from ..integrators import BarostatModifier, Integrator, ThermostatModifier
    from ..neighborlists import NeighborList
    from ..system import MDState

from .reporters import Reporter, ReporterGroup


class SimulationHook:
    """
    Hook for custom actions during simulation.

    Hooks can be called at different points in the simulation loop:
    - pre_step: Before each step
    - post_force: After force computation
    - post_step: After integration
    """

    def pre_step(self, engine: MDEngine, state: MDState) -> MDState | None:
        """Called before each step. Can modify state."""
        return None

    def post_force(
        self, engine: MDEngine, state: MDState, forces: np.ndarray
    ) -> np.ndarray | None:
        """Called after force computation. Can modify forces."""
        return None

    def post_step(self, engine: MDEngine, state: MDState) -> MDState | None:
        """Called after each step. Can modify state."""
        return None


class MDEngine:
    """
    Molecular dynamics simulation engine.

    Orchestrates the main simulation loop, managing:
    - State propagation (integrator)
    - Force computation (force provider)
    - Neighbor list updates
    - Temperature/pressure control (thermostats/barostats)
    - Output and checkpointing (reporters)
    - Extensibility (hooks)

    Example usage:
        engine = MDEngine(
            state=initial_state,
            integrator=VelocityVerletIntegrator(dt=0.002),
            force_provider=forcefield,
            neighbor_list=VerletList(cutoff=1.0),
        )
        engine.add_reporter(StateReporter(frequency=1000))
        engine.run(nsteps=100000)

    Attributes:
        state: Current simulation state.
        integrator: Time integration algorithm.
        force_provider: Force computation module.
        neighbor_list: Neighbor list for nonbonded interactions.
        thermostat: Optional temperature control.
        barostat: Optional pressure control.
    """

    def __init__(
        self,
        state: MDState,
        integrator: Integrator,
        force_provider: ForceProvider,
        neighbor_list: NeighborList | None = None,
        thermostat: ThermostatModifier | None = None,
        barostat: BarostatModifier | None = None,
    ) -> None:
        """
        Initialize MD engine.

        Args:
            state: Initial simulation state.
            integrator: Time integrator.
            force_provider: Force computation module.
            neighbor_list: Optional neighbor list for nonbonded forces.
            thermostat: Optional thermostat modifier.
            barostat: Optional barostat modifier.
        """
        self._state = state.copy()
        self._integrator = integrator
        self._force_provider = force_provider
        self._neighbor_list = neighbor_list
        self._thermostat = thermostat
        self._barostat = barostat

        self._reporters = ReporterGroup()
        self._hooks: list[SimulationHook] = []

        # Tracking
        self._running = False
        self._total_steps = 0
        self._wall_time = 0.0
        self._last_potential_energy = 0.0

        # Initialize neighbor list
        if self._neighbor_list is not None:
            self._neighbor_list.build(self._state.positions, self._state.box)

        # Compute initial forces
        self._compute_forces()

    @property
    def state(self) -> MDState:
        """Return current simulation state."""
        return self._state

    @state.setter
    def state(self, value: MDState) -> None:
        """Set simulation state."""
        self._state = value.copy()

    @property
    def integrator(self) -> Integrator:
        """Return integrator."""
        return self._integrator

    @property
    def force_provider(self) -> ForceProvider:
        """Return force provider."""
        return self._force_provider

    @property
    def neighbor_list(self) -> NeighborList | None:
        """Return neighbor list."""
        return self._neighbor_list

    @property
    def thermostat(self) -> ThermostatModifier | None:
        """Return thermostat."""
        return self._thermostat

    @thermostat.setter
    def thermostat(self, value: ThermostatModifier | None) -> None:
        """Set thermostat."""
        self._thermostat = value

    @property
    def barostat(self) -> BarostatModifier | None:
        """Return barostat."""
        return self._barostat

    @barostat.setter
    def barostat(self, value: BarostatModifier | None) -> None:
        """Set barostat."""
        self._barostat = value

    @property
    def potential_energy(self) -> float:
        """Return last computed potential energy."""
        return self._last_potential_energy

    @property
    def kinetic_energy(self) -> float:
        """Return current kinetic energy."""
        return self._state.kinetic_energy

    @property
    def total_energy(self) -> float:
        """Return total energy."""
        return self.kinetic_energy + self.potential_energy

    @property
    def temperature(self) -> float:
        """Return current temperature."""
        return self._state.temperature

    @property
    def performance(self) -> dict[str, float]:
        """Return performance statistics."""
        if self._wall_time == 0:
            return {"ns_per_day": 0.0, "steps_per_second": 0.0}

        steps_per_second = self._total_steps / self._wall_time
        # Assuming timestep in ps
        ns_per_day = (steps_per_second * self._integrator.timestep * 86400) / 1000.0

        return {
            "ns_per_day": ns_per_day,
            "steps_per_second": steps_per_second,
            "wall_time": self._wall_time,
            "total_steps": self._total_steps,
        }

    def add_reporter(self, reporter: Reporter) -> None:
        """Add a reporter."""
        self._reporters.add(reporter)

    def remove_reporter(self, reporter: Reporter) -> None:
        """Remove a reporter."""
        self._reporters.remove(reporter)

    def add_hook(self, hook: SimulationHook) -> None:
        """Add a simulation hook."""
        self._hooks.append(hook)

    def remove_hook(self, hook: SimulationHook) -> None:
        """Remove a simulation hook."""
        self._hooks.remove(hook)

    def _compute_forces(self) -> tuple[np.ndarray, float]:
        """
        Compute forces and potential energy.

        Returns:
            Tuple of (forces, potential_energy).
        """
        # Update neighbor list if needed
        if self._neighbor_list is not None:
            self._neighbor_list.update_if_needed(self._state.positions)

        # Compute forces
        forces, energy = self._force_provider.compute_with_energy(
            self._state, self._neighbor_list
        )

        self._last_potential_energy = energy
        self._state.forces = forces

        return forces, energy

    def _apply_hooks_pre_step(self) -> None:
        """Apply pre-step hooks."""
        for hook in self._hooks:
            result = hook.pre_step(self, self._state)
            if result is not None:
                self._state = result

    def _apply_hooks_post_force(self, forces: np.ndarray) -> np.ndarray:
        """Apply post-force hooks."""
        for hook in self._hooks:
            result = hook.post_force(self, self._state, forces)
            if result is not None:
                forces = result
        return forces

    def _apply_hooks_post_step(self) -> None:
        """Apply post-step hooks."""
        for hook in self._hooks:
            result = hook.post_step(self, self._state)
            if result is not None:
                self._state = result

    def step(self) -> None:
        """
        Perform a single simulation step.

        This is the core MD loop:
        1. Pre-step hooks
        2. Get neighbor list
        3. Compute forces
        4. Post-force hooks
        5. Integration step
        6. Apply thermostat
        7. Apply barostat
        8. Post-step hooks
        9. Report
        """
        # Pre-step hooks
        self._apply_hooks_pre_step()

        # Compute forces
        forces, energy = self._compute_forces()

        # Post-force hooks
        forces = self._apply_hooks_post_force(forces)

        # Integration step
        self._state = self._integrator.step(self._state, forces)

        # Apply thermostat
        if self._thermostat is not None:
            self._state = self._thermostat.apply(self._state)

        # Apply barostat
        if self._barostat is not None:
            self._state = self._barostat.apply(self._state)
            # Rebuild neighbor list after box change
            if self._neighbor_list is not None:
                self._neighbor_list.build(self._state.positions, self._state.box)

        # Post-step hooks
        self._apply_hooks_post_step()

        # Report
        self._reporters.report(
            self._state,
            potential_energy=self._last_potential_energy,
        )

    def run(
        self,
        nsteps: int,
        callback: Callable[[MDEngine], bool] | None = None,
    ) -> MDState:
        """
        Run simulation for specified number of steps.

        Args:
            nsteps: Number of steps to run.
            callback: Optional callback called each step.
                     Return True to stop simulation early.

        Returns:
            Final simulation state.
        """
        self._running = True
        self._reporters.initialize(self._state)

        start_time = time.perf_counter()

        try:
            for _ in range(nsteps):
                if not self._running:
                    break

                self.step()
                self._total_steps += 1

                if callback is not None and callback(self):
                    break
        finally:
            self._wall_time += time.perf_counter() - start_time
            self._reporters.finalize(self._state)
            self._running = False

        return self._state

    def minimize(
        self,
        max_iterations: int = 1000,
        force_tolerance: float = 1.0,
        energy_tolerance: float = 1e-5,
    ) -> MDState:
        """
        Perform energy minimization (steepest descent).

        A simple minimization to remove bad contacts before dynamics.

        Args:
            max_iterations: Maximum minimization steps.
            force_tolerance: Stop when max force < tolerance.
            energy_tolerance: Stop when energy change < tolerance.

        Returns:
            Minimized state.
        """
        step_size = 0.001  # Initial step size
        prev_energy = float("inf")

        for i in range(max_iterations):
            forces, energy = self._compute_forces()

            # Check convergence
            max_force = np.max(np.linalg.norm(forces, axis=1))
            energy_change = abs(prev_energy - energy)

            if max_force < force_tolerance:
                break
            if energy_change < energy_tolerance and i > 0:
                break

            # Steepest descent step
            # Normalize forces to prevent too large steps
            force_norm = np.linalg.norm(forces)
            if force_norm > 0:
                direction = forces / force_norm
            else:
                break

            # Line search (simple backtracking)
            alpha = step_size
            new_positions = self._state.positions + alpha * direction
            new_positions = self._state.box.wrap_positions(new_positions)

            # Store old positions
            old_positions = self._state.positions.copy()
            self._state.positions = new_positions

            _, new_energy = self._compute_forces()

            # Accept or reject
            if new_energy < energy:
                step_size = min(step_size * 1.2, 0.01)  # Increase step
                prev_energy = energy
            else:
                # Reject and reduce step
                self._state.positions = old_positions
                step_size *= 0.5
                if step_size < 1e-10:
                    break

        # Final force computation
        self._compute_forces()

        return self._state

    def stop(self) -> None:
        """Signal simulation to stop."""
        self._running = False

    def get_checkpoint(self) -> dict[str, Any]:
        """
        Get a checkpoint of current simulation state.

        Returns:
            Dictionary containing all state needed for restart.
        """
        return {
            "step": self._state.step,
            "time": self._state.time,
            "positions": self._state.positions.copy(),
            "velocities": self._state.velocities.copy(),
            "forces": self._state.forces.copy(),
            "masses": self._state.masses.copy(),
            "box_vectors": self._state.box.vectors.copy(),
            "potential_energy": self._last_potential_energy,
            "total_steps": self._total_steps,
            "wall_time": self._wall_time,
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        Load simulation state from checkpoint.

        Args:
            checkpoint: Checkpoint dictionary from get_checkpoint().
        """
        from ..system.box import Box
        from ..system.state import MDState

        box = Box.triclinic(checkpoint["box_vectors"])
        self._state = MDState(
            positions=checkpoint["positions"],
            velocities=checkpoint["velocities"],
            forces=checkpoint["forces"],
            masses=checkpoint["masses"],
            box=box,
            time=checkpoint["time"],
            step=checkpoint["step"],
        )

        self._last_potential_energy = checkpoint.get("potential_energy", 0.0)
        self._total_steps = checkpoint.get("total_steps", 0)
        self._wall_time = checkpoint.get("wall_time", 0.0)

        # Rebuild neighbor list
        if self._neighbor_list is not None:
            self._neighbor_list.build(self._state.positions, self._state.box)
