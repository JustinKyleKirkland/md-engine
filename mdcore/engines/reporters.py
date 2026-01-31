"""Reporter implementations for simulation output."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np

if TYPE_CHECKING:
    from ..system import MDState


class Reporter(ABC):
    """
    Abstract base class for simulation reporters.

    Reporters are called periodically during simulation to output
    information, save checkpoints, or perform analysis.
    """

    @abstractmethod
    def report(self, state: MDState, **kwargs: Any) -> None:
        """
        Generate report for current state.

        Args:
            state: Current simulation state.
            **kwargs: Additional information (e.g., energies).
        """
        ...

    @property
    @abstractmethod
    def frequency(self) -> int:
        """Return reporting frequency (every N steps)."""
        ...

    def should_report(self, step: int) -> bool:
        """Check if reporter should run at this step."""
        return step % self.frequency == 0

    def initialize(self, state: MDState) -> None:
        """Initialize reporter (called before simulation)."""
        pass

    def finalize(self, state: MDState) -> None:
        """Finalize reporter (called after simulation)."""
        pass


class ReporterGroup:
    """
    Collection of reporters with automatic frequency handling.
    """

    def __init__(self, reporters: list[Reporter] | None = None) -> None:
        """
        Initialize reporter group.

        Args:
            reporters: List of reporters to manage.
        """
        self._reporters: list[Reporter] = reporters if reporters else []

    def add(self, reporter: Reporter) -> None:
        """Add a reporter to the group."""
        self._reporters.append(reporter)

    def remove(self, reporter: Reporter) -> None:
        """Remove a reporter from the group."""
        self._reporters.remove(reporter)

    def initialize(self, state: MDState) -> None:
        """Initialize all reporters."""
        for reporter in self._reporters:
            reporter.initialize(state)

    def report(self, state: MDState, **kwargs: Any) -> None:
        """Run all reporters that should fire at this step."""
        for reporter in self._reporters:
            if reporter.should_report(state.step):
                reporter.report(state, **kwargs)

    def finalize(self, state: MDState) -> None:
        """Finalize all reporters."""
        for reporter in self._reporters:
            reporter.finalize(state)


class StateReporter(Reporter):
    """
    Reporter that prints simulation state to console or file.

    Outputs step, time, temperature, kinetic energy, potential energy,
    and total energy.
    """

    def __init__(
        self,
        frequency: int = 1000,
        file: TextIO | None = None,
        separator: str = "\t",
    ) -> None:
        """
        Initialize state reporter.

        Args:
            frequency: Reporting frequency (every N steps).
            file: Output file (defaults to stdout).
            separator: Field separator.
        """
        self._frequency = frequency
        self._file = file if file is not None else sys.stdout
        self._separator = separator
        self._header_written = False

    @property
    def frequency(self) -> int:
        return self._frequency

    def initialize(self, state: MDState) -> None:
        """Write header."""
        if not self._header_written:
            headers = ["Step", "Time", "Temperature", "KE", "PE", "Total"]
            self._file.write(self._separator.join(headers) + "\n")
            self._header_written = True

    def report(self, state: MDState, **kwargs: Any) -> None:
        """
        Report current state.

        Args:
            state: Current simulation state.
            **kwargs: Should include 'potential_energy' if available.
        """
        pe = kwargs.get("potential_energy", 0.0)
        ke = state.kinetic_energy
        temp = state.temperature
        total = ke + pe

        values = [
            f"{state.step}",
            f"{state.time:.4f}",
            f"{temp:.2f}",
            f"{ke:.4f}",
            f"{pe:.4f}",
            f"{total:.4f}",
        ]

        self._file.write(self._separator.join(values) + "\n")
        self._file.flush()


class TrajectoryReporter(Reporter):
    """
    Reporter that saves trajectory (positions) to memory or file.

    For production use, implement file-based output (XTC, DCD, etc.).
    This implementation stores in memory for simplicity.
    """

    def __init__(
        self,
        frequency: int = 100,
        include_velocities: bool = False,
        include_forces: bool = False,
    ) -> None:
        """
        Initialize trajectory reporter.

        Args:
            frequency: Reporting frequency.
            include_velocities: Also store velocities.
            include_forces: Also store forces.
        """
        self._frequency = frequency
        self._include_velocities = include_velocities
        self._include_forces = include_forces

        self._positions: list[np.ndarray] = []
        self._velocities: list[np.ndarray] = []
        self._forces: list[np.ndarray] = []
        self._times: list[float] = []
        self._steps: list[int] = []

    @property
    def frequency(self) -> int:
        return self._frequency

    def report(self, state: MDState, **kwargs: Any) -> None:
        """Store current frame."""
        self._positions.append(state.positions.copy())
        self._times.append(state.time)
        self._steps.append(state.step)

        if self._include_velocities:
            self._velocities.append(state.velocities.copy())
        if self._include_forces:
            self._forces.append(state.forces.copy())

    @property
    def n_frames(self) -> int:
        """Return number of stored frames."""
        return len(self._positions)

    @property
    def positions(self) -> np.ndarray:
        """Return positions as (n_frames, n_atoms, 3) array."""
        return np.array(self._positions)

    @property
    def velocities(self) -> np.ndarray | None:
        """Return velocities if stored."""
        if not self._include_velocities:
            return None
        return np.array(self._velocities)

    @property
    def forces(self) -> np.ndarray | None:
        """Return forces if stored."""
        if not self._include_forces:
            return None
        return np.array(self._forces)

    @property
    def times(self) -> np.ndarray:
        """Return times array."""
        return np.array(self._times)

    def clear(self) -> None:
        """Clear stored trajectory."""
        self._positions.clear()
        self._velocities.clear()
        self._forces.clear()
        self._times.clear()
        self._steps.clear()


class CheckpointReporter(Reporter):
    """
    Reporter that saves simulation checkpoints for restart.

    Stores the full state at regular intervals.
    """

    def __init__(
        self,
        frequency: int = 10000,
        keep_last: int = 3,
    ) -> None:
        """
        Initialize checkpoint reporter.

        Args:
            frequency: Checkpoint frequency.
            keep_last: Number of checkpoints to keep.
        """
        self._frequency = frequency
        self._keep_last = keep_last
        self._checkpoints: list[dict[str, Any]] = []

    @property
    def frequency(self) -> int:
        return self._frequency

    def report(self, state: MDState, **kwargs: Any) -> None:
        """Save checkpoint."""
        checkpoint = {
            "step": state.step,
            "time": state.time,
            "positions": state.positions.copy(),
            "velocities": state.velocities.copy(),
            "forces": state.forces.copy(),
            "masses": state.masses.copy(),
            "box_vectors": state.box.vectors.copy(),
        }
        checkpoint.update(kwargs)

        self._checkpoints.append(checkpoint)

        # Keep only last N checkpoints
        if len(self._checkpoints) > self._keep_last:
            self._checkpoints.pop(0)

    @property
    def latest_checkpoint(self) -> dict[str, Any] | None:
        """Return latest checkpoint."""
        if not self._checkpoints:
            return None
        return self._checkpoints[-1]

    @property
    def all_checkpoints(self) -> list[dict[str, Any]]:
        """Return all stored checkpoints."""
        return self._checkpoints.copy()


class CallbackReporter(Reporter):
    """
    Reporter that calls a user-defined function.

    Allows arbitrary custom reporting logic.
    """

    def __init__(
        self,
        callback: Callable[[MDState, dict[str, Any]], None],
        frequency: int = 1,
    ) -> None:
        """
        Initialize callback reporter.

        Args:
            callback: Function to call with (state, kwargs).
            frequency: Reporting frequency.
        """
        self._callback = callback
        self._frequency = frequency

    @property
    def frequency(self) -> int:
        return self._frequency

    def report(self, state: MDState, **kwargs: Any) -> None:
        """Call the callback function."""
        self._callback(state, kwargs)


class EnergyReporter(Reporter):
    """
    Reporter that tracks energy components over time.
    """

    def __init__(self, frequency: int = 100) -> None:
        """
        Initialize energy reporter.

        Args:
            frequency: Reporting frequency.
        """
        self._frequency = frequency
        self._steps: list[int] = []
        self._times: list[float] = []
        self._kinetic: list[float] = []
        self._potential: list[float] = []
        self._total: list[float] = []

    @property
    def frequency(self) -> int:
        return self._frequency

    def report(self, state: MDState, **kwargs: Any) -> None:
        """Record energies."""
        pe = kwargs.get("potential_energy", 0.0)
        ke = state.kinetic_energy

        self._steps.append(state.step)
        self._times.append(state.time)
        self._kinetic.append(ke)
        self._potential.append(pe)
        self._total.append(ke + pe)

    @property
    def kinetic_energy(self) -> np.ndarray:
        """Return kinetic energy time series."""
        return np.array(self._kinetic)

    @property
    def potential_energy(self) -> np.ndarray:
        """Return potential energy time series."""
        return np.array(self._potential)

    @property
    def total_energy(self) -> np.ndarray:
        """Return total energy time series."""
        return np.array(self._total)

    @property
    def times(self) -> np.ndarray:
        """Return times array."""
        return np.array(self._times)

    def clear(self) -> None:
        """Clear stored data."""
        self._steps.clear()
        self._times.clear()
        self._kinetic.clear()
        self._potential.clear()
        self._total.clear()
