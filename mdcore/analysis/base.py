"""Base classes for analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..system import MDState


class Analyzer(ABC):
    """
    Abstract base class for all analyzers.

    Analyzers process simulation data to compute properties
    like RDF, MSD, energies, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Analyzer name for identification."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset analyzer state."""
        ...

    @abstractmethod
    def result(self) -> dict[str, Any]:
        """
        Get analysis results.

        Returns:
            Dictionary of results (varies by analyzer type).
        """
        ...


class StreamingAnalyzer(Analyzer):
    """
    Base class for streaming (online) analyzers.

    Streaming analyzers process data frame-by-frame during
    simulation, maintaining running statistics without storing
    the full trajectory.

    This is memory-efficient for long simulations.

    Example:
        analyzer = MeanSquareDisplacement()
        for step in simulation:
            analyzer.update(state)
        msd = analyzer.result()
    """

    @abstractmethod
    def update(self, state: MDState, **kwargs: Any) -> None:
        """
        Update analyzer with new frame.

        Args:
            state: Current simulation state.
            **kwargs: Additional data (e.g., energies).
        """
        ...

    @property
    def n_frames(self) -> int:
        """Number of frames processed."""
        return getattr(self, "_n_frames", 0)


class TrajectoryAnalyzer(Analyzer):
    """
    Base class for trajectory (offline) analyzers.

    Trajectory analyzers operate on complete trajectories,
    enabling analyses that require multiple passes or
    non-local correlations.

    Supports various storage backends:
    - NumPy arrays (in-memory)
    - HDF5 files
    - Zarr arrays
    - Dask arrays (for parallel processing)
    """

    @abstractmethod
    def analyze(
        self,
        positions: NDArray[np.floating],
        times: NDArray[np.floating] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Analyze complete trajectory.

        Args:
            positions: Trajectory positions, shape (n_frames, n_atoms, 3).
            times: Frame times, shape (n_frames,).
            **kwargs: Additional trajectory data.

        Returns:
            Analysis results dictionary.
        """
        ...


class CompositeAnalyzer(StreamingAnalyzer):
    """
    Composite analyzer that runs multiple analyzers together.

    Useful for computing multiple properties in a single pass
    through the trajectory.
    """

    def __init__(self, analyzers: list[StreamingAnalyzer]) -> None:
        """
        Initialize composite analyzer.

        Args:
            analyzers: List of analyzers to run.
        """
        self.analyzers = analyzers
        self._n_frames = 0

    @property
    def name(self) -> str:
        """Return composite name."""
        return "composite"

    def update(self, state: MDState, **kwargs: Any) -> None:
        """Update all child analyzers."""
        for analyzer in self.analyzers:
            analyzer.update(state, **kwargs)
        self._n_frames += 1

    def reset(self) -> None:
        """Reset all child analyzers."""
        for analyzer in self.analyzers:
            analyzer.reset()
        self._n_frames = 0

    def result(self) -> dict[str, Any]:
        """Get results from all analyzers."""
        results = {}
        for analyzer in self.analyzers:
            results[analyzer.name] = analyzer.result()
        return results

    def get_analyzer(self, name: str) -> StreamingAnalyzer | None:
        """Get child analyzer by name."""
        for analyzer in self.analyzers:
            if analyzer.name == name:
                return analyzer
        return None
