"""XTC (GROMACS) compressed trajectory format (placeholder)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from ..base import BinaryTrajectoryReader, BinaryTrajectoryWriter

if TYPE_CHECKING:
    from ...system import MDState


class XTCWriter(BinaryTrajectoryWriter):
    """
    XTC format trajectory writer.

    XTC is GROMACS's compressed coordinate format using lossy compression.
    Provides ~10x compression vs DCD with configurable precision.

    Note: Requires XDR library or external package for full support.
    """

    def __init__(
        self,
        filename: str | Path,
        precision: float = 1000.0,
    ) -> None:
        """
        Initialize XTC writer.

        Args:
            filename: Output file path.
            precision: Coordinate precision (1000 = 0.001 nm).
        """
        super().__init__(filename)
        self.precision = precision

    def write(self, state: MDState, **kwargs) -> None:
        """
        Write a single frame.

        Args:
            state: MD state to write.
        """
        raise NotImplementedError(
            "XTC writing not yet implemented. "
            "Consider using MDAnalysis or mdtraj for XTC support."
        )


class XTCReader(BinaryTrajectoryReader):
    """
    XTC format trajectory reader.

    Note: Requires XDR library or external package for full support.
    """

    def __init__(self, filename: str | Path) -> None:
        """
        Initialize XTC reader.

        Args:
            filename: Input file path.
        """
        super().__init__(filename)
        self._n_frames = 0

    def read_frame(self, index: int) -> dict:
        """Read a specific frame."""
        raise NotImplementedError(
            "XTC reading not yet implemented. "
            "Consider using MDAnalysis or mdtraj for XTC support."
        )

    def __iter__(self) -> Iterator[dict]:
        """Iterate over all frames."""
        raise NotImplementedError("XTC iteration not yet implemented.")

    def __len__(self) -> int:
        """Return number of frames."""
        return self._n_frames
