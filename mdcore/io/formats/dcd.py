"""DCD (CHARMM/NAMD) binary trajectory format (placeholder)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from ..base import BinaryTrajectoryReader, BinaryTrajectoryWriter

if TYPE_CHECKING:
    from ...system import MDState


class DCDWriter(BinaryTrajectoryWriter):
    """
    DCD format trajectory writer.

    DCD is a binary format used by CHARMM, NAMD, and other MD packages.
    Stores coordinates only (not velocities) in single precision.

    Note: Full implementation requires proper Fortran record handling.
    """

    def __init__(self, filename: str | Path, n_atoms: int) -> None:
        """
        Initialize DCD writer.

        Args:
            filename: Output file path.
            n_atoms: Number of atoms (required for header).
        """
        super().__init__(filename)
        self.n_atoms = n_atoms
        self._header_written = False

    def write_header(self, **kwargs) -> None:
        """Write DCD file header."""
        if self._file is None:
            raise RuntimeError("File not open.")

        # DCD header structure (simplified)
        # Full implementation would follow CHARMM DCD spec
        self._header_written = True

    def write(self, state: MDState, **kwargs) -> None:
        """
        Write a single frame.

        Args:
            state: MD state to write.
        """
        if self._file is None:
            raise RuntimeError("File not open.")

        if not self._header_written:
            self.write_header()

        # Placeholder: actual DCD writing would go here
        # Requires Fortran-style record markers
        raise NotImplementedError(
            "DCD writing not yet implemented. "
            "Consider using MDAnalysis or mdtraj for DCD support."
        )


class DCDReader(BinaryTrajectoryReader):
    """
    DCD format trajectory reader.

    Note: Full implementation requires proper Fortran record handling.
    """

    def __init__(self, filename: str | Path) -> None:
        """
        Initialize DCD reader.

        Args:
            filename: Input file path.
        """
        super().__init__(filename)
        self._n_atoms = 0
        self._n_frames = 0

    def read_frame(self, index: int) -> dict:
        """Read a specific frame."""
        raise NotImplementedError(
            "DCD reading not yet implemented. "
            "Consider using MDAnalysis or mdtraj for DCD support."
        )

    def __iter__(self) -> Iterator[dict]:
        """Iterate over all frames."""
        raise NotImplementedError("DCD iteration not yet implemented.")

    def __len__(self) -> int:
        """Return number of frames."""
        return self._n_frames
