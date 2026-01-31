"""XYZ trajectory format implementation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..base import TrajectoryReader, TrajectoryWriter

if TYPE_CHECKING:
    from ...system import MDState


class XYZWriter(TrajectoryWriter):
    """
    XYZ format trajectory writer.

    XYZ is a simple text format:
        N
        comment line
        element x y z
        element x y z
        ...

    Supports extended XYZ with properties in the comment line.
    """

    def __init__(
        self,
        filename: str | Path,
        elements: list[str] | None = None,
        precision: int = 8,
    ) -> None:
        """
        Initialize XYZ writer.

        Args:
            filename: Output file path.
            elements: Element symbols for each atom. If None, uses "X".
            precision: Decimal places for coordinates.
        """
        super().__init__(filename)
        self.elements = elements
        self.precision = precision

    def write(self, state: MDState, comment: str = "", **kwargs) -> None:
        """
        Write a single frame in XYZ format.

        Args:
            state: MD state to write.
            comment: Comment line (can include properties).
        """
        if self._file is None:
            raise RuntimeError("File not open. Use context manager or call open().")

        n_atoms = state.n_atoms
        positions = state.positions

        # Determine element symbols
        if self.elements is not None:
            elements = self.elements
        elif hasattr(state, "elements") and state.elements is not None:
            elements = state.elements
        else:
            elements = ["X"] * n_atoms

        # Build comment line with optional properties
        if not comment:
            comment = f"Frame {self._n_frames}"
            if state.box is not None:
                # Add lattice info for extended XYZ
                lattice = state.box.vectors.flatten()
                lattice_str = " ".join(f"{v:.6f}" for v in lattice)
                comment = f'Lattice="{lattice_str}" {comment}'

        # Write frame
        self._file.write(f"{n_atoms}\n")
        self._file.write(f"{comment}\n")

        fmt = f"{{}} {{:.{self.precision}f}} {{:.{self.precision}f}} {{:.{self.precision}f}}\n"
        for i in range(n_atoms):
            self._file.write(fmt.format(elements[i], *positions[i]))

        self._n_frames += 1


class XYZReader(TrajectoryReader):
    """
    XYZ format trajectory reader.

    Supports standard and extended XYZ formats.
    """

    def __init__(self, filename: str | Path) -> None:
        """
        Initialize XYZ reader.

        Args:
            filename: Input file path.
        """
        super().__init__(filename)
        self._frame_offsets: list[int] = []

    def open(self) -> None:
        """Open file and index frame positions."""
        super().open()
        self._index_frames()

    def _index_frames(self) -> None:
        """Build index of frame positions in file."""
        self._frame_offsets = []

        if self._file is None:
            return

        self._file.seek(0)
        while True:
            offset = self._file.tell()
            line = self._file.readline()

            if not line:
                break

            try:
                n_atoms = int(line.strip())
            except ValueError:
                break

            self._frame_offsets.append(offset)

            # Skip comment and atom lines
            self._file.readline()  # Comment
            for _ in range(n_atoms):
                self._file.readline()

        self._n_frames = len(self._frame_offsets)

    def read_frame(self, index: int) -> dict:
        """
        Read a specific frame.

        Args:
            index: Frame index (0-based).

        Returns:
            Dictionary with 'positions', 'elements', 'comment', and optionally 'box'.
        """
        if self._file is None:
            raise RuntimeError("File not open.")

        if index < 0 or index >= len(self._frame_offsets):
            raise IndexError(f"Frame index {index} out of range")

        self._file.seek(self._frame_offsets[index])

        # Read number of atoms
        n_atoms = int(self._file.readline().strip())

        # Read comment line
        comment = self._file.readline().strip()

        # Parse extended XYZ lattice if present
        box = None
        if 'Lattice="' in comment:
            import re

            match = re.search(r'Lattice="([^"]+)"', comment)
            if match:
                lattice_str = match.group(1)
                lattice = np.array([float(x) for x in lattice_str.split()])
                if len(lattice) == 9:
                    box = lattice.reshape(3, 3)

        # Read atom data
        elements = []
        positions = np.zeros((n_atoms, 3))

        for i in range(n_atoms):
            parts = self._file.readline().split()
            elements.append(parts[0])
            positions[i] = [float(parts[1]), float(parts[2]), float(parts[3])]

        result = {
            "positions": positions,
            "elements": elements,
            "comment": comment,
            "n_atoms": n_atoms,
        }

        if box is not None:
            result["box"] = box

        return result

    def __iter__(self) -> Iterator[dict]:
        """Iterate over all frames."""
        for i in range(len(self)):
            yield self.read_frame(i)

    def __len__(self) -> int:
        """Return number of frames."""
        return len(self._frame_offsets)

    def __getitem__(self, index: int) -> dict:
        """Get frame by index."""
        return self.read_frame(index)
