"""PDB (Protein Data Bank) format implementation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..base import TrajectoryReader, TrajectoryWriter

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...system import MDState


class PDBWriter(TrajectoryWriter):
    """
    PDB format trajectory writer.

    Writes minimal PDB format suitable for visualization.
    For multi-frame trajectories, writes MODEL/ENDMDL records.
    """

    def __init__(
        self,
        filename: str | Path,
        elements: list[str] | None = None,
        atom_names: list[str] | None = None,
        residue_names: list[str] | None = None,
        residue_ids: list[int] | None = None,
    ) -> None:
        """
        Initialize PDB writer.

        Args:
            filename: Output file path.
            elements: Element symbols for each atom.
            atom_names: Atom names (e.g., "CA", "N").
            residue_names: Residue names (e.g., "ALA", "GLY").
            residue_ids: Residue IDs for each atom.
        """
        super().__init__(filename)
        self.elements = elements
        self.atom_names = atom_names
        self.residue_names = residue_names
        self.residue_ids = residue_ids

    def write(self, state: MDState, title: str = "", **kwargs) -> None:
        """
        Write a single frame in PDB format.

        Args:
            state: MD state to write.
            title: Title for the structure.
        """
        if self._file is None:
            raise RuntimeError("File not open.")

        n_atoms = state.n_atoms
        positions = state.positions

        # Write MODEL record for multi-frame
        if self._n_frames > 0 or kwargs.get("multiframe", True):
            self._file.write(f"MODEL     {self._n_frames + 1:4d}\n")

        # Write CRYST1 record if box is available
        if state.box is not None:
            lengths = state.box.lengths
            # Compute angles from box vectors
            angles = self._compute_box_angles(state.box.vectors)
            self._file.write(
                f"CRYST1{lengths[0]:9.3f}{lengths[1]:9.3f}{lengths[2]:9.3f}"
                f"{angles[0]:7.2f}{angles[1]:7.2f}{angles[2]:7.2f} P 1           1\n"
            )

        # Write ATOM records
        for i in range(n_atoms):
            atom_name = self._get_atom_name(i)
            res_name = self._get_residue_name(i)
            res_id = self._get_residue_id(i)
            element = self._get_element(i)

            x, y, z = positions[i]

            # PDB ATOM record format
            self._file.write(
                f"ATOM  {i + 1:5d} {atom_name:<4s} {res_name:3s}  "
                f"{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2s}\n"
            )

        # Write ENDMDL for multi-frame
        if self._n_frames > 0 or kwargs.get("multiframe", True):
            self._file.write("ENDMDL\n")

        self._n_frames += 1

    def write_footer(self, **kwargs) -> None:
        """Write END record."""
        if self._file is not None:
            self._file.write("END\n")

    def _get_atom_name(self, index: int) -> str:
        """Get atom name for given index."""
        if self.atom_names is not None and index < len(self.atom_names):
            return self.atom_names[index]
        return "X"

    def _get_residue_name(self, index: int) -> str:
        """Get residue name for given index."""
        if self.residue_names is not None and index < len(self.residue_names):
            return self.residue_names[index]
        return "UNK"

    def _get_residue_id(self, index: int) -> int:
        """Get residue ID for given index."""
        if self.residue_ids is not None and index < len(self.residue_ids):
            return self.residue_ids[index]
        return 1

    def _get_element(self, index: int) -> str:
        """Get element symbol for given index."""
        if self.elements is not None and index < len(self.elements):
            return self.elements[index]
        return "X"

    @staticmethod
    def _compute_box_angles(
        vectors: NDArray[np.floating],
    ) -> tuple[float, float, float]:
        """
        Compute box angles (alpha, beta, gamma) from box vectors.

        Args:
            vectors: Box vectors (3x3 array).

        Returns:
            Tuple of (alpha, beta, gamma) in degrees.
        """
        a, b, c = vectors[0], vectors[1], vectors[2]
        la = np.linalg.norm(a)
        lb = np.linalg.norm(b)
        lc = np.linalg.norm(c)

        # alpha = angle between b and c
        alpha = np.degrees(np.arccos(np.clip(np.dot(b, c) / (lb * lc), -1.0, 1.0)))
        # beta = angle between a and c
        beta = np.degrees(np.arccos(np.clip(np.dot(a, c) / (la * lc), -1.0, 1.0)))
        # gamma = angle between a and b
        gamma = np.degrees(np.arccos(np.clip(np.dot(a, b) / (la * lb), -1.0, 1.0)))

        return alpha, beta, gamma


class PDBReader(TrajectoryReader):
    """
    PDB format trajectory reader.

    Reads ATOM/HETATM records and MODEL/ENDMDL for multi-frame files.
    """

    def __init__(self, filename: str | Path) -> None:
        """
        Initialize PDB reader.

        Args:
            filename: Input file path.
        """
        super().__init__(filename)
        self._frame_offsets: list[int] = []
        self._single_frame = False

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
        current_offset = 0

        while True:
            line = self._file.readline()
            if not line:
                break

            if line.startswith("MODEL"):
                self._frame_offsets.append(current_offset)

            current_offset = self._file.tell()

        # Handle single-frame PDB (no MODEL records)
        if not self._frame_offsets:
            self._frame_offsets = [0]
            self._single_frame = True

        self._n_frames = len(self._frame_offsets)

    def read_frame(self, index: int) -> dict:
        """
        Read a specific frame.

        Args:
            index: Frame index (0-based).

        Returns:
            Dictionary with positions, elements, atom_names, etc.
        """
        if self._file is None:
            raise RuntimeError("File not open.")

        if index < 0 or index >= len(self._frame_offsets):
            raise IndexError(f"Frame index {index} out of range")

        self._file.seek(self._frame_offsets[index])

        positions = []
        elements = []
        atom_names = []
        residue_names = []
        residue_ids = []
        box = None

        for line in self._file:
            if line.startswith("ENDMDL") or line.startswith("END"):
                break

            if line.startswith("CRYST1"):
                # Parse unit cell
                a = float(line[6:15])
                b = float(line[15:24])
                c = float(line[24:33])
                alpha = float(line[33:40])
                beta = float(line[40:47])
                gamma = float(line[47:54])
                box = {
                    "lengths": np.array([a, b, c]),
                    "angles": np.array([alpha, beta, gamma]),
                }

            if line.startswith(("ATOM", "HETATM")):
                # Parse atom record
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                res_id = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip() if len(line) > 76 else atom_name[0]

                positions.append([x, y, z])
                elements.append(element)
                atom_names.append(atom_name)
                residue_names.append(res_name)
                residue_ids.append(res_id)

            # For single-frame files, stop at MODEL (next frame)
            if self._single_frame and line.startswith("MODEL") and positions:
                break

        result = {
            "positions": np.array(positions),
            "elements": elements,
            "atom_names": atom_names,
            "residue_names": residue_names,
            "residue_ids": residue_ids,
            "n_atoms": len(positions),
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
