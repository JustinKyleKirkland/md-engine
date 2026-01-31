"""Cell list neighbor list implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import NeighborList

if TYPE_CHECKING:
    from ..system import Box


class CellList(NeighborList):
    """
    Cell list (linked cell) neighbor list.

    Divides the simulation box into cells of size >= cutoff.
    Only atoms in the same or neighboring cells are considered
    as potential neighbors, reducing complexity from O(N^2) to O(N).

    Supports both orthorhombic and triclinic boxes (via transformation).

    Attributes:
        _cutoff: Interaction cutoff distance.
        skin: Additional buffer distance.
        _n_cells: Number of cells in each dimension.
        _cell_size: Size of cells in each dimension.
        _head: First atom in each cell.
        _list: Next atom in the same cell.
        _pairs: Cached neighbor pairs.
        _neighbors: Per-atom neighbor lists.
    """

    def __init__(self, cutoff: float, skin: float = 0.3) -> None:
        """
        Initialize cell list.

        Args:
            cutoff: Interaction cutoff distance.
            skin: Buffer distance for neighbor list validity.
        """
        self._cutoff = cutoff
        self.skin = skin
        self._list_cutoff = cutoff + skin

        self._n_cells: NDArray[np.integer] = np.array([1, 1, 1], dtype=np.int32)
        self._cell_size: NDArray[np.floating] = np.zeros(3, dtype=np.float64)

        self._head: NDArray[np.integer] | None = None
        self._list: NDArray[np.integer] | None = None

        self._pairs: NDArray[np.integer] = np.empty((0, 2), dtype=np.int32)
        self._neighbors: list[NDArray[np.integer]] = []
        self._positions_at_build: NDArray[np.floating] | None = None
        self._box: Box | None = None

    @property
    def cutoff(self) -> float:
        """Return the interaction cutoff distance."""
        return self._cutoff

    @property
    def list_cutoff(self) -> float:
        """Return the neighbor list cutoff (cutoff + skin)."""
        return self._list_cutoff

    @property
    def n_pairs(self) -> int:
        """Return the number of neighbor pairs."""
        return len(self._pairs)

    @property
    def n_cells(self) -> tuple[int, int, int]:
        """Return number of cells in each dimension."""
        return tuple(self._n_cells)

    def _get_cell_index(
        self, position: NDArray[np.floating], box: Box
    ) -> tuple[int, int, int]:
        """
        Get cell index for a position.

        Args:
            position: Position vector (3,).
            box: Simulation box.

        Returns:
            Tuple of (cx, cy, cz) cell indices.
        """
        # Wrap position into box
        wrapped = box.wrap_positions(position.reshape(1, 3))[0]

        # Get fractional coordinates (0 to 1)
        if box.is_orthorhombic:
            lengths = np.diag(box.vectors)
            frac = wrapped / lengths
        else:
            inv_vectors = np.linalg.inv(box.vectors)
            frac = wrapped @ inv_vectors.T

        # Convert to cell indices
        cell_idx = (frac * self._n_cells).astype(np.int32)

        # Handle edge cases
        cell_idx = np.clip(cell_idx, 0, self._n_cells - 1)

        return tuple(cell_idx)

    def _get_cell_linear_index(self, cx: int, cy: int, cz: int) -> int:
        """Convert 3D cell index to linear index."""
        # Apply periodic wrapping to cell indices
        cx = cx % self._n_cells[0]
        cy = cy % self._n_cells[1]
        cz = cz % self._n_cells[2]
        return cx * self._n_cells[1] * self._n_cells[2] + cy * self._n_cells[2] + cz

    def _get_neighbor_cells(
        self, cx: int, cy: int, cz: int
    ) -> list[tuple[int, int, int]]:
        """
        Get indices of neighboring cells (including self).

        Returns the 27 cells (3x3x3 cube) centered on the given cell,
        with periodic wrapping.
        """
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    nx = (cx + dx) % self._n_cells[0]
                    ny = (cy + dy) % self._n_cells[1]
                    nz = (cz + dz) % self._n_cells[2]
                    neighbors.append((nx, ny, nz))
        return neighbors

    def build(self, positions: ArrayLike, box: Box) -> None:
        """
        Build the cell list neighbor list.

        Args:
            positions: Atomic positions, shape (N, 3).
            box: Simulation box.
        """
        positions = np.asarray(positions, dtype=np.float64)
        n_atoms = len(positions)

        self._positions_at_build = positions.copy()
        self._box = box

        # Determine cell grid
        if box.is_orthorhombic:
            box_lengths = np.diag(box.vectors)
        else:
            # Use box vector norms as approximate lengths
            box_lengths = np.linalg.norm(box.vectors, axis=1)

        # Number of cells in each dimension (at least 3 for proper neighbor search)
        self._n_cells = np.maximum(
            (box_lengths / self._list_cutoff).astype(np.int32), 3
        )
        self._cell_size = box_lengths / self._n_cells

        total_cells = np.prod(self._n_cells)

        # Initialize linked list
        self._head = np.full(total_cells, -1, dtype=np.int32)
        self._list = np.full(n_atoms, -1, dtype=np.int32)

        # Assign atoms to cells
        atom_cells = np.zeros((n_atoms, 3), dtype=np.int32)
        for i in range(n_atoms):
            cx, cy, cz = self._get_cell_index(positions[i], box)
            atom_cells[i] = [cx, cy, cz]
            cell_idx = self._get_cell_linear_index(cx, cy, cz)

            # Insert at head of cell's linked list
            self._list[i] = self._head[cell_idx]
            self._head[cell_idx] = i

        # Build neighbor pairs using cell list
        pairs = []
        self._neighbors = [[] for _ in range(n_atoms)]

        cutoff_sq = self._list_cutoff**2

        # Iterate over all cells
        for cx in range(self._n_cells[0]):
            for cy in range(self._n_cells[1]):
                for cz in range(self._n_cells[2]):
                    cell_idx = self._get_cell_linear_index(cx, cy, cz)

                    # Get atoms in this cell
                    i = self._head[cell_idx]
                    while i != -1:
                        # Check neighbors in same cell (only j > i to avoid duplicates)
                        j = self._list[i]
                        while j != -1:
                            dr = box.minimum_image(positions[i], positions[j])
                            r_sq = np.sum(dr**2)

                            if r_sq < cutoff_sq:
                                pairs.append([min(i, j), max(i, j)])
                                self._neighbors[i].append(j)
                                self._neighbors[j].append(i)

                            j = self._list[j]

                        # Check neighboring cells
                        neighbor_cells = self._get_neighbor_cells(cx, cy, cz)
                        for ncx, ncy, ncz in neighbor_cells:
                            if (ncx, ncy, ncz) <= (cx, cy, cz):
                                continue  # Already checked or same cell

                            neighbor_idx = self._get_cell_linear_index(ncx, ncy, ncz)
                            j = self._head[neighbor_idx]

                            while j != -1:
                                dr = box.minimum_image(positions[i], positions[j])
                                r_sq = np.sum(dr**2)

                                if r_sq < cutoff_sq:
                                    pairs.append([min(i, j), max(i, j)])
                                    self._neighbors[i].append(j)
                                    self._neighbors[j].append(i)

                                j = self._list[j]

                        i = self._list[i]

        # Remove duplicates and sort
        if pairs:
            self._pairs = np.unique(np.array(pairs, dtype=np.int32), axis=0)
        else:
            self._pairs = np.empty((0, 2), dtype=np.int32)

        # Convert neighbor lists to arrays
        self._neighbors = [np.array(nbrs, dtype=np.int32) for nbrs in self._neighbors]

    def update_if_needed(self, positions: ArrayLike) -> bool:
        """
        Rebuild neighbor list if atoms have moved too far.

        Args:
            positions: Current atomic positions.

        Returns:
            True if the list was rebuilt.
        """
        if self._positions_at_build is None or self._box is None:
            raise RuntimeError("Neighbor list has not been built yet")

        positions = np.asarray(positions, dtype=np.float64)

        # Check maximum displacement
        dr = positions - self._positions_at_build
        max_displacement = np.max(np.linalg.norm(dr, axis=1))

        if max_displacement > self.skin / 2:
            self.build(positions, self._box)
            return True

        return False

    def get_pairs(self) -> NDArray[np.integer]:
        """Get all neighbor pairs."""
        return self._pairs

    def get_neighbors(self, atom_index: int) -> NDArray[np.integer]:
        """Get neighbors of a specific atom."""
        if not self._neighbors:
            return np.array([], dtype=np.int32)
        return self._neighbors[atom_index]
