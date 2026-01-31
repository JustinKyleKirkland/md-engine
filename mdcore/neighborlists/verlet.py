"""Verlet neighbor list implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import NeighborList

if TYPE_CHECKING:
    from ..system import Box


class VerletList(NeighborList):
    """
    Verlet neighbor list with skin distance.

    Uses a larger cutoff (cutoff + skin) for neighbor list construction,
    allowing the list to remain valid as atoms move small distances.
    The list is rebuilt when any atom has moved more than skin/2.

    Attributes:
        _cutoff: Interaction cutoff distance.
        skin: Additional buffer distance.
        _pairs: Cached neighbor pairs array.
        _neighbors: Per-atom neighbor lists.
        _positions_at_build: Positions when list was last built.
        _box: Box when list was last built.
    """

    def __init__(self, cutoff: float, skin: float = 0.3) -> None:
        """
        Initialize Verlet neighbor list.

        Args:
            cutoff: Interaction cutoff distance.
            skin: Buffer distance for neighbor list validity.
        """
        self._cutoff = cutoff
        self.skin = skin
        self._list_cutoff = cutoff + skin

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

    def build(self, positions: ArrayLike, box: Box) -> None:
        """
        Build the neighbor list from scratch.

        Uses O(N^2) distance calculations. For large systems,
        consider using CellList which has better scaling.

        Args:
            positions: Atomic positions, shape (N, 3).
            box: Simulation box.
        """
        positions = np.asarray(positions, dtype=np.float64)
        n_atoms = len(positions)

        self._positions_at_build = positions.copy()
        self._box = box

        # Build pair list
        pairs = []
        self._neighbors = [[] for _ in range(n_atoms)]

        # O(N^2) loop over all pairs
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dr = box.minimum_image(positions[i], positions[j])
                r = np.linalg.norm(dr)

                if r < self._list_cutoff:
                    pairs.append([i, j])
                    self._neighbors[i].append(j)
                    self._neighbors[j].append(i)

        if pairs:
            self._pairs = np.array(pairs, dtype=np.int32)
        else:
            self._pairs = np.empty((0, 2), dtype=np.int32)

        # Convert neighbor lists to arrays
        self._neighbors = [np.array(nbrs, dtype=np.int32) for nbrs in self._neighbors]

    def update_if_needed(self, positions: ArrayLike) -> bool:
        """
        Rebuild neighbor list if atoms have moved too far.

        The list is rebuilt if any atom has moved more than skin/2
        since the list was last built.

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

        # Rebuild if any atom moved more than skin/2
        # (factor of 2 because two atoms could move toward each other)
        if max_displacement > self.skin / 2:
            self.build(positions, self._box)
            return True

        return False

    def get_pairs(self) -> NDArray[np.integer]:
        """
        Get all neighbor pairs.

        Returns:
            Array of shape (N_pairs, 2) with (i, j) pairs where i < j.
        """
        return self._pairs

    def get_neighbors(self, atom_index: int) -> NDArray[np.integer]:
        """
        Get neighbors of a specific atom.

        Args:
            atom_index: Index of the atom to query.

        Returns:
            Array of neighbor atom indices.
        """
        if not self._neighbors:
            return np.array([], dtype=np.int32)
        return self._neighbors[atom_index]

    def get_distances(self, positions: ArrayLike, box: Box) -> NDArray[np.floating]:
        """
        Compute distances for all neighbor pairs.

        Args:
            positions: Current atomic positions.
            box: Simulation box.

        Returns:
            Array of distances for each pair in get_pairs().
        """
        if len(self._pairs) == 0:
            return np.array([], dtype=np.float64)

        positions = np.asarray(positions, dtype=np.float64)

        i_indices = self._pairs[:, 0]
        j_indices = self._pairs[:, 1]

        pos_i = positions[i_indices]
        pos_j = positions[j_indices]

        dr = box.minimum_image(pos_i, pos_j)
        return np.linalg.norm(dr, axis=1)
