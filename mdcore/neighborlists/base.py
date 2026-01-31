"""Base interface for neighbor lists."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from ..system import Box


class NeighborList(ABC):
    """
    Abstract base class for neighbor list implementations.

    Neighbor lists accelerate the computation of pairwise interactions
    by pre-computing and caching which atom pairs are within a cutoff distance.
    """

    @abstractmethod
    def build(self, positions: ArrayLike, box: Box) -> None:
        """
        Build the neighbor list from scratch.

        Args:
            positions: Atomic positions, shape (N, 3).
            box: Simulation box for periodic boundary conditions.
        """
        ...

    @abstractmethod
    def update_if_needed(self, positions: ArrayLike) -> bool:
        """
        Update neighbor list if atoms have moved significantly.

        Args:
            positions: Current atomic positions, shape (N, 3).

        Returns:
            True if the list was rebuilt, False if it was still valid.
        """
        ...

    @abstractmethod
    def get_pairs(self) -> NDArray[np.integer]:
        """
        Get all neighbor pairs.

        Returns:
            Array of shape (N_pairs, 2) containing (i, j) indices
            where i < j for all pairs.
        """
        ...

    @abstractmethod
    def get_neighbors(self, atom_index: int) -> NDArray[np.integer]:
        """
        Get neighbors of a specific atom.

        Args:
            atom_index: Index of the atom to query.

        Returns:
            Array of neighbor atom indices.
        """
        ...

    @property
    @abstractmethod
    def n_pairs(self) -> int:
        """Return the number of neighbor pairs."""
        ...

    @property
    @abstractmethod
    def cutoff(self) -> float:
        """Return the cutoff distance."""
        ...
