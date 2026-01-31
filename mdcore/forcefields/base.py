"""Base interface for force providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..neighborlists import NeighborList
    from ..system import MDState


class ForceProvider(ABC):
    """
    Abstract base class for all force computation modules.

    This is the most important abstraction in the MD engine.
    Everything that computes forces implements this interface:
    - Classical FF terms
    - ML potentials
    - Hybrid models
    """

    @abstractmethod
    def compute(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> NDArray[np.floating]:
        """
        Compute forces on all atoms.

        Args:
            state: Current MD state containing positions, box, etc.
            neighbors: Optional neighbor list for nonbonded interactions.

        Returns:
            Forces array of shape (N, 3).
        """
        ...

    def compute_with_energy(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> tuple[NDArray[np.floating], float]:
        """
        Compute forces and potential energy.

        Default implementation computes forces only; subclasses should
        override for efficiency if energy is needed.

        Args:
            state: Current MD state.
            neighbors: Optional neighbor list.

        Returns:
            Tuple of (forces array, potential energy).
        """
        forces = self.compute(state, neighbors)
        return forces, 0.0  # Subclasses should override with actual energy
