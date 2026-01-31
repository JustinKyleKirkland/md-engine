"""Composite force field combining multiple force providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .base import ForceProvider

if TYPE_CHECKING:
    from ..neighborlists import NeighborList
    from ..system import MDState


class ForceField(ForceProvider):
    """
    Composite force field combining multiple force providers.

    Implements the composite pattern: a ForceField is itself a ForceProvider
    that aggregates contributions from multiple terms.

    Example:
        ff = ForceField([
            HarmonicBondForce(topology, bond_params),
            LennardJonesForce(lj_params),
            CoulombForce(charges),
        ])
        forces = ff.compute(state, neighbors)
    """

    def __init__(self, terms: list[ForceProvider] | None = None) -> None:
        """
        Initialize composite force field.

        Args:
            terms: List of force providers to combine.
        """
        self.terms: list[ForceProvider] = terms if terms is not None else []

    def add_term(self, term: ForceProvider) -> None:
        """Add a force term to the force field."""
        self.terms.append(term)

    def remove_term(self, term: ForceProvider) -> None:
        """Remove a force term from the force field."""
        self.terms.remove(term)

    def compute(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> NDArray[np.floating]:
        """
        Compute total forces from all terms.

        Args:
            state: Current MD state.
            neighbors: Optional neighbor list for nonbonded terms.

        Returns:
            Total forces array of shape (N, 3).
        """
        total_forces = np.zeros((state.n_atoms, 3), dtype=np.float64)

        for term in self.terms:
            total_forces += term.compute(state, neighbors)

        return total_forces

    def compute_with_energy(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> tuple[NDArray[np.floating], float]:
        """
        Compute total forces and potential energy from all terms.

        Args:
            state: Current MD state.
            neighbors: Optional neighbor list for nonbonded terms.

        Returns:
            Tuple of (total forces array, total potential energy).
        """
        total_forces = np.zeros((state.n_atoms, 3), dtype=np.float64)
        total_energy = 0.0

        for term in self.terms:
            forces, energy = term.compute_with_energy(state, neighbors)
            total_forces += forces
            total_energy += energy

        return total_forces, total_energy

    def compute_per_term(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> list[tuple[NDArray[np.floating], float]]:
        """
        Compute forces and energies from each term separately.

        Useful for debugging and analysis.

        Args:
            state: Current MD state.
            neighbors: Optional neighbor list for nonbonded terms.

        Returns:
            List of (forces, energy) tuples, one per term.
        """
        results = []
        for term in self.terms:
            forces, energy = term.compute_with_energy(state, neighbors)
            results.append((forces, energy))
        return results
