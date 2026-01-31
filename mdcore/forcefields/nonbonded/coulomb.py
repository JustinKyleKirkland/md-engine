"""Coulomb electrostatic force implementation."""

from __future__ import annotations

from collections.abc import Set
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..base import ForceProvider

if TYPE_CHECKING:
    from ...neighborlists import NeighborList
    from ...system import MDState


# Coulomb constant in MD units (kJ*nm / (mol*e^2))
# This is 1/(4*pi*epsilon_0) in MD units
COULOMB_CONSTANT = 138.935458  # kJ*nm/(mol*e^2)


class CoulombForce(ForceProvider):
    """
    Direct Coulomb electrostatic interaction.

    V(r) = k_e * q_i * q_j / r

    where k_e is Coulomb's constant.

    Note: For periodic systems with long-range electrostatics, use PME instead.
    This implementation is for cutoff-based electrostatics or small systems.

    Attributes:
        charges: Atomic charges, shape (N,).
        cutoff: Cutoff distance.
        exclusions: Set of excluded atom pairs.
    """

    def __init__(
        self,
        charges: ArrayLike,
        cutoff: float = 1.0,
        exclusions: Set[tuple[int, int]] | None = None,
        coulomb_constant: float = COULOMB_CONSTANT,
    ) -> None:
        """
        Initialize Coulomb force.

        Args:
            charges: Atomic charges in elementary charge units, shape (N,).
            cutoff: Cutoff distance for interactions.
            exclusions: Set of excluded atom pairs.
            coulomb_constant: Coulomb constant in appropriate units.
        """
        self.charges = np.asarray(charges, dtype=np.float64)
        self.cutoff = cutoff
        self.exclusions = exclusions if exclusions is not None else set()
        self.coulomb_constant = coulomb_constant

    def compute(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> NDArray[np.floating]:
        """
        Compute Coulomb forces.

        Args:
            state: Current MD state.
            neighbors: Neighbor list (required for efficiency).

        Returns:
            Forces array of shape (N, 3).
        """
        forces = np.zeros((state.n_atoms, 3), dtype=np.float64)

        if neighbors is not None:
            pairs = neighbors.get_pairs()
            if len(pairs) == 0:
                return forces

            i_indices = pairs[:, 0]
            j_indices = pairs[:, 1]
        else:
            n = state.n_atoms
            i_indices, j_indices = np.triu_indices(n, k=1)

        # Filter excluded pairs
        if self.exclusions:
            mask = np.array(
                [
                    (min(i, j), max(i, j)) not in self.exclusions
                    for i, j in zip(i_indices, j_indices)
                ]
            )
            i_indices = i_indices[mask]
            j_indices = j_indices[mask]

        if len(i_indices) == 0:
            return forces

        # Get positions and compute displacements
        pos_i = state.positions[i_indices]
        pos_j = state.positions[j_indices]
        dr = state.box.minimum_image(pos_i, pos_j)
        r = np.linalg.norm(dr, axis=1)

        # Apply cutoff
        mask = r < self.cutoff
        if not np.any(mask):
            return forces

        i_indices = i_indices[mask]
        j_indices = j_indices[mask]
        dr = dr[mask]
        r = r[mask]

        # Get charges
        q_i = self.charges[i_indices]
        q_j = self.charges[j_indices]

        # Avoid division by zero
        r_safe = np.maximum(r, 1e-10)

        # Compute force magnitude
        # F = -dV/dr = k_e * q_i * q_j / r^2
        # Force points from i to j if charges have same sign (repulsive)
        force_mag = self.coulomb_constant * q_i * q_j / (r_safe**2)

        # Force vectors (pointing from i to j)
        unit_dr = dr / r_safe[:, np.newaxis]
        force_vectors = force_mag[:, np.newaxis] * unit_dr

        # Newton's third law
        np.add.at(forces, j_indices, force_vectors)
        np.add.at(forces, i_indices, -force_vectors)

        return forces

    def compute_with_energy(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> tuple[NDArray[np.floating], float]:
        """Compute Coulomb forces and potential energy."""
        forces = np.zeros((state.n_atoms, 3), dtype=np.float64)
        energy = 0.0

        if neighbors is not None:
            pairs = neighbors.get_pairs()
            if len(pairs) == 0:
                return forces, energy

            i_indices = pairs[:, 0]
            j_indices = pairs[:, 1]
        else:
            n = state.n_atoms
            i_indices, j_indices = np.triu_indices(n, k=1)

        # Filter excluded pairs
        if self.exclusions:
            mask = np.array(
                [
                    (min(i, j), max(i, j)) not in self.exclusions
                    for i, j in zip(i_indices, j_indices)
                ]
            )
            i_indices = i_indices[mask]
            j_indices = j_indices[mask]

        if len(i_indices) == 0:
            return forces, energy

        # Get positions and compute displacements
        pos_i = state.positions[i_indices]
        pos_j = state.positions[j_indices]
        dr = state.box.minimum_image(pos_i, pos_j)
        r = np.linalg.norm(dr, axis=1)

        # Apply cutoff
        mask = r < self.cutoff
        if not np.any(mask):
            return forces, energy

        i_indices = i_indices[mask]
        j_indices = j_indices[mask]
        dr = dr[mask]
        r = r[mask]

        # Get charges
        q_i = self.charges[i_indices]
        q_j = self.charges[j_indices]

        # Avoid division by zero
        r_safe = np.maximum(r, 1e-10)

        # Compute energy: V = k_e * q_i * q_j / r
        energy = float(self.coulomb_constant * np.sum(q_i * q_j / r_safe))

        # Compute force magnitude
        force_mag = self.coulomb_constant * q_i * q_j / (r_safe**2)

        # Force vectors
        unit_dr = dr / r_safe[:, np.newaxis]
        force_vectors = force_mag[:, np.newaxis] * unit_dr

        # Newton's third law
        np.add.at(forces, j_indices, force_vectors)
        np.add.at(forces, i_indices, -force_vectors)

        return forces, energy
