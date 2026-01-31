"""Lennard-Jones force implementation."""

from __future__ import annotations

from collections.abc import Set
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..base import ForceProvider

if TYPE_CHECKING:
    from ...neighborlists import NeighborList
    from ...system import MDState


class LennardJonesForce(ForceProvider):
    """
    Lennard-Jones 12-6 potential.

    V(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]

    Attributes:
        epsilon: Well depth per atom type, shape (n_types,).
        sigma: Size parameter per atom type, shape (n_types,).
        atom_types: Atom type index for each atom, shape (N,).
        cutoff: Cutoff distance.
        exclusions: Set of excluded atom pairs (i, j) where i < j.
    """

    def __init__(
        self,
        epsilon: ArrayLike,
        sigma: ArrayLike,
        atom_types: ArrayLike,
        cutoff: float = 1.0,
        exclusions: Set[tuple[int, int]] | None = None,
    ) -> None:
        """
        Initialize Lennard-Jones force.

        Args:
            epsilon: Well depth per atom type, shape (n_types,).
            sigma: Size parameter per atom type, shape (n_types,).
            atom_types: Atom type index for each atom, shape (N,).
            cutoff: Cutoff distance for interactions.
            exclusions: Set of excluded atom pairs.
        """
        self.epsilon = np.asarray(epsilon, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.atom_types = np.asarray(atom_types, dtype=np.int32)
        self.cutoff = cutoff
        self.exclusions = exclusions if exclusions is not None else set()

        if len(self.epsilon) != len(self.sigma):
            raise ValueError(
                f"epsilon length {len(self.epsilon)} != sigma length {len(self.sigma)}"
            )

    def _get_pair_params(
        self, type_i: NDArray[np.integer], type_j: NDArray[np.integer]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Get Lennard-Jones parameters for pairs using Lorentz-Berthelot combining rules.

        Args:
            type_i: Atom types for first atoms in pairs.
            type_j: Atom types for second atoms in pairs.

        Returns:
            Tuple of (epsilon_ij, sigma_ij) arrays.
        """
        eps_i = self.epsilon[type_i]
        eps_j = self.epsilon[type_j]
        sig_i = self.sigma[type_i]
        sig_j = self.sigma[type_j]

        # Lorentz-Berthelot combining rules
        epsilon_ij = np.sqrt(eps_i * eps_j)
        sigma_ij = 0.5 * (sig_i + sig_j)

        return epsilon_ij, sigma_ij

    def compute(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> NDArray[np.floating]:
        """
        Compute Lennard-Jones forces.

        Args:
            state: Current MD state.
            neighbors: Neighbor list (required for efficiency).

        Returns:
            Forces array of shape (N, 3).
        """
        forces = np.zeros((state.n_atoms, 3), dtype=np.float64)

        if neighbors is not None:
            # Use neighbor list for efficient computation
            pairs = neighbors.get_pairs()
            if len(pairs) == 0:
                return forces

            i_indices = pairs[:, 0]
            j_indices = pairs[:, 1]
        else:
            # Brute force all pairs (N^2)
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

        # Get pair parameters
        type_i = self.atom_types[i_indices]
        type_j = self.atom_types[j_indices]
        epsilon_ij, sigma_ij = self._get_pair_params(type_i, type_j)

        # Avoid division by zero
        r_safe = np.maximum(r, 1e-10)

        # Compute LJ force magnitude
        # F = -dV/dr = 24 * epsilon * [2*(sigma/r)^12 - (sigma/r)^6] / r
        sig_over_r = sigma_ij / r_safe
        sig_over_r_6 = sig_over_r**6
        sig_over_r_12 = sig_over_r_6**2

        force_mag = 24.0 * epsilon_ij * (2.0 * sig_over_r_12 - sig_over_r_6) / r_safe

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
        """Compute Lennard-Jones forces and potential energy."""
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

        # Get pair parameters
        type_i = self.atom_types[i_indices]
        type_j = self.atom_types[j_indices]
        epsilon_ij, sigma_ij = self._get_pair_params(type_i, type_j)

        # Avoid division by zero
        r_safe = np.maximum(r, 1e-10)

        # Compute LJ terms
        sig_over_r = sigma_ij / r_safe
        sig_over_r_6 = sig_over_r**6
        sig_over_r_12 = sig_over_r_6**2

        # Energy: V = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
        energy = float(4.0 * np.sum(epsilon_ij * (sig_over_r_12 - sig_over_r_6)))

        # Force magnitude
        force_mag = 24.0 * epsilon_ij * (2.0 * sig_over_r_12 - sig_over_r_6) / r_safe

        # Force vectors
        unit_dr = dr / r_safe[:, np.newaxis]
        force_vectors = force_mag[:, np.newaxis] * unit_dr

        # Newton's third law
        np.add.at(forces, j_indices, force_vectors)
        np.add.at(forces, i_indices, -force_vectors)

        return forces, energy
