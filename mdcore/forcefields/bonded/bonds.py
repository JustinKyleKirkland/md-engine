"""Harmonic bond force implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..base import ForceProvider

if TYPE_CHECKING:
    from ...neighborlists import NeighborList
    from ...system import MDState
    from ...topology import Topology


class HarmonicBondForce(ForceProvider):
    """
    Harmonic bond stretching force.

    V(r) = 0.5 * k * (r - r0)^2

    Attributes:
        bond_indices: Bond atom pairs, shape (N_bonds, 2).
        force_constants: Spring constants k, shape (N_bonds,).
        equilibrium_lengths: Equilibrium distances r0, shape (N_bonds,).
    """

    def __init__(
        self,
        bond_indices: ArrayLike,
        force_constants: ArrayLike,
        equilibrium_lengths: ArrayLike,
    ) -> None:
        """
        Initialize harmonic bond force.

        Args:
            bond_indices: Bond atom pairs, shape (N_bonds, 2).
            force_constants: Spring constants k, shape (N_bonds,).
            equilibrium_lengths: Equilibrium distances r0, shape (N_bonds,).
        """
        self.bond_indices = np.asarray(bond_indices, dtype=np.int32).reshape(-1, 2)
        self.force_constants = np.asarray(force_constants, dtype=np.float64)
        self.equilibrium_lengths = np.asarray(equilibrium_lengths, dtype=np.float64)

        n_bonds = len(self.bond_indices)
        if len(self.force_constants) != n_bonds:
            raise ValueError(
                f"force_constants length {len(self.force_constants)} != "
                f"number of bonds {n_bonds}"
            )
        if len(self.equilibrium_lengths) != n_bonds:
            raise ValueError(
                f"equilibrium_lengths length {len(self.equilibrium_lengths)} != "
                f"number of bonds {n_bonds}"
            )

    @classmethod
    def from_topology(
        cls,
        topology: Topology,
        force_constants: ArrayLike,
        equilibrium_lengths: ArrayLike,
    ) -> HarmonicBondForce:
        """
        Create from topology bond list.

        Args:
            topology: System topology.
            force_constants: Spring constants, one per bond or single value.
            equilibrium_lengths: Equilibrium distances, one per bond or single value.

        Returns:
            HarmonicBondForce instance.
        """
        n_bonds = topology.n_bonds

        force_constants = np.asarray(force_constants, dtype=np.float64)
        equilibrium_lengths = np.asarray(equilibrium_lengths, dtype=np.float64)

        # Broadcast single values
        if force_constants.ndim == 0:
            force_constants = np.full(n_bonds, force_constants)
        if equilibrium_lengths.ndim == 0:
            equilibrium_lengths = np.full(n_bonds, equilibrium_lengths)

        return cls(topology.bonds, force_constants, equilibrium_lengths)

    def compute(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> NDArray[np.floating]:
        """Compute bond forces."""
        forces = np.zeros((state.n_atoms, 3), dtype=np.float64)

        if len(self.bond_indices) == 0:
            return forces

        # Get positions of bonded atoms
        i_indices = self.bond_indices[:, 0]
        j_indices = self.bond_indices[:, 1]

        pos_i = state.positions[i_indices]
        pos_j = state.positions[j_indices]

        # Compute displacement vectors with minimum image convention
        dr = state.box.minimum_image(pos_i, pos_j)

        # Compute distances
        r = np.linalg.norm(dr, axis=1)

        # Avoid division by zero
        r = np.maximum(r, 1e-10)

        # Compute force magnitude: F = -k * (r - r0)
        # Force on j from i: F_ij = -k * (r - r0) * (r_j - r_i) / r
        force_mag = -self.force_constants * (r - self.equilibrium_lengths)

        # Unit vectors from i to j
        unit_dr = dr / r[:, np.newaxis]

        # Forces on each atom
        force_vectors = force_mag[:, np.newaxis] * unit_dr

        # Accumulate forces
        np.add.at(forces, j_indices, force_vectors)
        np.add.at(forces, i_indices, -force_vectors)

        return forces

    def compute_with_energy(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> tuple[NDArray[np.floating], float]:
        """Compute bond forces and potential energy."""
        forces = np.zeros((state.n_atoms, 3), dtype=np.float64)
        energy = 0.0

        if len(self.bond_indices) == 0:
            return forces, energy

        # Get positions of bonded atoms
        i_indices = self.bond_indices[:, 0]
        j_indices = self.bond_indices[:, 1]

        pos_i = state.positions[i_indices]
        pos_j = state.positions[j_indices]

        # Compute displacement vectors with minimum image convention
        dr = state.box.minimum_image(pos_i, pos_j)

        # Compute distances
        r = np.linalg.norm(dr, axis=1)

        # Avoid division by zero
        r_safe = np.maximum(r, 1e-10)

        # Compute energy: V = 0.5 * k * (r - r0)^2
        delta_r = r - self.equilibrium_lengths
        energy = float(0.5 * np.sum(self.force_constants * delta_r**2))

        # Compute force magnitude: F = -k * (r - r0)
        force_mag = -self.force_constants * delta_r

        # Unit vectors from i to j
        unit_dr = dr / r_safe[:, np.newaxis]

        # Forces on each atom
        force_vectors = force_mag[:, np.newaxis] * unit_dr

        # Accumulate forces
        np.add.at(forces, j_indices, force_vectors)
        np.add.at(forces, i_indices, -force_vectors)

        return forces, energy
