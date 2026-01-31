"""Periodic dihedral force implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..base import ForceProvider

if TYPE_CHECKING:
    from ...neighborlists import NeighborList
    from ...system import MDState
    from ...topology import Topology


class PeriodicDihedralForce(ForceProvider):
    """
    Periodic dihedral (torsion) force.

    V(phi) = k * (1 + cos(n*phi - phase))

    where phi is the dihedral angle i-j-k-l.

    Attributes:
        dihedral_indices: Dihedral atom quads (i, j, k, l), shape (N_dihedrals, 4).
        force_constants: Force constants k, shape (N_dihedrals,).
        periodicities: Periodicity n, shape (N_dihedrals,).
        phases: Phase shifts, shape (N_dihedrals,).
    """

    def __init__(
        self,
        dihedral_indices: ArrayLike,
        force_constants: ArrayLike,
        periodicities: ArrayLike,
        phases: ArrayLike,
    ) -> None:
        """
        Initialize periodic dihedral force.

        Args:
            dihedral_indices: Dihedral atom quads (i, j, k, l), shape (N_dihedrals, 4).
            force_constants: Force constants k, shape (N_dihedrals,).
            periodicities: Periodicity n (integer), shape (N_dihedrals,).
            phases: Phase shifts in radians, shape (N_dihedrals,).
        """
        self.dihedral_indices = np.asarray(dihedral_indices, dtype=np.int32).reshape(
            -1, 4
        )
        self.force_constants = np.asarray(force_constants, dtype=np.float64)
        self.periodicities = np.asarray(periodicities, dtype=np.int32)
        self.phases = np.asarray(phases, dtype=np.float64)

        n_dihedrals = len(self.dihedral_indices)
        if len(self.force_constants) != n_dihedrals:
            raise ValueError(
                f"force_constants length {len(self.force_constants)} != "
                f"number of dihedrals {n_dihedrals}"
            )
        if len(self.periodicities) != n_dihedrals:
            raise ValueError(
                f"periodicities length {len(self.periodicities)} != "
                f"number of dihedrals {n_dihedrals}"
            )
        if len(self.phases) != n_dihedrals:
            raise ValueError(
                f"phases length {len(self.phases)} != number of dihedrals {n_dihedrals}"
            )

    @classmethod
    def from_topology(
        cls,
        topology: Topology,
        force_constants: ArrayLike,
        periodicities: ArrayLike,
        phases: ArrayLike,
    ) -> PeriodicDihedralForce:
        """
        Create from topology dihedral list.

        Args:
            topology: System topology.
            force_constants: Force constants, one per dihedral or single value.
            periodicities: Periodicities, one per dihedral or single value.
            phases: Phase shifts in radians, one per dihedral or single value.

        Returns:
            PeriodicDihedralForce instance.
        """
        n_dihedrals = topology.n_dihedrals

        force_constants = np.asarray(force_constants, dtype=np.float64)
        periodicities = np.asarray(periodicities, dtype=np.int32)
        phases = np.asarray(phases, dtype=np.float64)

        # Broadcast single values
        if force_constants.ndim == 0:
            force_constants = np.full(n_dihedrals, force_constants)
        if periodicities.ndim == 0:
            periodicities = np.full(n_dihedrals, periodicities, dtype=np.int32)
        if phases.ndim == 0:
            phases = np.full(n_dihedrals, phases)

        return cls(topology.dihedrals, force_constants, periodicities, phases)

    def _compute_dihedral_angle(
        self,
        r_ij: NDArray[np.floating],
        r_jk: NDArray[np.floating],
        r_kl: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Compute dihedral angles from displacement vectors.

        Args:
            r_ij: Vector from i to j, shape (N, 3).
            r_jk: Vector from j to k, shape (N, 3).
            r_kl: Vector from k to l, shape (N, 3).

        Returns:
            Dihedral angles in radians, shape (N,).
        """
        # Normal vectors to the planes
        n1 = np.cross(r_ij, r_jk)
        n2 = np.cross(r_jk, r_kl)

        # Normalize
        n1_norm = np.linalg.norm(n1, axis=1, keepdims=True)
        n2_norm = np.linalg.norm(n2, axis=1, keepdims=True)

        n1_norm = np.maximum(n1_norm, 1e-10)
        n2_norm = np.maximum(n2_norm, 1e-10)

        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        # Compute angle
        cos_phi = np.sum(n1 * n2, axis=1)
        cos_phi = np.clip(cos_phi, -1.0, 1.0)

        # Determine sign using the cross product
        m1 = np.cross(n1, r_jk / np.linalg.norm(r_jk, axis=1, keepdims=True))
        sin_phi = np.sum(m1 * n2, axis=1)

        phi = np.arctan2(sin_phi, cos_phi)

        return phi

    def compute(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> NDArray[np.floating]:
        """Compute dihedral forces."""
        forces = np.zeros((state.n_atoms, 3), dtype=np.float64)

        if len(self.dihedral_indices) == 0:
            return forces

        # Get atom indices
        i_indices = self.dihedral_indices[:, 0]
        j_indices = self.dihedral_indices[:, 1]
        k_indices = self.dihedral_indices[:, 2]
        l_indices = self.dihedral_indices[:, 3]

        # Get positions
        pos_i = state.positions[i_indices]
        pos_j = state.positions[j_indices]
        pos_k = state.positions[k_indices]
        pos_l = state.positions[l_indices]

        # Compute displacement vectors
        r_ij = state.box.minimum_image(pos_i, pos_j)
        r_jk = state.box.minimum_image(pos_j, pos_k)
        r_kl = state.box.minimum_image(pos_k, pos_l)

        # Compute dihedral angle
        phi = self._compute_dihedral_angle(r_ij, r_jk, r_kl)

        # Compute torque: dV/dphi = -k * n * sin(n*phi - phase)
        n = self.periodicities.astype(np.float64)
        torque = -self.force_constants * n * np.sin(n * phi - self.phases)

        # Compute forces using the derivative of the dihedral angle
        # This follows the standard derivation for dihedral forces

        # Cross products
        n1 = np.cross(r_ij, r_jk)
        n2 = np.cross(r_jk, r_kl)

        # Lengths
        r_jk_norm = np.linalg.norm(r_jk, axis=1)
        n1_sq = np.sum(n1 * n1, axis=1)
        n2_sq = np.sum(n2 * n2, axis=1)

        # Avoid division by zero
        r_jk_norm = np.maximum(r_jk_norm, 1e-10)
        n1_sq = np.maximum(n1_sq, 1e-10)
        n2_sq = np.maximum(n2_sq, 1e-10)

        # Force contributions
        f_i = (
            torque[:, np.newaxis] * r_jk_norm[:, np.newaxis] * n1 / n1_sq[:, np.newaxis]
        )
        f_l = (
            -torque[:, np.newaxis]
            * r_jk_norm[:, np.newaxis]
            * n2
            / n2_sq[:, np.newaxis]
        )

        # Projection factors
        r_ij_dot_r_jk = np.sum(r_ij * r_jk, axis=1)
        r_kl_dot_r_jk = np.sum(r_kl * r_jk, axis=1)
        r_jk_sq = r_jk_norm**2

        factor_j = r_ij_dot_r_jk / r_jk_sq
        factor_k = r_kl_dot_r_jk / r_jk_sq

        f_j = -f_i + factor_j[:, np.newaxis] * f_i - factor_k[:, np.newaxis] * f_l
        f_k = -f_l - factor_j[:, np.newaxis] * f_i + factor_k[:, np.newaxis] * f_l

        # Accumulate forces
        np.add.at(forces, i_indices, f_i)
        np.add.at(forces, j_indices, f_j)
        np.add.at(forces, k_indices, f_k)
        np.add.at(forces, l_indices, f_l)

        return forces

    def compute_with_energy(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> tuple[NDArray[np.floating], float]:
        """Compute dihedral forces and potential energy."""
        forces = np.zeros((state.n_atoms, 3), dtype=np.float64)
        energy = 0.0

        if len(self.dihedral_indices) == 0:
            return forces, energy

        # Get atom indices
        i_indices = self.dihedral_indices[:, 0]
        j_indices = self.dihedral_indices[:, 1]
        k_indices = self.dihedral_indices[:, 2]
        l_indices = self.dihedral_indices[:, 3]

        # Get positions
        pos_i = state.positions[i_indices]
        pos_j = state.positions[j_indices]
        pos_k = state.positions[k_indices]
        pos_l = state.positions[l_indices]

        # Compute displacement vectors
        r_ij = state.box.minimum_image(pos_i, pos_j)
        r_jk = state.box.minimum_image(pos_j, pos_k)
        r_kl = state.box.minimum_image(pos_k, pos_l)

        # Compute dihedral angle
        phi = self._compute_dihedral_angle(r_ij, r_jk, r_kl)

        # Compute energy: V = k * (1 + cos(n*phi - phase))
        n = self.periodicities.astype(np.float64)
        energy = float(
            np.sum(self.force_constants * (1.0 + np.cos(n * phi - self.phases)))
        )

        # Compute torque
        torque = -self.force_constants * n * np.sin(n * phi - self.phases)

        # Cross products
        n1 = np.cross(r_ij, r_jk)
        n2 = np.cross(r_jk, r_kl)

        # Lengths
        r_jk_norm = np.linalg.norm(r_jk, axis=1)
        n1_sq = np.sum(n1 * n1, axis=1)
        n2_sq = np.sum(n2 * n2, axis=1)

        # Avoid division by zero
        r_jk_norm = np.maximum(r_jk_norm, 1e-10)
        n1_sq = np.maximum(n1_sq, 1e-10)
        n2_sq = np.maximum(n2_sq, 1e-10)

        # Force contributions
        f_i = (
            torque[:, np.newaxis] * r_jk_norm[:, np.newaxis] * n1 / n1_sq[:, np.newaxis]
        )
        f_l = (
            -torque[:, np.newaxis]
            * r_jk_norm[:, np.newaxis]
            * n2
            / n2_sq[:, np.newaxis]
        )

        # Projection factors
        r_ij_dot_r_jk = np.sum(r_ij * r_jk, axis=1)
        r_kl_dot_r_jk = np.sum(r_kl * r_jk, axis=1)
        r_jk_sq = r_jk_norm**2

        factor_j = r_ij_dot_r_jk / r_jk_sq
        factor_k = r_kl_dot_r_jk / r_jk_sq

        f_j = -f_i + factor_j[:, np.newaxis] * f_i - factor_k[:, np.newaxis] * f_l
        f_k = -f_l - factor_j[:, np.newaxis] * f_i + factor_k[:, np.newaxis] * f_l

        # Accumulate forces
        np.add.at(forces, i_indices, f_i)
        np.add.at(forces, j_indices, f_j)
        np.add.at(forces, k_indices, f_k)
        np.add.at(forces, l_indices, f_l)

        return forces, energy
