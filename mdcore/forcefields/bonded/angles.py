"""Harmonic angle force implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..base import ForceProvider

if TYPE_CHECKING:
    from ...neighborlists import NeighborList
    from ...system import MDState
    from ...topology import Topology


class HarmonicAngleForce(ForceProvider):
    """
    Harmonic angle bending force.

    V(theta) = 0.5 * k * (theta - theta0)^2

    where theta is the angle i-j-k (j is the central atom).

    Attributes:
        angle_indices: Angle atom triplets (i, j, k), shape (N_angles, 3).
        force_constants: Spring constants k, shape (N_angles,).
        equilibrium_angles: Equilibrium angles theta0 in radians, shape (N_angles,).
    """

    def __init__(
        self,
        angle_indices: ArrayLike,
        force_constants: ArrayLike,
        equilibrium_angles: ArrayLike,
    ) -> None:
        """
        Initialize harmonic angle force.

        Args:
            angle_indices: Angle atom triplets (i, j, k), shape (N_angles, 3).
            force_constants: Spring constants k, shape (N_angles,).
            equilibrium_angles: Equilibrium angles theta0 in radians, shape (N_angles,).
        """
        self.angle_indices = np.asarray(angle_indices, dtype=np.int32).reshape(-1, 3)
        self.force_constants = np.asarray(force_constants, dtype=np.float64)
        self.equilibrium_angles = np.asarray(equilibrium_angles, dtype=np.float64)

        n_angles = len(self.angle_indices)
        if len(self.force_constants) != n_angles:
            raise ValueError(
                f"force_constants length {len(self.force_constants)} != "
                f"number of angles {n_angles}"
            )
        if len(self.equilibrium_angles) != n_angles:
            raise ValueError(
                f"equilibrium_angles length {len(self.equilibrium_angles)} != "
                f"number of angles {n_angles}"
            )

    @classmethod
    def from_topology(
        cls,
        topology: Topology,
        force_constants: ArrayLike,
        equilibrium_angles: ArrayLike,
    ) -> HarmonicAngleForce:
        """
        Create from topology angle list.

        Args:
            topology: System topology.
            force_constants: Spring constants, one per angle or single value.
            equilibrium_angles: Equilibrium angles in radians, one per angle or single value.

        Returns:
            HarmonicAngleForce instance.
        """
        n_angles = topology.n_angles

        force_constants = np.asarray(force_constants, dtype=np.float64)
        equilibrium_angles = np.asarray(equilibrium_angles, dtype=np.float64)

        # Broadcast single values
        if force_constants.ndim == 0:
            force_constants = np.full(n_angles, force_constants)
        if equilibrium_angles.ndim == 0:
            equilibrium_angles = np.full(n_angles, equilibrium_angles)

        return cls(topology.angles, force_constants, equilibrium_angles)

    def compute(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> NDArray[np.floating]:
        """Compute angle forces."""
        forces = np.zeros((state.n_atoms, 3), dtype=np.float64)

        if len(self.angle_indices) == 0:
            return forces

        # Get atom indices
        i_indices = self.angle_indices[:, 0]
        j_indices = self.angle_indices[:, 1]  # Central atom
        k_indices = self.angle_indices[:, 2]

        # Get positions
        pos_i = state.positions[i_indices]
        pos_j = state.positions[j_indices]
        pos_k = state.positions[k_indices]

        # Compute vectors from central atom j
        r_ji = state.box.minimum_image(pos_j, pos_i)  # j -> i
        r_jk = state.box.minimum_image(pos_j, pos_k)  # j -> k

        # Compute distances
        d_ji = np.linalg.norm(r_ji, axis=1)
        d_jk = np.linalg.norm(r_jk, axis=1)

        # Avoid division by zero
        d_ji = np.maximum(d_ji, 1e-10)
        d_jk = np.maximum(d_jk, 1e-10)

        # Compute angle using dot product
        cos_theta = np.sum(r_ji * r_jk, axis=1) / (d_ji * d_jk)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # Compute torque coefficient: dV/dtheta = k * (theta - theta0)
        delta_theta = theta - self.equilibrium_angles
        torque = self.force_constants * delta_theta

        # Compute sin(theta) for force calculation
        sin_theta = np.sin(theta)
        sin_theta = np.maximum(sin_theta, 1e-10)  # Avoid division by zero

        # Force magnitude factor
        factor = -torque / sin_theta

        # Unit vectors
        r_ji_hat = r_ji / d_ji[:, np.newaxis]
        r_jk_hat = r_jk / d_jk[:, np.newaxis]

        # Force on atom i: perpendicular to r_ji in the i-j-k plane
        # F_i = factor * (cos(theta) * r_ji_hat - r_jk_hat) / d_ji
        f_i = (
            factor[:, np.newaxis]
            * (cos_theta[:, np.newaxis] * r_ji_hat - r_jk_hat)
            / d_ji[:, np.newaxis]
        )

        # Force on atom k: perpendicular to r_jk in the i-j-k plane
        # F_k = factor * (cos(theta) * r_jk_hat - r_ji_hat) / d_jk
        f_k = (
            factor[:, np.newaxis]
            * (cos_theta[:, np.newaxis] * r_jk_hat - r_ji_hat)
            / d_jk[:, np.newaxis]
        )

        # Force on central atom j: Newton's third law
        f_j = -(f_i + f_k)

        # Accumulate forces
        np.add.at(forces, i_indices, f_i)
        np.add.at(forces, j_indices, f_j)
        np.add.at(forces, k_indices, f_k)

        return forces

    def compute_with_energy(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> tuple[NDArray[np.floating], float]:
        """Compute angle forces and potential energy."""
        forces = np.zeros((state.n_atoms, 3), dtype=np.float64)
        energy = 0.0

        if len(self.angle_indices) == 0:
            return forces, energy

        # Get atom indices
        i_indices = self.angle_indices[:, 0]
        j_indices = self.angle_indices[:, 1]  # Central atom
        k_indices = self.angle_indices[:, 2]

        # Get positions
        pos_i = state.positions[i_indices]
        pos_j = state.positions[j_indices]
        pos_k = state.positions[k_indices]

        # Compute vectors from central atom j
        r_ji = state.box.minimum_image(pos_j, pos_i)
        r_jk = state.box.minimum_image(pos_j, pos_k)

        # Compute distances
        d_ji = np.linalg.norm(r_ji, axis=1)
        d_jk = np.linalg.norm(r_jk, axis=1)

        # Avoid division by zero
        d_ji = np.maximum(d_ji, 1e-10)
        d_jk = np.maximum(d_jk, 1e-10)

        # Compute angle
        cos_theta = np.sum(r_ji * r_jk, axis=1) / (d_ji * d_jk)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # Compute energy: V = 0.5 * k * (theta - theta0)^2
        delta_theta = theta - self.equilibrium_angles
        energy = float(0.5 * np.sum(self.force_constants * delta_theta**2))

        # Compute torque coefficient
        torque = self.force_constants * delta_theta

        # Compute sin(theta)
        sin_theta = np.sin(theta)
        sin_theta = np.maximum(sin_theta, 1e-10)

        # Force magnitude factor
        factor = -torque / sin_theta

        # Unit vectors
        r_ji_hat = r_ji / d_ji[:, np.newaxis]
        r_jk_hat = r_jk / d_jk[:, np.newaxis]

        # Forces
        f_i = (
            factor[:, np.newaxis]
            * (cos_theta[:, np.newaxis] * r_ji_hat - r_jk_hat)
            / d_ji[:, np.newaxis]
        )
        f_k = (
            factor[:, np.newaxis]
            * (cos_theta[:, np.newaxis] * r_jk_hat - r_ji_hat)
            / d_jk[:, np.newaxis]
        )
        f_j = -(f_i + f_k)

        # Accumulate forces
        np.add.at(forces, i_indices, f_i)
        np.add.at(forces, j_indices, f_j)
        np.add.at(forces, k_indices, f_k)

        return forces, energy
