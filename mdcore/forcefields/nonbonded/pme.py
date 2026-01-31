"""Particle Mesh Ewald (PME) electrostatics implementation."""

from __future__ import annotations

from collections.abc import Set
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..base import ForceProvider
from .coulomb import COULOMB_CONSTANT

if TYPE_CHECKING:
    from ...neighborlists import NeighborList
    from ...system import MDState


class PMEForce(ForceProvider):
    """
    Particle Mesh Ewald for long-range electrostatics.

    Splits the Coulomb sum into:
    - Short-range real space contribution (handled with error function)
    - Long-range reciprocal space contribution (handled with FFT)
    - Self-energy correction

    This is a simplified implementation. Production codes use optimized FFT
    libraries and more sophisticated charge spreading/interpolation.

    Attributes:
        charges: Atomic charges, shape (N,).
        alpha: Ewald splitting parameter.
        real_cutoff: Real-space cutoff.
        kmax: Maximum k-vectors in each dimension.
        exclusions: Set of excluded atom pairs.
    """

    def __init__(
        self,
        charges: ArrayLike,
        alpha: float = 0.3,
        real_cutoff: float = 1.0,
        kmax: int = 5,
        exclusions: Set[tuple[int, int]] | None = None,
        coulomb_constant: float = COULOMB_CONSTANT,
    ) -> None:
        """
        Initialize PME force.

        Args:
            charges: Atomic charges in elementary charge units, shape (N,).
            alpha: Ewald splitting parameter (1/nm).
            real_cutoff: Cutoff for real-space interactions.
            kmax: Maximum k-vector index in each dimension.
            exclusions: Set of excluded atom pairs.
            coulomb_constant: Coulomb constant in appropriate units.
        """
        self.charges = np.asarray(charges, dtype=np.float64)
        self.alpha = alpha
        self.real_cutoff = real_cutoff
        self.kmax = kmax
        self.exclusions = exclusions if exclusions is not None else set()
        self.coulomb_constant = coulomb_constant

    def _compute_real_space(
        self,
        state: MDState,
        neighbors: NeighborList | None,
    ) -> tuple[NDArray[np.floating], float]:
        """Compute real-space contribution to Ewald sum."""
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
        mask = r < self.real_cutoff
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
        alpha_r = self.alpha * r_safe

        # Compute erfc and its derivative
        from scipy.special import erfc

        erfc_ar = erfc(alpha_r)
        exp_ar2 = np.exp(-(alpha_r**2))

        # Real-space energy: V = k_e * q_i * q_j * erfc(alpha*r) / r
        energy = float(self.coulomb_constant * np.sum(q_i * q_j * erfc_ar / r_safe))

        # Real-space force
        # F = -dV/dr = k_e * q_i * q_j * [erfc(alpha*r)/r^2 + 2*alpha/sqrt(pi)*exp(-alpha^2*r^2)/r]
        two_alpha_sqrt_pi = 2.0 * self.alpha / np.sqrt(np.pi)
        force_mag = (
            self.coulomb_constant
            * q_i
            * q_j
            * (erfc_ar / r_safe**2 + two_alpha_sqrt_pi * exp_ar2 / r_safe)
        )

        # Force vectors
        unit_dr = dr / r_safe[:, np.newaxis]
        force_vectors = force_mag[:, np.newaxis] * unit_dr

        # Newton's third law
        np.add.at(forces, j_indices, force_vectors)
        np.add.at(forces, i_indices, -force_vectors)

        return forces, energy

    def _compute_reciprocal_space(
        self,
        state: MDState,
    ) -> tuple[NDArray[np.floating], float]:
        """Compute reciprocal-space contribution to Ewald sum."""
        forces = np.zeros((state.n_atoms, 3), dtype=np.float64)
        energy = 0.0

        if not state.box.is_orthorhombic:
            raise NotImplementedError("PME only supports orthorhombic boxes currently")

        box_lengths = np.diag(state.box.vectors)
        volume = state.box.volume

        # Generate k-vectors
        k_vectors = []
        for nx in range(-self.kmax, self.kmax + 1):
            for ny in range(-self.kmax, self.kmax + 1):
                for nz in range(-self.kmax, self.kmax + 1):
                    if nx == 0 and ny == 0 and nz == 0:
                        continue
                    k = 2.0 * np.pi * np.array([nx, ny, nz]) / box_lengths
                    k_vectors.append(k)

        if not k_vectors:
            return forces, energy

        k_vectors = np.array(k_vectors)
        k_sq = np.sum(k_vectors**2, axis=1)

        # Precompute structure factors
        # S(k) = sum_j q_j * exp(i * k . r_j)
        positions = state.positions
        charges = self.charges

        # k . r for all k and all atoms: shape (n_k, n_atoms)
        k_dot_r = k_vectors @ positions.T

        # Structure factor
        cos_kr = np.cos(k_dot_r)  # (n_k, n_atoms)
        sin_kr = np.sin(k_dot_r)  # (n_k, n_atoms)

        S_real = cos_kr @ charges  # (n_k,)
        S_imag = sin_kr @ charges  # (n_k,)
        S_sq = S_real**2 + S_imag**2  # (n_k,)

        # Reciprocal energy
        # V_k = (2*pi*k_e/V) * sum_k exp(-k^2/(4*alpha^2)) / k^2 * |S(k)|^2
        prefactor = 2.0 * np.pi * self.coulomb_constant / volume
        exp_factor = np.exp(-k_sq / (4.0 * self.alpha**2))

        energy = float(prefactor * np.sum(exp_factor / k_sq * S_sq))

        # Reciprocal forces
        # F_i = -(4*pi*k_e/V) * q_i * sum_k exp(-k^2/(4*alpha^2)) / k^2 * k * Im[S(k) * exp(-i*k.r_i)]
        for i in range(state.n_atoms):
            # Im[S(k) * exp(-i*k.r_i)] = S_real * sin(k.r_i) - S_imag * cos(k.r_i)
            im_part = S_real * sin_kr[:, i] - S_imag * cos_kr[:, i]
            force_factor = -2.0 * prefactor * charges[i] * exp_factor / k_sq * im_part
            forces[i] = np.sum(force_factor[:, np.newaxis] * k_vectors, axis=0)

        return forces, energy

    def _compute_self_energy(self, state: MDState) -> float:
        """Compute self-energy correction."""
        # Self energy: V_self = -k_e * alpha / sqrt(pi) * sum_i q_i^2
        return float(
            -self.coulomb_constant
            * self.alpha
            / np.sqrt(np.pi)
            * np.sum(self.charges**2)
        )

    def compute(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> NDArray[np.floating]:
        """Compute PME forces."""
        forces_real, _ = self._compute_real_space(state, neighbors)
        forces_recip, _ = self._compute_reciprocal_space(state)
        return forces_real + forces_recip

    def compute_with_energy(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> tuple[NDArray[np.floating], float]:
        """Compute PME forces and potential energy."""
        forces_real, energy_real = self._compute_real_space(state, neighbors)
        forces_recip, energy_recip = self._compute_reciprocal_space(state)
        energy_self = self._compute_self_energy(state)

        total_forces = forces_real + forces_recip
        total_energy = energy_real + energy_recip + energy_self

        return total_forces, total_energy
