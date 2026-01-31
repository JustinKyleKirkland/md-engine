"""Long-range dispersion correction force implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..base import ForceProvider

if TYPE_CHECKING:
    from ...neighborlists import NeighborList
    from ...system import MDState


class DispersionForce(ForceProvider):
    """
    Long-range dispersion correction for Lennard-Jones interactions.

    Computes analytical tail correction for LJ energy and pressure beyond
    the cutoff. This accounts for the missing interactions at distances
    greater than the cutoff.

    The energy correction is:
    V_tail = (8/3) * pi * N * rho * epsilon * sigma^6 *
             [(1/3)*(sigma/rc)^6 - 1] * (sigma/rc)^3

    where rho is the number density, rc is the cutoff, and the sum is
    over all type pairs weighted by their populations.

    Attributes:
        epsilon: LJ well depth per atom type, shape (n_types,).
        sigma: LJ size parameter per atom type, shape (n_types,).
        atom_types: Atom type index for each atom, shape (N,).
        cutoff: LJ cutoff distance (same as used in LJ force).
    """

    def __init__(
        self,
        epsilon: ArrayLike,
        sigma: ArrayLike,
        atom_types: ArrayLike,
        cutoff: float = 1.0,
    ) -> None:
        """
        Initialize dispersion correction.

        Args:
            epsilon: LJ well depth per atom type, shape (n_types,).
            sigma: LJ size parameter per atom type, shape (n_types,).
            atom_types: Atom type index for each atom, shape (N,).
            cutoff: LJ cutoff distance.
        """
        self.epsilon = np.asarray(epsilon, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.atom_types = np.asarray(atom_types, dtype=np.int32)
        self.cutoff = cutoff

        if len(self.epsilon) != len(self.sigma):
            raise ValueError(
                f"epsilon length {len(self.epsilon)} != sigma length {len(self.sigma)}"
            )

    def _get_type_counts(self) -> NDArray[np.integer]:
        """Count atoms of each type."""
        n_types = len(self.epsilon)
        counts = np.zeros(n_types, dtype=np.int32)
        for t in self.atom_types:
            counts[t] += 1
        return counts

    def compute(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> NDArray[np.floating]:
        """
        Compute dispersion correction forces.

        The dispersion correction is a mean-field approximation that
        assumes uniform density beyond the cutoff. As such, it does
        not contribute to individual atomic forces (only to the virial
        for pressure calculation).

        Returns:
            Zero forces array (correction doesn't contribute to atomic forces).
        """
        # Dispersion correction is a volume-dependent term
        # It contributes to pressure but not to atomic forces
        return np.zeros((state.n_atoms, 3), dtype=np.float64)

    def compute_with_energy(
        self, state: MDState, neighbors: NeighborList | None = None
    ) -> tuple[NDArray[np.floating], float]:
        """
        Compute dispersion correction energy.

        The tail correction energy is:
        V_tail = (8/3) * pi * rho * sum_{i,j} N_i * N_j * epsilon_ij * sigma_ij^3 *
                 [(1/3)*(sigma_ij/rc)^6 - 1]

        where the sum is over all type pairs.
        """
        forces = np.zeros((state.n_atoms, 3), dtype=np.float64)

        volume = state.box.volume

        # Get type populations
        type_counts = self._get_type_counts()
        n_types = len(self.epsilon)

        energy = 0.0
        rc = self.cutoff

        # Sum over all type pairs
        for i in range(n_types):
            for j in range(i, n_types):
                n_i = type_counts[i]
                n_j = type_counts[j]

                if n_i == 0 or n_j == 0:
                    continue

                # Lorentz-Berthelot combining rules
                eps_ij = np.sqrt(self.epsilon[i] * self.epsilon[j])
                sig_ij = 0.5 * (self.sigma[i] + self.sigma[j])

                sig_over_rc = sig_ij / rc
                sig_over_rc_3 = sig_over_rc**3
                sig_over_rc_9 = sig_over_rc_3**3

                # Pair count (avoid double counting)
                n_pairs = n_i * (n_i - 1) / 2.0 if i == j else n_i * n_j

                # Tail correction
                # V_pair = (8/3) * pi * eps * sig^3 * [(1/3)*(sig/rc)^6 - 1] / rc^3
                # Total = n_pairs * rho * V_pair
                v_pair = (
                    (8.0 / 3.0)
                    * np.pi
                    * eps_ij
                    * sig_ij**3
                    * ((1.0 / 3.0) * sig_over_rc_9 / sig_over_rc_3 - sig_over_rc_3)
                )

                energy += n_pairs * v_pair / volume

        return forces, float(energy)

    def compute_pressure_correction(self, state: MDState) -> float:
        """
        Compute virial correction to pressure from dispersion.

        The pressure correction is:
        P_tail = (16/3) * pi * rho^2 * sum_{i,j} x_i * x_j * epsilon_ij * sigma_ij^3 *
                 [(2/3)*(sigma_ij/rc)^6 - 1]

        where x_i is the mole fraction of type i.

        Returns:
            Pressure correction in energy/volume units.
        """
        n_atoms = state.n_atoms
        volume = state.box.volume
        rho = n_atoms / volume

        type_counts = self._get_type_counts()
        n_types = len(self.epsilon)

        p_correction = 0.0
        rc = self.cutoff

        for i in range(n_types):
            for j in range(i, n_types):
                x_i = type_counts[i] / n_atoms
                x_j = type_counts[j] / n_atoms

                if x_i == 0 or x_j == 0:
                    continue

                # Combining rules
                eps_ij = np.sqrt(self.epsilon[i] * self.epsilon[j])
                sig_ij = 0.5 * (self.sigma[i] + self.sigma[j])

                sig_over_rc = sig_ij / rc
                sig_over_rc_3 = sig_over_rc**3
                sig_over_rc_9 = sig_over_rc_3**3

                # Symmetry factor
                sym = 1.0 if i == j else 2.0

                p_pair = (
                    (16.0 / 3.0)
                    * np.pi
                    * rho**2
                    * eps_ij
                    * sig_ij**3
                    * ((2.0 / 3.0) * sig_over_rc_9 / sig_over_rc_3 - sig_over_rc_3)
                )

                p_correction += sym * x_i * x_j * p_pair

        return float(p_correction)
