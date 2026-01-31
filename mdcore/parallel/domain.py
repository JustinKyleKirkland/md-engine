"""Domain decomposition for spatial parallelization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..system import Box
    from .backends.base import ParallelBackend


@dataclass
class Domain:
    """
    A spatial domain for domain decomposition.

    Each MPI rank owns a domain containing local atoms plus
    ghost atoms from neighboring domains.

    Attributes:
        rank: Rank that owns this domain.
        lower_bounds: Lower corner of domain, shape (3,).
        upper_bounds: Upper corner of domain, shape (3,).
        local_atoms: Indices of atoms owned by this domain.
        ghost_atoms: Indices of ghost atoms from neighbors.
        neighbor_ranks: Ranks of neighboring domains.
    """

    rank: int
    lower_bounds: NDArray[np.floating]
    upper_bounds: NDArray[np.floating]
    local_atoms: NDArray[np.integer] = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    ghost_atoms: NDArray[np.integer] = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    neighbor_ranks: list[int] = field(default_factory=list)

    @property
    def n_local(self) -> int:
        """Number of local atoms."""
        return len(self.local_atoms)

    @property
    def n_ghost(self) -> int:
        """Number of ghost atoms."""
        return len(self.ghost_atoms)

    @property
    def n_total(self) -> int:
        """Total atoms (local + ghost)."""
        return self.n_local + self.n_ghost

    @property
    def all_atoms(self) -> NDArray[np.integer]:
        """All atom indices (local + ghost)."""
        return np.concatenate([self.local_atoms, self.ghost_atoms])

    def contains_point(self, point: NDArray[np.floating]) -> bool:
        """Check if a point is inside this domain."""
        return bool(
            np.all(point >= self.lower_bounds) and np.all(point < self.upper_bounds)
        )


class DomainDecomposition:
    """
    Spatial domain decomposition for parallel MD.

    Divides the simulation box into subdomains, one per MPI rank.
    Handles assignment of atoms to domains and ghost atom communication.
    """

    def __init__(
        self,
        backend: ParallelBackend,
        box: Box,
        cutoff: float,
        grid: tuple[int, int, int] | None = None,
    ) -> None:
        """
        Initialize domain decomposition.

        Args:
            backend: Parallel backend to use.
            box: Simulation box.
            cutoff: Interaction cutoff (determines ghost shell width).
            grid: Domain grid dimensions (nx, ny, nz).
                  If None, automatically determined.
        """
        self._backend = backend
        self._box = box
        self._cutoff = cutoff
        self._skin = cutoff * 0.5  # Ghost shell extends beyond cutoff

        if grid is None:
            self._grid = self._auto_grid(backend.n_workers)
        else:
            self._grid = grid

        if np.prod(self._grid) != backend.n_workers:
            raise ValueError(
                f"Grid {self._grid} has {np.prod(self._grid)} cells "
                f"but backend has {backend.n_workers} workers"
            )

        self._domains = self._create_domains()

    @property
    def grid(self) -> tuple[int, int, int]:
        """Return domain grid dimensions."""
        return self._grid

    @property
    def my_domain(self) -> Domain:
        """Return domain for current rank."""
        return self._domains[self._backend.rank]

    def _auto_grid(self, n_workers: int) -> tuple[int, int, int]:
        """
        Automatically determine domain grid.

        Tries to make domains as cubic as possible.

        Args:
            n_workers: Number of domains needed.

        Returns:
            Grid dimensions (nx, ny, nz).
        """
        # Simple factorization - prefer more divisions in larger dimensions
        lengths = self._box.lengths

        # Start with 1x1xN and improve
        best_grid = (1, 1, n_workers)
        best_ratio = float("inf")

        # Try all factorizations
        for nx in range(1, n_workers + 1):
            if n_workers % nx != 0:
                continue
            remaining = n_workers // nx

            for ny in range(1, remaining + 1):
                if remaining % ny != 0:
                    continue
                nz = remaining // ny

                # Compute domain dimensions
                dx = lengths[0] / nx
                dy = lengths[1] / ny
                dz = lengths[2] / nz

                # Prefer cubic domains (ratio close to 1)
                dims = sorted([dx, dy, dz])
                ratio = dims[2] / dims[0]  # max/min

                if ratio < best_ratio:
                    best_ratio = ratio
                    best_grid = (nx, ny, nz)

        return best_grid

    def _create_domains(self) -> list[Domain]:
        """Create domain objects for all ranks."""
        nx, ny, nz = self._grid
        lengths = self._box.lengths
        # Box origin defaults to (0, 0, 0)
        origin = np.zeros(3)

        dx = lengths[0] / nx
        dy = lengths[1] / ny
        dz = lengths[2] / nz

        domains = []

        for rank in range(self._backend.n_workers):
            # Convert rank to 3D index
            ix = rank % nx
            iy = (rank // nx) % ny
            iz = rank // (nx * ny)

            lower = origin + np.array([ix * dx, iy * dy, iz * dz])
            upper = lower + np.array([dx, dy, dz])

            # Find neighbor ranks (26 neighbors in 3D)
            neighbors = []
            for dix in [-1, 0, 1]:
                for diy in [-1, 0, 1]:
                    for diz in [-1, 0, 1]:
                        if dix == 0 and diy == 0 and diz == 0:
                            continue

                        # Periodic wrapping
                        nix = (ix + dix) % nx
                        niy = (iy + diy) % ny
                        niz = (iz + diz) % nz

                        neighbor_rank = nix + niy * nx + niz * nx * ny
                        if neighbor_rank not in neighbors:
                            neighbors.append(neighbor_rank)

            domains.append(
                Domain(
                    rank=rank,
                    lower_bounds=lower,
                    upper_bounds=upper,
                    neighbor_ranks=neighbors,
                )
            )

        return domains

    def assign_atoms(self, positions: NDArray[np.floating]) -> None:
        """
        Assign atoms to domains based on positions.

        Args:
            positions: Atomic positions, shape (N, 3).
        """
        n_atoms = len(positions)

        for domain in self._domains:
            # Find atoms in this domain
            in_domain = np.array(
                [i for i in range(n_atoms) if domain.contains_point(positions[i])],
                dtype=np.int64,
            )
            domain.local_atoms = in_domain

    def find_ghost_atoms(self, positions: NDArray[np.floating]) -> None:
        """
        Find ghost atoms for each domain.

        Ghost atoms are atoms in neighboring domains that are
        within cutoff + skin of the domain boundary.

        Args:
            positions: Atomic positions, shape (N, 3).
        """
        ghost_width = self._cutoff + self._skin

        for domain in self._domains:
            ghost_list = []

            # Check atoms from neighboring domains
            for neighbor_rank in domain.neighbor_ranks:
                neighbor_domain = self._domains[neighbor_rank]

                for atom_idx in neighbor_domain.local_atoms:
                    pos = positions[atom_idx]

                    # Check if within ghost shell of domain
                    dist_to_lower = pos - domain.lower_bounds
                    dist_to_upper = domain.upper_bounds - pos

                    # Handle periodic boundaries
                    dist_to_lower = np.minimum(
                        dist_to_lower, self._box.lengths - np.abs(dist_to_lower)
                    )
                    dist_to_upper = np.minimum(
                        dist_to_upper, self._box.lengths - np.abs(dist_to_upper)
                    )

                    min_dist = np.min(
                        np.abs(np.concatenate([dist_to_lower, dist_to_upper]))
                    )

                    if min_dist < ghost_width:
                        ghost_list.append(atom_idx)

            domain.ghost_atoms = np.array(ghost_list, dtype=np.int64)

    def exchange_forces(
        self,
        local_forces: NDArray[np.floating],
        global_forces: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Exchange and sum forces across domain boundaries.

        Forces on ghost atoms need to be sent back to their
        owning domain and summed.

        Args:
            local_forces: Forces computed locally, shape (n_total, 3).
            global_forces: Global force array to update.

        Returns:
            Updated global forces.
        """
        # In serial mode, just copy
        if self._backend.n_workers == 1:
            domain = self.my_domain
            for i, atom_idx in enumerate(domain.local_atoms):
                global_forces[atom_idx] = local_forces[i]
            return global_forces

        # For MPI, would need proper halo exchange
        # This is a simplified version
        return self._backend.allreduce_sum(global_forces)

    def get_local_positions(
        self,
        positions: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Get positions for local + ghost atoms.

        Args:
            positions: Global positions, shape (N, 3).

        Returns:
            Local positions, shape (n_local + n_ghost, 3).
        """
        domain = self.my_domain
        all_indices = domain.all_atoms

        if len(all_indices) == 0:
            return np.empty((0, 3), dtype=positions.dtype)

        return positions[all_indices]
