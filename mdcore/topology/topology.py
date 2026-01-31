"""Topology representation for molecular systems."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class Topology:
    """
    Topology representation for molecular systems.

    Index-based design (no objects per atom) for efficiency.
    Compatible with GROMACS / OpenMM semantics.

    Attributes:
        n_atoms: Number of atoms in the system.
        atom_types: Atom type indices, shape (N,).
        atom_names: Atom names, length N.
        residue_indices: Residue index for each atom, shape (N,).
        residue_names: Residue names, length N_residues.
        bonds: Bond pairs as (i, j) indices, shape (N_bonds, 2).
        angles: Angle triplets as (i, j, k) indices, shape (N_angles, 3).
        dihedrals: Dihedral quads as (i, j, k, l) indices, shape (N_dihedrals, 4).
        impropers: Improper dihedral quads, shape (N_impropers, 4).
        exclusions: Set of excluded atom pairs (i, j) where i < j.
    """

    n_atoms: int
    atom_types: NDArray[np.integer] = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )
    atom_names: list[str] = field(default_factory=list)
    residue_indices: NDArray[np.integer] = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )
    residue_names: list[str] = field(default_factory=list)
    bonds: NDArray[np.integer] = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int32)
    )
    angles: NDArray[np.integer] = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int32)
    )
    dihedrals: NDArray[np.integer] = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.int32)
    )
    impropers: NDArray[np.integer] = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.int32)
    )
    exclusions: set[tuple[int, int]] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Validate and convert arrays."""
        self.atom_types = np.asarray(self.atom_types, dtype=np.int32)
        self.residue_indices = np.asarray(self.residue_indices, dtype=np.int32)
        self.bonds = (
            np.asarray(self.bonds, dtype=np.int32).reshape(-1, 2)
            if len(self.bonds) > 0
            else np.empty((0, 2), dtype=np.int32)
        )
        self.angles = (
            np.asarray(self.angles, dtype=np.int32).reshape(-1, 3)
            if len(self.angles) > 0
            else np.empty((0, 3), dtype=np.int32)
        )
        self.dihedrals = (
            np.asarray(self.dihedrals, dtype=np.int32).reshape(-1, 4)
            if len(self.dihedrals) > 0
            else np.empty((0, 4), dtype=np.int32)
        )
        self.impropers = (
            np.asarray(self.impropers, dtype=np.int32).reshape(-1, 4)
            if len(self.impropers) > 0
            else np.empty((0, 4), dtype=np.int32)
        )

        # Initialize arrays to correct size if empty
        if len(self.atom_types) == 0:
            self.atom_types = np.zeros(self.n_atoms, dtype=np.int32)
        if len(self.atom_names) == 0:
            self.atom_names = [f"A{i}" for i in range(self.n_atoms)]
        if len(self.residue_indices) == 0:
            self.residue_indices = np.zeros(self.n_atoms, dtype=np.int32)
        if len(self.residue_names) == 0:
            self.residue_names = ["UNK"]

        # Validate sizes
        if len(self.atom_types) != self.n_atoms:
            raise ValueError(
                f"atom_types length {len(self.atom_types)} != n_atoms {self.n_atoms}"
            )
        if len(self.atom_names) != self.n_atoms:
            raise ValueError(
                f"atom_names length {len(self.atom_names)} != n_atoms {self.n_atoms}"
            )
        if len(self.residue_indices) != self.n_atoms:
            raise ValueError(
                f"residue_indices length {len(self.residue_indices)} != n_atoms {self.n_atoms}"
            )

    @property
    def n_bonds(self) -> int:
        """Return number of bonds."""
        return len(self.bonds)

    @property
    def n_angles(self) -> int:
        """Return number of angles."""
        return len(self.angles)

    @property
    def n_dihedrals(self) -> int:
        """Return number of dihedrals."""
        return len(self.dihedrals)

    @property
    def n_impropers(self) -> int:
        """Return number of impropers."""
        return len(self.impropers)

    @property
    def n_residues(self) -> int:
        """Return number of residues."""
        if self.n_atoms == 0:
            return 0
        return int(np.max(self.residue_indices)) + 1

    def add_bond(self, i: int, j: int) -> None:
        """Add a bond between atoms i and j."""
        self._validate_atom_index(i)
        self._validate_atom_index(j)
        new_bond = np.array([[min(i, j), max(i, j)]], dtype=np.int32)
        self.bonds = (
            np.vstack([self.bonds, new_bond]) if len(self.bonds) > 0 else new_bond
        )

    def add_angle(self, i: int, j: int, k: int) -> None:
        """Add an angle between atoms i-j-k."""
        self._validate_atom_index(i)
        self._validate_atom_index(j)
        self._validate_atom_index(k)
        new_angle = np.array([[i, j, k]], dtype=np.int32)
        self.angles = (
            np.vstack([self.angles, new_angle]) if len(self.angles) > 0 else new_angle
        )

    def add_dihedral(self, i: int, j: int, k: int, l: int) -> None:
        """Add a dihedral between atoms i-j-k-l."""
        self._validate_atom_index(i)
        self._validate_atom_index(j)
        self._validate_atom_index(k)
        self._validate_atom_index(l)
        new_dihedral = np.array([[i, j, k, l]], dtype=np.int32)
        self.dihedrals = (
            np.vstack([self.dihedrals, new_dihedral])
            if len(self.dihedrals) > 0
            else new_dihedral
        )

    def add_improper(self, i: int, j: int, k: int, l: int) -> None:
        """Add an improper dihedral between atoms i-j-k-l."""
        self._validate_atom_index(i)
        self._validate_atom_index(j)
        self._validate_atom_index(k)
        self._validate_atom_index(l)
        new_improper = np.array([[i, j, k, l]], dtype=np.int32)
        self.impropers = (
            np.vstack([self.impropers, new_improper])
            if len(self.impropers) > 0
            else new_improper
        )

    def add_exclusion(self, i: int, j: int) -> None:
        """Add an exclusion between atoms i and j."""
        self._validate_atom_index(i)
        self._validate_atom_index(j)
        self.exclusions.add((min(i, j), max(i, j)))

    def is_excluded(self, i: int, j: int) -> bool:
        """Check if atoms i and j are excluded from nonbonded interactions."""
        return (min(i, j), max(i, j)) in self.exclusions

    def build_exclusions_from_bonds(self, n_bonds: int = 3) -> None:
        """
        Build exclusion list from bond connectivity.

        Args:
            n_bonds: Exclude pairs within n_bonds bond distance.
                     1 = 1-2 pairs (bonded)
                     2 = 1-2 and 1-3 pairs
                     3 = 1-2, 1-3, and 1-4 pairs (default)
        """
        if n_bonds < 1:
            return

        # Build adjacency list
        adjacency: list[set[int]] = [set() for _ in range(self.n_atoms)]
        for bond in self.bonds:
            i, j = bond
            adjacency[i].add(j)
            adjacency[j].add(i)

        # BFS to find pairs within n_bonds
        for start in range(self.n_atoms):
            visited = {start}
            current_level = {start}
            for _ in range(n_bonds):
                next_level: set[int] = set()
                for atom in current_level:
                    for neighbor in adjacency[atom]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_level.add(neighbor)
                            if neighbor > start:
                                self.exclusions.add((start, neighbor))
                current_level = next_level

    def get_bonded_atoms(self, atom_index: int) -> NDArray[np.integer]:
        """Return indices of atoms bonded to given atom."""
        self._validate_atom_index(atom_index)
        mask = (self.bonds[:, 0] == atom_index) | (self.bonds[:, 1] == atom_index)
        bonded_pairs = self.bonds[mask]
        bonded = []
        for pair in bonded_pairs:
            if pair[0] == atom_index:
                bonded.append(pair[1])
            else:
                bonded.append(pair[0])
        return np.array(bonded, dtype=np.int32)

    def subset(self, atom_indices: ArrayLike) -> Topology:
        """
        Create a topology subset for domain decomposition.

        Args:
            atom_indices: Indices of atoms to include.

        Returns:
            New Topology containing only specified atoms.
        """
        atom_indices = np.asarray(atom_indices, dtype=np.int32)
        index_set = set(atom_indices)
        n_atoms_new = len(atom_indices)

        # Create mapping from old to new indices
        old_to_new = {old: new for new, old in enumerate(atom_indices)}

        # Filter and remap bonds
        new_bonds = []
        for bond in self.bonds:
            if bond[0] in index_set and bond[1] in index_set:
                new_bonds.append([old_to_new[bond[0]], old_to_new[bond[1]]])

        # Filter and remap angles
        new_angles = []
        for angle in self.angles:
            if all(idx in index_set for idx in angle):
                new_angles.append([old_to_new[idx] for idx in angle])

        # Filter and remap dihedrals
        new_dihedrals = []
        for dihedral in self.dihedrals:
            if all(idx in index_set for idx in dihedral):
                new_dihedrals.append([old_to_new[idx] for idx in dihedral])

        # Filter and remap impropers
        new_impropers = []
        for improper in self.impropers:
            if all(idx in index_set for idx in improper):
                new_impropers.append([old_to_new[idx] for idx in improper])

        # Filter and remap exclusions
        new_exclusions = set()
        for i, j in self.exclusions:
            if i in index_set and j in index_set:
                new_i, new_j = old_to_new[i], old_to_new[j]
                new_exclusions.add((min(new_i, new_j), max(new_i, new_j)))

        return Topology(
            n_atoms=n_atoms_new,
            atom_types=self.atom_types[atom_indices],
            atom_names=[self.atom_names[i] for i in atom_indices],
            residue_indices=self.residue_indices[atom_indices],
            residue_names=self.residue_names.copy(),
            bonds=np.array(new_bonds, dtype=np.int32)
            if new_bonds
            else np.empty((0, 2), dtype=np.int32),
            angles=np.array(new_angles, dtype=np.int32)
            if new_angles
            else np.empty((0, 3), dtype=np.int32),
            dihedrals=np.array(new_dihedrals, dtype=np.int32)
            if new_dihedrals
            else np.empty((0, 4), dtype=np.int32),
            impropers=np.array(new_impropers, dtype=np.int32)
            if new_impropers
            else np.empty((0, 4), dtype=np.int32),
            exclusions=new_exclusions,
        )

    def _validate_atom_index(self, index: int) -> None:
        """Validate that atom index is in range."""
        if index < 0 or index >= self.n_atoms:
            raise IndexError(f"Atom index {index} out of range [0, {self.n_atoms})")
