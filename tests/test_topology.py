"""Tests for Topology class."""

import numpy as np
import pytest

from mdcore.topology.topology import Topology


class TestTopologyCreation:
    """Test Topology creation."""

    def test_basic_creation(self):
        """Test creating a basic topology."""
        topo = Topology(n_atoms=10)

        assert topo.n_atoms == 10
        assert len(topo.atom_types) == 10
        assert len(topo.atom_names) == 10
        assert topo.n_bonds == 0
        assert topo.n_angles == 0
        assert topo.n_dihedrals == 0

    def test_creation_with_arrays(self):
        """Test creating topology with custom arrays."""
        topo = Topology(
            n_atoms=3,
            atom_types=np.array([0, 1, 0]),
            atom_names=["H", "O", "H"],
            bonds=np.array([[0, 1], [1, 2]]),
        )

        assert np.array_equal(topo.atom_types, [0, 1, 0])
        assert topo.atom_names == ["H", "O", "H"]
        assert topo.n_bonds == 2

    def test_invalid_atom_types_length(self):
        """Test that mismatched atom_types length raises error."""
        with pytest.raises(ValueError):
            Topology(
                n_atoms=5,
                atom_types=np.array([0, 1, 0]),  # Wrong length
            )


class TestTopologyBonds:
    """Test bond management."""

    def test_add_bond(self):
        """Test adding a bond."""
        topo = Topology(n_atoms=5)
        topo.add_bond(0, 1)
        topo.add_bond(1, 2)

        assert topo.n_bonds == 2
        assert [0, 1] in topo.bonds.tolist()
        assert [1, 2] in topo.bonds.tolist()

    def test_add_bond_normalizes_order(self):
        """Test that bonds are stored with i < j."""
        topo = Topology(n_atoms=5)
        topo.add_bond(3, 1)  # Out of order

        assert topo.bonds[0, 0] == 1
        assert topo.bonds[0, 1] == 3

    def test_add_bond_invalid_index(self):
        """Test that invalid bond indices raise error."""
        topo = Topology(n_atoms=5)

        with pytest.raises(IndexError):
            topo.add_bond(0, 10)

        with pytest.raises(IndexError):
            topo.add_bond(-1, 0)

    def test_get_bonded_atoms(self):
        """Test getting bonded atoms."""
        topo = Topology(n_atoms=5)
        topo.add_bond(0, 1)
        topo.add_bond(0, 2)
        topo.add_bond(1, 3)

        bonded_to_0 = topo.get_bonded_atoms(0)
        assert set(bonded_to_0) == {1, 2}

        bonded_to_1 = topo.get_bonded_atoms(1)
        assert set(bonded_to_1) == {0, 3}


class TestTopologyAngles:
    """Test angle management."""

    def test_add_angle(self):
        """Test adding an angle."""
        topo = Topology(n_atoms=5)
        topo.add_angle(0, 1, 2)

        assert topo.n_angles == 1
        assert list(topo.angles[0]) == [0, 1, 2]

    def test_add_angle_invalid_index(self):
        """Test that invalid angle indices raise error."""
        topo = Topology(n_atoms=5)

        with pytest.raises(IndexError):
            topo.add_angle(0, 1, 10)


class TestTopologyDihedrals:
    """Test dihedral management."""

    def test_add_dihedral(self):
        """Test adding a dihedral."""
        topo = Topology(n_atoms=5)
        topo.add_dihedral(0, 1, 2, 3)

        assert topo.n_dihedrals == 1
        assert list(topo.dihedrals[0]) == [0, 1, 2, 3]

    def test_add_improper(self):
        """Test adding an improper dihedral."""
        topo = Topology(n_atoms=5)
        topo.add_improper(0, 1, 2, 3)

        assert topo.n_impropers == 1
        assert list(topo.impropers[0]) == [0, 1, 2, 3]


class TestTopologyExclusions:
    """Test exclusion management."""

    def test_add_exclusion(self):
        """Test adding an exclusion."""
        topo = Topology(n_atoms=5)
        topo.add_exclusion(0, 1)

        assert (0, 1) in topo.exclusions
        assert topo.is_excluded(0, 1)
        assert topo.is_excluded(1, 0)  # Order shouldn't matter

    def test_build_exclusions_from_bonds(self):
        """Test building exclusions from bond connectivity."""
        # Linear chain: 0-1-2-3-4
        topo = Topology(n_atoms=5)
        topo.add_bond(0, 1)
        topo.add_bond(1, 2)
        topo.add_bond(2, 3)
        topo.add_bond(3, 4)

        # Build 1-2 and 1-3 exclusions
        topo.build_exclusions_from_bonds(n_bonds=2)

        # 1-2 pairs (bonded)
        assert topo.is_excluded(0, 1)
        assert topo.is_excluded(1, 2)

        # 1-3 pairs
        assert topo.is_excluded(0, 2)
        assert topo.is_excluded(1, 3)

        # 1-4 pairs should NOT be excluded with n_bonds=2
        assert not topo.is_excluded(0, 3)

    def test_build_exclusions_14(self):
        """Test building 1-4 exclusions."""
        # Linear chain: 0-1-2-3
        topo = Topology(n_atoms=4)
        topo.add_bond(0, 1)
        topo.add_bond(1, 2)
        topo.add_bond(2, 3)

        topo.build_exclusions_from_bonds(n_bonds=3)

        # Now 1-4 pairs should be excluded
        assert topo.is_excluded(0, 3)


class TestTopologySubset:
    """Test topology subset creation."""

    def test_subset_atoms(self):
        """Test creating a subset of atoms."""
        topo = Topology(
            n_atoms=5,
            atom_types=np.array([0, 1, 2, 1, 0]),
            atom_names=["A", "B", "C", "D", "E"],
        )
        topo.add_bond(0, 1)
        topo.add_bond(1, 2)
        topo.add_bond(2, 3)
        topo.add_bond(3, 4)

        # Take atoms 1, 2, 3
        subset = topo.subset([1, 2, 3])

        assert subset.n_atoms == 3
        assert list(subset.atom_types) == [1, 2, 1]
        assert subset.atom_names == ["B", "C", "D"]

        # Only bonds within the subset
        assert subset.n_bonds == 2

    def test_subset_preserves_internal_bonds(self):
        """Test that bonds within subset are preserved."""
        topo = Topology(n_atoms=4)
        topo.add_bond(0, 1)
        topo.add_bond(1, 2)
        topo.add_bond(2, 3)

        subset = topo.subset([1, 2])

        # Only bond 1-2 should be preserved (renumbered to 0-1)
        assert subset.n_bonds == 1
        assert list(subset.bonds[0]) == [0, 1]

    def test_subset_remaps_exclusions(self):
        """Test that exclusions are properly remapped."""
        topo = Topology(n_atoms=5)
        topo.add_exclusion(1, 3)
        topo.add_exclusion(0, 2)  # Not in subset

        subset = topo.subset([1, 2, 3])

        # Original (1, 3) -> new (0, 2)
        assert (0, 2) in subset.exclusions
        assert len(subset.exclusions) == 1


class TestTopologyResidues:
    """Test residue handling."""

    def test_n_residues(self):
        """Test counting residues."""
        topo = Topology(
            n_atoms=6,
            residue_indices=np.array([0, 0, 0, 1, 1, 1]),
        )

        assert topo.n_residues == 2

    def test_n_residues_empty(self):
        """Test n_residues for empty topology."""
        # Create a minimal topology with 0 atoms
        topo = Topology(n_atoms=0)
        assert topo.n_residues == 0
