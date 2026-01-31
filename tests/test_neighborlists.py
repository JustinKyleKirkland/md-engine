"""Tests for neighbor list implementations."""

import numpy as np
import pytest

from mdcore.neighborlists.cell import CellList
from mdcore.neighborlists.verlet import VerletList
from mdcore.system.box import Box


class TestVerletList:
    """Test Verlet neighbor list."""

    @pytest.fixture
    def simple_positions(self):
        """Create simple test positions."""
        # Three atoms: 0 and 1 are close, 2 is far
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        )

    def test_build_finds_close_pairs(self, simple_positions):
        """Test that build finds pairs within cutoff."""
        box = Box.cubic(10.0)
        nlist = VerletList(cutoff=1.0, skin=0.3)

        nlist.build(simple_positions, box)

        pairs = nlist.get_pairs()

        # Only pair (0, 1) should be within cutoff+skin
        assert len(pairs) == 1
        assert list(pairs[0]) == [0, 1]

    def test_n_pairs(self, simple_positions):
        """Test n_pairs property."""
        box = Box.cubic(10.0)
        nlist = VerletList(cutoff=1.0, skin=0.3)

        nlist.build(simple_positions, box)

        assert nlist.n_pairs == 1

    def test_get_neighbors(self, simple_positions):
        """Test getting neighbors of specific atom."""
        box = Box.cubic(10.0)
        nlist = VerletList(cutoff=1.0, skin=0.3)

        nlist.build(simple_positions, box)

        neighbors_0 = nlist.get_neighbors(0)
        assert 1 in neighbors_0
        assert 2 not in neighbors_0

        neighbors_2 = nlist.get_neighbors(2)
        assert len(neighbors_2) == 0

    def test_update_not_needed_small_displacement(self, simple_positions):
        """Test that small displacements don't trigger rebuild."""
        box = Box.cubic(10.0)
        nlist = VerletList(cutoff=1.0, skin=0.3)

        nlist.build(simple_positions, box)

        # Small displacement
        new_positions = simple_positions + 0.05

        rebuilt = nlist.update_if_needed(new_positions)

        assert not rebuilt

    def test_update_needed_large_displacement(self, simple_positions):
        """Test that large displacements trigger rebuild."""
        box = Box.cubic(10.0)
        nlist = VerletList(cutoff=1.0, skin=0.3)

        nlist.build(simple_positions, box)

        # Large displacement (> skin/2)
        new_positions = simple_positions.copy()
        new_positions[0, 0] += 0.2

        rebuilt = nlist.update_if_needed(new_positions)

        assert rebuilt

    def test_periodic_boundaries(self):
        """Test neighbor finding across periodic boundaries."""
        box = Box.cubic(10.0)

        # Atoms near opposite edges of box
        positions = np.array(
            [
                [0.5, 5.0, 5.0],
                [9.5, 5.0, 5.0],
            ]
        )

        nlist = VerletList(cutoff=2.0, skin=0.3)
        nlist.build(positions, box)

        # Distance is 1.0 across periodic boundary
        pairs = nlist.get_pairs()
        assert len(pairs) == 1

    def test_cutoff_property(self):
        """Test cutoff property."""
        nlist = VerletList(cutoff=1.5, skin=0.3)
        assert nlist.cutoff == 1.5
        assert nlist.list_cutoff == 1.8

    def test_get_distances(self, simple_positions):
        """Test computing distances for pairs."""
        box = Box.cubic(10.0)
        nlist = VerletList(cutoff=1.0, skin=0.3)
        nlist.build(simple_positions, box)

        distances = nlist.get_distances(simple_positions, box)

        assert len(distances) == 1
        assert np.isclose(distances[0], 0.5)


class TestCellList:
    """Test cell list neighbor list."""

    @pytest.fixture
    def simple_positions(self):
        """Create simple test positions."""
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        )

    def test_build_finds_close_pairs(self, simple_positions):
        """Test that build finds pairs within cutoff."""
        box = Box.cubic(10.0)
        nlist = CellList(cutoff=1.0, skin=0.3)

        nlist.build(simple_positions, box)

        pairs = nlist.get_pairs()

        # Only pair (0, 1) should be within cutoff+skin
        assert len(pairs) == 1
        assert list(pairs[0]) == [0, 1]

    def test_n_cells(self, simple_positions):
        """Test cell grid dimensions."""
        box = Box.cubic(10.0)
        nlist = CellList(cutoff=1.0, skin=0.3)

        nlist.build(simple_positions, box)

        # With cutoff+skin=1.3, expect ~7 cells per dimension
        n_cells = nlist.n_cells
        assert all(n >= 3 for n in n_cells)

    def test_get_neighbors(self, simple_positions):
        """Test getting neighbors of specific atom."""
        box = Box.cubic(10.0)
        nlist = CellList(cutoff=1.0, skin=0.3)

        nlist.build(simple_positions, box)

        neighbors_0 = nlist.get_neighbors(0)
        assert 1 in neighbors_0
        assert 2 not in neighbors_0

    def test_update_not_needed(self, simple_positions):
        """Test that small displacements don't trigger rebuild."""
        box = Box.cubic(10.0)
        nlist = CellList(cutoff=1.0, skin=0.3)

        nlist.build(simple_positions, box)

        new_positions = simple_positions + 0.05
        rebuilt = nlist.update_if_needed(new_positions)

        assert not rebuilt

    def test_update_needed(self, simple_positions):
        """Test that large displacements trigger rebuild."""
        box = Box.cubic(10.0)
        nlist = CellList(cutoff=1.0, skin=0.3)

        nlist.build(simple_positions, box)

        new_positions = simple_positions.copy()
        new_positions[0, 0] += 0.2
        rebuilt = nlist.update_if_needed(new_positions)

        assert rebuilt

    def test_periodic_boundaries(self):
        """Test neighbor finding across periodic boundaries."""
        box = Box.cubic(10.0)

        positions = np.array(
            [
                [0.5, 5.0, 5.0],
                [9.5, 5.0, 5.0],
            ]
        )

        nlist = CellList(cutoff=2.0, skin=0.3)
        nlist.build(positions, box)

        pairs = nlist.get_pairs()
        assert len(pairs) == 1

    def test_many_atoms(self):
        """Test with many atoms (stress test)."""
        box = Box.cubic(10.0)
        n_atoms = 100
        np.random.seed(42)
        positions = np.random.uniform(0, 10, (n_atoms, 3))

        nlist = CellList(cutoff=1.0, skin=0.3)
        nlist.build(positions, box)

        # Just check it doesn't crash and returns valid pairs
        pairs = nlist.get_pairs()
        assert pairs.shape[1] == 2
        assert np.all(pairs[:, 0] < pairs[:, 1])  # i < j


class TestNeighborListConsistency:
    """Test that Verlet and Cell lists give same results."""

    def test_same_pairs(self):
        """Test that both implementations find the same pairs."""
        box = Box.cubic(10.0)
        n_atoms = 20
        np.random.seed(42)
        positions = np.random.uniform(0, 10, (n_atoms, 3))

        verlet = VerletList(cutoff=2.0, skin=0.3)
        cell = CellList(cutoff=2.0, skin=0.3)

        verlet.build(positions, box)
        cell.build(positions, box)

        # Convert to sets of tuples for comparison
        verlet_pairs = set(map(tuple, verlet.get_pairs()))
        cell_pairs = set(map(tuple, cell.get_pairs()))

        assert verlet_pairs == cell_pairs

    def test_same_n_pairs(self):
        """Test that both implementations find same number of pairs."""
        box = Box.cubic(10.0)
        n_atoms = 50
        np.random.seed(123)
        positions = np.random.uniform(0, 10, (n_atoms, 3))

        verlet = VerletList(cutoff=1.5, skin=0.2)
        cell = CellList(cutoff=1.5, skin=0.2)

        verlet.build(positions, box)
        cell.build(positions, box)

        assert verlet.n_pairs == cell.n_pairs
