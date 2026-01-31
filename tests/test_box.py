"""Tests for Box class."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mdcore.system.box import Box


class TestBoxCreation:
    """Test box creation methods."""

    def test_cubic_box(self):
        """Test creating a cubic box."""
        box = Box.cubic(10.0)
        assert np.allclose(box.lengths, [10.0, 10.0, 10.0])
        assert box.is_orthorhombic
        assert np.isclose(box.volume, 1000.0)

    def test_orthorhombic_box(self):
        """Test creating an orthorhombic box."""
        box = Box.orthorhombic(10.0, 20.0, 30.0)
        assert np.allclose(box.lengths, [10.0, 20.0, 30.0])
        assert box.is_orthorhombic
        assert np.isclose(box.volume, 6000.0)

    def test_triclinic_box(self):
        """Test creating a triclinic box."""
        vectors = [
            [10.0, 0.0, 0.0],
            [2.0, 10.0, 0.0],
            [1.0, 1.0, 10.0],
        ]
        box = Box.triclinic(vectors)
        assert not box.is_orthorhombic
        assert np.isclose(box.volume, 1000.0)  # det = 10*10*10 = 1000

    def test_from_array(self):
        """Test creating a box from a length array."""
        box = Box(np.array([5.0, 6.0, 7.0]))
        assert np.allclose(np.diag(box.vectors), [5.0, 6.0, 7.0])

    def test_invalid_shape(self):
        """Test that invalid shapes raise errors."""
        with pytest.raises(ValueError):
            Box(np.array([1.0, 2.0]))  # Wrong shape


class TestBoxProperties:
    """Test box properties."""

    def test_lengths_orthorhombic(self):
        """Test box lengths for orthorhombic box."""
        box = Box.orthorhombic(3.0, 4.0, 5.0)
        assert np.allclose(box.lengths, [3.0, 4.0, 5.0])

    def test_lengths_triclinic(self):
        """Test box lengths for triclinic box."""
        vectors = [
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [1.0, 0.0, 5.0],
        ]
        box = Box.triclinic(vectors)
        expected = [3.0, 4.0, np.sqrt(1 + 25)]
        assert np.allclose(box.lengths, expected)

    def test_volume(self):
        """Test volume calculation."""
        box = Box.cubic(2.0)
        assert np.isclose(box.volume, 8.0)


class TestPeriodicBoundaries:
    """Test periodic boundary condition methods."""

    def test_wrap_positions_orthorhombic(self):
        """Test position wrapping for orthorhombic box."""
        box = Box.cubic(10.0)

        # Position inside box
        pos = np.array([[5.0, 5.0, 5.0]])
        wrapped = box.wrap_positions(pos)
        assert np.allclose(wrapped, pos)

        # Position outside box
        pos = np.array([[15.0, -3.0, 25.0]])
        wrapped = box.wrap_positions(pos)
        assert np.allclose(wrapped, [[5.0, 7.0, 5.0]])

    def test_wrap_positions_batch(self):
        """Test wrapping multiple positions."""
        box = Box.cubic(10.0)
        pos = np.array(
            [
                [5.0, 5.0, 5.0],
                [15.0, 5.0, 5.0],
                [-5.0, 5.0, 5.0],
            ]
        )
        wrapped = box.wrap_positions(pos)
        expected = np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0],
                [5.0, 5.0, 5.0],
            ]
        )
        assert np.allclose(wrapped, expected)

    def test_minimum_image_orthorhombic(self):
        """Test minimum image displacement for orthorhombic box."""
        box = Box.cubic(10.0)

        # Close atoms
        r1 = np.array([1.0, 1.0, 1.0])
        r2 = np.array([2.0, 1.0, 1.0])
        dr = box.minimum_image(r1, r2)
        assert np.allclose(dr, [1.0, 0.0, 0.0])

        # Atoms across periodic boundary
        r1 = np.array([1.0, 1.0, 1.0])
        r2 = np.array([9.0, 1.0, 1.0])
        dr = box.minimum_image(r1, r2)
        assert np.allclose(dr, [-2.0, 0.0, 0.0])

    def test_minimum_image_distance(self):
        """Test minimum image distance calculation."""
        box = Box.cubic(10.0)

        r1 = np.array([1.0, 1.0, 1.0])
        r2 = np.array([9.0, 1.0, 1.0])

        dist = box.minimum_image_distance(r1, r2)
        assert np.isclose(dist, 2.0)

    def test_minimum_image_batch(self):
        """Test minimum image for multiple pairs."""
        box = Box.cubic(10.0)

        r1 = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        r2 = np.array([[9.0, 1.0, 1.0], [9.5, 0.0, 0.0]])

        dr = box.minimum_image(r1, r2)
        expected = np.array([[-2.0, 0.0, 0.0], [-0.5, 0.0, 0.0]])
        assert np.allclose(dr, expected)


class TestBoxImmutability:
    """Test that Box is immutable."""

    def test_frozen_dataclass(self):
        """Test that box attributes cannot be modified."""
        box = Box.cubic(10.0)

        with pytest.raises(FrozenInstanceError):
            box.vectors = np.eye(3)
