"""Tests for MDState class."""

import numpy as np
import pytest

from mdcore.system.box import Box
from mdcore.system.state import FrozenMDState, MDState


class TestMDStateCreation:
    """Test MDState creation."""

    def test_basic_creation(self):
        """Test creating an MDState with all arrays."""
        n_atoms = 10
        box = Box.cubic(5.0)

        state = MDState(
            positions=np.random.rand(n_atoms, 3),
            velocities=np.random.rand(n_atoms, 3),
            forces=np.random.rand(n_atoms, 3),
            masses=np.ones(n_atoms),
            box=box,
        )

        assert state.n_atoms == n_atoms
        assert state.positions.shape == (n_atoms, 3)
        assert state.velocities.shape == (n_atoms, 3)
        assert state.forces.shape == (n_atoms, 3)
        assert state.masses.shape == (n_atoms,)
        assert state.time == 0.0
        assert state.step == 0

    def test_create_factory(self):
        """Test MDState.create factory method."""
        n_atoms = 5
        positions = np.random.rand(n_atoms, 3)
        masses = np.ones(n_atoms) * 12.0
        box = Box.cubic(10.0)

        state = MDState.create(positions, masses, box)

        assert state.n_atoms == n_atoms
        assert np.allclose(state.positions, positions)
        assert np.allclose(state.masses, masses)
        assert np.allclose(state.velocities, 0)
        assert np.allclose(state.forces, 0)

    def test_create_with_velocities(self):
        """Test creating state with initial velocities."""
        n_atoms = 3
        positions = np.random.rand(n_atoms, 3)
        velocities = np.random.rand(n_atoms, 3)
        masses = np.ones(n_atoms)
        box = Box.cubic(10.0)

        state = MDState.create(positions, masses, box, velocities=velocities)

        assert np.allclose(state.velocities, velocities)

    def test_shape_mismatch_positions(self):
        """Test that mismatched array shapes raise errors."""
        box = Box.cubic(10.0)

        with pytest.raises(ValueError):
            MDState(
                positions=np.zeros((5, 3)),
                velocities=np.zeros((5, 3)),
                forces=np.zeros((5, 3)),
                masses=np.ones(10),  # Mismatch!
                box=box,
            )

    def test_shape_mismatch_velocities(self):
        """Test that mismatched velocity shape raises error."""
        box = Box.cubic(10.0)

        with pytest.raises(ValueError):
            MDState(
                positions=np.zeros((5, 3)),
                velocities=np.zeros((3, 3)),  # Mismatch!
                forces=np.zeros((5, 3)),
                masses=np.ones(5),
                box=box,
            )


class TestMDStateProperties:
    """Test MDState computed properties."""

    def test_kinetic_energy(self):
        """Test kinetic energy calculation."""
        box = Box.cubic(10.0)

        # Two atoms with mass 1, velocity 1 in x direction
        # KE = 2 * (0.5 * 1 * 1^2) = 1.0
        state = MDState.create(
            positions=np.array([[0, 0, 0], [1, 0, 0]]),
            masses=np.array([1.0, 1.0]),
            box=box,
            velocities=np.array([[1.0, 0, 0], [1.0, 0, 0]]),
        )

        assert np.isclose(state.kinetic_energy, 1.0)

    def test_center_of_mass(self):
        """Test center of mass calculation."""
        box = Box.cubic(10.0)

        # Two equal mass atoms
        state = MDState.create(
            positions=np.array([[0, 0, 0], [2, 0, 0]]),
            masses=np.array([1.0, 1.0]),
            box=box,
        )

        com = state.center_of_mass
        assert np.allclose(com, [1.0, 0.0, 0.0])

    def test_center_of_mass_weighted(self):
        """Test center of mass with different masses."""
        box = Box.cubic(10.0)

        # Mass-weighted COM
        state = MDState.create(
            positions=np.array([[0, 0, 0], [4, 0, 0]]),
            masses=np.array([3.0, 1.0]),
            box=box,
        )

        com = state.center_of_mass
        assert np.allclose(com, [1.0, 0.0, 0.0])  # (0*3 + 4*1) / 4 = 1

    def test_center_of_mass_velocity(self):
        """Test center of mass velocity."""
        box = Box.cubic(10.0)

        state = MDState.create(
            positions=np.array([[0, 0, 0], [1, 0, 0]]),
            masses=np.array([1.0, 1.0]),
            box=box,
            velocities=np.array([[1.0, 0, 0], [3.0, 0, 0]]),
        )

        com_vel = state.center_of_mass_velocity
        assert np.allclose(com_vel, [2.0, 0.0, 0.0])


class TestMDStateCopy:
    """Test MDState copying functionality."""

    def test_copy_is_independent(self):
        """Test that copy creates independent arrays."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.zeros((5, 3)),
            masses=np.ones(5),
            box=box,
        )

        state_copy = state.copy()

        # Modify original
        state.positions[0, 0] = 999.0

        # Copy should be unchanged
        assert state_copy.positions[0, 0] == 0.0

    def test_freeze_creates_immutable(self):
        """Test that freeze creates read-only state."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.zeros((5, 3)),
            masses=np.ones(5),
            box=box,
        )

        frozen = state.freeze()

        assert isinstance(frozen, FrozenMDState)

        with pytest.raises(ValueError):
            frozen.positions[0, 0] = 1.0

    def test_thaw_frozen_state(self):
        """Test that thaw creates mutable copy."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.zeros((5, 3)),
            masses=np.ones(5),
            box=box,
        )

        frozen = state.freeze()
        thawed = frozen.thaw()

        assert isinstance(thawed, MDState)

        # Should be modifiable
        thawed.positions[0, 0] = 1.0
        assert thawed.positions[0, 0] == 1.0


class TestMDStateTemperature:
    """Test temperature calculations."""

    def test_temperature_zero_velocity(self):
        """Test that zero velocity gives zero temperature."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.zeros((10, 3)),
            masses=np.ones(10),
            box=box,
        )

        assert state.temperature == 0.0

    def test_temperature_single_atom(self):
        """Test that single atom returns zero temperature."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.zeros((1, 3)),
            masses=np.ones(1),
            box=box,
            velocities=np.array([[1.0, 0, 0]]),
        )

        assert state.temperature == 0.0
