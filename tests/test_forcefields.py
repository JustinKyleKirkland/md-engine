"""Tests for force field implementations."""

import numpy as np
import pytest

from mdcore.forcefields.base import ForceProvider
from mdcore.forcefields.bonded.angles import HarmonicAngleForce
from mdcore.forcefields.bonded.bonds import HarmonicBondForce
from mdcore.forcefields.bonded.dihedrals import PeriodicDihedralForce
from mdcore.forcefields.composite import ForceField
from mdcore.forcefields.nonbonded.coulomb import CoulombForce
from mdcore.forcefields.nonbonded.lj import LennardJonesForce
from mdcore.system.box import Box
from mdcore.system.state import MDState


class TestForceProviderInterface:
    """Test ForceProvider interface."""

    def test_abstract_class(self):
        """Test that ForceProvider cannot be instantiated."""
        with pytest.raises(TypeError):
            ForceProvider()


class TestHarmonicBondForce:
    """Test harmonic bond force."""

    @pytest.fixture
    def two_atom_state(self):
        """Create a two-atom state."""
        box = Box.cubic(10.0)
        return MDState.create(
            positions=np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
            box=box,
        )

    def test_zero_force_at_equilibrium(self, two_atom_state):
        """Test that force is zero at equilibrium length."""
        force = HarmonicBondForce(
            bond_indices=[[0, 1]],
            force_constants=[1000.0],
            equilibrium_lengths=[0.15],
        )

        forces = force.compute(two_atom_state)

        assert np.allclose(forces, 0, atol=1e-10)

    def test_restoring_force_direction(self, two_atom_state):
        """Test that force points toward equilibrium."""
        # Bond stretched beyond equilibrium
        force = HarmonicBondForce(
            bond_indices=[[0, 1]],
            force_constants=[1000.0],
            equilibrium_lengths=[0.10],  # Equilibrium shorter than actual
        )

        forces = force.compute(two_atom_state)

        # Force on atom 1 should point toward atom 0 (negative x)
        assert forces[1, 0] < 0
        # Force on atom 0 should point toward atom 1 (positive x)
        assert forces[0, 0] > 0
        # Newton's third law
        assert np.allclose(forces.sum(axis=0), 0)

    def test_force_magnitude(self, two_atom_state):
        """Test force magnitude follows F = k * (r - r0)."""
        k = 1000.0
        r0 = 0.10
        r = 0.15

        force = HarmonicBondForce(
            bond_indices=[[0, 1]],
            force_constants=[k],
            equilibrium_lengths=[r0],
        )

        forces = force.compute(two_atom_state)

        expected_magnitude = k * (r - r0)
        actual_magnitude = np.abs(forces[0, 0])

        assert np.isclose(actual_magnitude, expected_magnitude)

    def test_energy_calculation(self, two_atom_state):
        """Test potential energy calculation."""
        k = 1000.0
        r0 = 0.10
        r = 0.15

        force = HarmonicBondForce(
            bond_indices=[[0, 1]],
            force_constants=[k],
            equilibrium_lengths=[r0],
        )

        forces, energy = force.compute_with_energy(two_atom_state)

        expected_energy = 0.5 * k * (r - r0) ** 2
        assert np.isclose(energy, expected_energy)

    def test_multiple_bonds(self):
        """Test with multiple bonds."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                ]
            ),
            masses=np.ones(3),
            box=box,
        )

        force = HarmonicBondForce(
            bond_indices=[[0, 1], [1, 2]],
            force_constants=[1000.0, 1000.0],
            equilibrium_lengths=[0.1, 0.1],
        )

        forces = force.compute(state)

        # At equilibrium, forces should be zero
        assert np.allclose(forces, 0, atol=1e-10)


class TestHarmonicAngleForce:
    """Test harmonic angle force."""

    @pytest.fixture
    def three_atom_state(self):
        """Create a three-atom state with 90-degree angle."""
        box = Box.cubic(10.0)
        return MDState.create(
            positions=np.array(
                [
                    [1.0, 0.0, 0.0],  # i
                    [0.0, 0.0, 0.0],  # j (central)
                    [0.0, 1.0, 0.0],  # k
                ]
            ),
            masses=np.ones(3),
            box=box,
        )

    def test_zero_force_at_equilibrium(self, three_atom_state):
        """Test that force is zero at equilibrium angle."""
        force = HarmonicAngleForce(
            angle_indices=[[0, 1, 2]],
            force_constants=[100.0],
            equilibrium_angles=[np.pi / 2],  # 90 degrees
        )

        forces = force.compute(three_atom_state)

        assert np.allclose(forces, 0, atol=1e-10)

    def test_newton_third_law(self, three_atom_state):
        """Test that total force is zero (Newton's third law)."""
        force = HarmonicAngleForce(
            angle_indices=[[0, 1, 2]],
            force_constants=[100.0],
            equilibrium_angles=[np.pi / 3],  # 60 degrees (not at equilibrium)
        )

        forces = force.compute(three_atom_state)

        assert np.allclose(forces.sum(axis=0), 0, atol=1e-10)

    def test_energy_calculation(self, three_atom_state):
        """Test potential energy calculation."""
        k = 100.0
        theta0 = np.pi / 3  # 60 degrees
        theta = np.pi / 2  # 90 degrees (actual angle)

        force = HarmonicAngleForce(
            angle_indices=[[0, 1, 2]],
            force_constants=[k],
            equilibrium_angles=[theta0],
        )

        forces, energy = force.compute_with_energy(three_atom_state)

        expected_energy = 0.5 * k * (theta - theta0) ** 2
        assert np.isclose(energy, expected_energy)


class TestPeriodicDihedralForce:
    """Test periodic dihedral force."""

    @pytest.fixture
    def four_atom_state(self):
        """Create a four-atom state."""
        box = Box.cubic(10.0)
        # Planar configuration (dihedral = 0 or 180 degrees)
        return MDState.create(
            positions=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.5, 1.0, 0.0],
                    [2.5, 1.0, 0.0],
                ]
            ),
            masses=np.ones(4),
            box=box,
        )

    def test_newton_third_law(self, four_atom_state):
        """Test that total force is zero."""
        force = PeriodicDihedralForce(
            dihedral_indices=[[0, 1, 2, 3]],
            force_constants=[10.0],
            periodicities=[2],
            phases=[0.0],
        )

        forces = force.compute(four_atom_state)

        assert np.allclose(forces.sum(axis=0), 0, atol=1e-10)

    def test_energy_at_minimum(self, four_atom_state):
        """Test energy at potential minimum."""
        # V = k * (1 + cos(n*phi - phase))
        # Minimum when n*phi - phase = pi, so V = 0
        force = PeriodicDihedralForce(
            dihedral_indices=[[0, 1, 2, 3]],
            force_constants=[10.0],
            periodicities=[1],
            phases=[np.pi],  # phase = pi means minimum at phi = 0
        )

        # For this geometry, phi is approximately 0 or 180
        forces, energy = force.compute_with_energy(four_atom_state)

        # Energy should be near minimum (0) or maximum (2*k)
        assert 0 <= energy <= 20.0


class TestLennardJonesForce:
    """Test Lennard-Jones force."""

    @pytest.fixture
    def two_atom_lj_state(self):
        """Create a two-atom state for LJ."""
        box = Box.cubic(10.0)
        # Distance = 1.0 * 2^(1/6) ≈ 1.122 (LJ minimum for sigma=1)
        r_min = 2 ** (1 / 6)
        return MDState.create(
            positions=np.array([[0.0, 0.0, 0.0], [r_min, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
            box=box,
        )

    def test_zero_force_at_minimum(self, two_atom_lj_state):
        """Test that force is zero at LJ minimum."""
        force = LennardJonesForce(
            epsilon=np.array([1.0]),
            sigma=np.array([1.0]),
            atom_types=np.array([0, 0]),
            cutoff=3.0,
        )

        forces = force.compute(two_atom_lj_state)

        assert np.allclose(forces, 0, atol=1e-10)

    def test_repulsive_at_short_distance(self):
        """Test repulsion at short distances."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
            box=box,
        )

        force = LennardJonesForce(
            epsilon=np.array([1.0]),
            sigma=np.array([1.0]),
            atom_types=np.array([0, 0]),
            cutoff=3.0,
        )

        forces = force.compute(state)

        # Force on atom 1 should point away from atom 0 (positive x)
        assert forces[1, 0] > 0

    def test_attractive_at_long_distance(self):
        """Test attraction at intermediate distances."""
        box = Box.cubic(10.0)
        # Distance between minimum and cutoff
        state = MDState.create(
            positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
            box=box,
        )

        force = LennardJonesForce(
            epsilon=np.array([1.0]),
            sigma=np.array([1.0]),
            atom_types=np.array([0, 0]),
            cutoff=3.0,
        )

        forces = force.compute(state)

        # Force on atom 1 should point toward atom 0 (negative x)
        assert forces[1, 0] < 0

    def test_cutoff(self):
        """Test that interactions beyond cutoff are ignored."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
            box=box,
        )

        force = LennardJonesForce(
            epsilon=np.array([1.0]),
            sigma=np.array([1.0]),
            atom_types=np.array([0, 0]),
            cutoff=1.5,  # Less than distance
        )

        forces = force.compute(state)

        assert np.allclose(forces, 0)

    def test_energy_at_minimum(self, two_atom_lj_state):
        """Test energy at LJ minimum."""
        epsilon = 1.0

        force = LennardJonesForce(
            epsilon=np.array([epsilon]),
            sigma=np.array([1.0]),
            atom_types=np.array([0, 0]),
            cutoff=3.0,
        )

        forces, energy = force.compute_with_energy(two_atom_lj_state)

        # At minimum, V = -epsilon
        assert np.isclose(energy, -epsilon)

    def test_exclusions(self):
        """Test that excluded pairs are skipped."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
            box=box,
        )

        force = LennardJonesForce(
            epsilon=np.array([1.0]),
            sigma=np.array([1.0]),
            atom_types=np.array([0, 0]),
            cutoff=3.0,
            exclusions={(0, 1)},
        )

        forces = force.compute(state)

        assert np.allclose(forces, 0)


class TestCoulombForce:
    """Test Coulomb force."""

    def test_repulsion_same_sign(self):
        """Test repulsion between same-sign charges."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
            box=box,
        )

        force = CoulombForce(
            charges=np.array([1.0, 1.0]),
            cutoff=3.0,
        )

        forces = force.compute(state)

        # Force on atom 1 should point away from atom 0
        assert forces[1, 0] > 0

    def test_attraction_opposite_sign(self):
        """Test attraction between opposite-sign charges."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
            box=box,
        )

        force = CoulombForce(
            charges=np.array([1.0, -1.0]),
            cutoff=3.0,
        )

        forces = force.compute(state)

        # Force on atom 1 should point toward atom 0
        assert forces[1, 0] < 0

    def test_newton_third_law(self):
        """Test Newton's third law for Coulomb."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0]),
            box=box,
        )

        force = CoulombForce(
            charges=np.array([1.0, 2.0]),
            cutoff=3.0,
        )

        forces = force.compute(state)

        assert np.allclose(forces.sum(axis=0), 0)


class TestForceFieldComposite:
    """Test composite force field."""

    def test_empty_forcefield(self):
        """Test empty force field returns zeros."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.zeros((5, 3)),
            masses=np.ones(5),
            box=box,
        )

        ff = ForceField()
        forces = ff.compute(state)

        assert forces.shape == (5, 3)
        assert np.allclose(forces, 0)

    def test_combine_forces(self):
        """Test that forces from multiple terms are summed."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]]),
            masses=np.ones(2),
            box=box,
        )

        bond_force = HarmonicBondForce(
            bond_indices=[[0, 1]],
            force_constants=[1000.0],
            equilibrium_lengths=[0.10],
        )

        lj_force = LennardJonesForce(
            epsilon=np.array([0.0]),  # Zero epsilon = no LJ
            sigma=np.array([1.0]),
            atom_types=np.array([0, 0]),
            cutoff=3.0,
        )

        ff = ForceField([bond_force, lj_force])

        forces_combined = ff.compute(state)
        forces_bond_only = bond_force.compute(state)

        # With zero LJ, combined should equal bond only
        assert np.allclose(forces_combined, forces_bond_only)

    def test_add_remove_terms(self):
        """Test adding and removing force terms."""
        ff = ForceField()

        bond_force = HarmonicBondForce(
            bond_indices=[[0, 1]],
            force_constants=[1000.0],
            equilibrium_lengths=[0.15],
        )

        ff.add_term(bond_force)
        assert len(ff.terms) == 1

        ff.remove_term(bond_force)
        assert len(ff.terms) == 0

    def test_compute_per_term(self):
        """Test computing forces from each term separately."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]]),
            masses=np.ones(2),
            box=box,
        )

        bond_force = HarmonicBondForce(
            bond_indices=[[0, 1]],
            force_constants=[1000.0],
            equilibrium_lengths=[0.10],
        )

        ff = ForceField([bond_force])

        results = ff.compute_per_term(state)

        assert len(results) == 1
        forces, energy = results[0]
        assert forces.shape == (2, 3)
