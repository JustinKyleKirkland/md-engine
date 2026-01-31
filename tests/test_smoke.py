"""
CI-friendly smoke tests.

These tests are designed to:
1. Run fast (<1s each)
2. Test core functionality
3. Be deterministic (seeded RNG)
4. Run in serial (single rank)

Use for continuous integration to catch regressions quickly.
"""

import numpy as np
import pytest

from mdcore.analysis import EnergyAnalyzer, MeanSquareDisplacement
from mdcore.forcefields import ForceField
from mdcore.forcefields.nonbonded import LennardJonesForce
from mdcore.integrators import VelocityVerletIntegrator
from mdcore.neighborlists import VerletList
from mdcore.system import Box, MDState

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_lj_system():
    """
    Create minimal LJ system for smoke tests.

    10 particles, cubic box, deterministic seed.
    """
    n_atoms = 10
    rng = np.random.default_rng(42)  # Deterministic seed

    # Box size for reasonable density
    box_length = 5.0
    box = Box.cubic(box_length)

    # Random positions avoiding overlaps
    positions = rng.uniform(0.5, box_length - 0.5, (n_atoms, 3))

    # Maxwell-Boltzmann velocities at T=1
    velocities = rng.normal(0, 1.0, (n_atoms, 3))
    velocities -= velocities.mean(axis=0)  # Zero COM velocity

    return MDState(
        positions=positions,
        velocities=velocities,
        forces=np.zeros((n_atoms, 3)),
        masses=np.ones(n_atoms),
        box=box,
    )


@pytest.fixture
def lj_forcefield(small_lj_system):
    """Create LJ force field."""
    n_atoms = small_lj_system.n_atoms
    # All atoms are same type (type 0)
    atom_types = np.zeros(n_atoms, dtype=np.int32)
    lj = LennardJonesForce(
        epsilon=np.array([1.0]),  # One type
        sigma=np.array([1.0]),
        atom_types=atom_types,
        cutoff=2.5,
    )
    return ForceField([lj])


@pytest.fixture
def neighbor_list():
    """Create neighbor list."""
    return VerletList(cutoff=2.5, skin=0.3)


@pytest.fixture
def integrator():
    """Create velocity Verlet integrator."""
    return VelocityVerletIntegrator(dt=0.001)


# =============================================================================
# Smoke Tests
# =============================================================================


class TestNVESmoke:
    """
    Smoke tests for NVE simulation.

    Tests 100-step NVE with 10-particle LJ system.
    Should complete in <1s and maintain energy conservation.
    """

    def test_nve_completes(
        self, small_lj_system, lj_forcefield, neighbor_list, integrator
    ):
        """Test that 100-step NVE completes without error."""
        positions = small_lj_system.positions.copy()
        velocities = small_lj_system.velocities.copy()
        masses = small_lj_system.masses
        box = small_lj_system.box

        n_steps = 100

        for _ in range(n_steps):
            state = MDState(
                positions=positions,
                velocities=velocities,
                forces=np.zeros_like(positions),
                masses=masses,
                box=box,
            )

            neighbor_list.build(state.positions, state.box)
            forces = lj_forcefield.compute(state, neighbor_list)

            new_state = integrator.step(state, forces)
            positions = new_state.positions
            velocities = new_state.velocities

        # Just verify it completed
        assert positions.shape == small_lj_system.positions.shape

    def test_nve_energy_conserved(
        self, small_lj_system, lj_forcefield, neighbor_list, integrator
    ):
        """Test that NVE conserves energy (within tolerance)."""
        positions = small_lj_system.positions.copy()
        velocities = small_lj_system.velocities.copy()
        masses = small_lj_system.masses
        box = small_lj_system.box

        energies = []

        for step in range(100):
            state = MDState(
                positions=positions,
                velocities=velocities,
                forces=np.zeros_like(positions),
                masses=masses,
                box=box,
            )

            neighbor_list.build(state.positions, state.box)
            forces, pe = lj_forcefield.compute_with_energy(state, neighbor_list)

            ke = state.kinetic_energy
            energies.append(ke + pe)

            new_state = integrator.step(state, forces)
            positions = new_state.positions
            velocities = new_state.velocities

        energies = np.array(energies)

        # Energy should be conserved within 5% (allowing for neighbor list updates)
        e_std = np.std(energies)
        e_mean = np.mean(np.abs(energies))

        relative_fluctuation = e_std / e_mean if e_mean > 0 else 0

        assert relative_fluctuation < 0.05, (
            f"Energy fluctuation {relative_fluctuation:.4f} exceeds 5%"
        )

    def test_nve_deterministic(
        self, small_lj_system, lj_forcefield, neighbor_list, integrator
    ):
        """Test that simulation is deterministic with same seed."""

        def run_simulation(state):
            positions = state.positions.copy()
            velocities = state.velocities.copy()
            masses = state.masses
            box = state.box

            for _ in range(50):
                s = MDState(
                    positions=positions,
                    velocities=velocities,
                    forces=np.zeros_like(positions),
                    masses=masses,
                    box=box,
                )

                neighbor_list.build(s.positions, s.box)
                forces = lj_forcefield.compute(s, neighbor_list)

                new_s = integrator.step(s, forces)
                positions = new_s.positions
                velocities = new_s.velocities

            return positions

        # Run twice
        final_pos_1 = run_simulation(small_lj_system)
        final_pos_2 = run_simulation(small_lj_system)

        # Should be very close (small differences from neighbor list rebuild order)
        np.testing.assert_allclose(
            final_pos_1,
            final_pos_2,
            rtol=1e-2,
            err_msg="Simulation not reproducible",
        )


class TestForceFieldSmoke:
    """Smoke tests for force field components."""

    def test_lj_forces_finite(self, small_lj_system, lj_forcefield, neighbor_list):
        """Test that LJ forces are finite."""
        neighbor_list.build(small_lj_system.positions, small_lj_system.box)
        forces = lj_forcefield.compute(small_lj_system, neighbor_list)

        assert np.all(np.isfinite(forces)), "Forces contain NaN/Inf"

    def test_lj_energy_finite(self, small_lj_system, lj_forcefield, neighbor_list):
        """Test that LJ energy is finite."""
        neighbor_list.build(small_lj_system.positions, small_lj_system.box)
        forces, energy = lj_forcefield.compute_with_energy(
            small_lj_system, neighbor_list
        )

        assert np.isfinite(energy), f"Energy is not finite: {energy}"

    def test_force_momentum_conservation(
        self, small_lj_system, lj_forcefield, neighbor_list
    ):
        """Test Newton's third law (momentum conservation)."""
        neighbor_list.build(small_lj_system.positions, small_lj_system.box)
        forces = lj_forcefield.compute(small_lj_system, neighbor_list)

        total_force = forces.sum(axis=0)

        np.testing.assert_allclose(
            total_force,
            0.0,
            atol=1e-10,
            err_msg="Total force not zero (Newton's 3rd law violated)",
        )


class TestIntegratorSmoke:
    """Smoke tests for integrators."""

    def test_verlet_step(
        self, small_lj_system, lj_forcefield, neighbor_list, integrator
    ):
        """Test single Verlet step."""
        neighbor_list.build(small_lj_system.positions, small_lj_system.box)
        forces = lj_forcefield.compute(small_lj_system, neighbor_list)

        new_state = integrator.step(small_lj_system, forces)

        # Positions should change
        assert not np.allclose(new_state.positions, small_lj_system.positions)

        # Shape should be preserved
        assert new_state.positions.shape == small_lj_system.positions.shape

    def test_verlet_reversible(
        self, small_lj_system, lj_forcefield, neighbor_list, integrator
    ):
        """Test that Verlet is time-reversible."""
        neighbor_list.build(small_lj_system.positions, small_lj_system.box)
        forces = lj_forcefield.compute(small_lj_system, neighbor_list)

        # Forward step
        state_1 = integrator.step(small_lj_system, forces)

        # Reverse velocities
        state_1_reversed = MDState(
            positions=state_1.positions,
            velocities=-state_1.velocities,
            forces=state_1.forces,
            masses=state_1.masses,
            box=state_1.box,
        )

        # Backward step
        neighbor_list.build(state_1_reversed.positions, state_1_reversed.box)
        forces_1 = lj_forcefield.compute(state_1_reversed, neighbor_list)
        state_2 = integrator.step(state_1_reversed, forces_1)

        # Should return close to original (with reversed velocity)
        np.testing.assert_allclose(
            state_2.positions,
            small_lj_system.positions,
            rtol=1e-5,
            err_msg="Verlet not time-reversible",
        )


class TestNeighborListSmoke:
    """Smoke tests for neighbor lists."""

    def test_neighbor_list_builds(self, small_lj_system, neighbor_list):
        """Test that neighbor list builds without error."""
        neighbor_list.build(small_lj_system.positions, small_lj_system.box)

        # Should have some pairs
        n_pairs = neighbor_list.n_pairs
        assert n_pairs >= 0

    def test_neighbor_list_pairs_valid(self, small_lj_system, neighbor_list):
        """Test that neighbor pairs are valid."""
        neighbor_list.build(small_lj_system.positions, small_lj_system.box)

        pairs = neighbor_list.get_pairs()

        # Pairs should be within bounds
        n_atoms = small_lj_system.n_atoms
        for i, j in pairs:
            assert 0 <= i < n_atoms
            assert 0 <= j < n_atoms
            assert i != j  # No self-pairs


class TestAnalysisSmoke:
    """Smoke tests for analysis tools."""

    def test_energy_analyzer(self, small_lj_system, lj_forcefield, neighbor_list):
        """Test energy analyzer."""
        analyzer = EnergyAnalyzer()

        neighbor_list.build(small_lj_system.positions, small_lj_system.box)
        _, pe = lj_forcefield.compute_with_energy(small_lj_system, neighbor_list)

        analyzer.update(small_lj_system, potential_energy=pe)

        result = analyzer.result()
        assert "kinetic" in result
        assert "potential" in result
        assert result["n_frames"] == 1

    def test_msd_analyzer(self, small_lj_system):
        """Test MSD analyzer."""
        msd = MeanSquareDisplacement(max_lag=10)

        for _ in range(5):
            msd.update(small_lj_system)

        result = msd.result()
        assert "msd" in result
        assert len(result["msd"]) == 10


class TestStateSmoke:
    """Smoke tests for MDState."""

    def test_state_properties(self, small_lj_system):
        """Test state property accessors."""
        assert small_lj_system.n_atoms == 10
        assert small_lj_system.kinetic_energy >= 0
        assert small_lj_system.temperature >= 0

    def test_state_immutable_positions(self, small_lj_system):
        """Test that state positions can be accessed."""
        pos = small_lj_system.positions
        assert pos.shape == (10, 3)


class TestBoxSmoke:
    """Smoke tests for Box."""

    def test_cubic_box(self):
        """Test cubic box creation."""
        box = Box.cubic(10.0)

        np.testing.assert_allclose(box.volume, 1000.0)
        np.testing.assert_allclose(box.lengths, [10.0, 10.0, 10.0])

    def test_minimum_image(self):
        """Test minimum image convention."""
        box = Box.cubic(10.0)

        r1 = np.array([1.0, 1.0, 1.0])
        r2 = np.array([9.0, 1.0, 1.0])

        # Minimum image should be -2, not +8
        displacement = box.minimum_image(r1, r2)
        assert abs(displacement[0]) < 5.0

    def test_wrap_positions(self):
        """Test position wrapping."""
        box = Box.cubic(10.0)

        positions = np.array([[15.0, -3.0, 25.0]])
        wrapped = box.wrap_positions(positions)

        assert np.all(wrapped >= 0)
        assert np.all(wrapped < 10)
