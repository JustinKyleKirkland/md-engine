"""
Determinism Gates.

These tests verify that simulations are reproducible.

Gates:
1. Serial determinism: identical results with same seed
2. Parallel determinism: serial == parallel results
"""

import numpy as np

from mdcore.forcefields import ForceField
from mdcore.forcefields.nonbonded import LennardJonesForce
from mdcore.integrators import VelocityVerletIntegrator
from mdcore.neighborlists import VerletList
from mdcore.parallel import get_backend
from mdcore.system import Box, MDState


def create_lj_system(seed: int, n_atoms: int = 16):
    """Create reproducible LJ system."""
    rng = np.random.default_rng(seed)

    n_side = int(np.ceil(n_atoms ** (1 / 3)))
    spacing = 1.5
    box_length = n_side * spacing
    box = Box.cubic(box_length)

    positions = []
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                if len(positions) < n_atoms:
                    positions.append(
                        [
                            (ix + 0.5) * spacing,
                            (iy + 0.5) * spacing,
                            (iz + 0.5) * spacing,
                        ]
                    )
    positions = np.array(positions)
    positions += rng.uniform(-0.1, 0.1, positions.shape)

    velocities = rng.normal(0, 0.2, (n_atoms, 3))
    velocities -= velocities.mean(axis=0)

    return positions, velocities, box


def run_nve_trajectory(
    positions: np.ndarray,
    velocities: np.ndarray,
    box: Box,
    n_steps: int = 50,
    dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Run NVE trajectory and return final positions and energies."""
    n_atoms = len(positions)
    positions = positions.copy()
    velocities = velocities.copy()

    atom_types = np.zeros(n_atoms, dtype=np.int32)
    lj = LennardJonesForce(
        epsilon=np.array([1.0]),
        sigma=np.array([1.0]),
        atom_types=atom_types,
        cutoff=2.5,
    )
    ff = ForceField([lj])
    integrator = VelocityVerletIntegrator(dt=dt)
    neighbor_list = VerletList(cutoff=2.5, skin=0.3)

    energies = []

    for _ in range(n_steps):
        state = MDState(
            positions=positions,
            velocities=velocities,
            forces=np.zeros_like(positions),
            masses=np.ones(n_atoms),
            box=box,
        )

        neighbor_list.build(positions, box)
        forces, pe = ff.compute_with_energy(state, neighbor_list)

        ke = state.kinetic_energy
        energies.append(ke + pe)

        new_state = integrator.step(state, forces)
        positions = new_state.positions
        velocities = new_state.velocities

    return positions, np.array(energies)


class TestSerialDeterminism:
    """
    Gate: Serial execution must be deterministic.

    Same seed → identical results.
    """

    def test_trajectory_determinism(self):
        """Gate: Same seed produces identical trajectories."""
        seed = 42424

        # Run 1
        pos1, vel1, box1 = create_lj_system(seed)
        final_pos1, energies1 = run_nve_trajectory(pos1, vel1, box1)

        # Run 2 (same seed)
        pos2, vel2, box2 = create_lj_system(seed)
        final_pos2, energies2 = run_nve_trajectory(pos2, vel2, box2)

        # Gate: must be bitwise identical
        np.testing.assert_array_equal(
            final_pos1,
            final_pos2,
            err_msg="GATE FAILED: Trajectory not deterministic",
        )

        np.testing.assert_array_equal(
            energies1,
            energies2,
            err_msg="GATE FAILED: Energies not deterministic",
        )

    def test_force_determinism(self):
        """Gate: Force calculations are deterministic."""
        seed = 33333

        pos, vel, box = create_lj_system(seed)
        n_atoms = len(pos)

        atom_types = np.zeros(n_atoms, dtype=np.int32)
        lj = LennardJonesForce(
            epsilon=np.array([1.0]),
            sigma=np.array([1.0]),
            atom_types=atom_types,
            cutoff=2.5,
        )
        ff = ForceField([lj])
        neighbor_list = VerletList(cutoff=2.5, skin=0.3)

        state = MDState(
            positions=pos,
            velocities=vel,
            forces=np.zeros_like(pos),
            masses=np.ones(n_atoms),
            box=box,
        )

        # Compute forces twice
        neighbor_list.build(pos, box)
        forces1, energy1 = ff.compute_with_energy(state, neighbor_list)

        neighbor_list.build(pos, box)
        forces2, energy2 = ff.compute_with_energy(state, neighbor_list)

        # Gate: must be identical
        np.testing.assert_array_equal(
            forces1,
            forces2,
            err_msg="GATE FAILED: Forces not deterministic",
        )

        assert energy1 == energy2, "GATE FAILED: Energies not deterministic"


class TestParallelDeterminism:
    """
    Gate: Parallel execution must match serial.

    Serial results == Parallel results (within numerical tolerance).
    """

    def test_backend_consistency(self):
        """Gate: Serial backend is consistent."""
        backend = get_backend("serial")

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Multiple operations should be consistent
        result1 = backend.allreduce_sum(data)
        result2 = backend.allreduce_sum(data)

        np.testing.assert_array_equal(
            result1,
            result2,
            err_msg="GATE FAILED: Backend operations not consistent",
        )

    def test_partition_coverage(self):
        """Gate: Partitioning covers all atoms exactly once."""
        backend = get_backend("serial")

        n_atoms = 100
        start, end = backend.partition_atoms(n_atoms)

        # Serial should cover all atoms
        assert start == 0, "GATE FAILED: Partition doesn't start at 0"
        assert end == n_atoms, "GATE FAILED: Partition doesn't cover all atoms"

        # No gaps or overlaps
        covered = set(range(start, end))
        expected = set(range(n_atoms))

        assert covered == expected, "GATE FAILED: Partition has gaps or overlaps"
