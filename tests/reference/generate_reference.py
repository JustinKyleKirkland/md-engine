"""
Generate reference data for regression testing.

This script generates reference forces, energies, and trajectories
using our implementation. In a production scenario, you would also
generate reference data from OpenMM/LAMMPS for comparison.

Usage:
    python -m tests.reference.generate_reference
"""

import json
from pathlib import Path

import numpy as np

from mdcore.forcefields import ForceField
from mdcore.forcefields.nonbonded import LennardJonesForce
from mdcore.integrators import VelocityVerletIntegrator
from mdcore.neighborlists import VerletList
from mdcore.system import Box, MDState

REFERENCE_DIR = Path(__file__).parent


def generate_lj_reference_forces():
    """Generate reference LJ forces for a small system."""
    print("Generating LJ reference forces...")

    # Fixed configuration for reproducibility using lattice
    n_atoms = 27  # 3x3x3 cube
    n_side = 3
    rng = np.random.default_rng(12345)

    spacing = 1.5
    box_length = n_side * spacing
    box = Box.cubic(box_length)

    # Lattice positions to avoid overlaps
    positions = []
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                positions.append(
                    [
                        (ix + 0.5) * spacing,
                        (iy + 0.5) * spacing,
                        (iz + 0.5) * spacing,
                    ]
                )
    positions = np.array(positions)
    positions += rng.uniform(-0.1, 0.1, positions.shape)

    state = MDState(
        positions=positions,
        velocities=np.zeros((n_atoms, 3)),
        forces=np.zeros((n_atoms, 3)),
        masses=np.ones(n_atoms),
        box=box,
    )

    # Compute forces
    atom_types = np.zeros(n_atoms, dtype=np.int32)
    lj = LennardJonesForce(
        epsilon=np.array([1.0]),
        sigma=np.array([1.0]),
        atom_types=atom_types,
        cutoff=2.5,
    )
    ff = ForceField([lj])
    neighbor_list = VerletList(cutoff=2.5, skin=0.3)
    neighbor_list.build(positions, box)

    forces, energy = ff.compute_with_energy(state, neighbor_list)

    # Save reference data
    ref_data = {
        "positions": positions.tolist(),
        "forces": forces.tolist(),
        "energy": float(energy),
        "n_atoms": n_atoms,
        "box_length": box_length,
        "cutoff": 2.5,
        "seed": 12345,
    }

    ref_path = REFERENCE_DIR / "lj_forces.json"
    with open(ref_path, "w") as f:
        json.dump(ref_data, f, indent=2)

    # Also save as numpy for faster loading
    np.save(REFERENCE_DIR / "lj_positions.npy", positions)
    np.save(REFERENCE_DIR / "lj_forces.npy", forces)

    print(f"  Saved to {ref_path}")
    print(f"  Energy: {energy:.6f}")
    print(f"  Force RMS: {np.sqrt(np.mean(forces**2)):.6f}")


def generate_nve_reference_trajectory():
    """Generate reference NVE trajectory."""
    print("Generating NVE reference trajectory...")

    n_atoms = 16
    n_steps = 100
    dt = 0.001
    rng = np.random.default_rng(54321)

    # Lattice initialization to avoid overlaps
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

    velocities = rng.normal(0, 0.3, (n_atoms, 3))
    velocities -= velocities.mean(axis=0)

    # Setup
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

    # Run trajectory
    trajectory_positions = [positions.copy()]
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

        trajectory_positions.append(positions.copy())

    trajectory_positions = np.array(trajectory_positions)
    energies = np.array(energies)

    # Save
    np.save(REFERENCE_DIR / "nve_trajectory.npy", trajectory_positions)
    np.save(REFERENCE_DIR / "nve_energies.npy", energies)

    energy_drift = (energies[-1] - energies[0]) / n_steps

    print(f"  Trajectory shape: {trajectory_positions.shape}")
    print(f"  Energy drift per step: {energy_drift:.2e}")


if __name__ == "__main__":
    generate_lj_reference_forces()
    generate_nve_reference_trajectory()
    print("\nReference data generation complete!")
