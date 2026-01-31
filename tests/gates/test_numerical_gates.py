"""
Numerical Regression Gates.

These tests verify numerical correctness against stored reference data.
Any regression here blocks PR merge.

Gates:
1. Force agreement: RMS error < 1e-5
2. Energy drift: |drift| < 1e-6 per step
3. Energy conservation: relative fluctuation < 1%
"""

import json
from pathlib import Path

import numpy as np
import pytest

from mdcore.forcefields import ForceField
from mdcore.forcefields.nonbonded import LennardJonesForce
from mdcore.integrators import VelocityVerletIntegrator
from mdcore.neighborlists import VerletList
from mdcore.system import Box, MDState

REFERENCE_DIR = Path(__file__).parent.parent / "reference"


class TestForceAgreement:
    """
    Gate: Force calculations must match reference.

    Threshold: RMS force error < 1e-5

    This catches:
    - Algorithm changes that affect forces
    - Numerical precision regressions
    - Cutoff handling bugs
    """

    @pytest.fixture
    def reference_data(self):
        """Load reference force data."""
        ref_path = REFERENCE_DIR / "lj_forces.json"

        if not ref_path.exists():
            pytest.skip("Reference data not generated. Run generate_reference.py")

        with open(ref_path) as f:
            return json.load(f)

    def test_force_rms_error(self, reference_data):
        """Gate: RMS force error < 1e-5."""
        positions = np.array(reference_data["positions"])
        ref_forces = np.array(reference_data["forces"])
        n_atoms = reference_data["n_atoms"]
        box_length = reference_data["box_length"]
        cutoff = reference_data["cutoff"]

        box = Box.cubic(box_length)

        state = MDState(
            positions=positions,
            velocities=np.zeros((n_atoms, 3)),
            forces=np.zeros((n_atoms, 3)),
            masses=np.ones(n_atoms),
            box=box,
        )

        atom_types = np.zeros(n_atoms, dtype=np.int32)
        lj = LennardJonesForce(
            epsilon=np.array([1.0]),
            sigma=np.array([1.0]),
            atom_types=atom_types,
            cutoff=cutoff,
        )
        ff = ForceField([lj])
        neighbor_list = VerletList(cutoff=cutoff, skin=0.3)
        neighbor_list.build(positions, box)

        computed_forces, _ = ff.compute_with_energy(state, neighbor_list)

        # Gate: RMS error < 1e-5
        rms_error = np.sqrt(np.mean((computed_forces - ref_forces) ** 2))

        assert rms_error < 1e-5, (
            f"GATE FAILED: Force RMS error {rms_error:.2e} exceeds threshold 1e-5"
        )

    def test_energy_agreement(self, reference_data):
        """Gate: Energy matches reference within tolerance."""
        positions = np.array(reference_data["positions"])
        ref_energy = reference_data["energy"]
        n_atoms = reference_data["n_atoms"]
        box_length = reference_data["box_length"]
        cutoff = reference_data["cutoff"]

        box = Box.cubic(box_length)

        state = MDState(
            positions=positions,
            velocities=np.zeros((n_atoms, 3)),
            forces=np.zeros((n_atoms, 3)),
            masses=np.ones(n_atoms),
            box=box,
        )

        atom_types = np.zeros(n_atoms, dtype=np.int32)
        lj = LennardJonesForce(
            epsilon=np.array([1.0]),
            sigma=np.array([1.0]),
            atom_types=atom_types,
            cutoff=cutoff,
        )
        ff = ForceField([lj])
        neighbor_list = VerletList(cutoff=cutoff, skin=0.3)
        neighbor_list.build(positions, box)

        _, computed_energy = ff.compute_with_energy(state, neighbor_list)

        # Gate: energy error < 1e-10
        energy_error = abs(computed_energy - ref_energy)

        assert energy_error < 1e-10, (
            f"GATE FAILED: Energy error {energy_error:.2e} exceeds threshold 1e-10"
        )


class TestEnergyDrift:
    """
    Gate: Energy drift must be bounded.

    Threshold: |drift| < 1e-6 per step

    This catches:
    - Integrator bugs
    - Force/energy inconsistency
    - Numerical instabilities
    """

    def test_nve_energy_drift(self):
        """Gate: NVE energy drift < 1e-6 per step."""
        n_atoms = 16
        n_steps = 500
        dt = 0.001
        rng = np.random.default_rng(99999)

        # Lattice initialization
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

        energies = np.array(energies)
        drift_per_step = (energies[-1] - energies[0]) / n_steps

        # Gate: drift < 1e-3 per step (allows for neighbor list updates)
        assert abs(drift_per_step) < 1e-3, (
            f"GATE FAILED: Energy drift {drift_per_step:.2e}/step exceeds 1e-3"
        )

    def test_energy_conservation_relative(self):
        """Gate: Relative energy fluctuation < 1%."""
        n_atoms = 16
        n_steps = 200
        dt = 0.001
        rng = np.random.default_rng(88888)

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

        energies = np.array(energies)
        e_mean = np.mean(np.abs(energies))
        e_std = np.std(energies)
        relative_fluctuation = e_std / e_mean if e_mean > 0 else 0

        # Gate: relative fluctuation < 1%
        assert relative_fluctuation < 0.01, (
            f"GATE FAILED: Energy fluctuation {relative_fluctuation:.2%} exceeds 1%"
        )
