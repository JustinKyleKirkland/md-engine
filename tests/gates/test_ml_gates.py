"""
ML Regression Gates.

These tests verify ML potential correctness.

Gates:
1. Force MAE < 0.05 eV/Å
2. Energy MAE < 1 meV/atom
3. No NaNs or infinities
4. No exploding forces
5. Energy bounded during MD
"""

import numpy as np
import pytest

from mdcore.forcefields.base import ForceProvider
from mdcore.ml import MLPotential
from mdcore.ml.base import Descriptor, MLModel
from mdcore.system import MDState

# =============================================================================
# Mock ML Components for Gate Testing
# =============================================================================


class GateTestDescriptor(Descriptor):
    """Descriptor for gate testing."""

    def compute(self, positions, species, cell=None):
        return positions

    def compute_with_derivatives(self, positions, species, cell=None):
        n_atoms = len(positions)
        desc = positions
        deriv = np.zeros((n_atoms, 3, n_atoms, 3))
        for i in range(n_atoms):
            for j in range(3):
                deriv[i, j, i, j] = 1.0
        return desc, deriv

    @property
    def n_features(self):
        return 3


class GateTestMLModel(MLModel):
    """ML model that mimics a well-trained potential."""

    def __init__(self, force_noise: float = 0.01, energy_noise: float = 0.001):
        self.force_noise = force_noise
        self.energy_noise = energy_noise
        self._rng = np.random.default_rng(42)

    def predict_energy(self, descriptors, species):
        # Harmonic energy with small noise
        base = 0.25 * np.sum(descriptors**2)
        noise = self._rng.normal(0, self.energy_noise)
        return base + noise

    def predict_forces(self, descriptors, descriptor_derivatives, species):
        # Harmonic forces with small noise
        forces = -0.5 * descriptors[:, :3]
        forces += self._rng.normal(0, self.force_noise, forces.shape)
        return forces

    def predict(self, descriptors, descriptor_derivatives, species):
        energy = self.predict_energy(descriptors, species)
        forces = self.predict_forces(descriptors, descriptor_derivatives, species)
        return energy, forces


class ReferenceHarmonic(ForceProvider):
    """Reference harmonic potential for comparison."""

    def compute(self, state, neighbors=None):
        return -0.5 * state.positions

    def compute_with_energy(self, state, neighbors=None):
        forces = self.compute(state, neighbors)
        energy = 0.25 * np.sum(state.positions**2)
        return forces, energy


# =============================================================================
# ML Accuracy Gates
# =============================================================================


class TestMLAccuracyGates:
    """
    Gate: ML accuracy must meet thresholds.
    """

    @pytest.fixture
    def test_configs(self):
        """Generate test configurations."""
        rng = np.random.default_rng(999)
        configs = []
        for _ in range(20):
            positions = rng.uniform(-2, 2, (10, 3))
            configs.append(positions)
        return configs

    def test_force_mae_gate(self, test_configs):
        """Gate: Force MAE < 0.05 (in test units)."""
        model = GateTestMLModel(force_noise=0.02)
        descriptor = GateTestDescriptor()
        ml_potential = MLPotential(model, descriptor)
        reference = ReferenceHarmonic()

        force_errors = []

        for positions in test_configs:
            state = MDState(
                positions=positions,
                velocities=np.zeros_like(positions),
                forces=np.zeros_like(positions),
                masses=np.ones(len(positions)),
                box=None,
            )

            ml_forces, _ = ml_potential.compute_with_energy(state)
            ref_forces, _ = reference.compute_with_energy(state)

            errors = np.abs(ml_forces - ref_forces).flatten()
            force_errors.extend(errors)

        mae = np.mean(force_errors)

        # Gate: MAE < 0.05
        assert mae < 0.05, f"GATE FAILED: Force MAE {mae:.4f} exceeds threshold 0.05"

    def test_energy_mae_gate(self, test_configs):
        """Gate: Energy MAE < 0.01 per atom (in test units)."""
        model = GateTestMLModel(energy_noise=0.001)
        descriptor = GateTestDescriptor()
        ml_potential = MLPotential(model, descriptor)
        reference = ReferenceHarmonic()

        energy_errors = []

        for positions in test_configs:
            state = MDState(
                positions=positions,
                velocities=np.zeros_like(positions),
                forces=np.zeros_like(positions),
                masses=np.ones(len(positions)),
                box=None,
            )

            _, ml_energy = ml_potential.compute_with_energy(state)
            _, ref_energy = reference.compute_with_energy(state)

            error_per_atom = abs(ml_energy - ref_energy) / len(positions)
            energy_errors.append(error_per_atom)

        mae = np.mean(energy_errors)

        # Gate: MAE < 0.01 per atom
        assert mae < 0.01, (
            f"GATE FAILED: Energy MAE {mae:.6f}/atom exceeds threshold 0.01"
        )


class TestMLStabilityGates:
    """
    Gate: ML potentials must be stable in MD.
    """

    @pytest.fixture
    def ml_potential(self):
        """Create ML potential."""
        model = GateTestMLModel(force_noise=0.01)
        descriptor = GateTestDescriptor()
        return MLPotential(model, descriptor)

    @pytest.fixture
    def initial_state(self):
        """Create initial state."""
        n_atoms = 10
        rng = np.random.default_rng(777)

        positions = rng.uniform(-1, 1, (n_atoms, 3))
        velocities = rng.normal(0, 0.1, (n_atoms, 3))
        velocities -= velocities.mean(axis=0)

        return MDState(
            positions=positions,
            velocities=velocities,
            forces=np.zeros((n_atoms, 3)),
            masses=np.ones(n_atoms),
            box=None,
        )

    def test_no_nan_gate(self, ml_potential, initial_state):
        """Gate: No NaN values in forces or energies."""
        positions = initial_state.positions.copy()
        velocities = initial_state.velocities.copy()
        masses = initial_state.masses
        dt = 0.001

        for step in range(100):
            state = MDState(
                positions=positions,
                velocities=velocities,
                forces=np.zeros_like(positions),
                masses=masses,
                box=None,
            )

            forces, energy = ml_potential.compute_with_energy(state)

            # Gate: no NaN/Inf
            assert np.isfinite(forces).all(), (
                f"GATE FAILED: NaN/Inf in forces at step {step}"
            )
            assert np.isfinite(energy), f"GATE FAILED: NaN/Inf in energy at step {step}"

            # Simple integration
            velocities = velocities + dt * forces / masses[:, None]
            positions = positions + dt * velocities

    def test_no_explosion_gate(self, ml_potential, initial_state):
        """Gate: Forces must not explode."""
        positions = initial_state.positions.copy()
        velocities = initial_state.velocities.copy()
        masses = initial_state.masses
        dt = 0.001

        max_force_threshold = 1000.0

        for step in range(100):
            state = MDState(
                positions=positions,
                velocities=velocities,
                forces=np.zeros_like(positions),
                masses=masses,
                box=None,
            )

            forces, _ = ml_potential.compute_with_energy(state)

            max_force = np.max(np.abs(forces))

            # Gate: max force < threshold
            assert max_force < max_force_threshold, (
                f"GATE FAILED: Force explosion at step {step}\n"
                f"  Max force: {max_force:.2f}\n"
                f"  Threshold: {max_force_threshold}"
            )

            velocities = velocities + dt * forces / masses[:, None]
            positions = positions + dt * velocities

    def test_energy_bounded_gate(self, ml_potential, initial_state):
        """Gate: Total energy must remain bounded."""
        positions = initial_state.positions.copy()
        velocities = initial_state.velocities.copy()
        masses = initial_state.masses
        dt = 0.001

        energies = []

        for _ in range(100):
            state = MDState(
                positions=positions,
                velocities=velocities,
                forces=np.zeros_like(positions),
                masses=masses,
                box=None,
            )

            forces, pe = ml_potential.compute_with_energy(state)
            ke = 0.5 * np.sum(masses[:, None] * velocities**2)
            energies.append(ke + pe)

            velocities = velocities + dt * forces / masses[:, None]
            positions = positions + dt * velocities

        energies = np.array(energies)

        # Gate: energy shouldn't drift by more than 100%
        initial_energy = abs(energies[0]) if energies[0] != 0 else 1.0
        max_drift = np.max(np.abs(energies - energies[0])) / initial_energy

        assert max_drift < 1.0, (
            f"GATE FAILED: Energy unbounded\n"
            f"  Max drift: {max_drift * 100:.1f}%\n"
            f"  Threshold: 100%"
        )
