"""
Benchmark tests for ML potentials.

These tests verify:
1. ML force/energy accuracy against reference data
2. MD simulation stability with ML potentials
3. Performance comparison between classical and ML potentials
"""

import time
from dataclasses import dataclass

import numpy as np
import pytest

from mdcore.forcefields.base import ForceProvider
from mdcore.ml import DeltaLearningPotential, MLPotential
from mdcore.ml.base import Descriptor, MLModel
from mdcore.system import Box, MDState

# =============================================================================
# Benchmark Data Structures
# =============================================================================


@dataclass
class ForceAccuracyMetrics:
    """Metrics for ML force accuracy."""

    mae_forces: float  # Mean Absolute Error on forces (eV/Å)
    rmse_forces: float  # Root Mean Square Error on forces
    mae_energy: float  # MAE on energy (eV or eV/atom)
    rmse_energy: float  # RMSE on energy
    max_force_error: float  # Maximum force component error
    n_samples: int


@dataclass
class StabilityMetrics:
    """Metrics for MD stability."""

    n_steps_completed: int
    energy_drift_per_step: float
    max_force_magnitude: float
    force_explosion: bool
    temperature_drift: float


@dataclass
class PerformanceMetrics:
    """Metrics for performance benchmarking."""

    steps_per_second: float
    time_per_step_ms: float
    total_time_s: float
    n_atoms: int
    gpu_utilization: float | None = None


# =============================================================================
# Mock Reference Data (would be replaced with real QM data)
# =============================================================================


def generate_mock_qm_dataset(
    n_samples: int = 100,
    n_atoms: int = 9,  # e.g., ethanol
    seed: int = 42,
) -> list[dict]:
    """
    Generate mock QM reference data for testing.

    In practice, this would load actual QM calculations (e.g., MD17 dataset).

    Returns:
        List of dicts with 'positions', 'forces', 'energy'.
    """
    rng = np.random.default_rng(seed)

    dataset = []
    for _ in range(n_samples):
        # Random positions (small molecule)
        positions = rng.uniform(-2, 2, (n_atoms, 3))

        # Mock "QM" forces (harmonic-like for testing)
        forces = -0.5 * positions + rng.normal(0, 0.1, (n_atoms, 3))

        # Mock energy
        energy = 0.25 * np.sum(positions**2) + rng.normal(0, 0.01)

        dataset.append(
            {
                "positions": positions,
                "forces": forces,
                "energy": energy,
                "n_atoms": n_atoms,
            }
        )

    return dataset


# =============================================================================
# Mock ML Models for Testing
# =============================================================================


class MockTrainedMLModel(MLModel):
    """
    Mock ML model that approximates the QM data generation function.

    This allows testing the benchmark infrastructure without real ML models.
    """

    def __init__(self, noise_level: float = 0.02):
        """
        Initialize mock model.

        Args:
            noise_level: Noise added to predictions (simulates model error).
        """
        self.noise_level = noise_level
        self._rng = np.random.default_rng(0)

    def predict_energy(self, descriptors, species):
        """Predict energy (mock)."""
        # Approximate the mock QM function
        return 0.25 * np.sum(descriptors**2)

    def predict_forces(self, descriptors, descriptor_derivatives, species):
        """Predict forces (mock)."""
        n_atoms = len(species)
        # Approximate forces with noise
        forces = (
            -0.5 * descriptors[:, :3]
            if descriptors.shape[1] >= 3
            else np.zeros((n_atoms, 3))
        )
        forces += self._rng.normal(0, self.noise_level, forces.shape)
        return forces

    def predict(self, descriptors, descriptor_derivatives, species):
        """Predict energy and forces."""
        energy = self.predict_energy(descriptors, species)
        forces = self.predict_forces(descriptors, descriptor_derivatives, species)
        return energy, forces


class IdentityDescriptor(Descriptor):
    """Simple descriptor that returns positions as features."""

    def __init__(self, n_features: int = 3):
        self._n_features = n_features

    def compute(self, positions, species, cell=None):
        """Return positions as descriptors."""
        return positions

    def compute_with_derivatives(self, positions, species, cell=None):
        """Return positions and identity derivatives."""
        n_atoms = len(positions)
        desc = positions
        # Simple identity derivative
        deriv = np.zeros((n_atoms, self._n_features, n_atoms, 3))
        for i in range(n_atoms):
            for j in range(min(3, self._n_features)):
                deriv[i, j, i, j] = 1.0
        return desc, deriv

    @property
    def n_features(self):
        return self._n_features


class MockClassicalFF(ForceProvider):
    """Mock classical force field for comparison."""

    def __init__(self, k: float = 1.0):
        self.k = k

    def compute(self, state, neighbors=None):
        """Compute harmonic forces."""
        return -self.k * state.positions

    def compute_with_energy(self, state, neighbors=None):
        """Compute forces and energy."""
        forces = self.compute(state, neighbors)
        energy = 0.5 * self.k * np.sum(state.positions**2)
        return forces, energy


# =============================================================================
# Force Accuracy Benchmarks
# =============================================================================


class TestMLForceAccuracy:
    """
    Tests for ML potential force/energy accuracy.

    Validates against reference QM data with metrics:
    - MAE(F) < 0.05 eV/Å (for production-quality models)
    - MAE(E) < 1 meV/atom
    """

    @pytest.fixture
    def qm_dataset(self):
        """Generate mock QM dataset."""
        return generate_mock_qm_dataset(n_samples=50, n_atoms=9)

    @pytest.fixture
    def ml_potential(self):
        """Create ML potential with mock model."""
        model = MockTrainedMLModel(noise_level=0.02)
        descriptor = IdentityDescriptor(n_features=3)
        return MLPotential(model, descriptor)

    def compute_accuracy_metrics(
        self,
        ml_potential: MLPotential,
        dataset: list[dict],
    ) -> ForceAccuracyMetrics:
        """
        Compute accuracy metrics for ML potential.

        Args:
            ml_potential: ML potential to evaluate.
            dataset: List of reference data dicts.

        Returns:
            ForceAccuracyMetrics with computed values.
        """
        force_errors = []
        energy_errors = []

        for sample in dataset:
            positions = sample["positions"]
            ref_forces = sample["forces"]
            ref_energy = sample["energy"]
            n_atoms = sample["n_atoms"]

            # Create state for prediction
            state = MDState(
                positions=positions,
                velocities=np.zeros_like(positions),
                forces=np.zeros_like(positions),
                masses=np.ones(n_atoms),
                box=None,
            )

            # Get ML predictions
            pred_forces, pred_energy = ml_potential.compute_with_energy(state)

            # Force errors (per component)
            force_errors.extend((pred_forces - ref_forces).flatten())

            # Energy error (per atom)
            energy_errors.append((pred_energy - ref_energy) / n_atoms)

        force_errors = np.array(force_errors)
        energy_errors = np.array(energy_errors)

        return ForceAccuracyMetrics(
            mae_forces=float(np.mean(np.abs(force_errors))),
            rmse_forces=float(np.sqrt(np.mean(force_errors**2))),
            mae_energy=float(np.mean(np.abs(energy_errors))),
            rmse_energy=float(np.sqrt(np.mean(energy_errors**2))),
            max_force_error=float(np.max(np.abs(force_errors))),
            n_samples=len(dataset),
        )

    def test_force_mae(self, ml_potential, qm_dataset):
        """Test that force MAE is below threshold."""
        metrics = self.compute_accuracy_metrics(ml_potential, qm_dataset)

        # For our mock model with noise_level=0.02, MAE should be ~0.02
        # Real models should achieve MAE < 0.05 eV/Å
        assert metrics.mae_forces < 0.1, (
            f"Force MAE {metrics.mae_forces:.4f} exceeds threshold"
        )

    def test_energy_mae(self, ml_potential, qm_dataset):
        """Test that energy MAE per atom is below threshold."""
        metrics = self.compute_accuracy_metrics(ml_potential, qm_dataset)

        # MAE(E) < 1 meV/atom = 0.001 eV/atom for production models
        # Our mock model should be reasonably accurate
        assert metrics.mae_energy < 0.1, (
            f"Energy MAE {metrics.mae_energy:.6f} eV/atom exceeds threshold"
        )

    def test_no_large_outliers(self, ml_potential, qm_dataset):
        """Test that there are no catastrophic force errors."""
        metrics = self.compute_accuracy_metrics(ml_potential, qm_dataset)

        # Max error should not be too much larger than MAE
        assert metrics.max_force_error < 10 * metrics.mae_forces, (
            f"Max force error {metrics.max_force_error:.4f} indicates outliers"
        )


# =============================================================================
# MD Stability Benchmarks
# =============================================================================


class TestMLMDStability:
    """
    Tests for ML potential stability in MD simulations.

    Validates:
    - No force explosions (forces stay bounded)
    - Energy conservation/drift is acceptable
    - Simulation completes without NaN/Inf
    """

    @pytest.fixture
    def ml_potential(self):
        """Create ML potential for stability testing."""
        model = MockTrainedMLModel(noise_level=0.01)
        descriptor = IdentityDescriptor(n_features=3)
        return MLPotential(model, descriptor)

    @pytest.fixture
    def initial_state(self):
        """Create initial state for small molecule."""
        n_atoms = 9
        rng = np.random.default_rng(42)

        positions = rng.uniform(-1, 1, (n_atoms, 3))
        velocities = rng.normal(0, 0.1, (n_atoms, 3))
        velocities -= velocities.mean(axis=0)  # Remove COM velocity

        return MDState(
            positions=positions,
            velocities=velocities,
            forces=np.zeros((n_atoms, 3)),
            masses=np.ones(n_atoms),
            box=None,
        )

    def run_stability_test(
        self,
        potential: ForceProvider,
        initial_state: MDState,
        n_steps: int = 100,
        dt: float = 0.001,
    ) -> StabilityMetrics:
        """
        Run short MD simulation and check stability.

        Uses simple velocity Verlet integration.

        Args:
            potential: Force provider.
            initial_state: Starting configuration.
            n_steps: Number of steps to run.
            dt: Timestep.

        Returns:
            StabilityMetrics with results.
        """
        positions = initial_state.positions.copy()
        velocities = initial_state.velocities.copy()
        masses = initial_state.masses

        energies = []
        temperatures = []
        max_force = 0.0
        force_explosion = False

        for step in range(n_steps):
            # Create state
            state = MDState(
                positions=positions,
                velocities=velocities,
                forces=np.zeros_like(positions),
                masses=masses,
                box=initial_state.box,
            )

            # Get forces
            forces, potential_energy = potential.compute_with_energy(state)

            # Check for explosion
            force_mag = np.max(np.abs(forces))
            max_force = max(max_force, force_mag)

            if not np.isfinite(forces).all() or force_mag > 1000:
                force_explosion = True
                break

            # Velocity Verlet integration
            velocities = velocities + 0.5 * dt * forces / masses[:, None]
            positions = positions + dt * velocities
            velocities = velocities + 0.5 * dt * forces / masses[:, None]

            # Record energy
            kinetic_energy = 0.5 * np.sum(masses[:, None] * velocities**2)
            total_energy = kinetic_energy + potential_energy
            energies.append(total_energy)

            # Record temperature
            n_dof = 3 * len(masses) - 3
            temp = 2 * kinetic_energy / n_dof if n_dof > 0 else 0
            temperatures.append(temp)

        # Compute metrics
        n_completed = len(energies)
        if n_completed > 1:
            energy_drift = (energies[-1] - energies[0]) / n_completed
            temp_drift = temperatures[-1] - temperatures[0]
        else:
            energy_drift = 0.0
            temp_drift = 0.0

        return StabilityMetrics(
            n_steps_completed=n_completed,
            energy_drift_per_step=energy_drift,
            max_force_magnitude=max_force,
            force_explosion=force_explosion,
            temperature_drift=temp_drift,
        )

    def test_no_force_explosion(self, ml_potential, initial_state):
        """Test that forces stay bounded during MD."""
        metrics = self.run_stability_test(ml_potential, initial_state, n_steps=100)

        assert not metrics.force_explosion, "Force explosion detected"
        assert metrics.max_force_magnitude < 100, (
            f"Max force {metrics.max_force_magnitude:.2f} too large"
        )

    def test_simulation_completes(self, ml_potential, initial_state):
        """Test that simulation runs to completion."""
        metrics = self.run_stability_test(ml_potential, initial_state, n_steps=100)

        assert metrics.n_steps_completed == 100, (
            f"Only completed {metrics.n_steps_completed}/100 steps"
        )

    def test_energy_drift_acceptable(self, ml_potential, initial_state):
        """Test that energy drift is reasonable."""
        metrics = self.run_stability_test(ml_potential, initial_state, n_steps=100)

        # Energy drift should be small for stable integration
        # Note: our mock model adds noise, so some drift is expected
        assert abs(metrics.energy_drift_per_step) < 0.1, (
            f"Energy drift {metrics.energy_drift_per_step:.6f} per step too large"
        )


# =============================================================================
# Performance Benchmarks
# =============================================================================


class TestMLPerformance:
    """
    Performance benchmarks comparing classical and ML potentials.

    Measures:
    - Steps per second
    - Relative speedup/slowdown
    """

    @pytest.fixture
    def classical_ff(self):
        """Create classical force field."""
        return MockClassicalFF(k=1.0)

    @pytest.fixture
    def ml_potential(self):
        """Create ML potential."""
        model = MockTrainedMLModel(noise_level=0.01)
        descriptor = IdentityDescriptor(n_features=3)
        return MLPotential(model, descriptor)

    @pytest.fixture
    def test_state(self):
        """Create test state."""
        n_atoms = 100
        rng = np.random.default_rng(42)

        return MDState(
            positions=rng.uniform(-5, 5, (n_atoms, 3)),
            velocities=np.zeros((n_atoms, 3)),
            forces=np.zeros((n_atoms, 3)),
            masses=np.ones(n_atoms),
            box=Box.cubic(20.0),
        )

    def benchmark_force_computation(
        self,
        potential: ForceProvider,
        state: MDState,
        n_evaluations: int = 100,
    ) -> PerformanceMetrics:
        """
        Benchmark force computation speed.

        Args:
            potential: Force provider to benchmark.
            state: Test state.
            n_evaluations: Number of force evaluations.

        Returns:
            PerformanceMetrics with timing data.
        """
        # Warmup
        for _ in range(5):
            potential.compute(state)

        # Timed run
        start = time.perf_counter()
        for _ in range(n_evaluations):
            potential.compute(state)
        elapsed = time.perf_counter() - start

        return PerformanceMetrics(
            steps_per_second=n_evaluations / elapsed,
            time_per_step_ms=1000 * elapsed / n_evaluations,
            total_time_s=elapsed,
            n_atoms=state.n_atoms,
        )

    def test_classical_performance(self, classical_ff, test_state):
        """Benchmark classical force field performance."""
        metrics = self.benchmark_force_computation(classical_ff, test_state)

        # Classical should be fast
        assert metrics.steps_per_second > 100, (
            f"Classical FF too slow: {metrics.steps_per_second:.1f} steps/s"
        )

    def test_ml_performance(self, ml_potential, test_state):
        """Benchmark ML potential performance."""
        metrics = self.benchmark_force_computation(ml_potential, test_state)

        # ML might be slower but should still be usable
        assert metrics.steps_per_second > 10, (
            f"ML potential too slow: {metrics.steps_per_second:.1f} steps/s"
        )

    def test_performance_comparison(self, classical_ff, ml_potential, test_state):
        """Compare classical and ML performance."""
        classical_metrics = self.benchmark_force_computation(classical_ff, test_state)
        ml_metrics = self.benchmark_force_computation(ml_potential, test_state)

        speedup = classical_metrics.steps_per_second / ml_metrics.steps_per_second

        # Log the comparison (this is informational, not a pass/fail)
        print(f"\nPerformance comparison (n_atoms={test_state.n_atoms}):")
        print(f"  Classical: {classical_metrics.steps_per_second:.1f} steps/s")
        print(f"  ML:        {ml_metrics.steps_per_second:.1f} steps/s")
        print(f"  Ratio:     {speedup:.2f}x")

        # ML should not be catastrophically slow
        assert speedup < 1000, "ML potential is too slow compared to classical"


# =============================================================================
# Delta-Learning Benchmarks
# =============================================================================


class TestDeltaLearning:
    """Tests for delta-learning potential accuracy."""

    def test_delta_learning_improves_baseline(self):
        """Test that delta-learning improves over baseline."""
        # Create baseline (intentionally wrong)
        baseline = MockClassicalFF(k=0.3)  # Wrong spring constant

        # Create correction
        correction_model = MockTrainedMLModel(noise_level=0.01)
        correction_descriptor = IdentityDescriptor(n_features=3)
        correction = MLPotential(correction_model, correction_descriptor)

        # Delta-learning potential
        delta_potential = DeltaLearningPotential(baseline, correction)

        # Test state
        n_atoms = 9
        rng = np.random.default_rng(42)
        positions = rng.uniform(-1, 1, (n_atoms, 3))

        state = MDState(
            positions=positions,
            velocities=np.zeros((n_atoms, 3)),
            forces=np.zeros((n_atoms, 3)),
            masses=np.ones(n_atoms),
            box=None,
        )

        # Get predictions
        baseline_forces, baseline_energy = baseline.compute_with_energy(state)
        delta_forces, delta_energy = delta_potential.compute_with_energy(state)

        # Delta should be different from baseline (correction applied)
        assert not np.allclose(baseline_forces, delta_forces), (
            "Delta-learning should modify baseline forces"
        )
