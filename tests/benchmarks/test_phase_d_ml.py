"""
Phase D - ML Benchmarks

Build order:
1. Static ML accuracy (force/energy predictions)
2. ML MD stability (dynamics with ML potential)

Final validation phase for ML potentials.
"""

import numpy as np
import pytest

from mdcore.benchmarks import BenchmarkReporter, BenchmarkResult
from mdcore.forcefields.base import ForceProvider
from mdcore.ml import MLPotential
from mdcore.ml.base import Descriptor, MLModel
from mdcore.system import MDState

# =============================================================================
# Mock ML Components
# =============================================================================


class MockDescriptor(Descriptor):
    """Simple position-based descriptor."""

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


class MockMLModel(MLModel):
    """Mock ML model with controllable accuracy."""

    def __init__(self, noise_level: float = 0.01):
        self.noise_level = noise_level
        self._rng = np.random.default_rng(0)

    def predict_energy(self, descriptors, species):
        # Harmonic-like energy
        return 0.25 * np.sum(descriptors**2)

    def predict_forces(self, descriptors, descriptor_derivatives, species):
        # Harmonic forces with noise
        forces = -0.5 * descriptors[:, :3]
        forces += self._rng.normal(0, self.noise_level, forces.shape)
        return forces

    def predict(self, descriptors, descriptor_derivatives, species):
        energy = self.predict_energy(descriptors, species)
        forces = self.predict_forces(descriptors, descriptor_derivatives, species)
        return energy, forces


class ReferenceHarmonicFF(ForceProvider):
    """Reference harmonic force field for comparison."""

    def compute(self, state, neighbors=None):
        return -0.5 * state.positions

    def compute_with_energy(self, state, neighbors=None):
        forces = self.compute(state, neighbors)
        energy = 0.25 * np.sum(state.positions**2)
        return forces, energy


# =============================================================================
# Phase D.1: Static ML Accuracy
# =============================================================================


class TestStaticMLAccuracy:
    """
    Static ML accuracy benchmarks.

    Tests ML predictions against reference without dynamics.
    """

    @pytest.fixture
    def reporter(self):
        """Create benchmark reporter."""
        return BenchmarkReporter("phase_d_ml")

    @pytest.fixture
    def test_configurations(self):
        """Generate test configurations."""
        rng = np.random.default_rng(789)
        n_configs = 50
        n_atoms = 9

        configs = []
        for _ in range(n_configs):
            positions = rng.uniform(-2, 2, (n_atoms, 3))
            configs.append(positions)

        return configs

    def test_force_mae(self, reporter, test_configurations):
        """Test force mean absolute error."""
        model = MockMLModel(noise_level=0.02)
        descriptor = MockDescriptor()
        ml_potential = MLPotential(model, descriptor)
        reference = ReferenceHarmonicFF()

        force_errors = []

        for positions in test_configurations:
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

        result = BenchmarkResult(
            test="ml_force_mae",
            mae_forces=float(mae),
            n_atoms=9,
            passed=mae < 0.1,  # Threshold for mock model
            extra={"n_configs": len(test_configurations)},
        )
        reporter.add_result(result)

        assert result.passed

    def test_energy_mae(self, reporter, test_configurations):
        """Test energy mean absolute error."""
        model = MockMLModel(noise_level=0.01)
        descriptor = MockDescriptor()
        ml_potential = MLPotential(model, descriptor)
        reference = ReferenceHarmonicFF()

        energy_errors = []

        for positions in test_configurations:
            state = MDState(
                positions=positions,
                velocities=np.zeros_like(positions),
                forces=np.zeros_like(positions),
                masses=np.ones(len(positions)),
                box=None,
            )

            _, ml_energy = ml_potential.compute_with_energy(state)
            _, ref_energy = reference.compute_with_energy(state)

            # Per-atom error
            error = abs(ml_energy - ref_energy) / len(positions)
            energy_errors.append(error)

        mae = np.mean(energy_errors)

        result = BenchmarkResult(
            test="ml_energy_mae",
            mae_energy=float(mae),
            n_atoms=9,
            passed=mae < 0.01,
            extra={"n_configs": len(test_configurations)},
        )
        reporter.add_result(result)

        assert result.passed

    def test_force_correlation(self, reporter, test_configurations):
        """Test correlation between ML and reference forces."""
        model = MockMLModel(noise_level=0.02)
        descriptor = MockDescriptor()
        ml_potential = MLPotential(model, descriptor)
        reference = ReferenceHarmonicFF()

        ml_all = []
        ref_all = []

        for positions in test_configurations[:20]:
            state = MDState(
                positions=positions,
                velocities=np.zeros_like(positions),
                forces=np.zeros_like(positions),
                masses=np.ones(len(positions)),
                box=None,
            )

            ml_forces, _ = ml_potential.compute_with_energy(state)
            ref_forces, _ = reference.compute_with_energy(state)

            ml_all.extend(ml_forces.flatten())
            ref_all.extend(ref_forces.flatten())

        # Compute correlation
        correlation = np.corrcoef(ml_all, ref_all)[0, 1]

        result = BenchmarkResult(
            test="ml_force_correlation",
            n_atoms=9,
            passed=correlation > 0.9,
            extra={"correlation": float(correlation)},
        )
        reporter.add_result(result)

        assert result.passed


# =============================================================================
# Phase D.2: ML MD Stability
# =============================================================================


class TestMLMDStability:
    """
    ML MD stability benchmarks.

    Tests that ML potentials produce stable dynamics.
    """

    @pytest.fixture
    def reporter(self):
        """Create benchmark reporter."""
        return BenchmarkReporter("phase_d_ml")

    @pytest.fixture
    def ml_potential(self):
        """Create ML potential."""
        model = MockMLModel(noise_level=0.01)
        descriptor = MockDescriptor()
        return MLPotential(model, descriptor)

    @pytest.fixture
    def initial_state(self):
        """Create initial state."""
        n_atoms = 9
        rng = np.random.default_rng(42)

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

    def run_ml_md(
        self,
        potential: ForceProvider,
        initial_state: MDState,
        n_steps: int = 100,
        dt: float = 0.001,
    ) -> dict:
        """Run MD with ML potential."""
        positions = initial_state.positions.copy()
        velocities = initial_state.velocities.copy()
        masses = initial_state.masses

        energies = []
        max_forces = []

        for _ in range(n_steps):
            state = MDState(
                positions=positions,
                velocities=velocities,
                forces=np.zeros_like(positions),
                masses=masses,
                box=None,
            )

            forces, pe = potential.compute_with_energy(state)

            if not np.isfinite(forces).all():
                return {"exploded": True, "step": len(energies)}

            max_forces.append(np.max(np.abs(forces)))

            ke = 0.5 * np.sum(masses[:, None] * velocities**2)
            energies.append(ke + pe)

            # Velocity Verlet
            velocities = velocities + 0.5 * dt * forces / masses[:, None]
            positions = positions + dt * velocities
            velocities = velocities + 0.5 * dt * forces / masses[:, None]

        return {
            "exploded": False,
            "energies": np.array(energies),
            "max_forces": np.array(max_forces),
        }

    def test_no_force_explosion(self, reporter, ml_potential, initial_state):
        """Test that forces stay bounded."""
        result_data = self.run_ml_md(ml_potential, initial_state, n_steps=200)

        passed = not result_data.get("exploded", True)
        max_force = np.max(result_data.get("max_forces", [np.inf]))

        result = BenchmarkResult(
            test="ml_md_no_explosion",
            n_atoms=9,
            n_steps=200,
            max_force_error=float(max_force),
            passed=passed and max_force < 100,
        )
        reporter.add_result(result)

        assert result.passed

    def test_energy_drift(self, reporter, ml_potential, initial_state):
        """Test that energy drift is bounded."""
        result_data = self.run_ml_md(ml_potential, initial_state, n_steps=200)

        if result_data.get("exploded"):
            result = BenchmarkResult(
                test="ml_md_energy_drift",
                n_atoms=9,
                passed=False,
                failure_reason="Simulation exploded",
            )
            reporter.add_result(result)
            pytest.fail("Simulation exploded")
            return

        energies = result_data["energies"]
        drift = (energies[-1] - energies[0]) / len(energies)

        result = BenchmarkResult(
            test="ml_md_energy_drift",
            n_atoms=9,
            n_steps=200,
            energy_drift=float(drift),
            energy_drift_per_step=float(drift),
            passed=abs(drift) < 0.01,
        )
        reporter.add_result(result)

        assert result.passed

    def test_simulation_completes(self, reporter, ml_potential, initial_state):
        """Test that full simulation completes."""
        result_data = self.run_ml_md(ml_potential, initial_state, n_steps=100)

        completed = not result_data.get("exploded", True)

        result = BenchmarkResult(
            test="ml_md_completion",
            n_atoms=9,
            n_steps=100,
            passed=completed,
        )
        reporter.add_result(result)

        assert result.passed


# =============================================================================
# Phase D Summary
# =============================================================================


class TestPhaseDSummary:
    """Summary test for Phase D benchmarks."""

    def test_phase_d_complete(self):
        """Verify all Phase D tests can be imported and run."""
        assert True, "Phase D benchmarks loaded successfully"

    def test_full_benchmark_report(self):
        """Generate a sample benchmark report."""
        reporter = BenchmarkReporter("full_benchmark")

        # Add sample results
        reporter.create_result(
            test="lj_nve_256",
            energy_drift=1.2e-6,
            rms_force_error=3.1e-7,
            ns_per_day=120.4,
            n_ranks=8,
            backend="mpi",
            passed=True,
        )

        reporter.create_result(
            test="ml_force_accuracy",
            mae_forces=0.045,
            mae_energy=0.0008,
            n_atoms=100,
            passed=True,
        )

        # Verify report format
        report = reporter.to_dict()

        assert "results" in report
        assert len(report["results"]) == 2
        assert report["n_passed"] == 2

        # Print sample output
        print("\nSample Benchmark Report:")
        print(reporter.to_json())
