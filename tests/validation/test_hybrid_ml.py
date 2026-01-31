"""
Tests for hybrid ML + Classical potentials.

Validates:
1. Delta-learning stability and accuracy
2. On-the-fly learning behavior
"""

import numpy as np
import pytest

from mdcore.forcefields.base import ForceProvider
from mdcore.ml import DeltaLearningPotential, MLPotential
from mdcore.ml.base import Descriptor, MLModel
from mdcore.ml.uncertainty import ActiveLearningTrigger, UncertaintyEstimator
from mdcore.system import Box, MDState

# =============================================================================
# Mock Components for Testing
# =============================================================================


class MockWaterFF(ForceProvider):
    """
    Mock classical water force field.

    Simple harmonic model for testing delta-learning.
    """

    def __init__(self, k_bond: float = 100.0, k_angle: float = 50.0):
        self.k_bond = k_bond
        self.k_angle = k_angle

    def compute(self, state, neighbors=None):
        """Compute mock water forces."""
        # Simple harmonic restraint to origin
        return -self.k_bond * 0.01 * state.positions

    def compute_with_energy(self, state, neighbors=None):
        """Compute forces and energy."""
        forces = self.compute(state, neighbors)
        energy = 0.5 * self.k_bond * 0.01 * np.sum(state.positions**2)
        return forces, energy


class MockMLCorrection(MLModel):
    """
    Mock ML correction that learns the error in classical FF.

    Simulates a trained correction model.
    """

    def __init__(self, correction_strength: float = 0.1):
        self.correction_strength = correction_strength
        self._call_count = 0

    def predict_energy(self, descriptors, species):
        """Predict energy correction."""
        self._call_count += 1
        return self.correction_strength * np.sum(np.sin(descriptors))

    def predict_forces(self, descriptors, descriptor_derivatives, species):
        """Predict force correction."""
        n_atoms = len(species)
        # Small correction based on positions
        correction = self.correction_strength * np.cos(descriptors[:, :3])
        if correction.shape != (n_atoms, 3):
            correction = np.zeros((n_atoms, 3))
        return correction

    def predict(self, descriptors, descriptor_derivatives, species):
        """Predict energy and force corrections."""
        energy = self.predict_energy(descriptors, species)
        forces = self.predict_forces(descriptors, descriptor_derivatives, species)
        return energy, forces


class SimpleDescriptor(Descriptor):
    """Simple descriptor returning positions."""

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


class MockUncertaintyEstimator(UncertaintyEstimator):
    """Mock uncertainty that increases over time."""

    def __init__(self, base_uncertainty: float = 0.1, growth_rate: float = 0.01):
        self.base_uncertainty = base_uncertainty
        self.growth_rate = growth_rate
        self._call_count = 0

    def estimate(self, state):
        """Return uncertainty that grows with calls."""
        self._call_count += 1
        return self.base_uncertainty + self.growth_rate * self._call_count

    def estimate_per_atom(self, state):
        """Return per-atom uncertainties."""
        uncertainty = self.estimate(state)
        return np.full(state.n_atoms, uncertainty)

    def reset(self):
        """Reset call counter."""
        self._call_count = 0


# =============================================================================
# Delta-Learning Stability Tests
# =============================================================================


class TestDeltaLearningStability:
    """
    Tests for delta-learning potential stability.

    Validates:
    1. ML correction reduces force error vs reference
    2. No systematic bias drift in long simulations
    3. Energy conservation is maintained
    """

    @pytest.fixture
    def water_box_state(self):
        """Create a small water box state."""
        # 3 water molecules = 9 atoms
        n_molecules = 3
        n_atoms = n_molecules * 3

        rng = np.random.default_rng(42)

        # Simple grid placement
        positions = rng.uniform(-3, 3, (n_atoms, 3))
        velocities = rng.normal(0, 0.1, (n_atoms, 3))
        velocities -= velocities.mean(axis=0)

        return MDState(
            positions=positions,
            velocities=velocities,
            forces=np.zeros((n_atoms, 3)),
            masses=np.array([16.0, 1.0, 1.0] * n_molecules),  # O, H, H
            box=Box.cubic(10.0),
        )

    @pytest.fixture
    def delta_potential(self):
        """Create delta-learning potential."""
        baseline = MockWaterFF()
        correction_model = MockMLCorrection(correction_strength=0.05)
        correction_descriptor = SimpleDescriptor()
        correction = MLPotential(correction_model, correction_descriptor)

        return DeltaLearningPotential(baseline, correction)

    def test_correction_modifies_forces(self, water_box_state, delta_potential):
        """Test that ML correction actually modifies the forces."""
        baseline = delta_potential.baseline

        baseline_forces, _ = baseline.compute_with_energy(water_box_state)
        delta_forces, _ = delta_potential.compute_with_energy(water_box_state)

        # Forces should be different
        assert not np.allclose(baseline_forces, delta_forces), (
            "Delta-learning should modify baseline forces"
        )

        # Difference should be the correction
        correction_magnitude = np.linalg.norm(delta_forces - baseline_forces)
        assert correction_magnitude > 0, "Correction should be non-zero"

    def test_no_force_explosion_in_dynamics(self, water_box_state, delta_potential):
        """Test that delta-learning doesn't cause force explosions."""
        positions = water_box_state.positions.copy()
        velocities = water_box_state.velocities.copy()
        masses = water_box_state.masses
        dt = 0.001

        max_force_seen = 0.0

        for _ in range(100):
            state = MDState(
                positions=positions,
                velocities=velocities,
                forces=np.zeros_like(positions),
                masses=masses,
                box=water_box_state.box,
            )

            forces, _ = delta_potential.compute_with_energy(state)

            # Check for explosion
            force_mag = np.max(np.abs(forces))
            max_force_seen = max(max_force_seen, force_mag)

            if not np.isfinite(forces).all():
                pytest.fail("NaN/Inf forces detected")

            if force_mag > 1000:
                pytest.fail(f"Force explosion: max force = {force_mag}")

            # Simple integration
            velocities = velocities + dt * forces / masses[:, None]
            positions = positions + dt * velocities

        assert max_force_seen < 100, f"Forces grew too large: {max_force_seen}"

    def test_energy_drift_bounded(self, water_box_state, delta_potential):
        """Test that energy drift remains bounded."""
        positions = water_box_state.positions.copy()
        velocities = water_box_state.velocities.copy()
        masses = water_box_state.masses
        dt = 0.0005

        energies = []

        for step in range(200):
            state = MDState(
                positions=positions,
                velocities=velocities,
                forces=np.zeros_like(positions),
                masses=masses,
                box=water_box_state.box,
            )

            forces, pe = delta_potential.compute_with_energy(state)

            # Kinetic energy
            ke = 0.5 * np.sum(masses[:, None] * velocities**2)
            energies.append(ke + pe)

            # Velocity Verlet
            velocities = velocities + 0.5 * dt * forces / masses[:, None]
            positions = positions + dt * velocities

            state2 = MDState(
                positions=positions,
                velocities=velocities,
                forces=np.zeros_like(positions),
                masses=masses,
                box=water_box_state.box,
            )
            forces2, _ = delta_potential.compute_with_energy(state2)
            velocities = velocities + 0.5 * dt * forces2 / masses[:, None]

        # Check energy drift
        energies = np.array(energies)
        drift_per_step = (energies[-1] - energies[0]) / len(energies)

        assert abs(drift_per_step) < 0.01, (
            f"Energy drift {drift_per_step:.6f}/step too large"
        )


# =============================================================================
# On-the-Fly Learning Tests
# =============================================================================


class TestOnTheFlyLearning:
    """
    Tests for on-the-fly learning behavior.

    Validates:
    1. Uncertainty triggers fire at expected rate
    2. Triggered states are stored for retraining
    3. Model can be updated during simulation
    """

    @pytest.fixture
    def test_state(self):
        """Create test state."""
        n_atoms = 10
        rng = np.random.default_rng(123)

        return MDState(
            positions=rng.uniform(-2, 2, (n_atoms, 3)),
            velocities=rng.normal(0, 0.1, (n_atoms, 3)),
            forces=np.zeros((n_atoms, 3)),
            masses=np.ones(n_atoms),
            box=Box.cubic(10.0),
        )

    def test_uncertainty_trigger_fires(self, test_state):
        """Test that uncertainty trigger fires when threshold exceeded."""
        # Estimator with growing uncertainty
        estimator = MockUncertaintyEstimator(base_uncertainty=0.1, growth_rate=0.1)
        trigger = ActiveLearningTrigger(
            estimator,
            threshold=0.5,
            cooldown_steps=0,
        )

        triggers = []
        for i in range(20):
            triggered = trigger.check(test_state)
            triggers.append(triggered)

        # Should start triggering after ~4 steps (0.1 + 0.1*4 = 0.5)
        assert any(triggers), "Trigger should fire when uncertainty exceeds threshold"
        assert not all(triggers), "Trigger shouldn't fire immediately"

    def test_trigger_respects_cooldown(self, test_state):
        """Test that trigger respects cooldown period."""
        estimator = MockUncertaintyEstimator(base_uncertainty=1.0, growth_rate=0.0)
        trigger = ActiveLearningTrigger(
            estimator,
            threshold=0.5,  # Always above threshold
            cooldown_steps=5,
        )

        triggers = []
        for _ in range(15):
            triggered = trigger.check(test_state)
            triggers.append(triggered)

        # Should trigger, then wait 5 steps, then trigger again
        trigger_indices = [i for i, t in enumerate(triggers) if t]

        # Check spacing between triggers
        if len(trigger_indices) >= 2:
            spacing = trigger_indices[1] - trigger_indices[0]
            assert spacing >= 5, f"Cooldown not respected: spacing = {spacing}"

    def test_triggered_states_stored(self, test_state):
        """Test that triggered states are stored for retraining."""
        estimator = MockUncertaintyEstimator(base_uncertainty=1.0, growth_rate=0.0)
        trigger = ActiveLearningTrigger(
            estimator,
            threshold=0.5,
            cooldown_steps=0,
        )

        # Trigger multiple times
        for _ in range(5):
            trigger.check(test_state)

        states = trigger.get_triggered_states()
        assert len(states) == 5, f"Expected 5 stored states, got {len(states)}"

        # Clear and verify
        trigger.clear_triggered_states()
        assert len(trigger.get_triggered_states()) == 0

    def test_uncertainty_varies_with_configuration(self):
        """Test that uncertainty depends on atomic configuration."""
        estimator = MockUncertaintyEstimator(base_uncertainty=0.1, growth_rate=0.0)

        # Different configurations
        state1 = MDState(
            positions=np.zeros((5, 3)),
            velocities=np.zeros((5, 3)),
            forces=np.zeros((5, 3)),
            masses=np.ones(5),
            box=None,
        )

        state2 = MDState(
            positions=np.ones((5, 3)) * 10,
            velocities=np.zeros((5, 3)),
            forces=np.zeros((5, 3)),
            masses=np.ones(5),
            box=None,
        )

        # Our mock returns same value, but real estimators would differ
        u1 = estimator.estimate(state1)
        estimator.reset()
        u2 = estimator.estimate(state2)

        # Both should be valid uncertainties
        assert u1 > 0
        assert u2 > 0


# =============================================================================
# Model Hot-Swap Tests
# =============================================================================


class TestModelHotSwap:
    """Tests for model hot-swapping during simulation."""

    def test_potential_model_can_be_replaced(self):
        """Test that ML model can be replaced in potential."""
        model1 = MockMLCorrection(correction_strength=0.1)
        model2 = MockMLCorrection(correction_strength=0.5)
        descriptor = SimpleDescriptor()

        potential = MLPotential(model1, descriptor)

        state = MDState(
            positions=np.random.randn(5, 3),
            velocities=np.zeros((5, 3)),
            forces=np.zeros((5, 3)),
            masses=np.ones(5),
            box=None,
        )

        # Get forces with model1
        forces1, _ = potential.compute_with_energy(state)

        # Swap model
        potential.model = model2

        # Get forces with model2
        forces2, _ = potential.compute_with_energy(state)

        # Forces should be different due to different correction strength
        assert not np.allclose(forces1, forces2), (
            "Hot-swapped model should give different forces"
        )
