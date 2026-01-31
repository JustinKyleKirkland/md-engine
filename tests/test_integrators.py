"""Tests for integrator implementations."""

import numpy as np
import pytest

from mdcore.integrators.baoab import BAOABIntegrator
from mdcore.integrators.langevin import LangevinIntegrator
from mdcore.integrators.thermostats import (
    AndersenThermostat,
    BerendsenThermostat,
    NoseHooverThermostat,
    VelocityRescaleThermostat,
)
from mdcore.integrators.velocity_verlet import (
    LeapfrogIntegrator,
    VelocityVerletIntegrator,
)
from mdcore.system.box import Box
from mdcore.system.state import MDState


@pytest.fixture
def harmonic_state():
    """Create a state with two atoms in a harmonic potential."""
    box = Box.cubic(10.0)
    # Two atoms at equilibrium distance with some velocity
    return MDState.create(
        positions=np.array([[4.5, 5.0, 5.0], [5.5, 5.0, 5.0]]),
        masses=np.array([1.0, 1.0]),
        box=box,
        velocities=np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
    )


@pytest.fixture
def zero_force():
    """Zero forces for testing."""
    return np.zeros((2, 3))


@pytest.fixture
def simple_force():
    """Simple forces pushing atoms apart."""
    return np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


class TestVelocityVerletIntegrator:
    """Test Velocity Verlet integrator (leapfrog formulation)."""

    def test_timestep_property(self):
        """Test timestep property."""
        integrator = VelocityVerletIntegrator(dt=0.002)
        assert integrator.timestep == 0.002

    def test_step_updates_time(self, harmonic_state, zero_force):
        """Test that step updates time correctly."""
        dt = 0.002
        integrator = VelocityVerletIntegrator(dt=dt)

        new_state = integrator.step(harmonic_state, zero_force)

        assert new_state.time == pytest.approx(dt)
        assert new_state.step == 1

    def test_step_updates_positions(self, harmonic_state, zero_force):
        """Test that positions are updated."""
        dt = 0.002
        integrator = VelocityVerletIntegrator(dt=dt)

        initial_pos = harmonic_state.positions.copy()
        new_state = integrator.step(harmonic_state, zero_force)

        # With velocity and no force, position changes
        assert not np.allclose(new_state.positions, initial_pos)

    def test_constant_velocity_no_force(self, harmonic_state, zero_force):
        """Test constant velocity motion with no forces."""
        dt = 0.001
        integrator = VelocityVerletIntegrator(dt=dt)

        # Run several steps
        state = harmonic_state
        for _ in range(10):
            state = integrator.step(state, zero_force)

        # Velocity should remain constant (leapfrog preserves this)
        assert np.allclose(state.velocities, harmonic_state.velocities, atol=1e-10)

    def test_momentum_conservation_no_external_force(self, harmonic_state):
        """Test momentum conservation with internal forces only."""
        dt = 0.001
        integrator = VelocityVerletIntegrator(dt=dt)

        initial_momentum = np.sum(
            harmonic_state.masses[:, np.newaxis] * harmonic_state.velocities, axis=0
        )

        # Internal force (Newton's third law)
        internal_force = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])

        state = harmonic_state
        for _ in range(100):
            state = integrator.step(state, internal_force)

        final_momentum = np.sum(state.masses[:, np.newaxis] * state.velocities, axis=0)

        assert np.allclose(initial_momentum, final_momentum, atol=1e-8)

    def test_reset(self, harmonic_state, simple_force):
        """Test reset clears internal state."""
        integrator = VelocityVerletIntegrator(dt=0.001)

        # Run a step
        integrator.step(harmonic_state, simple_force)

        # Reset
        integrator.reset()

        # Should be able to start fresh
        new_state = integrator.step(harmonic_state, simple_force)
        assert new_state.step == 1

    def test_energy_conservation_harmonic_oscillator(self):
        """Test energy conservation for a harmonic oscillator."""
        # Single particle in harmonic potential
        box = Box.cubic(20.0)
        k = 1.0  # Spring constant
        m = 1.0  # Mass

        # Initial conditions: displaced from equilibrium
        x0 = 1.0
        v0 = 0.0
        state = MDState.create(
            positions=np.array([[10.0 + x0, 10.0, 10.0]]),
            masses=np.array([m]),
            box=box,
            velocities=np.array([[v0, 0.0, 0.0]]),
        )

        # Smaller timestep for better energy conservation
        dt = 0.001
        integrator = VelocityVerletIntegrator(dt=dt)

        # Track energy
        energies = []
        n_steps = 1000

        for _ in range(n_steps):
            # Harmonic force: F = -k * (x - x_eq)
            displacement = state.positions[0, 0] - 10.0
            force = np.array([[-k * displacement, 0.0, 0.0]])

            # Compute energy before step
            ke = 0.5 * m * np.sum(state.velocities**2)
            pe = 0.5 * k * displacement**2
            energies.append(ke + pe)

            state = integrator.step(state, force)

        energies = np.array(energies)

        # Energy should be conserved to high precision with small dt
        rel_fluctuation = np.std(energies) / np.mean(energies)
        assert rel_fluctuation < 0.001, f"Energy fluctuation {rel_fluctuation} too high"

    def test_deterministic_integrator(self, harmonic_state, simple_force):
        """Test that integrator is deterministic when reset between runs."""
        integrator = VelocityVerletIntegrator(dt=0.001)

        # First run
        state1 = integrator.step(harmonic_state, simple_force)
        state2 = integrator.step(state1, simple_force)

        # Reset integrator for second run
        integrator.reset()

        # Second run with same inputs should give identical results
        state1_again = integrator.step(harmonic_state, simple_force)
        state2_again = integrator.step(state1_again, simple_force)

        # Should be exactly equal after reset
        np.testing.assert_array_equal(state2.positions, state2_again.positions)
        np.testing.assert_array_equal(state2.velocities, state2_again.velocities)

    def test_symplecticity_phase_space_volume(self):
        """Test symplectic property by checking phase space volume conservation."""
        # Two nearby trajectories should maintain their separation
        box = Box.cubic(20.0)

        state1 = MDState.create(
            positions=np.array([[10.0, 10.0, 10.0]]),
            masses=np.array([1.0]),
            box=box,
            velocities=np.array([[1.0, 0.0, 0.0]]),
        )

        state2 = MDState.create(
            positions=np.array([[10.0 + 1e-6, 10.0, 10.0]]),
            masses=np.array([1.0]),
            box=box,
            velocities=np.array([[1.0, 0.0, 0.0]]),
        )

        dt = 0.001
        integrator1 = VelocityVerletIntegrator(dt=dt)
        integrator2 = VelocityVerletIntegrator(dt=dt)

        # Run with no forces (free particle)
        zero_force = np.zeros((1, 3))
        for _ in range(100):
            state1 = integrator1.step(state1, zero_force)
            state2 = integrator2.step(state2, zero_force)

        # Phase space separation should be preserved (linear growth is OK)
        pos_diff = np.linalg.norm(state1.positions - state2.positions)
        assert pos_diff < 1e-4, "Trajectories diverged too much"


class TestLeapfrogIntegrator:
    """Test Leapfrog integrator."""

    def test_timestep_property(self):
        """Test timestep property."""
        integrator = LeapfrogIntegrator(dt=0.002)
        assert integrator.timestep == 0.002

    def test_step_updates_state(self, harmonic_state, simple_force):
        """Test that step updates state."""
        integrator = LeapfrogIntegrator(dt=0.001)

        new_state = integrator.step(harmonic_state, simple_force)

        assert new_state.step == 1
        assert not np.allclose(new_state.positions, harmonic_state.positions)


class TestLangevinIntegrator:
    """Test Langevin integrator."""

    def test_temperature_property(self):
        """Test temperature property."""
        integrator = LangevinIntegrator(dt=0.002, temperature=300.0, friction=1.0)
        assert integrator.temperature == 300.0

    def test_friction_property(self):
        """Test friction property."""
        integrator = LangevinIntegrator(dt=0.002, temperature=300.0, friction=2.0)
        assert integrator.friction == 2.0

    def test_reproducibility_with_seed(self, harmonic_state, zero_force):
        """Test that same seed gives same results."""
        integrator1 = LangevinIntegrator(
            dt=0.001, temperature=300.0, friction=1.0, seed=42
        )
        integrator2 = LangevinIntegrator(
            dt=0.001, temperature=300.0, friction=1.0, seed=42
        )

        state1 = integrator1.step(harmonic_state, zero_force)
        state2 = integrator2.step(harmonic_state, zero_force)

        assert np.allclose(state1.velocities, state2.velocities)

    def test_different_seeds_different_results(self, harmonic_state, zero_force):
        """Test that different seeds give different results."""
        integrator1 = LangevinIntegrator(
            dt=0.001, temperature=300.0, friction=1.0, seed=42
        )
        integrator2 = LangevinIntegrator(
            dt=0.001, temperature=300.0, friction=1.0, seed=123
        )

        state1 = integrator1.step(harmonic_state, zero_force)
        state2 = integrator2.step(harmonic_state, zero_force)

        assert not np.allclose(state1.velocities, state2.velocities)


class TestBAOABIntegrator:
    """Test BAOAB integrator."""

    def test_temperature_property(self):
        """Test temperature property."""
        integrator = BAOABIntegrator(dt=0.002, temperature=300.0, friction=1.0)
        assert integrator.temperature == 300.0

    def test_temperature_setter(self):
        """Test temperature setter updates coefficients."""
        integrator = BAOABIntegrator(dt=0.002, temperature=300.0, friction=1.0)
        integrator.temperature = 350.0
        assert integrator.temperature == 350.0

    def test_friction_setter(self):
        """Test friction setter updates coefficients."""
        integrator = BAOABIntegrator(dt=0.002, temperature=300.0, friction=1.0)
        integrator.friction = 2.0
        assert integrator.friction == 2.0

    def test_step_updates_state(self, harmonic_state, zero_force):
        """Test that step updates state."""
        integrator = BAOABIntegrator(dt=0.001, temperature=300.0, friction=1.0, seed=42)

        new_state = integrator.step(harmonic_state, zero_force)

        assert new_state.step == 1
        assert new_state.time > harmonic_state.time

    def test_step_with_new_forces(self, harmonic_state, simple_force):
        """Test step_with_new_forces method."""
        integrator = BAOABIntegrator(dt=0.001, temperature=300.0, friction=1.0, seed=42)

        new_force = simple_force * 1.1
        new_state = integrator.step_with_new_forces(
            harmonic_state, simple_force, new_force
        )

        assert new_state.step == 1


class TestVelocityRescaleThermostat:
    """Test velocity rescaling thermostat."""

    @pytest.fixture
    def moving_state(self):
        """Create state with kinetic energy."""
        box = Box.cubic(10.0)
        return MDState.create(
            positions=np.array([[5.0, 5.0, 5.0]] * 10),
            masses=np.ones(10),
            box=box,
            velocities=np.random.randn(10, 3) * 0.1,
        )

    def test_rescales_to_target(self, moving_state):
        """Test that temperature is rescaled to target."""
        thermostat = VelocityRescaleThermostat(temperature=300.0)

        new_state = thermostat.apply(moving_state)

        assert np.isclose(new_state.temperature, 300.0, rtol=1e-6)

    def test_handles_zero_velocity(self):
        """Test that zero velocity state is handled."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.zeros((5, 3)),
            masses=np.ones(5),
            box=box,
        )

        thermostat = VelocityRescaleThermostat(temperature=300.0)
        new_state = thermostat.apply(state)

        # Should return unchanged
        assert np.allclose(new_state.velocities, 0)


class TestBerendsenThermostat:
    """Test Berendsen thermostat."""

    @pytest.fixture
    def hot_state(self):
        """Create state with high kinetic energy."""
        box = Box.cubic(10.0)
        return MDState.create(
            positions=np.zeros((10, 3)),
            masses=np.ones(10),
            box=box,
            velocities=np.random.randn(10, 3) * 1.0,
        )

    def test_cools_hot_system(self, hot_state):
        """Test that hot system cools toward target."""
        target_temp = 300.0
        thermostat = BerendsenThermostat(
            temperature=target_temp,
            tau=0.1,
            dt=0.001,
        )

        initial_temp = hot_state.temperature

        state = hot_state
        for _ in range(1000):
            state = thermostat.apply(state)

        # Should be closer to target
        final_temp = state.temperature
        assert abs(final_temp - target_temp) < abs(initial_temp - target_temp)


class TestAndersenThermostat:
    """Test Andersen thermostat."""

    def test_changes_some_velocities(self):
        """Test that some velocities are changed."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.zeros((100, 3)),
            masses=np.ones(100),
            box=box,
            velocities=np.zeros((100, 3)),
        )

        thermostat = AndersenThermostat(
            temperature=300.0,
            collision_frequency=100.0,  # High frequency
            dt=0.001,
            seed=42,
        )

        new_state = thermostat.apply(state)

        # Some velocities should have changed
        assert not np.allclose(new_state.velocities, 0)


class TestNoseHooverThermostat:
    """Test Nose-Hoover thermostat."""

    def test_initialization(self):
        """Test thermostat initializes correctly."""
        thermostat = NoseHooverThermostat(
            temperature=300.0,
            tau=0.1,
            dt=0.001,
        )

        assert thermostat.target_temperature == 300.0

    def test_reset(self):
        """Test reset method."""
        thermostat = NoseHooverThermostat(
            temperature=300.0,
            tau=0.1,
            dt=0.001,
        )

        thermostat.reset()
        assert np.allclose(thermostat._xi, 0)
