"""Tests for MD simulation engine."""

import numpy as np
import pytest

from mdcore.engines.engine import MDEngine, SimulationHook
from mdcore.engines.reporters import (
    CallbackReporter,
    CheckpointReporter,
    EnergyReporter,
    TrajectoryReporter,
)
from mdcore.forcefields.bonded.bonds import HarmonicBondForce
from mdcore.forcefields.composite import ForceField
from mdcore.forcefields.nonbonded.lj import LennardJonesForce
from mdcore.integrators.thermostats import VelocityRescaleThermostat
from mdcore.integrators.velocity_verlet import VelocityVerletIntegrator
from mdcore.neighborlists.verlet import VerletList
from mdcore.system.box import Box
from mdcore.system.state import MDState


@pytest.fixture
def simple_system():
    """Create a simple two-atom system with harmonic bond."""
    box = Box.cubic(10.0)
    state = MDState.create(
        positions=np.array([[4.5, 5.0, 5.0], [5.5, 5.0, 5.0]]),
        masses=np.array([1.0, 1.0]),
        box=box,
        velocities=np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
    )

    forcefield = HarmonicBondForce(
        bond_indices=[[0, 1]],
        force_constants=[1000.0],
        equilibrium_lengths=[1.0],
    )

    integrator = VelocityVerletIntegrator(dt=0.001)

    return state, forcefield, integrator


@pytest.fixture
def lj_system():
    """Create a Lennard-Jones system."""
    box = Box.cubic(5.0)
    np.random.seed(42)
    n_atoms = 20

    # Place atoms on a grid
    positions = []
    for i in range(n_atoms):
        x = (i % 4) * 1.2 + 0.5
        y = ((i // 4) % 4) * 1.2 + 0.5
        z = (i // 16) * 1.2 + 0.5
        positions.append([x, y, z])
    positions = np.array(positions[:n_atoms])

    state = MDState.create(
        positions=positions,
        masses=np.ones(n_atoms),
        box=box,
        velocities=np.random.randn(n_atoms, 3) * 0.1,
    )

    forcefield = LennardJonesForce(
        epsilon=np.array([1.0]),
        sigma=np.array([0.3]),
        atom_types=np.zeros(n_atoms, dtype=np.int32),
        cutoff=2.0,
    )

    neighbor_list = VerletList(cutoff=2.0, skin=0.3)

    integrator = VelocityVerletIntegrator(dt=0.001)

    return state, forcefield, neighbor_list, integrator


class TestMDEngine:
    """Test MD engine."""

    def test_initialization(self, simple_system):
        """Test engine initialization."""
        state, forcefield, integrator = simple_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        assert engine.state is not None
        assert engine.integrator is integrator
        assert engine.force_provider is forcefield

    def test_single_step(self, simple_system):
        """Test single simulation step."""
        state, forcefield, integrator = simple_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        initial_step = engine.state.step
        engine.step()

        assert engine.state.step == initial_step + 1

    def test_run_multiple_steps(self, simple_system):
        """Test running multiple steps."""
        state, forcefield, integrator = simple_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        nsteps = 100
        final_state = engine.run(nsteps)

        assert final_state.step == nsteps

    def test_with_neighbor_list(self, lj_system):
        """Test engine with neighbor list."""
        state, forcefield, neighbor_list, integrator = lj_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
            neighbor_list=neighbor_list,
        )

        engine.run(10)

        assert engine.state.step == 10

    def test_energy_properties(self, simple_system):
        """Test energy property access."""
        state, forcefield, integrator = simple_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        assert engine.kinetic_energy >= 0
        assert isinstance(engine.potential_energy, float)
        assert engine.total_energy == engine.kinetic_energy + engine.potential_energy

    def test_temperature_property(self, simple_system):
        """Test temperature property."""
        state, forcefield, integrator = simple_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        assert engine.temperature >= 0

    def test_with_thermostat(self, lj_system):
        """Test engine with thermostat."""
        state, forcefield, neighbor_list, integrator = lj_system

        thermostat = VelocityRescaleThermostat(temperature=300.0)

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
            neighbor_list=neighbor_list,
            thermostat=thermostat,
        )

        # Run some steps
        engine.run(10)

        # Temperature should be controlled
        assert engine.temperature > 0

    def test_callback_stops_simulation(self, simple_system):
        """Test callback can stop simulation early."""
        state, forcefield, integrator = simple_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        stop_at = 50

        def stop_callback(eng):
            return eng.state.step >= stop_at

        engine.run(1000, callback=stop_callback)

        assert engine.state.step == stop_at

    def test_checkpoint_save_load(self, simple_system):
        """Test checkpoint save and load."""
        state, forcefield, integrator = simple_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        engine.run(100)

        # Save checkpoint
        checkpoint = engine.get_checkpoint()

        # Run more steps
        engine.run(50)
        step_before_load = engine.state.step

        # Load checkpoint
        engine.load_checkpoint(checkpoint)

        assert engine.state.step == 100
        assert engine.state.step != step_before_load

    def test_stop_method(self, simple_system):
        """Test stop method."""
        state, forcefield, integrator = simple_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        # Use callback to stop after 20 steps
        def stop_after_20(eng):
            if eng.state.step >= 20:
                eng.stop()
            return False

        engine.run(1000, callback=stop_after_20)

        assert engine.state.step == 20

    def test_performance_stats(self, simple_system):
        """Test performance statistics."""
        state, forcefield, integrator = simple_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        engine.run(100)

        perf = engine.performance

        assert "steps_per_second" in perf
        assert "wall_time" in perf
        assert perf["total_steps"] == 100


class TestMinimization:
    """Test energy minimization."""

    def test_minimize_reduces_energy(self, lj_system):
        """Test that minimization reduces potential energy."""
        state, forcefield, neighbor_list, integrator = lj_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
            neighbor_list=neighbor_list,
        )

        initial_pe = engine.potential_energy

        engine.minimize(max_iterations=50)

        final_pe = engine.potential_energy

        # Energy should decrease or stay same
        assert final_pe <= initial_pe + 1e-6


class TestReporters:
    """Test reporter implementations."""

    def test_trajectory_reporter(self, simple_system):
        """Test trajectory reporter stores frames."""
        state, forcefield, integrator = simple_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        reporter = TrajectoryReporter(frequency=10)
        engine.add_reporter(reporter)

        engine.run(100)

        assert reporter.n_frames == 10
        assert reporter.positions.shape == (10, 2, 3)

    def test_trajectory_reporter_with_velocities(self, simple_system):
        """Test trajectory reporter with velocities."""
        state, forcefield, integrator = simple_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        reporter = TrajectoryReporter(frequency=10, include_velocities=True)
        engine.add_reporter(reporter)

        engine.run(50)

        assert reporter.velocities is not None
        assert reporter.velocities.shape[0] == 5

    def test_energy_reporter(self, simple_system):
        """Test energy reporter tracks energies."""
        state, forcefield, integrator = simple_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        reporter = EnergyReporter(frequency=10)
        engine.add_reporter(reporter)

        engine.run(100)

        assert len(reporter.kinetic_energy) == 10
        assert len(reporter.potential_energy) == 10
        assert len(reporter.total_energy) == 10

    def test_checkpoint_reporter(self, simple_system):
        """Test checkpoint reporter saves state."""
        state, forcefield, integrator = simple_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        reporter = CheckpointReporter(frequency=25, keep_last=2)
        engine.add_reporter(reporter)

        engine.run(100)

        assert reporter.latest_checkpoint is not None
        assert reporter.latest_checkpoint["step"] == 100
        assert len(reporter.all_checkpoints) == 2

    def test_callback_reporter(self, simple_system):
        """Test callback reporter calls function."""
        state, forcefield, integrator = simple_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        call_count = [0]

        def my_callback(state, kwargs):
            call_count[0] += 1

        reporter = CallbackReporter(callback=my_callback, frequency=5)
        engine.add_reporter(reporter)

        engine.run(50)

        assert call_count[0] == 10

    def test_remove_reporter(self, simple_system):
        """Test removing a reporter."""
        state, forcefield, integrator = simple_system

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        reporter = TrajectoryReporter(frequency=1)
        engine.add_reporter(reporter)
        engine.remove_reporter(reporter)

        engine.run(10)

        assert reporter.n_frames == 0


class TestSimulationHooks:
    """Test simulation hooks."""

    def test_pre_step_hook(self, simple_system):
        """Test pre-step hook is called."""
        state, forcefield, integrator = simple_system

        class CountingHook(SimulationHook):
            def __init__(self):
                self.count = 0

            def pre_step(self, engine, state):
                self.count += 1
                return None

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        hook = CountingHook()
        engine.add_hook(hook)

        engine.run(50)

        assert hook.count == 50

    def test_post_force_hook_modifies_forces(self, simple_system):
        """Test post-force hook can modify forces."""
        state, forcefield, integrator = simple_system

        class ZeroForceHook(SimulationHook):
            def post_force(self, engine, state, forces):
                return np.zeros_like(forces)

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        hook = ZeroForceHook()
        engine.add_hook(hook)

        # Record initial velocity
        initial_vel = engine.state.velocities.copy()

        # Run with zero forces
        engine.run(10)

        # With zero forces, velocity should remain constant
        np.testing.assert_allclose(engine.state.velocities, initial_vel, rtol=1e-10)

    def test_remove_hook(self, simple_system):
        """Test removing a hook."""
        state, forcefield, integrator = simple_system

        class CountingHook(SimulationHook):
            def __init__(self):
                self.count = 0

            def pre_step(self, engine, state):
                self.count += 1
                return None

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        hook = CountingHook()
        engine.add_hook(hook)
        engine.remove_hook(hook)

        engine.run(10)

        assert hook.count == 0


class TestCompositeForceField:
    """Test engine with composite force field."""

    def test_multiple_force_terms(self):
        """Test engine with multiple force terms."""
        box = Box.cubic(10.0)
        state = MDState.create(
            positions=np.array([[4.5, 5.0, 5.0], [5.5, 5.0, 5.0]]),
            masses=np.array([1.0, 1.0]),
            box=box,
        )

        bond_force = HarmonicBondForce(
            bond_indices=[[0, 1]],
            force_constants=[1000.0],
            equilibrium_lengths=[1.0],
        )

        lj_force = LennardJonesForce(
            epsilon=np.array([0.1]),
            sigma=np.array([0.3]),
            atom_types=np.array([0, 0]),
            cutoff=2.0,
            exclusions={(0, 1)},  # Exclude bonded pair
        )

        forcefield = ForceField([bond_force, lj_force])
        integrator = VelocityVerletIntegrator(dt=0.001)

        engine = MDEngine(
            state=state,
            integrator=integrator,
            force_provider=forcefield,
        )

        engine.run(100)

        assert engine.state.step == 100
