"""Tests for analysis subsystem."""

import numpy as np
import pytest

from mdcore.analysis import (
    CompositeAnalyzer,
    EnergyAnalyzer,
    MeanSquareDisplacement,
    PressureTensor,
    RadialDistributionFunction,
    StreamingAnalyzer,
    VelocityAutocorrelation,
)
from mdcore.analysis.base import Analyzer
from mdcore.system import Box, MDState


@pytest.fixture
def simple_state():
    """Create a simple MD state for testing."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    velocities = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    forces = np.zeros((4, 3))
    masses = np.ones(4)
    box = Box.cubic(10.0)

    return MDState(
        positions=positions,
        velocities=velocities,
        forces=forces,
        masses=masses,
        box=box,
    )


class TestRadialDistributionFunction:
    """Tests for RDF analyzer."""

    def test_rdf_creation(self):
        """Test RDF creation."""
        rdf = RadialDistributionFunction(r_max=5.0, n_bins=50)

        assert rdf.name == "rdf"
        assert rdf.r_max == 5.0
        assert rdf.n_bins == 50
        assert len(rdf.r) == 50

    def test_rdf_reset(self):
        """Test RDF reset."""
        rdf = RadialDistributionFunction(r_max=5.0)
        rdf._n_frames = 10
        rdf.reset()

        assert rdf._n_frames == 0

    def test_rdf_update(self, simple_state):
        """Test RDF update with simple state."""
        rdf = RadialDistributionFunction(r_max=5.0, n_bins=50)
        rdf.update(simple_state)

        assert rdf._n_frames == 1
        result = rdf.result()
        assert "r" in result
        assert "g_r" in result
        assert len(result["g_r"]) == 50

    def test_rdf_multiple_frames(self, simple_state):
        """Test RDF with multiple frames."""
        rdf = RadialDistributionFunction(r_max=5.0)

        for _ in range(10):
            rdf.update(simple_state)

        assert rdf._n_frames == 10
        result = rdf.result()
        assert result["n_frames"] == 10


class TestMeanSquareDisplacement:
    """Tests for MSD analyzer."""

    def test_msd_creation(self):
        """Test MSD creation."""
        msd = MeanSquareDisplacement(max_lag=100)

        assert msd.name == "msd"
        assert msd.max_lag == 100

    def test_msd_update(self, simple_state):
        """Test MSD update."""
        msd = MeanSquareDisplacement(max_lag=100)
        msd.update(simple_state)

        assert msd._n_frames == 1
        assert msd._initial_positions is not None

    def test_msd_diffusion(self):
        """Test MSD with diffusing particles."""
        msd = MeanSquareDisplacement(max_lag=50, store_positions=True)

        # Simulate diffusion
        positions = np.zeros((4, 3))
        box = Box.cubic(100.0)
        masses = np.ones(4)

        for i in range(50):
            # Random walk
            positions = positions + np.random.randn(4, 3) * 0.1

            state = MDState(
                positions=positions.copy(),
                velocities=np.zeros((4, 3)),
                forces=np.zeros((4, 3)),
                masses=masses,
                box=box,
            )
            msd.update(state, time=float(i))

        result = msd.result()
        assert result["n_frames"] == 50

        # MSD should increase with time (roughly)
        msd_values = result["msd"]
        # Check that later values are generally larger (with some tolerance)
        assert msd_values[10] <= msd_values[20] or np.isclose(
            msd_values[10], msd_values[20], rtol=0.5
        )


class TestVelocityAutocorrelation:
    """Tests for VACF analyzer."""

    def test_vacf_creation(self):
        """Test VACF creation."""
        vacf = VelocityAutocorrelation(max_lag=100)

        assert vacf.name == "vacf"
        assert vacf.max_lag == 100

    def test_vacf_update(self, simple_state):
        """Test VACF update."""
        vacf = VelocityAutocorrelation(max_lag=100)
        vacf.update(simple_state)

        assert vacf._n_frames == 1

    def test_vacf_normalization(self, simple_state):
        """Test VACF normalization."""
        vacf = VelocityAutocorrelation(max_lag=100)

        for _ in range(10):
            vacf.update(simple_state)

        result = vacf.result()

        # VACF(0) normalized should be 1.0
        np.testing.assert_allclose(result["vacf_normalized"][0], 1.0)

    def test_vacf_requires_velocities(self, simple_state):
        """Test that VACF requires velocities."""
        vacf = VelocityAutocorrelation()

        # Create a state without velocities by setting them to None after creation
        state = simple_state
        state.velocities = None

        with pytest.raises(ValueError, match="velocities"):
            vacf.update(state)


class TestEnergyAnalyzer:
    """Tests for energy analyzer."""

    def test_energy_creation(self):
        """Test energy analyzer creation."""
        analyzer = EnergyAnalyzer()
        assert analyzer.name == "energy"

    def test_energy_update(self, simple_state):
        """Test energy update."""
        analyzer = EnergyAnalyzer()
        analyzer.update(simple_state, potential_energy=10.0)

        assert analyzer._n_frames == 1
        assert len(analyzer._kinetic) == 1
        assert len(analyzer._potential) == 1

    def test_energy_statistics(self, simple_state):
        """Test energy statistics."""
        analyzer = EnergyAnalyzer()

        for i in range(10):
            analyzer.update(simple_state, potential_energy=float(i), time=float(i))

        result = analyzer.result()

        assert result["n_frames"] == 10
        assert "kinetic_mean" in result
        assert "potential_mean" in result
        assert "temperature_mean" in result


class TestPressureTensor:
    """Tests for pressure tensor analyzer."""

    def test_pressure_creation(self):
        """Test pressure analyzer creation."""
        analyzer = PressureTensor()
        assert analyzer.name == "pressure"

    def test_pressure_update(self, simple_state):
        """Test pressure update."""
        analyzer = PressureTensor()
        forces = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )

        analyzer.update(simple_state, forces=forces)

        assert analyzer._n_frames == 1

    def test_pressure_result(self, simple_state):
        """Test pressure result."""
        analyzer = PressureTensor()
        forces = np.zeros((4, 3))

        for _ in range(5):
            analyzer.update(simple_state, forces=forces)

        result = analyzer.result()

        assert result["n_frames"] == 5
        assert "pressure" in result
        assert "pressure_tensor" in result
        assert result["pressure_tensor"].shape == (3, 3)


class TestCompositeAnalyzer:
    """Tests for composite analyzer."""

    def test_composite_creation(self):
        """Test composite analyzer creation."""
        analyzers = [
            EnergyAnalyzer(),
            MeanSquareDisplacement(),
        ]
        composite = CompositeAnalyzer(analyzers)

        assert composite.name == "composite"
        assert len(composite.analyzers) == 2

    def test_composite_update(self, simple_state):
        """Test composite update runs all analyzers."""
        analyzers = [
            EnergyAnalyzer(),
            VelocityAutocorrelation(),
        ]
        composite = CompositeAnalyzer(analyzers)

        composite.update(simple_state, potential_energy=5.0)

        assert composite._n_frames == 1
        # Each child should also have been updated
        assert analyzers[0]._n_frames == 1
        assert analyzers[1]._n_frames == 1

    def test_composite_result(self, simple_state):
        """Test composite result aggregates all results."""
        analyzers = [
            EnergyAnalyzer(),
            VelocityAutocorrelation(),
        ]
        composite = CompositeAnalyzer(analyzers)

        composite.update(simple_state, potential_energy=5.0)
        result = composite.result()

        assert "energy" in result
        assert "vacf" in result

    def test_composite_get_analyzer(self):
        """Test getting child analyzer by name."""
        energy = EnergyAnalyzer()
        vacf = VelocityAutocorrelation()
        composite = CompositeAnalyzer([energy, vacf])

        found = composite.get_analyzer("energy")
        assert found is energy

        not_found = composite.get_analyzer("nonexistent")
        assert not_found is None


class TestAnalyzerInterface:
    """Test analyzer interface compliance."""

    @pytest.fixture
    def analyzers(self):
        """Get list of streaming analyzers."""
        return [
            RadialDistributionFunction(r_max=5.0),
            MeanSquareDisplacement(),
            VelocityAutocorrelation(),
            EnergyAnalyzer(),
            PressureTensor(),
        ]

    def test_all_have_name(self, analyzers):
        """Test all analyzers have name property."""
        for analyzer in analyzers:
            assert isinstance(analyzer.name, str)
            assert len(analyzer.name) > 0

    def test_all_are_streaming_analyzers(self, analyzers):
        """Test all analyzers inherit from StreamingAnalyzer."""
        for analyzer in analyzers:
            assert isinstance(analyzer, StreamingAnalyzer)
            assert isinstance(analyzer, Analyzer)

    def test_all_have_reset(self, analyzers):
        """Test all analyzers have reset method."""
        for analyzer in analyzers:
            analyzer.reset()  # Should not raise

    def test_all_have_result(self, analyzers):
        """Test all analyzers have result method."""
        for analyzer in analyzers:
            result = analyzer.result()
            assert isinstance(result, dict)
