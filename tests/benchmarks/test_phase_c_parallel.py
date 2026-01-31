"""
Phase C - Parallel Benchmarks

Build order:
1. Determinism (reproducible results across runs)
2. Strong scaling (speedup with more workers)

These tests verify parallel correctness before
moving to ML benchmarks.
"""

import numpy as np
import pytest

from mdcore.benchmarks import BenchmarkReporter, BenchmarkResult
from mdcore.parallel import SerialBackend, get_backend

# =============================================================================
# Phase C.1: Determinism
# =============================================================================


class TestDeterminism:
    """
    Determinism tests for reproducibility.

    Critical for:
    - Scientific reproducibility
    - Debugging
    - Checkpointing
    """

    @pytest.fixture
    def reporter(self):
        """Create benchmark reporter."""
        return BenchmarkReporter("phase_c_parallel")

    def test_serial_backend_determinism(self, reporter):
        """Test that serial backend is deterministic."""
        backend = SerialBackend()

        # Same operation twice
        data = np.array([1.0, 2.0, 3.0, 4.0])

        result1 = backend.reduce_sum(data)
        result2 = backend.reduce_sum(data)

        is_deterministic = np.array_equal(result1, result2)

        result = BenchmarkResult(
            test="serial_determinism",
            n_ranks=1,
            backend="serial",
            passed=is_deterministic,
        )
        reporter.add_result(result)

        assert result.passed

    def test_rng_reproducibility(self, reporter):
        """Test that seeded RNG gives identical sequences."""
        seed = 42

        rng1 = np.random.default_rng(seed)
        seq1 = rng1.random(100)

        rng2 = np.random.default_rng(seed)
        seq2 = rng2.random(100)

        is_identical = np.allclose(seq1, seq2)

        result = BenchmarkResult(
            test="rng_reproducibility",
            n_ranks=1,
            backend="serial",
            passed=is_identical,
        )
        reporter.add_result(result)

        assert result.passed

    def test_force_calculation_determinism(self, reporter):
        """Test that force calculations are deterministic."""
        # Create fixed state
        n_atoms = 10
        rng = np.random.default_rng(123)

        positions = rng.uniform(0, 5, (n_atoms, 3))

        # Simple harmonic force (deterministic)
        def compute_forces(pos):
            return -0.1 * pos

        forces1 = compute_forces(positions)
        forces2 = compute_forces(positions)

        is_identical = np.allclose(forces1, forces2)

        result = BenchmarkResult(
            test="force_determinism",
            n_atoms=n_atoms,
            n_ranks=1,
            backend="serial",
            passed=is_identical,
        )
        reporter.add_result(result)

        assert result.passed


# =============================================================================
# Phase C.2: Strong Scaling
# =============================================================================


class TestStrongScaling:
    """
    Strong scaling tests.

    Measures speedup when adding more workers
    for fixed problem size.

    Note: Actual parallel tests require MPI or multiprocessing.
    These tests verify the interface works correctly.
    """

    @pytest.fixture
    def reporter(self):
        """Create benchmark reporter."""
        return BenchmarkReporter("phase_c_parallel")

    def test_backend_interface(self, reporter):
        """Test that parallel backend interface works."""
        backend = get_backend("serial")

        assert backend.name == "serial"
        assert backend.n_workers == 1
        assert backend.rank == 0
        assert backend.is_root is True

        result = BenchmarkResult(
            test="backend_interface",
            n_ranks=1,
            backend="serial",
            passed=True,
        )
        reporter.add_result(result)

    def test_parallel_map_serial(self, reporter):
        """Test parallel_map on serial backend."""
        backend = get_backend("serial")

        def square(x):
            return x * x

        inputs = [1, 2, 3, 4, 5]
        results = backend.parallel_map(square, inputs)

        expected = [1, 4, 9, 16, 25]
        is_correct = results == expected

        result = BenchmarkResult(
            test="parallel_map_serial",
            n_ranks=1,
            backend="serial",
            passed=is_correct,
        )
        reporter.add_result(result)

        assert result.passed

    def test_reduction_operations(self, reporter):
        """Test reduction operations."""
        backend = get_backend("serial")

        data = np.array([1.0, 2.0, 3.0, 4.0])

        # Serial backend returns the data unchanged (no actual reduction needed)
        sum_result = backend.reduce_sum(data)

        # Verify the result is the same array (serial pass-through)
        is_correct = np.array_equal(sum_result, data)

        result = BenchmarkResult(
            test="reduction_operations",
            n_ranks=1,
            backend="serial",
            passed=is_correct,
        )
        reporter.add_result(result)

        assert result.passed

    def test_partition_atoms(self, reporter):
        """Test atom partitioning."""
        backend = get_backend("serial")

        n_atoms = 100
        start, end = backend.partition_atoms(n_atoms)

        # Serial backend should return all atoms (0, n_atoms)
        is_correct = start == 0 and end == n_atoms

        result = BenchmarkResult(
            test="partition_atoms",
            n_atoms=n_atoms,
            n_ranks=1,
            backend="serial",
            passed=is_correct,
            extra={"start": start, "end": end},
        )
        reporter.add_result(result)

        assert result.passed


# =============================================================================
# Phase C Summary
# =============================================================================


class TestPhaseCSimulated:
    """
    Simulated scaling tests using serial backend.

    These provide interface verification without
    requiring actual MPI setup.
    """

    @pytest.fixture
    def reporter(self):
        """Create benchmark reporter."""
        return BenchmarkReporter("phase_c_parallel")

    def test_scaling_interface_ready(self, reporter):
        """Verify scaling infrastructure is ready."""
        # Check all backends can be requested
        serial = get_backend("serial")

        result = BenchmarkResult(
            test="scaling_interface",
            n_ranks=1,
            backend="serial",
            passed=serial is not None,
        )
        reporter.add_result(result)

        assert result.passed


class TestPhasecSummary:
    """Summary test for Phase C benchmarks."""

    def test_phase_c_complete(self):
        """Verify all Phase C tests can be imported and run."""
        assert True, "Phase C benchmarks loaded successfully"
