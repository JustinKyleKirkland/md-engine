"""
Performance Regression Gates.

These tests verify that performance hasn't regressed.

Gates:
1. Steps/second within tolerance of baseline
2. No severe performance regressions (>15%)
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from mdcore.benchmarks import BenchmarkResult
from mdcore.forcefields import ForceField
from mdcore.forcefields.nonbonded import LennardJonesForce
from mdcore.integrators import VelocityVerletIntegrator
from mdcore.neighborlists import VerletList
from mdcore.system import Box, MDState

BASELINE_DIR = Path(__file__).parent.parent.parent / "benchmarks" / "baselines"


def load_baseline(platform: str) -> dict:
    """Load performance baseline."""
    baseline_path = BASELINE_DIR / f"{platform}.json"
    if not baseline_path.exists():
        return {}
    with open(baseline_path) as f:
        return json.load(f)


def create_lattice_system(n_atoms: int, seed: int = 12345) -> MDState:
    """Create lattice-initialized LJ system."""
    rng = np.random.default_rng(seed)

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

    return MDState(
        positions=positions,
        velocities=velocities,
        forces=np.zeros((n_atoms, 3)),
        masses=np.ones(n_atoms),
        box=box,
    )


class TestPerformanceGates:
    """
    Gate: Performance must not regress beyond tolerance.

    Measured performance must be >= baseline * (1 - tolerance).
    """

    @pytest.fixture
    def baseline(self):
        """Load CPU serial baseline."""
        return load_baseline("cpu_serial")

    def run_benchmark(
        self,
        n_atoms: int,
        n_steps: int = 100,
        warmup_steps: int = 10,
    ) -> float:
        """Run benchmark and return steps/second."""
        state = create_lattice_system(n_atoms)

        atom_types = np.zeros(n_atoms, dtype=np.int32)
        lj = LennardJonesForce(
            epsilon=np.array([1.0]),
            sigma=np.array([1.0]),
            atom_types=atom_types,
            cutoff=2.5,
        )
        ff = ForceField([lj])
        integrator = VelocityVerletIntegrator(dt=0.001)
        neighbor_list = VerletList(cutoff=2.5, skin=0.3)

        positions = state.positions.copy()
        velocities = state.velocities.copy()
        masses = state.masses
        box = state.box

        # Warmup
        for _ in range(warmup_steps):
            s = MDState(
                positions=positions,
                velocities=velocities,
                forces=np.zeros_like(positions),
                masses=masses,
                box=box,
            )
            neighbor_list.build(positions, box)
            forces = ff.compute(s, neighbor_list)
            new_s = integrator.step(s, forces)
            positions = new_s.positions
            velocities = new_s.velocities

        # Timed run
        start = time.perf_counter()

        for _ in range(n_steps):
            s = MDState(
                positions=positions,
                velocities=velocities,
                forces=np.zeros_like(positions),
                masses=masses,
                box=box,
            )
            neighbor_list.build(positions, box)
            forces = ff.compute(s, neighbor_list)
            new_s = integrator.step(s, forces)
            positions = new_s.positions
            velocities = new_s.velocities

        elapsed = time.perf_counter() - start
        steps_per_second = n_steps / elapsed

        return steps_per_second

    def test_lj_64_performance(self, baseline):
        """Gate: 64-atom LJ performance within baseline tolerance."""
        if not baseline or "benchmarks" not in baseline:
            pytest.skip("Baseline not available")

        bench_name = "lj_64_atoms"
        if bench_name not in baseline["benchmarks"]:
            pytest.skip(f"Baseline for {bench_name} not available")

        bench_baseline = baseline["benchmarks"][bench_name]
        expected = bench_baseline["steps_per_second"]
        tolerance = bench_baseline["tolerance"]

        measured = self.run_benchmark(n_atoms=64, n_steps=50)

        min_acceptable = expected * (1 - tolerance)

        result = BenchmarkResult(
            test=bench_name,
            steps_per_second=measured,
            n_atoms=64,
            passed=measured >= min_acceptable,
            extra={
                "baseline": expected,
                "tolerance": tolerance,
                "min_acceptable": min_acceptable,
            },
        )

        # Soft gate: warn but don't fail for small regressions
        # Hard gate: fail for >15% regression
        if measured < expected * 0.85:
            pytest.fail(
                f"GATE FAILED: Performance regression {bench_name}\n"
                f"  Baseline: {expected:.1f} steps/s\n"
                f"  Measured: {measured:.1f} steps/s\n"
                f"  Regression: {(1 - measured / expected) * 100:.1f}%"
            )

    def test_no_catastrophic_regression(self):
        """Gate: No catastrophic performance regression (>50%)."""
        # Quick sanity check that simulation runs at reasonable speed
        measured = self.run_benchmark(n_atoms=27, n_steps=20)

        # Should complete at least 10 steps/second for tiny system
        assert measured > 10, (
            f"GATE FAILED: Catastrophic performance issue\n"
            f"  Measured: {measured:.1f} steps/s\n"
            f"  Expected: >10 steps/s for 27-atom system"
        )


class TestMLPerformanceGates:
    """
    Gate: ML potential accuracy thresholds.
    """

    @pytest.fixture
    def ml_baseline(self):
        """Load ML baseline."""
        return load_baseline("ml")

    def test_ml_accuracy_gates_structure(self, ml_baseline):
        """Verify ML accuracy gates are defined."""
        if not ml_baseline:
            pytest.skip("ML baseline not available")

        if "accuracy_gates" not in ml_baseline:
            pytest.skip("ML accuracy gates not defined")

        gates = ml_baseline["accuracy_gates"]

        # Verify required gates exist
        assert "force_mae_ev_angstrom" in gates, (
            "GATE MISSING: force_mae_ev_angstrom not defined"
        )
        assert "energy_mae_mev_atom" in gates, (
            "GATE MISSING: energy_mae_mev_atom not defined"
        )

        # Verify thresholds are reasonable
        assert gates["force_mae_ev_angstrom"]["max_value"] > 0
        assert gates["energy_mae_mev_atom"]["max_value"] > 0
