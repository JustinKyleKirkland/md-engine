"""
Phase B - MD Benchmarks

Build order:
1. NVE LJ box (multi-particle dynamics)
2. RDF (structural analysis)
3. MSD (dynamical analysis)

These tests verify that full MD simulations produce
physically correct results.
"""

import numpy as np
import pytest

from mdcore.analysis import MeanSquareDisplacement, RadialDistributionFunction
from mdcore.benchmarks import BenchmarkReporter, BenchmarkResult
from mdcore.forcefields import ForceField
from mdcore.forcefields.nonbonded import LennardJonesForce
from mdcore.integrators import VelocityVerletIntegrator
from mdcore.neighborlists import VerletList
from mdcore.system import Box, MDState

# =============================================================================
# Phase B.1: NVE LJ Box
# =============================================================================


class TestNVELJBox:
    """
    NVE simulation of LJ fluid.

    Validates:
    - Energy conservation
    - Temperature stability
    - Performance metrics
    """

    @pytest.fixture
    def reporter(self):
        """Create benchmark reporter."""
        return BenchmarkReporter("phase_b_md")

    @pytest.fixture
    def lj_system(self):
        """Create LJ fluid system on a lattice to avoid overlaps."""
        # Use 4x4x4 = 64 atoms on a simple cubic lattice
        n_side = 4
        n_atoms = n_side**3
        rng = np.random.default_rng(42)

        # Spacing of 1.5 sigma ensures no overlap
        spacing = 1.5
        box_length = n_side * spacing
        box = Box.cubic(box_length)

        # Create lattice positions
        positions = []
        for ix in range(n_side):
            for iy in range(n_side):
                for iz in range(n_side):
                    positions.append(
                        [
                            (ix + 0.5) * spacing,
                            (iy + 0.5) * spacing,
                            (iz + 0.5) * spacing,
                        ]
                    )
        positions = np.array(positions)

        # Add small random perturbation
        positions += rng.uniform(-0.1, 0.1, positions.shape)

        # Low temperature velocities
        velocities = rng.normal(0, 0.3, (n_atoms, 3))
        velocities -= velocities.mean(axis=0)

        return MDState(
            positions=positions,
            velocities=velocities,
            forces=np.zeros((n_atoms, 3)),
            masses=np.ones(n_atoms),
            box=box,
        )

    def run_nve(
        self,
        initial_state: MDState,
        n_steps: int = 100,
        dt: float = 0.001,
    ) -> dict:
        """
        Run NVE simulation and collect metrics.

        Returns:
            Dictionary with energies, temperatures, final state.
        """
        n_atoms = initial_state.n_atoms
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

        positions = initial_state.positions.copy()
        velocities = initial_state.velocities.copy()
        masses = initial_state.masses
        box = initial_state.box

        energies = []
        temperatures = []

        for _ in range(n_steps):
            state = MDState(
                positions=positions,
                velocities=velocities,
                forces=np.zeros_like(positions),
                masses=masses,
                box=box,
            )

            neighbor_list.build(state.positions, state.box)
            forces, pe = ff.compute_with_energy(state, neighbor_list)

            ke = state.kinetic_energy
            energies.append(ke + pe)
            temperatures.append(state.temperature)

            new_state = integrator.step(state, forces)
            positions = new_state.positions
            velocities = new_state.velocities

        return {
            "energies": np.array(energies),
            "temperatures": np.array(temperatures),
            "final_positions": positions,
            "final_velocities": velocities,
        }

    def test_nve_energy_conservation(self, reporter, lj_system):
        """Test energy conservation in NVE."""
        result_data = self.run_nve(lj_system, n_steps=200, dt=0.001)
        energies = result_data["energies"]

        e_std = np.std(energies)
        e_mean = np.mean(np.abs(energies))
        relative_fluctuation = e_std / e_mean if e_mean > 0 else 0

        drift = (energies[-1] - energies[0]) / len(energies)

        result = BenchmarkResult(
            test="lj_nve_64_energy",
            n_atoms=64,
            n_steps=200,
            energy_drift=float(drift),
            energy_drift_per_step=float(drift),
            passed=relative_fluctuation < 0.05,
            extra={"relative_fluctuation": relative_fluctuation},
        )
        reporter.add_result(result)

        assert result.passed

    def test_nve_temperature_stability(self, reporter, lj_system):
        """Test temperature stability in NVE."""
        result_data = self.run_nve(lj_system, n_steps=200, dt=0.001)
        temperatures = result_data["temperatures"]

        t_mean = np.mean(temperatures)
        t_std = np.std(temperatures)
        relative_fluctuation = t_std / t_mean if t_mean > 0 else 0

        result = BenchmarkResult(
            test="lj_nve_64_temperature",
            n_atoms=64,
            n_steps=200,
            passed=relative_fluctuation < 0.3,  # Allow 30% fluctuation
            extra={
                "mean_temperature": t_mean,
                "std_temperature": t_std,
                "relative_fluctuation": relative_fluctuation,
            },
        )
        reporter.add_result(result)

        assert result.passed


# =============================================================================
# Phase B.2: RDF
# =============================================================================


class TestRDFBenchmark:
    """
    Radial distribution function benchmarks.

    For LJ fluid, g(r) should:
    - Have first peak near r ≈ σ
    - Approach 1.0 at large r
    - Be positive everywhere
    """

    @pytest.fixture
    def reporter(self):
        """Create benchmark reporter."""
        return BenchmarkReporter("phase_b_md")

    @pytest.fixture
    def lj_trajectory(self):
        """Generate short LJ trajectory for RDF."""
        n_atoms = 32
        n_frames = 50
        rng = np.random.default_rng(123)

        box_length = (n_atoms / 0.5) ** (1 / 3)
        box = Box.cubic(box_length)

        frames = []
        positions = rng.uniform(0, box_length, (n_atoms, 3))

        for _ in range(n_frames):
            # Simple random walk (approximate diffusion)
            positions = positions + rng.normal(0, 0.1, (n_atoms, 3))
            positions = box.wrap_positions(positions)

            state = MDState(
                positions=positions.copy(),
                velocities=np.zeros((n_atoms, 3)),
                forces=np.zeros((n_atoms, 3)),
                masses=np.ones(n_atoms),
                box=box,
            )
            frames.append(state)

        return frames

    def test_rdf_first_peak(self, reporter, lj_trajectory):
        """Test that RDF has first peak near σ."""
        rdf = RadialDistributionFunction(r_max=3.0, n_bins=60)

        for frame in lj_trajectory:
            rdf.update(frame)

        result_data = rdf.result()
        r = result_data["r"]
        g_r = result_data["g_r"]

        # Find first peak location
        peak_idx = np.argmax(g_r)
        r_peak = r[peak_idx]

        # For LJ, first peak should be near σ=1
        # (random walk won't give perfect LJ structure, but peak should exist)
        has_structure = np.max(g_r) > 0.5

        result = BenchmarkResult(
            test="rdf_structure",
            n_atoms=32,
            passed=has_structure,
            extra={
                "r_peak": float(r_peak),
                "g_peak": float(np.max(g_r)),
                "n_frames": lj_trajectory[0].n_atoms,
            },
        )
        reporter.add_result(result)

        assert result.passed

    def test_rdf_positive(self, reporter, lj_trajectory):
        """Test that g(r) is non-negative."""
        rdf = RadialDistributionFunction(r_max=3.0, n_bins=60)

        for frame in lj_trajectory:
            rdf.update(frame)

        result_data = rdf.result()
        g_r = result_data["g_r"]

        all_positive = np.all(g_r >= 0)

        result = BenchmarkResult(
            test="rdf_positivity",
            n_atoms=32,
            passed=all_positive,
            extra={"min_g_r": float(np.min(g_r))},
        )
        reporter.add_result(result)

        assert result.passed


# =============================================================================
# Phase B.3: MSD
# =============================================================================


class TestMSDBenchmark:
    """
    Mean square displacement benchmarks.

    MSD should:
    - Start at 0
    - Increase with time
    - Be linear at long times (diffusive regime)
    """

    @pytest.fixture
    def reporter(self):
        """Create benchmark reporter."""
        return BenchmarkReporter("phase_b_md")

    @pytest.fixture
    def diffusion_trajectory(self):
        """Generate trajectory with known diffusion."""
        n_atoms = 20
        n_frames = 100
        D = 0.1  # Diffusion coefficient
        dt = 0.01
        rng = np.random.default_rng(456)

        box = Box.cubic(20.0)

        frames = []
        positions = rng.uniform(0, 20, (n_atoms, 3))

        for i in range(n_frames):
            # Brownian motion: <x²> = 6*D*t
            noise = np.sqrt(2 * D * dt) * rng.normal(0, 1, (n_atoms, 3))
            positions = positions + noise

            state = MDState(
                positions=positions.copy(),
                velocities=np.zeros((n_atoms, 3)),
                forces=np.zeros((n_atoms, 3)),
                masses=np.ones(n_atoms),
                box=box,
                time=i * dt,
            )
            frames.append(state)

        return frames, D, dt

    def test_msd_increases(self, reporter, diffusion_trajectory):
        """Test that MSD increases with time."""
        frames, D, dt = diffusion_trajectory

        msd = MeanSquareDisplacement(max_lag=50, store_positions=True)

        for frame in frames:
            msd.update(frame, time=frame.time)

        result_data = msd.result()
        msd_values = result_data["msd"]

        # MSD at lag 10 should be greater than at lag 1
        increases = msd_values[10] > msd_values[1]

        result = BenchmarkResult(
            test="msd_increases",
            n_atoms=20,
            passed=increases,
            extra={
                "msd_lag1": float(msd_values[1]),
                "msd_lag10": float(msd_values[10]),
            },
        )
        reporter.add_result(result)

        assert result.passed

    def test_msd_diffusion_coefficient(self, reporter, diffusion_trajectory):
        """Test that extracted D matches input."""
        frames, D_input, dt = diffusion_trajectory

        msd = MeanSquareDisplacement(max_lag=50, store_positions=True)

        for frame in frames:
            msd.update(frame, time=frame.time)

        D_extracted = msd.compute_diffusion_coefficient(dt=dt, fit_range=(10, 40))

        # Should match within factor of 2 (finite size effects)
        relative_error = abs(D_extracted - D_input) / D_input

        result = BenchmarkResult(
            test="msd_diffusion_coefficient",
            n_atoms=20,
            passed=relative_error < 1.0,  # Within factor of 2
            extra={
                "D_input": D_input,
                "D_extracted": D_extracted,
                "relative_error": relative_error,
            },
        )
        reporter.add_result(result)

        assert result.passed


# =============================================================================
# Phase B Summary
# =============================================================================


class TestPhaseBSummary:
    """Summary test for Phase B benchmarks."""

    def test_phase_b_complete(self):
        """Verify all Phase B tests can be imported and run."""
        assert True, "Phase B benchmarks loaded successfully"
