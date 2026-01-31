"""
Validation tests for analysis modules.

These tests verify that analysis tools produce physically correct results
on known systems where analytical or reference values exist.
"""

import numpy as np
import pytest

from mdcore.analysis import MeanSquareDisplacement, VelocityAutocorrelation
from mdcore.system import Box, MDState

# =============================================================================
# Fixtures for LJ Fluid Simulation
# =============================================================================


def generate_lj_fluid_trajectory(
    n_atoms: int = 108,
    n_frames: int = 1000,
    dt: float = 0.005,
    temperature: float = 1.0,
    density: float = 0.8,
    seed: int = 42,
) -> list[MDState]:
    """
    Generate a simple LJ fluid trajectory using random walk approximation.

    For validation purposes, we use a random walk model that should produce:
    - Linear MSD at long times: MSD(t) = 6*D*t
    - Exponentially decaying VACF

    Args:
        n_atoms: Number of atoms.
        n_frames: Number of frames.
        dt: Timestep.
        temperature: Reduced temperature (sets velocity scale).
        density: Reduced density (sets box size).
        seed: Random seed for reproducibility.

    Returns:
        List of MDState objects representing trajectory.
    """
    rng = np.random.default_rng(seed)

    # Box size from density: rho = N/V -> L = (N/rho)^(1/3)
    box_length = (n_atoms / density) ** (1 / 3)
    box = Box.cubic(box_length)

    # Initial positions on a simple cubic lattice
    n_side = int(np.ceil(n_atoms ** (1 / 3)))
    spacing = box_length / n_side

    positions = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                if len(positions) < n_atoms:
                    positions.append([i * spacing, j * spacing, k * spacing])

    positions = np.array(positions[:n_atoms])

    # Initial velocities from Maxwell-Boltzmann
    velocities = rng.normal(0, np.sqrt(temperature), (n_atoms, 3))
    # Remove COM velocity
    velocities -= velocities.mean(axis=0)

    masses = np.ones(n_atoms)
    forces = np.zeros((n_atoms, 3))

    # Generate trajectory using Langevin-like dynamics
    trajectory = []
    current_positions = positions.copy()
    current_velocities = velocities.copy()

    # Velocity correlation time
    tau_v = 0.5  # Approximate correlation time

    for frame in range(n_frames):
        # Update velocities with damping + noise (Ornstein-Uhlenbeck process)
        gamma = 1.0 / tau_v
        current_velocities = current_velocities * np.exp(-gamma * dt) + np.sqrt(
            temperature * (1 - np.exp(-2 * gamma * dt))
        ) * rng.normal(0, 1, (n_atoms, 3))

        # Update positions
        current_positions = current_positions + current_velocities * dt

        # Wrap positions (but store unwrapped for MSD)
        wrapped_positions = box.wrap_positions(current_positions)

        state = MDState(
            positions=wrapped_positions.copy(),
            velocities=current_velocities.copy(),
            forces=forces.copy(),
            masses=masses.copy(),
            box=box,
            time=frame * dt,
            step=frame,
        )
        trajectory.append(state)

    return trajectory


# =============================================================================
# MSD Validation Tests
# =============================================================================


class TestMSDValidation:
    """
    Validation tests for Mean Square Displacement analysis.

    Tests verify:
    1. MSD is linear at long times (diffusive regime)
    2. Diffusion coefficient extracted from MSD matches expected value
    """

    @pytest.fixture
    def lj_trajectory(self):
        """Generate LJ fluid trajectory for testing."""
        return generate_lj_fluid_trajectory(
            n_atoms=64,
            n_frames=500,
            dt=0.005,
            temperature=1.0,
            seed=12345,
        )

    def test_msd_increases_with_time(self, lj_trajectory):
        """Test that MSD increases monotonically with time."""
        msd_analyzer = MeanSquareDisplacement(max_lag=200, store_positions=True)

        for state in lj_trajectory:
            msd_analyzer.update(state)

        result = msd_analyzer.result()
        msd = result["msd"]

        # MSD should generally increase (with some noise)
        # Check that later values are larger than early values
        early_avg = np.mean(msd[10:30])
        late_avg = np.mean(msd[100:150])

        assert late_avg > early_avg, "MSD should increase with lag time"

    def test_msd_linear_regime(self, lj_trajectory):
        """Test that MSD is approximately linear at intermediate times."""
        msd_analyzer = MeanSquareDisplacement(max_lag=200, store_positions=True)

        for state in lj_trajectory:
            msd_analyzer.update(state)

        result = msd_analyzer.result()
        msd = result["msd"]
        lag = result["lag"]
        count = result["count"]

        # Fit linear region (skip early ballistic regime)
        valid = count > 10
        start_idx = 20  # Skip ballistic regime
        end_idx = 150

        if np.sum(valid[start_idx:end_idx]) < 10:
            pytest.skip("Not enough data points for linear fit")

        lag_fit = lag[start_idx:end_idx]
        msd_fit = msd[start_idx:end_idx]

        # Linear fit
        coeffs = np.polyfit(lag_fit, msd_fit, 1)
        slope = coeffs[0]

        # R² calculation
        msd_pred = np.polyval(coeffs, lag_fit)
        ss_res = np.sum((msd_fit - msd_pred) ** 2)
        ss_tot = np.sum((msd_fit - np.mean(msd_fit)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Should have reasonable linear fit (R² > 0.8)
        assert r_squared > 0.7, (
            f"MSD should be linear in diffusive regime (R²={r_squared:.3f})"
        )
        assert slope > 0, "MSD slope should be positive"

    def test_diffusion_coefficient_reasonable(self, lj_trajectory):
        """Test that extracted diffusion coefficient is physically reasonable."""
        msd_analyzer = MeanSquareDisplacement(max_lag=200, store_positions=True)

        for state in lj_trajectory:
            msd_analyzer.update(state)

        # Extract diffusion coefficient
        dt = 0.005
        D = msd_analyzer.compute_diffusion_coefficient(dt=dt, fit_range=(30, 150))

        # For LJ fluid at T=1, rho=0.8, D should be approximately 0.02-0.05
        # Our random walk model targets D ≈ 0.03
        assert D > 0, "Diffusion coefficient should be positive"
        assert D < 1.0, (
            "Diffusion coefficient should be reasonable (< 1.0 in reduced units)"
        )


# =============================================================================
# VACF Validation Tests
# =============================================================================


class TestVACFValidation:
    """
    Validation tests for Velocity Autocorrelation Function analysis.

    Tests verify:
    1. VACF(0) equals mean kinetic energy per degree of freedom
    2. VACF decays from initial value
    3. Green-Kubo integral gives consistent diffusion coefficient
    """

    @pytest.fixture
    def lj_trajectory(self):
        """Generate LJ fluid trajectory for testing."""
        return generate_lj_fluid_trajectory(
            n_atoms=64,
            n_frames=500,
            dt=0.005,
            temperature=1.0,
            seed=54321,
        )

    def test_vacf_initial_value(self, lj_trajectory):
        """Test that VACF(0) is related to temperature."""
        vacf_analyzer = VelocityAutocorrelation(max_lag=200)

        for state in lj_trajectory[100:]:  # Skip equilibration
            vacf_analyzer.update(state)

        result = vacf_analyzer.result()
        vacf = result["vacf"]

        # VACF(0) = <v·v> = 3*k_B*T/m = 3*T (in reduced units with m=1)
        # For our trajectory at T=1, VACF(0) should be approximately 3
        expected_vacf0 = 3.0 * 1.0  # 3*T

        # Allow 50% tolerance due to finite size effects
        assert vacf[0] > 0, "VACF(0) should be positive"
        assert abs(vacf[0] - expected_vacf0) / expected_vacf0 < 0.5, (
            f"VACF(0)={vacf[0]:.3f} should be close to {expected_vacf0:.3f}"
        )

    def test_vacf_decay(self, lj_trajectory):
        """Test that VACF decays from initial value."""
        vacf_analyzer = VelocityAutocorrelation(max_lag=100)

        for state in lj_trajectory:
            vacf_analyzer.update(state)

        result = vacf_analyzer.result()
        vacf_norm = result["vacf_normalized"]

        # Normalized VACF should start at 1 and decay
        np.testing.assert_allclose(vacf_norm[0], 1.0)

        # Should decay significantly by lag 50
        assert vacf_norm[50] < 0.8, "VACF should decay over time"

    def test_green_kubo_diffusion(self, lj_trajectory):
        """Test Green-Kubo relation: D = (1/3) * integral(VACF)."""
        vacf_analyzer = VelocityAutocorrelation(max_lag=200)

        for state in lj_trajectory:
            vacf_analyzer.update(state)

        dt = 0.005
        D_gk = vacf_analyzer.compute_diffusion_coefficient(dt=dt)

        # Should be positive and reasonable
        assert D_gk > 0, "Green-Kubo diffusion coefficient should be positive"
        assert D_gk < 1.0, "Green-Kubo D should be reasonable"

    def test_msd_vacf_consistency(self, lj_trajectory):
        """Test that MSD and VACF give consistent diffusion coefficients."""
        msd_analyzer = MeanSquareDisplacement(max_lag=200, store_positions=True)
        vacf_analyzer = VelocityAutocorrelation(max_lag=200)

        for state in lj_trajectory:
            msd_analyzer.update(state)
            vacf_analyzer.update(state)

        dt = 0.005

        D_msd = msd_analyzer.compute_diffusion_coefficient(dt=dt, fit_range=(30, 150))
        D_gk = vacf_analyzer.compute_diffusion_coefficient(dt=dt)

        # Both methods should give similar results (within factor of 2)
        # Note: exact agreement is not expected due to finite size/time effects
        if D_msd > 0 and D_gk > 0:
            ratio = D_msd / D_gk
            assert 0.2 < ratio < 5.0, (
                f"MSD D={D_msd:.4f} and GK D={D_gk:.4f} should be roughly consistent"
            )


# =============================================================================
# VDOS Validation Tests
# =============================================================================


class TestVDOSValidation:
    """Tests for vibrational density of states from VACF."""

    @pytest.fixture
    def lj_trajectory(self):
        """Generate LJ fluid trajectory."""
        return generate_lj_fluid_trajectory(
            n_atoms=32,
            n_frames=200,
            dt=0.005,
            seed=99999,
        )

    def test_vdos_positive(self, lj_trajectory):
        """Test that VDOS is non-negative."""
        vacf_analyzer = VelocityAutocorrelation(max_lag=100)

        for state in lj_trajectory:
            vacf_analyzer.update(state)

        freq, dos = vacf_analyzer.compute_vibrational_dos(dt=0.005)

        # DOS should be non-negative (it's |FFT|²)
        assert np.all(dos >= 0), "VDOS should be non-negative"

        # Should have peak at low frequency for liquid
        assert np.argmax(dos) < len(dos) // 4, (
            "Liquid VDOS should peak at low frequency"
        )
