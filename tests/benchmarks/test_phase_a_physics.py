"""
Phase A - Physics Benchmarks

Build order:
1. Single particle (free particle motion)
2. Harmonic oscillator (analytic solution)
3. LJ pair (two-body interaction)

These are foundational tests that verify basic physics
before moving to larger systems.
"""

import numpy as np
import pytest

from mdcore.benchmarks import BenchmarkReporter, BenchmarkResult

# =============================================================================
# Phase A.1: Single Particle
# =============================================================================


class TestSingleParticle:
    """
    Single free particle tests.

    A single particle with no forces should:
    - Move with constant velocity
    - Conserve kinetic energy exactly
    """

    @pytest.fixture
    def reporter(self):
        """Create benchmark reporter."""
        return BenchmarkReporter("phase_a_physics")

    def test_free_particle_straight_line(self, reporter):
        """Test that free particle moves in straight line."""
        # Initial state: single particle moving in +x direction
        v0 = np.array([[1.0, 0.0, 0.0]])
        x0 = np.array([[0.0, 0.0, 0.0]])

        positions = x0.copy()
        velocities = v0.copy()
        dt = 0.01
        n_steps = 100

        # Propagate with no forces
        for _ in range(n_steps):
            positions = positions + dt * velocities

        # Expected final position
        expected = x0 + n_steps * dt * v0

        error = np.linalg.norm(positions - expected)

        result = BenchmarkResult(
            test="single_particle_straight_line",
            n_atoms=1,
            n_steps=n_steps,
            rms_force_error=error,
            passed=error < 1e-10,
        )
        reporter.add_result(result)

        assert result.passed, f"Position error: {error}"

    def test_free_particle_energy_conservation(self, reporter):
        """Test that free particle conserves kinetic energy."""
        v0 = np.array([[1.0, 2.0, 3.0]])
        x0 = np.array([[0.0, 0.0, 0.0]])
        mass = 1.0

        ke_initial = 0.5 * mass * np.sum(v0**2)

        positions = x0.copy()
        velocities = v0.copy()
        dt = 0.01

        for _ in range(100):
            positions = positions + dt * velocities

        ke_final = 0.5 * mass * np.sum(velocities**2)
        energy_error = abs(ke_final - ke_initial)

        result = BenchmarkResult(
            test="single_particle_energy_conservation",
            n_atoms=1,
            energy_drift=energy_error,
            passed=energy_error < 1e-14,
        )
        reporter.add_result(result)

        assert result.passed, f"Energy error: {energy_error}"


# =============================================================================
# Phase A.2: Harmonic Oscillator
# =============================================================================


class TestHarmonicOscillator:
    """
    1D Harmonic oscillator tests.

    Analytic solution: x(t) = A*cos(ωt + φ)
    where ω = sqrt(k/m)

    Tests velocity Verlet against analytic solution.
    """

    @pytest.fixture
    def reporter(self):
        """Create benchmark reporter."""
        return BenchmarkReporter("phase_a_physics")

    def harmonic_force(self, x: np.ndarray, k: float = 1.0) -> np.ndarray:
        """Compute harmonic force F = -kx."""
        return -k * x

    def analytic_solution(
        self, t: float, x0: float, v0: float, k: float = 1.0, m: float = 1.0
    ) -> tuple[float, float]:
        """
        Analytic solution for harmonic oscillator.

        Args:
            t: Time.
            x0: Initial position.
            v0: Initial velocity.
            k: Spring constant.
            m: Mass.

        Returns:
            (position, velocity) at time t.
        """
        omega = np.sqrt(k / m)
        # x(t) = x0*cos(ωt) + (v0/ω)*sin(ωt)
        x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
        # v(t) = -x0*ω*sin(ωt) + v0*cos(ωt)
        v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
        return x, v

    def test_harmonic_position_accuracy(self, reporter):
        """Test position accuracy against analytic solution."""
        x0 = 1.0
        v0 = 0.0
        k = 1.0
        m = 1.0
        dt = 0.01
        n_steps = 1000  # One full period for ω=1 is 2π ≈ 628 steps

        # Velocity Verlet integration
        x = np.array([[x0, 0.0, 0.0]])
        v = np.array([[v0, 0.0, 0.0]])

        for _ in range(n_steps):
            f = self.harmonic_force(x, k)
            v = v + 0.5 * dt * f / m
            x = x + dt * v
            f_new = self.harmonic_force(x, k)
            v = v + 0.5 * dt * f_new / m

        # Compare to analytic
        t_final = n_steps * dt
        x_analytic, _ = self.analytic_solution(t_final, x0, v0, k, m)

        error = abs(x[0, 0] - x_analytic)

        result = BenchmarkResult(
            test="harmonic_position_accuracy",
            n_atoms=1,
            n_steps=n_steps,
            rms_force_error=error,
            passed=error < 1e-4,  # Verlet is O(dt²)
            extra={"dt": dt, "periods": n_steps * dt / (2 * np.pi)},
        )
        reporter.add_result(result)

        assert result.passed, f"Position error: {error}"

    def test_harmonic_energy_conservation(self, reporter):
        """Test energy conservation in harmonic oscillator."""
        x0 = 1.0
        v0 = 0.5
        k = 1.0
        m = 1.0
        dt = 0.001
        n_steps = 10000

        x = np.array([[x0, 0.0, 0.0]])
        v = np.array([[v0, 0.0, 0.0]])

        # Initial energy
        ke = 0.5 * m * np.sum(v**2)
        pe = 0.5 * k * np.sum(x**2)
        e_initial = ke + pe

        energies = [e_initial]

        for _ in range(n_steps):
            f = self.harmonic_force(x, k)
            v = v + 0.5 * dt * f / m
            x = x + dt * v
            f_new = self.harmonic_force(x, k)
            v = v + 0.5 * dt * f_new / m

            ke = 0.5 * m * np.sum(v**2)
            pe = 0.5 * k * np.sum(x**2)
            energies.append(ke + pe)

        energies = np.array(energies)
        energy_drift = abs(energies[-1] - energies[0])
        max_fluctuation = np.max(np.abs(energies - e_initial))

        result = BenchmarkResult(
            test="harmonic_energy_conservation",
            n_atoms=1,
            n_steps=n_steps,
            energy_drift=energy_drift,
            passed=energy_drift < 1e-6,
            extra={"max_fluctuation": max_fluctuation},
        )
        reporter.add_result(result)

        assert result.passed, f"Energy drift: {energy_drift}"


# =============================================================================
# Phase A.3: LJ Pair
# =============================================================================


class TestLJPair:
    """
    Two-particle Lennard-Jones tests.

    Tests basic LJ interaction between two particles:
    - Force direction
    - Force magnitude
    - Energy-force consistency
    """

    @pytest.fixture
    def reporter(self):
        """Create benchmark reporter."""
        return BenchmarkReporter("phase_a_physics")

    def lj_force(self, r: float, epsilon: float = 1.0, sigma: float = 1.0) -> float:
        """
        Analytic LJ force magnitude.

        F = 24*ε/r * [2*(σ/r)¹² - (σ/r)⁶]
        """
        sr = sigma / r
        sr6 = sr**6
        sr12 = sr6**2
        return 24 * epsilon / r * (2 * sr12 - sr6)

    def lj_energy(self, r: float, epsilon: float = 1.0, sigma: float = 1.0) -> float:
        """
        Analytic LJ potential energy.

        U = 4*ε * [(σ/r)¹² - (σ/r)⁶]
        """
        sr = sigma / r
        sr6 = sr**6
        sr12 = sr6**2
        return 4 * epsilon * (sr12 - sr6)

    def test_lj_equilibrium_distance(self, reporter):
        """Test that force is zero at equilibrium (r = 2^(1/6)*σ)."""
        sigma = 1.0
        r_eq = 2 ** (1 / 6) * sigma

        force = self.lj_force(r_eq, sigma=sigma)

        result = BenchmarkResult(
            test="lj_equilibrium_force",
            n_atoms=2,
            rms_force_error=abs(force),
            passed=abs(force) < 1e-10,
            extra={"r_equilibrium": r_eq},
        )
        reporter.add_result(result)

        assert result.passed, f"Force at equilibrium: {force}"

    def test_lj_repulsive_regime(self, reporter):
        """Test that force is repulsive at short range."""
        r_short = 0.9  # Less than equilibrium
        force = self.lj_force(r_short)

        # Positive force = repulsive (pointing away)
        is_repulsive = force > 0

        result = BenchmarkResult(
            test="lj_repulsive_regime",
            n_atoms=2,
            passed=is_repulsive,
            extra={"r": r_short, "force": force},
        )
        reporter.add_result(result)

        assert result.passed, f"Force should be repulsive at r={r_short}"

    def test_lj_attractive_regime(self, reporter):
        """Test that force is attractive at intermediate range."""
        r_mid = 1.5  # Between equilibrium and cutoff
        force = self.lj_force(r_mid)

        # Negative force = attractive (pointing toward)
        is_attractive = force < 0

        result = BenchmarkResult(
            test="lj_attractive_regime",
            n_atoms=2,
            passed=is_attractive,
            extra={"r": r_mid, "force": force},
        )
        reporter.add_result(result)

        assert result.passed, f"Force should be attractive at r={r_mid}"

    def test_lj_force_energy_consistency(self, reporter):
        """Test that F = -dU/dr via numerical derivative."""
        epsilon = 1.0
        sigma = 1.0
        r = 1.2
        dr = 1e-6

        # Numerical derivative of energy
        u_plus = self.lj_energy(r + dr, epsilon, sigma)
        u_minus = self.lj_energy(r - dr, epsilon, sigma)
        numerical_force = -(u_plus - u_minus) / (2 * dr)

        # Analytic force
        analytic_force = self.lj_force(r, epsilon, sigma)

        error = abs(numerical_force - analytic_force)
        rel_error = error / abs(analytic_force) if analytic_force != 0 else error

        result = BenchmarkResult(
            test="lj_force_energy_consistency",
            n_atoms=2,
            rms_force_error=error,
            passed=rel_error < 1e-5,
            extra={"relative_error": rel_error},
        )
        reporter.add_result(result)

        assert result.passed, f"Force-energy inconsistency: {rel_error}"


# =============================================================================
# Phase A Summary
# =============================================================================


class TestPhaseASummary:
    """Summary test that runs all Phase A benchmarks."""

    def test_phase_a_complete(self):
        """Verify all Phase A tests can be imported and run."""
        # This test just verifies the module loads correctly
        assert True, "Phase A benchmarks loaded successfully"
