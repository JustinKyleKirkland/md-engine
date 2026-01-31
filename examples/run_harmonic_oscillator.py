#!/usr/bin/env python
"""
Harmonic oscillator simulation.

This is a simple test case for verifying the integrator. The analytical
solution is known, so we can compare numerical results.

For a harmonic oscillator:
- Position: x(t) = A*cos(ωt + φ)
- Period: T = 2π√(m/k)
- Energy: E = ½kA² (constant)

Usage:
    python examples/run_harmonic_oscillator.py
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np

from mdcore import simulate


def main():
    print("=" * 60)
    print("Harmonic Oscillator Simulation")
    print("=" * 60)

    # Run simulation
    result = simulate.harmonic_oscillator(
        n_steps=2000,
        timestep=0.01,
        spring_constant=1.0,
        mass=1.0,
        initial_displacement=1.0,
    )

    # Extract data
    positions = result.positions[:, 0, 0]  # x-coordinate
    pe = result.potential_energy
    ke = result.kinetic_energy
    total = result.total_energy
    times = np.arange(len(positions)) * result.timestep

    # Analytical solution
    omega = np.sqrt(1.0)  # √(k/m)
    x_analytical = 1.0 * np.cos(omega * times)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Position vs time
    ax = axes[0, 0]
    ax.plot(times, positions, "b-", label="Numerical", linewidth=1)
    ax.plot(times, x_analytical, "r--", label="Analytical", linewidth=1, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Position")
    ax.set_title("Position vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy vs time
    ax = axes[0, 1]
    ax.plot(times, ke, "b-", label="Kinetic", linewidth=1)
    ax.plot(times, pe, "r-", label="Potential", linewidth=1)
    ax.plot(times, total, "k-", label="Total", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title("Energy vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy conservation (zoomed)
    ax = axes[1, 0]
    energy_error = (total - total[0]) / total[0] * 100
    ax.plot(times, energy_error, "k-", linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy Error (%)")
    ax.set_title("Energy Conservation")
    ax.grid(True, alpha=0.3)

    # Phase space
    ax = axes[1, 1]
    velocities = np.gradient(positions, result.timestep)
    ax.plot(positions, velocities, "b-", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title("Phase Space")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("harmonic_oscillator.png", dpi=150)
    print("\nPlot saved to harmonic_oscillator.png")

    # Print summary
    print("\nSummary:")
    print(f"  Max energy error: {np.max(np.abs(energy_error)):.2e}%")
    print(f"  Position RMSE: {np.sqrt(np.mean((positions - x_analytical) ** 2)):.4f}")


if __name__ == "__main__":
    main()
