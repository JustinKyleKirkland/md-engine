#!/usr/bin/env python
"""
NVT (constant temperature) LJ fluid simulation using Langevin dynamics.

This example demonstrates:
- Temperature control via Langevin thermostat
- Equilibration to target temperature
- Temperature fluctuations in NVT ensemble

Usage:
    python examples/run_nvt_simulation.py
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np

from mdcore import simulate


def main():
    print("=" * 60)
    print("NVT LJ Fluid Simulation (Langevin Dynamics)")
    print("=" * 60)

    # Run NVT simulation
    result = simulate.nvt_lj_fluid(
        n_atoms=64,
        temperature=1.0,
        density=0.5,
        n_steps=5000,
        timestep=0.001,
        friction=1.0,
    )

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    times = np.arange(len(result.temperature)) * result.timestep

    # Temperature vs time
    ax = axes[0, 0]
    ax.plot(times, result.temperature, "b-", linewidth=0.5, alpha=0.7)
    ax.axhline(y=1.0, color="r", linestyle="--", label="Target T*=1.0")
    ax.axhline(
        y=result.mean_temperature,
        color="g",
        linestyle="-",
        label=f"Mean T*={result.mean_temperature:.3f}",
    )
    ax.set_xlabel("Time (τ)")
    ax.set_ylabel("Temperature T*")
    ax.set_title("Temperature vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy vs time
    ax = axes[0, 1]
    ax.plot(times, result.kinetic_energy, "b-", label="KE", linewidth=0.5, alpha=0.7)
    ax.plot(times, result.potential_energy, "r-", label="PE", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Time (τ)")
    ax.set_ylabel("Energy (ε)")
    ax.set_title("Energy vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Temperature histogram
    ax = axes[1, 0]
    ax.hist(result.temperature, bins=50, density=True, alpha=0.7, color="blue")
    ax.axvline(x=1.0, color="r", linestyle="--", label="Target")
    ax.set_xlabel("Temperature T*")
    ax.set_ylabel("Probability Density")
    ax.set_title("Temperature Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Running average of temperature
    ax = axes[1, 1]
    window = 100
    running_avg = np.convolve(
        result.temperature, np.ones(window) / window, mode="valid"
    )
    ax.plot(times[window - 1 :], running_avg, "b-", linewidth=1)
    ax.axhline(y=1.0, color="r", linestyle="--", label="Target T*=1.0")
    ax.set_xlabel("Time (τ)")
    ax.set_ylabel("Temperature T*")
    ax.set_title(f"Running Average (window={window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("nvt_simulation.png", dpi=150)
    print("\nPlot saved to nvt_simulation.png")

    # Print summary
    print("\nSummary:")
    print("  Target temperature: 1.0")
    print(f"  Mean temperature: {result.mean_temperature:.4f}")
    print(f"  Std temperature: {np.std(result.temperature):.4f}")
    print(f"  Temperature within 5%: {abs(result.mean_temperature - 1.0) < 0.05}")


if __name__ == "__main__":
    main()
