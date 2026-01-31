#!/usr/bin/env python
"""
LJ fluid simulation with full analysis.

This example demonstrates:
- Radial distribution function (RDF)
- Mean square displacement (MSD)
- Diffusion coefficient calculation

Usage:
    python examples/run_analysis.py
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np

from mdcore import simulate


def main():
    print("=" * 60)
    print("LJ Fluid Analysis")
    print("=" * 60)

    # Run longer simulation for better statistics
    result = simulate.lj_fluid(
        n_atoms=108,
        temperature=1.0,
        density=0.6,
        n_steps=5000,
        n_equil=1000,
        timestep=0.001,
        compute_rdf=True,
        compute_msd=True,
    )

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # RDF
    ax = axes[0, 0]
    if result.rdf:
        r = result.rdf.get("r", np.array([]))
        g_r = result.rdf.get("g_r", np.array([]))
        if len(r) > 0 and len(g_r) > 0:
            ax.plot(r, g_r, "b-", linewidth=1.5)
            ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
            ax.set_xlabel("r (σ)")
            ax.set_ylabel("g(r)")
            ax.set_title("Radial Distribution Function")
            ax.set_xlim(0, 2.5)
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "RDF not computed", ha="center", va="center")

    # MSD
    ax = axes[0, 1]
    if result.msd:
        msd = result.msd.get("msd", np.array([]))
        if len(msd) > 0:
            times = np.arange(len(msd)) * result.timestep
            ax.plot(times, msd, "b-", linewidth=1.5)

            # Linear fit for diffusion coefficient
            if len(msd) > 30:
                fit_start, fit_end = 20, min(100, len(msd) - 1)
                t_fit = times[fit_start:fit_end]
                msd_fit = msd[fit_start:fit_end]
                coeffs = np.polyfit(t_fit, msd_fit, 1)
                ax.plot(
                    t_fit,
                    np.polyval(coeffs, t_fit),
                    "r--",
                    linewidth=2,
                    label=f"Linear fit (D*={coeffs[0] / 6:.3f})",
                )
                ax.legend()

            ax.set_xlabel("Time (τ)")
            ax.set_ylabel("MSD (σ²)")
            ax.set_title("Mean Square Displacement")
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "MSD not computed", ha="center", va="center")

    # Energy
    ax = axes[1, 0]
    times = np.arange(len(result.total_energy)) * result.timestep
    ax.plot(times, result.total_energy, "k-", linewidth=0.5, alpha=0.7)
    ax.axhline(
        y=np.mean(result.total_energy),
        color="r",
        linestyle="--",
        label=f"Mean = {np.mean(result.total_energy):.2f}",
    )
    ax.set_xlabel("Time (τ)")
    ax.set_ylabel("Total Energy (ε)")
    ax.set_title("Total Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Temperature
    ax = axes[1, 1]
    ax.plot(times, result.temperature, "b-", linewidth=0.5, alpha=0.7)
    ax.axhline(
        y=result.mean_temperature,
        color="r",
        linestyle="--",
        label=f"Mean T* = {result.mean_temperature:.3f}",
    )
    ax.set_xlabel("Time (τ)")
    ax.set_ylabel("Temperature T*")
    ax.set_title("Temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lj_analysis.png", dpi=150)
    print("\nPlot saved to lj_analysis.png")

    # Print summary
    print("\nAnalysis Summary:")
    print(f"  Mean temperature: {result.mean_temperature:.4f}")
    print(f"  Energy fluctuation: {result.energy_fluctuation:.2e}")
    print(f"  Diffusion coefficient D*: {result.diffusion_coefficient:.4f}")

    # Physical interpretation
    print("\nPhysical Interpretation:")
    if result.diffusion_coefficient > 0.01:
        print("  ✓ System is in fluid/liquid phase (diffusive motion)")
    else:
        print("  ✗ System may be in solid phase (low diffusion)")

    if result.energy_fluctuation < 0.01:
        print("  ✓ Energy is well conserved")
    else:
        print("  ! Energy fluctuation is high (may need smaller timestep)")


if __name__ == "__main__":
    main()
