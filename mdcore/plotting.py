"""
Built-in plotting utilities for simulation results.

Provides simple one-line plotting functions for common visualizations.

Example:
    >>> from mdcore import simulate, plotting
    >>> result = simulate.lj_fluid(n_atoms=64)
    >>> plotting.energy(result)
    >>> plotting.save("my_simulation.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .simulate import SimulationResult

# Try to import matplotlib, but don't fail if not available
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )


def energy(
    result: SimulationResult,
    show: bool = True,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """
    Plot energy time series.

    Shows kinetic, potential, and total energy vs time.

    Args:
        result: SimulationResult from a simulation.
        show: Whether to display the plot immediately.
        figsize: Figure size (width, height) in inches.

    Example:
        >>> result = simulate.lj_fluid()
        >>> plotting.energy(result)
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    times = np.arange(len(result.total_energy)) * result.timestep

    # Energy vs time
    ax = axes[0]
    ax.plot(times, result.kinetic_energy, "b-", label="Kinetic", alpha=0.7, lw=0.8)
    ax.plot(times, result.potential_energy, "r-", label="Potential", alpha=0.7, lw=0.8)
    ax.plot(times, result.total_energy, "k-", label="Total", lw=1.5)
    ax.set_xlabel("Time (τ)")
    ax.set_ylabel("Energy (ε)")
    ax.set_title("Energy vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy conservation
    ax = axes[1]
    if len(result.total_energy) > 0:
        e0 = result.total_energy[0]
        rel_error = (
            (result.total_energy - e0) / abs(e0) * 100
            if e0 != 0
            else result.total_energy * 0
        )
        ax.plot(times, rel_error, "k-", lw=1)
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (τ)")
    ax.set_ylabel("Relative Energy Error (%)")
    ax.set_title(f"Energy Conservation (fluct: {result.energy_fluctuation:.2e})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()


def temperature(
    result: SimulationResult,
    show: bool = True,
    figsize: tuple[float, float] = (10, 4),
) -> None:
    """
    Plot temperature time series.

    Args:
        result: SimulationResult from a simulation.
        show: Whether to display the plot immediately.
        figsize: Figure size (width, height) in inches.

    Example:
        >>> result = simulate.lj_fluid()
        >>> plotting.temperature(result)
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)
    times = np.arange(len(result.temperature)) * result.timestep

    ax.plot(times, result.temperature, "b-", alpha=0.7, lw=0.5)
    ax.axhline(
        y=result.mean_temperature,
        color="r",
        linestyle="--",
        lw=2,
        label=f"Mean T* = {result.mean_temperature:.3f}",
    )
    ax.fill_between(
        times,
        result.mean_temperature - np.std(result.temperature),
        result.mean_temperature + np.std(result.temperature),
        alpha=0.2,
        color="r",
        label=f"±1σ = {np.std(result.temperature):.3f}",
    )

    ax.set_xlabel("Time (τ)")
    ax.set_ylabel("Temperature T*")
    ax.set_title("Temperature vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()


def rdf(
    result: SimulationResult,
    show: bool = True,
    figsize: tuple[float, float] = (8, 5),
) -> None:
    """
    Plot radial distribution function g(r).

    Args:
        result: SimulationResult with RDF data.
        show: Whether to display the plot immediately.
        figsize: Figure size (width, height) in inches.

    Example:
        >>> result = simulate.lj_fluid(compute_rdf=True)
        >>> plotting.rdf(result)
    """
    _check_matplotlib()

    if not result.rdf:
        print("No RDF data available. Run simulation with compute_rdf=True")
        return

    r = result.rdf.get("r", np.array([]))
    g_r = result.rdf.get("g_r", np.array([]))

    if len(r) == 0 or len(g_r) == 0:
        print("RDF data is empty")
        return

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(r, g_r, "b-", lw=1.5)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Ideal gas")
    ax.set_xlabel("r (σ)")
    ax.set_ylabel("g(r)")
    ax.set_title("Radial Distribution Function")
    ax.set_xlim(0, max(r))
    ax.set_ylim(0, None)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()


def msd(
    result: SimulationResult,
    show: bool = True,
    figsize: tuple[float, float] = (8, 5),
) -> None:
    """
    Plot mean square displacement.

    Shows MSD vs time with linear fit for diffusion coefficient.

    Args:
        result: SimulationResult with MSD data.
        show: Whether to display the plot immediately.
        figsize: Figure size (width, height) in inches.

    Example:
        >>> result = simulate.lj_fluid(compute_msd=True)
        >>> plotting.msd(result)
    """
    _check_matplotlib()

    if not result.msd:
        print("No MSD data available. Run simulation with compute_msd=True")
        return

    msd_data = result.msd.get("msd", np.array([]))
    if len(msd_data) == 0:
        print("MSD data is empty")
        return

    times = np.arange(len(msd_data)) * result.timestep

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(times, msd_data, "b-", lw=1.5, label="MSD")

    # Linear fit for diffusion
    if len(msd_data) > 30:
        fit_start, fit_end = 20, min(100, len(msd_data) - 1)
        t_fit = times[fit_start:fit_end]
        msd_fit = msd_data[fit_start:fit_end]
        coeffs = np.polyfit(t_fit, msd_fit, 1)
        D = coeffs[0] / 6.0

        ax.plot(
            t_fit,
            np.polyval(coeffs, t_fit),
            "r--",
            lw=2,
            label=f"Linear fit (D* = {D:.4f})",
        )

    ax.set_xlabel("Time (τ)")
    ax.set_ylabel("MSD (σ²)")
    ax.set_title("Mean Square Displacement")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()


def trajectory_2d(
    result: SimulationResult,
    atom_indices: list[int] | None = None,
    projection: str = "xy",
    show: bool = True,
    figsize: tuple[float, float] = (8, 8),
) -> None:
    """
    Plot 2D projection of atom trajectories.

    Args:
        result: SimulationResult with trajectory data.
        atom_indices: List of atom indices to plot (default: first 5).
        projection: Projection plane ("xy", "xz", or "yz").
        show: Whether to display the plot immediately.
        figsize: Figure size (width, height) in inches.

    Example:
        >>> result = simulate.lj_fluid()
        >>> plotting.trajectory_2d(result, atom_indices=[0, 1, 2])
    """
    _check_matplotlib()

    if len(result.positions) == 0:
        print("No trajectory data available")
        return

    proj_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if projection not in proj_map:
        print(f"Invalid projection '{projection}'. Use 'xy', 'xz', or 'yz'")
        return

    idx1, idx2 = proj_map[projection]
    labels = ["x", "y", "z"]

    if atom_indices is None:
        atom_indices = list(range(min(5, result.n_atoms)))

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(atom_indices)))

    for i, atom_idx in enumerate(atom_indices):
        if atom_idx >= result.n_atoms:
            continue
        traj = result.positions[:, atom_idx, :]
        ax.plot(
            traj[:, idx1],
            traj[:, idx2],
            "-",
            color=colors[i],
            alpha=0.5,
            lw=0.5,
            label=f"Atom {atom_idx}",
        )
        # Mark start and end
        ax.scatter(traj[0, idx1], traj[0, idx2], color=colors[i], s=50, marker="o")
        ax.scatter(traj[-1, idx1], traj[-1, idx2], color=colors[i], s=50, marker="x")

    ax.set_xlabel(f"{labels[idx1]} (σ)")
    ax.set_ylabel(f"{labels[idx2]} (σ)")
    ax.set_title(f"Atom Trajectories ({projection} projection)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()


def summary(
    result: SimulationResult,
    show: bool = True,
    figsize: tuple[float, float] = (14, 10),
) -> None:
    """
    Plot a comprehensive summary of simulation results.

    Shows energy, temperature, RDF, and MSD in a single figure.

    Args:
        result: SimulationResult from a simulation.
        show: Whether to display the plot immediately.
        figsize: Figure size (width, height) in inches.

    Example:
        >>> result = simulate.lj_fluid(compute_rdf=True, compute_msd=True)
        >>> plotting.summary(result)
    """
    _check_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    times = np.arange(len(result.total_energy)) * result.timestep

    # Energy
    ax = axes[0, 0]
    ax.plot(times, result.kinetic_energy, "b-", label="KE", alpha=0.7, lw=0.5)
    ax.plot(times, result.potential_energy, "r-", label="PE", alpha=0.7, lw=0.5)
    ax.plot(times, result.total_energy, "k-", label="Total", lw=1)
    ax.set_xlabel("Time (τ)")
    ax.set_ylabel("Energy (ε)")
    ax.set_title(f"Energy (fluctuation: {result.energy_fluctuation:.2e})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Temperature
    ax = axes[0, 1]
    ax.plot(times, result.temperature, "b-", alpha=0.7, lw=0.5)
    ax.axhline(y=result.mean_temperature, color="r", linestyle="--", lw=1.5)
    ax.set_xlabel("Time (τ)")
    ax.set_ylabel("Temperature T*")
    ax.set_title(f"Temperature (mean: {result.mean_temperature:.3f})")
    ax.grid(True, alpha=0.3)

    # RDF
    ax = axes[1, 0]
    if result.rdf:
        r = result.rdf.get("r", np.array([]))
        g_r = result.rdf.get("g_r", np.array([]))
        if len(r) > 0 and len(g_r) > 0:
            ax.plot(r, g_r, "b-", lw=1.5)
            ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
            ax.set_xlim(0, max(r))
    ax.set_xlabel("r (σ)")
    ax.set_ylabel("g(r)")
    ax.set_title("Radial Distribution Function")
    ax.grid(True, alpha=0.3)

    # MSD
    ax = axes[1, 1]
    if result.msd:
        msd_data = result.msd.get("msd", np.array([]))
        if len(msd_data) > 0:
            msd_times = np.arange(len(msd_data)) * result.timestep
            ax.plot(msd_times, msd_data, "b-", lw=1.5)

            if result.diffusion_coefficient > 0:
                ax.set_title(f"MSD (D* = {result.diffusion_coefficient:.4f})")
            else:
                ax.set_title("Mean Square Displacement")
    else:
        ax.set_title("Mean Square Displacement")
    ax.set_xlabel("Time (τ)")
    ax.set_ylabel("MSD (σ²)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()


def save(filename: str | Path, dpi: int = 150) -> None:
    """
    Save the current figure to a file.

    Args:
        filename: Output filename (e.g., "plot.png", "plot.pdf").
        dpi: Resolution in dots per inch.

    Example:
        >>> plotting.energy(result, show=False)
        >>> plotting.save("energy.png")
    """
    _check_matplotlib()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to {filename}")


def show() -> None:
    """
    Display all pending plots.

    Use this after creating plots with show=False.

    Example:
        >>> plotting.energy(result, show=False)
        >>> plotting.temperature(result, show=False)
        >>> plotting.show()  # Display both
    """
    _check_matplotlib()
    plt.show()
