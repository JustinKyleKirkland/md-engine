"""
Simple high-level simulation API.

This module provides a user-friendly interface for running MD simulations
with minimal configuration.

Example:
    >>> from mdcore import simulate
    >>> results = simulate.lj_fluid(n_atoms=64, temperature=1.0, n_steps=1000)
    >>> print(results.mean_temperature)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .analysis import EnergyAnalyzer, MeanSquareDisplacement, RadialDistributionFunction
from .forcefields import ForceField
from .forcefields.bonded import HarmonicBondForce
from .forcefields.nonbonded import LennardJonesForce
from .integrators import VelocityVerletIntegrator
from .integrators.langevin import LangevinIntegrator
from .neighborlists import VerletList
from .system import Box, MDState

if TYPE_CHECKING:
    pass


@dataclass
class SimulationResult:
    """Results from a simulation run."""

    # Trajectory data
    positions: NDArray[np.floating] = field(default_factory=lambda: np.array([]))
    velocities: NDArray[np.floating] = field(default_factory=lambda: np.array([]))

    # Energy time series
    kinetic_energy: NDArray[np.floating] = field(default_factory=lambda: np.array([]))
    potential_energy: NDArray[np.floating] = field(default_factory=lambda: np.array([]))
    total_energy: NDArray[np.floating] = field(default_factory=lambda: np.array([]))
    temperature: NDArray[np.floating] = field(default_factory=lambda: np.array([]))

    # Summary statistics
    mean_temperature: float = 0.0
    mean_potential_energy: float = 0.0
    mean_kinetic_energy: float = 0.0
    energy_drift: float = 0.0
    energy_fluctuation: float = 0.0

    # Analysis results
    rdf: dict[str, Any] = field(default_factory=dict)
    msd: dict[str, Any] = field(default_factory=dict)
    diffusion_coefficient: float = 0.0

    # Metadata
    n_atoms: int = 0
    n_steps: int = 0
    timestep: float = 0.0
    box_size: float = 0.0


def _compute_temperature_reduced(
    velocities: NDArray[np.floating], masses: NDArray[np.floating]
) -> float:
    """Compute temperature in reduced units (kB = 1)."""
    n_atoms = len(velocities)
    if n_atoms <= 1:
        return 0.0
    ke = 0.5 * np.sum(masses[:, None] * velocities**2)
    n_dof = 3 * n_atoms - 3
    return 2.0 * ke / n_dof


def _rescale_velocities(
    velocities: NDArray[np.floating],
    masses: NDArray[np.floating],
    target_temp: float,
) -> NDArray[np.floating]:
    """Rescale velocities to target temperature."""
    current_temp = _compute_temperature_reduced(velocities, masses)
    if current_temp > 0:
        scale = np.sqrt(target_temp / current_temp)
        return velocities * scale
    return velocities


def _create_lattice_positions(
    n_atoms: int, box_length: float, seed: int = 42
) -> NDArray[np.floating]:
    """Create positions on a cubic lattice with small random displacements."""
    rng = np.random.default_rng(seed)
    n_side = int(np.ceil(n_atoms ** (1 / 3)))
    spacing = box_length / n_side

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
    return positions


def lj_fluid(
    n_atoms: int = 64,
    temperature: float = 1.0,
    density: float = 0.5,
    n_steps: int = 1000,
    n_equil: int = 500,
    timestep: float = 0.001,
    cutoff: float = 2.5,
    seed: int = 42,
    verbose: bool = True,
    compute_rdf: bool = True,
    compute_msd: bool = True,
) -> SimulationResult:
    """
    Run a Lennard-Jones fluid simulation.

    This is the simplest way to run an LJ simulation. All parameters have
    sensible defaults for a liquid state point.

    Args:
        n_atoms: Number of atoms (default: 64).
        temperature: Reduced temperature T* = kT/ε (default: 1.0).
        density: Reduced density ρ* = Nσ³/V (default: 0.5).
        n_steps: Number of production steps (default: 1000).
        n_equil: Number of equilibration steps (default: 500).
        timestep: Integration timestep in reduced units (default: 0.001).
        cutoff: LJ cutoff distance in σ (default: 2.5).
        seed: Random seed for reproducibility (default: 42).
        verbose: Print progress (default: True).
        compute_rdf: Compute radial distribution function (default: True).
        compute_msd: Compute mean square displacement (default: True).

    Returns:
        SimulationResult with trajectory and analysis data.

    Example:
        >>> result = lj_fluid(n_atoms=64, temperature=1.0, n_steps=1000)
        >>> print(f"Mean temperature: {result.mean_temperature:.3f}")
        >>> print(f"Diffusion coefficient: {result.diffusion_coefficient:.4f}")
    """
    rng = np.random.default_rng(seed)

    # Setup system
    volume = n_atoms / density
    box_length = volume ** (1 / 3)
    box = Box.cubic(box_length)

    positions = _create_lattice_positions(n_atoms, box_length, seed)
    velocities = rng.normal(0, np.sqrt(temperature), (n_atoms, 3))
    velocities -= velocities.mean(axis=0)
    masses = np.ones(n_atoms)
    velocities = _rescale_velocities(velocities, masses, temperature)

    # Setup force field
    atom_types = np.zeros(n_atoms, dtype=np.int32)
    lj_force = LennardJonesForce(
        epsilon=np.array([1.0]),
        sigma=np.array([1.0]),
        atom_types=atom_types,
        cutoff=cutoff,
    )
    forcefield = ForceField([lj_force])
    neighbor_list = VerletList(cutoff=cutoff, skin=0.3)
    integrator = VelocityVerletIntegrator(dt=timestep)

    if verbose:
        print(f"LJ Fluid: N={n_atoms}, ρ*={density}, T*={temperature}")

    # Equilibration
    if verbose:
        print(f"Equilibrating ({n_equil} steps)...", end=" ", flush=True)

    for step in range(n_equil):
        state = MDState(
            positions=positions,
            velocities=velocities,
            forces=np.zeros_like(positions),
            masses=masses,
            box=box,
        )
        neighbor_list.build(positions, box)
        forces, _ = forcefield.compute_with_energy(state, neighbor_list)
        new_state = integrator.step(state, forces)
        positions = new_state.positions
        velocities = new_state.velocities

        if step % 10 == 0:
            velocities = _rescale_velocities(velocities, masses, temperature)

    if verbose:
        print("done")

    # Reset integrator for production
    integrator.reset()

    # Setup analyzers
    energy_analyzer = EnergyAnalyzer()
    rdf_analyzer = (
        RadialDistributionFunction(r_max=cutoff, n_bins=100) if compute_rdf else None
    )
    msd_analyzer = (
        MeanSquareDisplacement(max_lag=min(200, n_steps // 2), store_positions=True)
        if compute_msd
        else None
    )

    # Production
    if verbose:
        print(f"Running production ({n_steps} steps)...", end=" ", flush=True)

    pe_values = []
    ke_values = []
    temp_values = []
    traj_positions = []
    traj_velocities = []

    for step in range(n_steps):
        state = MDState(
            positions=positions,
            velocities=velocities,
            forces=np.zeros_like(positions),
            masses=masses,
            box=box,
            step=step,
            time=step * timestep,
        )

        neighbor_list.build(positions, box)
        forces, pe = forcefield.compute_with_energy(state, neighbor_list)

        ke = 0.5 * np.sum(masses[:, None] * velocities**2)
        temp = _compute_temperature_reduced(velocities, masses)

        pe_values.append(pe)
        ke_values.append(ke)
        temp_values.append(temp)
        traj_positions.append(positions.copy())
        traj_velocities.append(velocities.copy())

        energy_analyzer.update(state, potential_energy=pe)
        if rdf_analyzer:
            rdf_analyzer.update(state)
        if msd_analyzer:
            msd_analyzer.update(state, time=step * timestep)

        new_state = integrator.step(state, forces)
        positions = new_state.positions
        velocities = new_state.velocities

    if verbose:
        print("done")

    # Compute results
    pe_arr = np.array(pe_values)
    ke_arr = np.array(ke_values)
    total_arr = pe_arr + ke_arr
    temp_arr = np.array(temp_values)

    result = SimulationResult(
        positions=np.array(traj_positions),
        velocities=np.array(traj_velocities),
        kinetic_energy=ke_arr,
        potential_energy=pe_arr,
        total_energy=total_arr,
        temperature=temp_arr,
        mean_temperature=float(np.mean(temp_arr)),
        mean_potential_energy=float(np.mean(pe_arr)),
        mean_kinetic_energy=float(np.mean(ke_arr)),
        energy_drift=float((total_arr[-1] - total_arr[0]) / n_steps),
        energy_fluctuation=float(np.std(total_arr) / np.abs(np.mean(total_arr))),
        n_atoms=n_atoms,
        n_steps=n_steps,
        timestep=timestep,
        box_size=box_length,
    )

    if rdf_analyzer:
        result.rdf = rdf_analyzer.result()

    if msd_analyzer:
        result.msd = msd_analyzer.result()
        if len(result.msd["msd"]) > 30:
            fit_start, fit_end = 20, min(100, len(result.msd["msd"]) - 1)
            t_fit = np.arange(fit_start, fit_end) * timestep
            msd_fit = result.msd["msd"][fit_start:fit_end]
            result.diffusion_coefficient = float(np.polyfit(t_fit, msd_fit, 1)[0] / 6.0)

    if verbose:
        print("\nResults:")
        print(f"  Mean T*: {result.mean_temperature:.3f}")
        print(f"  Energy fluctuation: {result.energy_fluctuation:.2e}")
        if result.diffusion_coefficient > 0:
            print(f"  Diffusion coefficient D*: {result.diffusion_coefficient:.4f}")

    return result


def harmonic_oscillator(
    n_steps: int = 1000,
    timestep: float = 0.01,
    spring_constant: float = 1.0,
    mass: float = 1.0,
    initial_displacement: float = 1.0,
    verbose: bool = True,
) -> SimulationResult:
    """
    Run a simple harmonic oscillator simulation.

    Great for testing and understanding the integrator behavior.

    Args:
        n_steps: Number of steps (default: 1000).
        timestep: Integration timestep (default: 0.01).
        spring_constant: Spring constant k (default: 1.0).
        mass: Particle mass (default: 1.0).
        initial_displacement: Initial displacement from equilibrium (default: 1.0).
        verbose: Print progress (default: True).

    Returns:
        SimulationResult with trajectory data.

    Example:
        >>> result = harmonic_oscillator(n_steps=1000, timestep=0.01)
        >>> print(f"Energy fluctuation: {result.energy_fluctuation:.2e}")
    """
    box = Box.cubic(20.0)
    equilibrium = 10.0
    k = spring_constant
    m = mass

    positions = np.array([[equilibrium + initial_displacement, 10.0, 10.0]])
    velocities = np.zeros((1, 3))
    masses = np.array([m])

    integrator = VelocityVerletIntegrator(dt=timestep)

    if verbose:
        print(f"Harmonic oscillator: k={k}, m={m}, x0={initial_displacement}")
        print(f"Running {n_steps} steps...", end=" ", flush=True)

    pe_values = []
    ke_values = []
    traj_positions = []

    state = MDState(
        positions=positions,
        velocities=velocities,
        forces=np.zeros_like(positions),
        masses=masses,
        box=box,
    )

    for step in range(n_steps):
        displacement = state.positions[0, 0] - equilibrium
        force = np.array([[-k * displacement, 0.0, 0.0]])

        pe = 0.5 * k * displacement**2
        ke = 0.5 * m * np.sum(state.velocities**2)

        pe_values.append(pe)
        ke_values.append(ke)
        traj_positions.append(state.positions.copy())

        state = integrator.step(state, force)

    if verbose:
        print("done")

    pe_arr = np.array(pe_values)
    ke_arr = np.array(ke_values)
    total_arr = pe_arr + ke_arr

    result = SimulationResult(
        positions=np.array(traj_positions),
        kinetic_energy=ke_arr,
        potential_energy=pe_arr,
        total_energy=total_arr,
        energy_drift=float((total_arr[-1] - total_arr[0]) / n_steps),
        energy_fluctuation=float(np.std(total_arr) / np.abs(np.mean(total_arr))),
        n_atoms=1,
        n_steps=n_steps,
        timestep=timestep,
    )

    if verbose:
        print("\nResults:")
        print(f"  Energy fluctuation: {result.energy_fluctuation:.2e}")
        # Analytical period: T = 2π√(m/k)
        period = 2 * np.pi * np.sqrt(m / k)
        print(f"  Analytical period: {period:.3f}")
        print(f"  Simulated periods: {n_steps * timestep / period:.1f}")

    return result


def diatomic_molecule(
    n_steps: int = 1000,
    timestep: float = 0.001,
    bond_length: float = 1.0,
    spring_constant: float = 100.0,
    temperature: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> SimulationResult:
    """
    Run a diatomic molecule simulation with a harmonic bond.

    Args:
        n_steps: Number of steps (default: 1000).
        timestep: Integration timestep (default: 0.001).
        bond_length: Equilibrium bond length (default: 1.0).
        spring_constant: Bond spring constant (default: 100.0).
        temperature: Initial temperature (default: 1.0).
        seed: Random seed (default: 42).
        verbose: Print progress (default: True).

    Returns:
        SimulationResult with trajectory data.
    """
    rng = np.random.default_rng(seed)

    box = Box.cubic(10.0)
    center = 5.0

    positions = np.array(
        [
            [center - bond_length / 2, center, center],
            [center + bond_length / 2, center, center],
        ]
    )
    velocities = rng.normal(0, np.sqrt(temperature), (2, 3))
    velocities -= velocities.mean(axis=0)  # Remove COM velocity
    masses = np.ones(2)

    bond_force = HarmonicBondForce(
        bond_indices=np.array([[0, 1]]),
        force_constants=np.array([spring_constant]),
        equilibrium_lengths=np.array([bond_length]),
    )
    forcefield = ForceField([bond_force])
    integrator = VelocityVerletIntegrator(dt=timestep)

    if verbose:
        print(f"Diatomic molecule: r0={bond_length}, k={spring_constant}")
        print(f"Running {n_steps} steps...", end=" ", flush=True)

    pe_values = []
    ke_values = []
    bond_lengths = []

    for step in range(n_steps):
        state = MDState(
            positions=positions,
            velocities=velocities,
            forces=np.zeros_like(positions),
            masses=masses,
            box=box,
            step=step,
            time=step * timestep,
        )

        forces, pe = forcefield.compute_with_energy(state, neighbors=None)
        ke = 0.5 * np.sum(masses[:, None] * velocities**2)

        pe_values.append(pe)
        ke_values.append(ke)
        bond_lengths.append(np.linalg.norm(positions[1] - positions[0]))

        new_state = integrator.step(state, forces)
        positions = new_state.positions
        velocities = new_state.velocities

    if verbose:
        print("done")

    pe_arr = np.array(pe_values)
    ke_arr = np.array(ke_values)
    total_arr = pe_arr + ke_arr
    bond_arr = np.array(bond_lengths)

    result = SimulationResult(
        kinetic_energy=ke_arr,
        potential_energy=pe_arr,
        total_energy=total_arr,
        energy_drift=float((total_arr[-1] - total_arr[0]) / n_steps),
        energy_fluctuation=float(np.std(total_arr) / np.abs(np.mean(total_arr))),
        n_atoms=2,
        n_steps=n_steps,
        timestep=timestep,
    )

    if verbose:
        print("\nResults:")
        print(f"  Mean bond length: {np.mean(bond_arr):.4f} ± {np.std(bond_arr):.4f}")
        print(f"  Energy fluctuation: {result.energy_fluctuation:.2e}")

    return result


def nvt_lj_fluid(
    n_atoms: int = 64,
    temperature: float = 1.0,
    density: float = 0.5,
    n_steps: int = 1000,
    timestep: float = 0.001,
    friction: float = 1.0,
    cutoff: float = 2.5,
    seed: int = 42,
    verbose: bool = True,
) -> SimulationResult:
    """
    Run an NVT (constant temperature) LJ fluid simulation using Langevin dynamics.

    Args:
        n_atoms: Number of atoms (default: 64).
        temperature: Target reduced temperature (default: 1.0).
        density: Reduced density (default: 0.5).
        n_steps: Number of steps (default: 1000).
        timestep: Integration timestep (default: 0.001).
        friction: Langevin friction coefficient (default: 1.0).
        cutoff: LJ cutoff distance (default: 2.5).
        seed: Random seed (default: 42).
        verbose: Print progress (default: True).

    Returns:
        SimulationResult with trajectory data.
    """
    rng = np.random.default_rng(seed)

    volume = n_atoms / density
    box_length = volume ** (1 / 3)
    box = Box.cubic(box_length)

    positions = _create_lattice_positions(n_atoms, box_length, seed)
    velocities = rng.normal(0, np.sqrt(temperature), (n_atoms, 3))
    velocities -= velocities.mean(axis=0)
    masses = np.ones(n_atoms)

    atom_types = np.zeros(n_atoms, dtype=np.int32)
    lj_force = LennardJonesForce(
        epsilon=np.array([1.0]),
        sigma=np.array([1.0]),
        atom_types=atom_types,
        cutoff=cutoff,
    )
    forcefield = ForceField([lj_force])
    neighbor_list = VerletList(cutoff=cutoff, skin=0.3)
    integrator = LangevinIntegrator(
        dt=timestep, temperature=temperature, friction=friction, seed=seed
    )

    if verbose:
        print(f"NVT LJ Fluid (Langevin): N={n_atoms}, T*={temperature}, γ={friction}")
        print(f"Running {n_steps} steps...", end=" ", flush=True)

    pe_values = []
    ke_values = []
    temp_values = []

    for step in range(n_steps):
        state = MDState(
            positions=positions,
            velocities=velocities,
            forces=np.zeros_like(positions),
            masses=masses,
            box=box,
            step=step,
            time=step * timestep,
        )

        neighbor_list.build(positions, box)
        forces, pe = forcefield.compute_with_energy(state, neighbor_list)

        ke = 0.5 * np.sum(masses[:, None] * velocities**2)
        temp = _compute_temperature_reduced(velocities, masses)

        pe_values.append(pe)
        ke_values.append(ke)
        temp_values.append(temp)

        new_state = integrator.step(state, forces)
        positions = new_state.positions
        velocities = new_state.velocities

    if verbose:
        print("done")

    pe_arr = np.array(pe_values)
    ke_arr = np.array(ke_values)
    temp_arr = np.array(temp_values)

    result = SimulationResult(
        kinetic_energy=ke_arr,
        potential_energy=pe_arr,
        total_energy=pe_arr + ke_arr,
        temperature=temp_arr,
        mean_temperature=float(np.mean(temp_arr)),
        mean_potential_energy=float(np.mean(pe_arr)),
        mean_kinetic_energy=float(np.mean(ke_arr)),
        n_atoms=n_atoms,
        n_steps=n_steps,
        timestep=timestep,
        box_size=box_length,
    )

    if verbose:
        print("\nResults:")
        print(f"  Mean T*: {result.mean_temperature:.3f} ± {np.std(temp_arr):.3f}")
        print(f"  Target T*: {temperature}")

    return result
