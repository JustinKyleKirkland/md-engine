#!/usr/bin/env python
"""
Example: Running an LJ fluid simulation in reduced units.

This script demonstrates how to:
1. Create a system (positions, velocities, box)
2. Set up force field and neighbor list
3. Equilibrate with velocity rescaling
4. Run production NVE simulation
5. Analyze results

Reduced LJ units:
- Length: σ (LJ size parameter)
- Energy: ε (LJ well depth)
- Mass: m (particle mass)
- Time: τ = σ√(m/ε)
- Temperature: T* = kT/ε (so kB = 1)

Usage:
    python examples/run_lj_simulation.py
"""

import numpy as np

from mdcore.analysis import EnergyAnalyzer, MeanSquareDisplacement
from mdcore.forcefields import ForceField
from mdcore.forcefields.nonbonded import LennardJonesForce
from mdcore.integrators import VelocityVerletIntegrator
from mdcore.neighborlists import VerletList
from mdcore.system import Box, MDState


def compute_temperature_reduced(velocities: np.ndarray, masses: np.ndarray) -> float:
    """
    Compute temperature in reduced LJ units.

    T* = (2/3) * KE / N  (with kB = 1)

    Args:
        velocities: Velocity array (N, 3).
        masses: Mass array (N,).

    Returns:
        Reduced temperature T*.
    """
    n_atoms = len(velocities)
    if n_atoms <= 1:
        return 0.0
    ke = 0.5 * np.sum(masses[:, None] * velocities**2)
    # T* = 2*KE / (3*N - 3) for 3D with COM constraint removed
    n_dof = 3 * n_atoms - 3
    return 2.0 * ke / n_dof


def rescale_velocities(
    velocities: np.ndarray,
    masses: np.ndarray,
    target_temp: float,
) -> np.ndarray:
    """
    Rescale velocities to target temperature.

    Args:
        velocities: Current velocities (N, 3).
        masses: Masses (N,).
        target_temp: Target reduced temperature T*.

    Returns:
        Rescaled velocities.
    """
    current_temp = compute_temperature_reduced(velocities, masses)
    if current_temp > 0:
        scale = np.sqrt(target_temp / current_temp)
        return velocities * scale
    return velocities


def create_lj_system(
    n_atoms: int = 64,
    density: float = 0.5,
    temperature: float = 1.0,
    seed: int = 42,
) -> MDState:
    """
    Create a Lennard-Jones fluid system.

    Args:
        n_atoms: Number of atoms.
        density: Reduced number density ρ* = N*σ³/V.
        temperature: Target reduced temperature T* = kT/ε.
        seed: Random seed for reproducibility.

    Returns:
        Initial MDState.
    """
    rng = np.random.default_rng(seed)

    # Calculate box size from density
    # ρ* = N/V, so V = N/ρ*, L = V^(1/3)
    volume = n_atoms / density
    box_length = volume ** (1 / 3)
    box = Box.cubic(box_length)

    # Create FCC-like lattice positions to avoid overlaps
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

    # Add small random displacement to break symmetry
    positions += rng.uniform(-0.1, 0.1, positions.shape)

    # Maxwell-Boltzmann velocities in reduced units
    # For T* = 1: <v²> = 3*T*/m = 3 (with m=1)
    # Each component: σ_v = √(T*/m) = √T*
    sigma_v = np.sqrt(temperature)
    velocities = rng.normal(0, sigma_v, (n_atoms, 3))
    velocities -= velocities.mean(axis=0)  # Remove net momentum

    masses = np.ones(n_atoms)

    # Rescale to exact target temperature
    velocities = rescale_velocities(velocities, masses, temperature)

    return MDState(
        positions=positions,
        velocities=velocities,
        forces=np.zeros((n_atoms, 3)),
        masses=masses,
        box=box,
    )


def run_simulation(
    n_atoms: int = 64,
    n_equil: int = 500,
    n_prod: int = 1000,
    dt: float = 0.002,
    temperature: float = 1.0,
    density: float = 0.8,
    print_freq: int = 100,
    rescale_freq: int = 10,
):
    """
    Run LJ fluid simulation with equilibration and production.

    Args:
        n_atoms: Number of atoms.
        n_equil: Number of equilibration steps (with thermostat).
        n_prod: Number of production NVE steps.
        dt: Timestep in reduced units (τ = σ√(m/ε)).
        temperature: Target reduced temperature T* = kT/ε.
        density: Reduced density ρ* = Nσ³/V.
        print_freq: Print frequency.
        rescale_freq: Velocity rescaling frequency during equilibration.
    """
    print("=" * 60)
    print("LJ Fluid Simulation (Reduced Units)")
    print("=" * 60)
    print("\nParameters:")
    print(f"  N = {n_atoms} atoms")
    print(f"  ρ* = {density} (reduced density)")
    print(f"  T* = {temperature} (reduced temperature)")
    print(f"  dt = {dt} τ")

    # 1. Create initial system
    print("\nCreating system...")
    state = create_lj_system(
        n_atoms=n_atoms,
        density=density,
        temperature=temperature,
    )
    box_l = state.box.lengths[0]
    print(f"  Box: {box_l:.2f}σ × {box_l:.2f}σ × {box_l:.2f}σ")
    print(
        f"  Initial T*: {compute_temperature_reduced(state.velocities, state.masses):.3f}"
    )

    # 2. Set up force field
    atom_types = np.zeros(n_atoms, dtype=np.int32)
    lj_force = LennardJonesForce(
        epsilon=np.array([1.0]),
        sigma=np.array([1.0]),
        atom_types=atom_types,
        cutoff=2.5,
    )
    forcefield = ForceField([lj_force])

    # 3. Set up neighbor list and integrator
    neighbor_list = VerletList(cutoff=2.5, skin=0.3)
    integrator = VelocityVerletIntegrator(dt=dt)

    positions = state.positions.copy()
    velocities = state.velocities.copy()
    masses = state.masses
    box = state.box

    # =========================================
    # PHASE 1: Equilibration (with thermostat)
    # =========================================
    print(f"\n--- Equilibration ({n_equil} steps) ---")
    print(f"{'Step':>8} {'PE/N':>10} {'KE/N':>10} {'T*':>8}")
    print("-" * 40)

    for step in range(n_equil):
        current_state = MDState(
            positions=positions,
            velocities=velocities,
            forces=np.zeros_like(positions),
            masses=masses,
            box=box,
        )

        neighbor_list.build(positions, box)
        forces, pe = forcefield.compute_with_energy(current_state, neighbor_list)

        # Print progress
        if step % print_freq == 0:
            ke = current_state.kinetic_energy
            temp = compute_temperature_reduced(velocities, masses)
            print(f"{step:>8} {pe / n_atoms:>10.4f} {ke / n_atoms:>10.4f} {temp:>8.3f}")

        # Integrate
        new_state = integrator.step(current_state, forces)
        positions = new_state.positions
        velocities = new_state.velocities

        # Velocity rescaling thermostat
        if step % rescale_freq == 0:
            velocities = rescale_velocities(velocities, masses, temperature)

    print("-" * 40)
    final_temp = compute_temperature_reduced(velocities, masses)
    print(f"Equilibration complete. Final T* = {final_temp:.3f}")

    # =========================================
    # PHASE 2: Production (NVE)
    # =========================================
    print(f"\n--- Production NVE ({n_prod} steps) ---")
    print(f"{'Step':>8} {'PE/N':>10} {'KE/N':>10} {'Total/N':>10} {'T*':>8}")
    print("-" * 52)

    # Set up analyzers for production only
    energy_analyzer = EnergyAnalyzer()
    msd_analyzer = MeanSquareDisplacement(
        max_lag=min(200, n_prod // 2), store_positions=True
    )

    pe_values = []
    ke_values = []
    temp_values = []

    for step in range(n_prod):
        current_state = MDState(
            positions=positions,
            velocities=velocities,
            forces=np.zeros_like(positions),
            masses=masses,
            box=box,
            step=step,
            time=step * dt,
        )

        neighbor_list.build(positions, box)
        forces, pe = forcefield.compute_with_energy(current_state, neighbor_list)

        ke = current_state.kinetic_energy
        temp = compute_temperature_reduced(velocities, masses)

        pe_values.append(pe)
        ke_values.append(ke)
        temp_values.append(temp)

        energy_analyzer.update(current_state, potential_energy=pe)
        msd_analyzer.update(current_state, time=step * dt)

        # Print progress
        if step % print_freq == 0:
            total = pe + ke
            print(
                f"{step:>8} {pe / n_atoms:>10.4f} {ke / n_atoms:>10.4f} {total / n_atoms:>10.4f} {temp:>8.3f}"
            )

        # Integrate (pure NVE, no thermostat)
        new_state = integrator.step(current_state, forces)
        positions = new_state.positions
        velocities = new_state.velocities

    print("-" * 52)

    # =========================================
    # RESULTS
    # =========================================
    print("\n" + "=" * 60)
    print("Results (Production Phase)")
    print("=" * 60)

    pe_arr = np.array(pe_values)
    ke_arr = np.array(ke_values)
    total_arr = pe_arr + ke_arr
    temp_arr = np.array(temp_values)

    print(f"\nThermodynamics ({n_prod} frames):")
    print(
        f"  <PE/N>  = {np.mean(pe_arr) / n_atoms:.4f} ± {np.std(pe_arr) / n_atoms:.4f} ε"
    )
    print(
        f"  <KE/N>  = {np.mean(ke_arr) / n_atoms:.4f} ± {np.std(ke_arr) / n_atoms:.4f} ε"
    )
    print(
        f"  <E/N>   = {np.mean(total_arr) / n_atoms:.4f} ± {np.std(total_arr) / n_atoms:.4f} ε"
    )
    print(f"  <T*>    = {np.mean(temp_arr):.4f} ± {np.std(temp_arr):.4f}")

    # Energy conservation check
    drift_per_step = (total_arr[-1] - total_arr[0]) / n_prod
    rel_fluct = np.std(total_arr) / np.abs(np.mean(total_arr))
    print("\nEnergy Conservation:")
    print(f"  Drift/step:        {drift_per_step:.2e} ε")
    print(f"  Relative fluct:    {rel_fluct:.2e}")
    if rel_fluct > 0.01:
        print("  Note: Drift due to sharp LJ cutoff (needs switching function)")

    # MSD and Diffusion
    msd_result = msd_analyzer.result()
    msd_arr = msd_result["msd"]
    lag_times = np.arange(len(msd_arr)) * dt

    print("\nDynamics:")
    lag_50 = min(50, len(msd_arr) - 1)
    print(f"  MSD(t={lag_50 * dt:.2f}τ) = {msd_arr[lag_50]:.4f} σ²")

    if len(msd_arr) > 30:
        # Fit MSD = 6*D*t in diffusive regime
        fit_start, fit_end = 20, min(100, len(msd_arr) - 1)
        t_fit = lag_times[fit_start:fit_end]
        msd_fit = msd_arr[fit_start:fit_end]
        # Linear fit: MSD = 6*D*t
        D = np.polyfit(t_fit, msd_fit, 1)[0] / 6.0
        print(f"  D* = {D:.4f} σ²/τ (from MSD slope)")

    # Physical sanity checks
    print("\nPhysical Checks:")
    mean_temp = np.mean(temp_arr)
    temp_ok = 0.8 * temperature < mean_temp < 1.2 * temperature
    print(
        f"  Temperature stable: {'✓' if temp_ok else '✗'} (T* = {mean_temp:.3f}, target = {temperature})"
    )

    energy_ok = rel_fluct < 0.01
    print(
        f"  Energy conserved:   {'✓' if energy_ok else '✗'} (fluct = {rel_fluct:.1e})"
    )

    is_fluid = D > 0.01 if len(msd_arr) > 30 else msd_arr[lag_50] > 0.1
    print(f"  Fluid behavior:     {'✓' if is_fluid else '✗'} (diffusive motion)")

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    # LJ fluid state points:
    # - ρ* = 0.8, T* = 1.0: dense liquid (may need small dt)
    # - ρ* = 0.5, T* = 1.0: moderate liquid
    # - ρ* = 0.3, T* = 1.5: gas
    run_simulation(
        n_atoms=64,
        n_equil=2000,
        n_prod=5000,
        dt=0.0005,  # Very small timestep
        temperature=1.0,
        density=0.5,
        print_freq=500,
        rescale_freq=5,
    )
