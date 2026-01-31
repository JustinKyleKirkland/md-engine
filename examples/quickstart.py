#!/usr/bin/env python
"""
Quick start example - the simplest way to run a simulation.

This demonstrates the high-level API for users who just want results
without dealing with the internal details.

Usage:
    python examples/quickstart.py
"""

from mdcore import simulate


def main():
    print("=" * 60)
    print("MD Engine Quick Start")
    print("=" * 60)

    # 1. Simplest possible simulation - just 1 line!
    print("\n1. LJ Fluid (simplest usage):")
    print("-" * 40)
    result = simulate.lj_fluid()
    print(f"   All checks passed: {result.energy_fluctuation < 0.01}")

    # 2. Customize parameters
    print("\n2. LJ Fluid (custom parameters):")
    print("-" * 40)
    result = simulate.lj_fluid(
        n_atoms=108,
        temperature=0.8,
        density=0.6,
        n_steps=2000,
    )

    # 3. Harmonic oscillator - great for testing
    print("\n3. Harmonic Oscillator:")
    print("-" * 40)
    result = simulate.harmonic_oscillator(n_steps=5000)
    print(f"   Energy conserved: {result.energy_fluctuation < 1e-6}")

    # 4. Diatomic molecule
    print("\n4. Diatomic Molecule:")
    print("-" * 40)
    result = simulate.diatomic_molecule(n_steps=1000)

    # 5. NVT simulation with thermostat
    print("\n5. NVT LJ Fluid (Langevin thermostat):")
    print("-" * 40)
    result = simulate.nvt_lj_fluid(
        n_atoms=64,
        temperature=1.0,
        n_steps=1000,
    )

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
