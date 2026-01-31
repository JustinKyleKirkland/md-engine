"""
mdcore - A modular, composable molecular dynamics engine.

Design Principles:
- Modular, composable MD engine
- Deterministic + reproducible
- CPU/GPU parity
- MPI + shared-memory parallelism
- ML potentials as drop-in force providers
- Analysis as first-class citizens

Quick Start:
    >>> from mdcore import simulate
    >>> result = simulate.lj_fluid(n_atoms=64, temperature=1.0)
    >>> print(f"Mean temperature: {result.mean_temperature:.3f}")
"""

__version__ = "0.1.0"

# High-level APIs
from . import plotting, simulate
from .forcefields import ForceField
from .integrators import VelocityVerletIntegrator
from .neighborlists import VerletList

# Core components for advanced users
from .system import Box, MDState

__all__ = [
    "simulate",
    "plotting",
    "Box",
    "MDState",
    "ForceField",
    "VelocityVerletIntegrator",
    "VerletList",
]
