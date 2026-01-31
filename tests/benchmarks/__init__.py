"""
Phased benchmark tests organized by build order.

Build Order (Agent-Friendly):
  Phase A - Physics:     Single-particle, Harmonic oscillator, LJ pair
  Phase B - MD:          NVE LJ box, RDF, MSD
  Phase C - Parallel:    Determinism, Strong scaling
  Phase D - ML:          Static ML accuracy, ML MD stability
"""

__all__: list[str] = []
