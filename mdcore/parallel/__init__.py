"""Parallelization infrastructure for MD simulations."""

from .backends.base import ParallelBackend
from .backends.serial import SerialBackend
from .dispatcher import get_backend, set_default_backend
from .domain import Domain, DomainDecomposition

__all__ = [
    "ParallelBackend",
    "SerialBackend",
    "Domain",
    "DomainDecomposition",
    "get_backend",
    "set_default_backend",
]
