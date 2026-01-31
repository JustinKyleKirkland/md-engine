"""Backend dispatcher for selecting and managing parallel backends."""

from __future__ import annotations

from typing import Literal

from .backends.base import ParallelBackend
from .backends.serial import SerialBackend

# Global default backend
_default_backend: ParallelBackend | None = None

# Available backend types
BackendType = Literal["serial", "multiprocessing", "mpi4py", "ray"]


def get_backend(
    backend: BackendType | ParallelBackend | None = None,
    **kwargs,
) -> ParallelBackend:
    """
    Get a parallel backend instance.

    Args:
        backend: Backend specification. Can be:
            - None: Use default backend (serial if not set)
            - String: Create backend by name
            - ParallelBackend: Use provided instance directly
        **kwargs: Additional arguments for backend initialization.

    Returns:
        ParallelBackend instance.

    Examples:
        >>> backend = get_backend()  # Default (serial)
        >>> backend = get_backend("multiprocessing", n_workers=4)
        >>> backend = get_backend("mpi4py")
    """
    global _default_backend

    # Use provided backend instance directly
    if isinstance(backend, ParallelBackend):
        return backend

    # Use default if None
    if backend is None:
        if _default_backend is None:
            _default_backend = SerialBackend()
        return _default_backend

    # Create backend by name
    return create_backend(backend, **kwargs)


def create_backend(name: BackendType, **kwargs) -> ParallelBackend:
    """
    Create a parallel backend by name.

    Args:
        name: Backend name.
        **kwargs: Backend-specific arguments.

    Returns:
        ParallelBackend instance.

    Raises:
        ValueError: If backend name is unknown.
        ImportError: If required package is not installed.
    """
    if name == "serial":
        return SerialBackend()

    elif name == "multiprocessing":
        from .backends.multiprocessing_backend import MultiprocessingBackend

        return MultiprocessingBackend(**kwargs)

    elif name == "mpi4py":
        from .backends.mpi4py_backend import MPI4PyBackend

        return MPI4PyBackend()

    elif name == "ray":
        raise NotImplementedError("Ray backend not yet implemented")

    else:
        raise ValueError(
            f"Unknown backend: {name}. Available: serial, multiprocessing, mpi4py, ray"
        )


def set_default_backend(
    backend: BackendType | ParallelBackend,
    **kwargs,
) -> ParallelBackend:
    """
    Set the default parallel backend.

    Args:
        backend: Backend specification (name or instance).
        **kwargs: Arguments for backend creation.

    Returns:
        The new default backend.
    """
    global _default_backend

    if isinstance(backend, ParallelBackend):
        _default_backend = backend
    else:
        _default_backend = create_backend(backend, **kwargs)

    return _default_backend


def reset_default_backend() -> None:
    """Reset default backend to None (will use serial on next get)."""
    global _default_backend
    _default_backend = None


def detect_best_backend() -> BackendType:
    """
    Detect the best available backend.

    Checks for MPI environment, then falls back to multiprocessing or serial.

    Returns:
        Name of recommended backend.
    """
    import os

    # Check for MPI environment variables
    mpi_vars = [
        "OMPI_COMM_WORLD_SIZE",  # OpenMPI
        "PMI_SIZE",  # MPICH
        "SLURM_NTASKS",  # SLURM
        "PBS_NP",  # PBS
    ]

    for var in mpi_vars:
        if var in os.environ:
            try:
                from mpi4py import MPI  # noqa: F401

                return "mpi4py"
            except ImportError:
                pass
            break

    # Check CPU count for multiprocessing benefit
    import multiprocessing as mp

    if mp.cpu_count() > 1:
        return "multiprocessing"

    return "serial"


def auto_backend(**kwargs) -> ParallelBackend:
    """
    Automatically select and create the best backend.

    Args:
        **kwargs: Arguments passed to backend creation.

    Returns:
        ParallelBackend instance.
    """
    backend_type = detect_best_backend()
    return create_backend(backend_type, **kwargs)
