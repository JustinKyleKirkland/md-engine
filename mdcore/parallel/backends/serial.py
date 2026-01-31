"""Serial (single-process) backend."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import ParallelBackend


class SerialBackend(ParallelBackend):
    """
    Serial backend for single-process execution.

    This is the default backend and provides a reference implementation.
    All parallel operations are no-ops or simple pass-throughs.
    """

    @property
    def name(self) -> str:
        """Return backend name."""
        return "serial"

    @property
    def n_workers(self) -> int:
        """Return number of parallel workers."""
        return 1

    @property
    def rank(self) -> int:
        """Return rank of current process."""
        return 0

    def scatter(
        self,
        data: NDArray[np.floating] | None,
        root: int = 0,
    ) -> NDArray[np.floating]:
        """Scatter is a no-op in serial; just return data."""
        if data is None:
            raise ValueError("Data must be provided in serial mode")
        return data

    def gather(
        self,
        local_data: NDArray[np.floating],
        root: int = 0,
    ) -> NDArray[np.floating]:
        """Gather is a no-op in serial; just return local data."""
        return local_data

    def allgather(
        self,
        local_data: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Allgather is a no-op in serial; just return local data."""
        return local_data

    def reduce_sum(
        self,
        local_data: NDArray[np.floating],
        root: int = 0,
    ) -> NDArray[np.floating]:
        """Reduce is a no-op in serial; just return local data."""
        return local_data

    def allreduce_sum(
        self,
        local_data: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Allreduce is a no-op in serial; just return local data."""
        return local_data

    def broadcast(
        self,
        data: NDArray[np.floating] | None,
        root: int = 0,
    ) -> NDArray[np.floating]:
        """Broadcast is a no-op in serial; just return data."""
        if data is None:
            raise ValueError("Data must be provided in serial mode")
        return data

    def barrier(self) -> None:
        """Barrier is a no-op in serial."""
        pass
