"""Multiprocessing backend using Python's multiprocessing module."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import ParallelBackend


class MultiprocessingBackend(ParallelBackend):
    """
    Multiprocessing backend for shared-memory parallelism.

    Uses Python's multiprocessing for process-based parallelism.
    Best for CPU-bound work on a single node.

    Note: This backend is limited - scatter/gather operations
    work differently than MPI. It's primarily useful for
    parallel_map style operations.
    """

    def __init__(self, n_workers: int | None = None) -> None:
        """
        Initialize multiprocessing backend.

        Args:
            n_workers: Number of worker processes. Defaults to CPU count.
        """
        self._n_workers = n_workers or mp.cpu_count()
        self._executor: ProcessPoolExecutor | None = None

    @property
    def name(self) -> str:
        """Return backend name."""
        return "multiprocessing"

    @property
    def n_workers(self) -> int:
        """Return number of parallel workers."""
        return self._n_workers

    @property
    def rank(self) -> int:
        """Return rank (always 0 for main process)."""
        return 0

    def scatter(
        self,
        data: NDArray[np.floating] | None,
        root: int = 0,
    ) -> NDArray[np.floating]:
        """Scatter not supported in multiprocessing; return data."""
        if data is None:
            raise ValueError("Data must be provided")
        return data

    def gather(
        self,
        local_data: NDArray[np.floating],
        root: int = 0,
    ) -> NDArray[np.floating]:
        """Gather not supported in multiprocessing; return data."""
        return local_data

    def allgather(
        self,
        local_data: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Allgather not supported in multiprocessing; return data."""
        return local_data

    def reduce_sum(
        self,
        local_data: NDArray[np.floating],
        root: int = 0,
    ) -> NDArray[np.floating]:
        """Reduce not directly supported; return data."""
        return local_data

    def allreduce_sum(
        self,
        local_data: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Allreduce not directly supported; return data."""
        return local_data

    def broadcast(
        self,
        data: NDArray[np.floating] | None,
        root: int = 0,
    ) -> NDArray[np.floating]:
        """Broadcast not supported in multiprocessing; return data."""
        if data is None:
            raise ValueError("Data must be provided")
        return data

    def barrier(self) -> None:
        """Barrier not applicable in multiprocessing."""
        pass

    def parallel_map(
        self,
        func: Callable[..., Any],
        items: list[Any],
    ) -> list[Any]:
        """
        Apply function to items in parallel using process pool.

        Args:
            func: Function to apply (must be picklable).
            items: Items to process.

        Returns:
            Results for each item.
        """
        if len(items) == 0:
            return []

        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            results = list(executor.map(func, items))

        return results

    def parallel_starmap(
        self,
        func: Callable[..., Any],
        args_list: list[tuple[Any, ...]],
    ) -> list[Any]:
        """
        Apply function to argument tuples in parallel.

        Args:
            func: Function to apply (must be picklable).
            args_list: List of argument tuples.

        Returns:
            Results for each argument tuple.
        """
        if len(args_list) == 0:
            return []

        # Use starmap-like behavior
        def wrapper(args: tuple[Any, ...]) -> Any:
            return func(*args)

        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            results = list(executor.map(wrapper, args_list))

        return results
