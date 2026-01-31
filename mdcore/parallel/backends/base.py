"""Abstract base class for parallel backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


class ParallelBackend(ABC):
    """
    Abstract base class for parallelization backends.

    All parallel operations go through this interface, allowing
    transparent switching between serial, multiprocessing, MPI, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return backend name."""
        ...

    @property
    @abstractmethod
    def n_workers(self) -> int:
        """Return number of parallel workers."""
        ...

    @property
    @abstractmethod
    def rank(self) -> int:
        """Return rank of current process (0 for serial)."""
        ...

    @property
    def is_root(self) -> bool:
        """Check if this is the root process."""
        return self.rank == 0

    @abstractmethod
    def scatter(
        self,
        data: NDArray[np.floating] | None,
        root: int = 0,
    ) -> NDArray[np.floating]:
        """
        Scatter data from root to all workers.

        Args:
            data: Data to scatter (only needed on root).
            root: Rank of the root process.

        Returns:
            Local chunk of scattered data.
        """
        ...

    @abstractmethod
    def gather(
        self,
        local_data: NDArray[np.floating],
        root: int = 0,
    ) -> NDArray[np.floating] | None:
        """
        Gather data from all workers to root.

        Args:
            local_data: Local data chunk to gather.
            root: Rank of the root process.

        Returns:
            Gathered data on root, None on other ranks.
        """
        ...

    @abstractmethod
    def allgather(
        self,
        local_data: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Gather data from all workers to all workers.

        Args:
            local_data: Local data chunk.

        Returns:
            Gathered data on all ranks.
        """
        ...

    @abstractmethod
    def reduce_sum(
        self,
        local_data: NDArray[np.floating],
        root: int = 0,
    ) -> NDArray[np.floating] | None:
        """
        Sum-reduce data from all workers to root.

        Args:
            local_data: Local data to reduce.
            root: Rank of the root process.

        Returns:
            Reduced sum on root, None on other ranks.
        """
        ...

    @abstractmethod
    def allreduce_sum(
        self,
        local_data: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Sum-reduce data from all workers to all workers.

        Args:
            local_data: Local data to reduce.

        Returns:
            Reduced sum on all ranks.
        """
        ...

    @abstractmethod
    def broadcast(
        self,
        data: NDArray[np.floating] | None,
        root: int = 0,
    ) -> NDArray[np.floating]:
        """
        Broadcast data from root to all workers.

        Args:
            data: Data to broadcast (only needed on root).
            root: Rank of the root process.

        Returns:
            Broadcasted data on all ranks.
        """
        ...

    @abstractmethod
    def barrier(self) -> None:
        """Synchronize all workers."""
        ...

    def reduce_forces(
        self,
        local_forces: NDArray[np.floating],
        n_atoms: int,
    ) -> NDArray[np.floating]:
        """
        Reduce forces from all workers.

        This is the key operation for parallel force computation.
        Each worker computes forces on its subset of interactions,
        then forces are summed across all workers.

        Args:
            local_forces: Local force contributions, shape (n_atoms, 3).
            n_atoms: Total number of atoms.

        Returns:
            Total forces on all ranks, shape (n_atoms, 3).
        """
        return self.allreduce_sum(local_forces)

    def parallel_map(
        self,
        func: Callable[..., Any],
        items: list[Any],
    ) -> list[Any]:
        """
        Apply function to items in parallel.

        Default implementation is serial; backends can override.

        Args:
            func: Function to apply.
            items: Items to process.

        Returns:
            Results for each item.
        """
        return [func(item) for item in items]

    def partition_atoms(
        self,
        n_atoms: int,
    ) -> tuple[int, int]:
        """
        Get atom range for this worker.

        Args:
            n_atoms: Total number of atoms.

        Returns:
            Tuple of (start_index, end_index) for this worker.
        """
        atoms_per_worker = n_atoms // self.n_workers
        remainder = n_atoms % self.n_workers

        if self.rank < remainder:
            start = self.rank * (atoms_per_worker + 1)
            end = start + atoms_per_worker + 1
        else:
            start = self.rank * atoms_per_worker + remainder
            end = start + atoms_per_worker

        return start, end

    def partition_pairs(
        self,
        n_pairs: int,
    ) -> tuple[int, int]:
        """
        Get pair range for this worker.

        Args:
            n_pairs: Total number of pairs.

        Returns:
            Tuple of (start_index, end_index) for this worker.
        """
        pairs_per_worker = n_pairs // self.n_workers
        remainder = n_pairs % self.n_workers

        if self.rank < remainder:
            start = self.rank * (pairs_per_worker + 1)
            end = start + pairs_per_worker + 1
        else:
            start = self.rank * pairs_per_worker + remainder
            end = start + pairs_per_worker

        return start, end
