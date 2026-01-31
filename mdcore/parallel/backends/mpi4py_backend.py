"""MPI backend using mpi4py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .base import ParallelBackend

if TYPE_CHECKING:
    from mpi4py import MPI as MPI_TYPE


class MPI4PyBackend(ParallelBackend):
    """
    MPI backend using mpi4py for distributed-memory parallelism.

    This backend requires mpi4py to be installed and the program
    to be launched with mpirun/mpiexec.

    Example:
        mpirun -n 4 python simulation.py
    """

    def __init__(self) -> None:
        """Initialize MPI backend."""
        try:
            from mpi4py import MPI
        except ImportError as e:
            raise ImportError(
                "mpi4py is required for MPI backend. Install with: pip install mpi4py"
            ) from e

        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._size = self._comm.Get_size()

    @property
    def name(self) -> str:
        """Return backend name."""
        return "mpi4py"

    @property
    def n_workers(self) -> int:
        """Return number of MPI processes."""
        return self._size

    @property
    def rank(self) -> int:
        """Return MPI rank of current process."""
        return self._rank

    @property
    def comm(self) -> MPI_TYPE.Comm:
        """Return MPI communicator."""
        return self._comm

    def scatter(
        self,
        data: NDArray[np.floating] | None,
        root: int = 0,
    ) -> NDArray[np.floating]:
        """
        Scatter data from root to all ranks.

        Args:
            data: Data to scatter (only needed on root).
                  Must be evenly divisible by n_workers.
            root: Rank of the root process.

        Returns:
            Local chunk of scattered data.
        """
        if self._rank == root:
            if data is None:
                raise ValueError("Root must provide data")
            # Split data for scattering
            chunks = np.array_split(data, self._size)
        else:
            chunks = None

        local_data = self._comm.scatter(chunks, root=root)
        return np.asarray(local_data)

    def gather(
        self,
        local_data: NDArray[np.floating],
        root: int = 0,
    ) -> NDArray[np.floating] | None:
        """
        Gather data from all ranks to root.

        Args:
            local_data: Local data chunk.
            root: Rank of the root process.

        Returns:
            Concatenated data on root, None elsewhere.
        """
        gathered = self._comm.gather(local_data, root=root)

        if self._rank == root:
            return np.concatenate(gathered)
        return None

    def allgather(
        self,
        local_data: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Gather data from all ranks to all ranks.

        Args:
            local_data: Local data chunk.

        Returns:
            Concatenated data on all ranks.
        """
        gathered = self._comm.allgather(local_data)
        return np.concatenate(gathered)

    def reduce_sum(
        self,
        local_data: NDArray[np.floating],
        root: int = 0,
    ) -> NDArray[np.floating] | None:
        """
        Sum-reduce data to root.

        Args:
            local_data: Local data to reduce.
            root: Rank of the root process.

        Returns:
            Reduced sum on root, None elsewhere.
        """
        result = np.zeros_like(local_data)
        self._comm.Reduce(local_data, result, op=self._MPI.SUM, root=root)

        if self._rank == root:
            return result
        return None

    def allreduce_sum(
        self,
        local_data: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Sum-reduce data to all ranks.

        Args:
            local_data: Local data to reduce.

        Returns:
            Reduced sum on all ranks.
        """
        result = np.zeros_like(local_data)
        self._comm.Allreduce(local_data, result, op=self._MPI.SUM)
        return result

    def broadcast(
        self,
        data: NDArray[np.floating] | None,
        root: int = 0,
    ) -> NDArray[np.floating]:
        """
        Broadcast data from root to all ranks.

        Args:
            data: Data to broadcast (only needed on root).
            root: Rank of the root process.

        Returns:
            Broadcasted data on all ranks.
        """
        return self._comm.bcast(data, root=root)

    def barrier(self) -> None:
        """Synchronize all MPI processes."""
        self._comm.Barrier()

    def sendrecv(
        self,
        sendbuf: NDArray[np.floating],
        dest: int,
        source: int,
    ) -> NDArray[np.floating]:
        """
        Send and receive data simultaneously.

        Useful for halo exchange in domain decomposition.

        Args:
            sendbuf: Data to send.
            dest: Destination rank.
            source: Source rank.

        Returns:
            Received data.
        """
        recvbuf = np.empty_like(sendbuf)
        self._comm.Sendrecv(sendbuf, dest, recvbuf=recvbuf, source=source)
        return recvbuf
