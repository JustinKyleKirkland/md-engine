"""Base classes for trajectory I/O."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..system import MDState


class TrajectoryWriter(ABC):
    """
    Abstract base class for trajectory writers.

    Trajectory writers serialize simulation states to various file formats.
    They support both single-frame and streaming (append) modes.

    Example:
        with XYZWriter("trajectory.xyz") as writer:
            for step in simulation:
                writer.write(state)
    """

    def __init__(self, filename: str | Path) -> None:
        """
        Initialize trajectory writer.

        Args:
            filename: Output file path.
        """
        self.filename = Path(filename)
        self._file = None
        self._n_frames = 0

    @abstractmethod
    def write(self, state: MDState, **kwargs) -> None:
        """
        Write a single frame.

        Args:
            state: MD state to write.
            **kwargs: Format-specific options.
        """
        ...

    def write_header(self, **kwargs) -> None:
        """Write file header (optional, format-dependent)."""
        pass

    def write_footer(self, **kwargs) -> None:
        """Write file footer (optional, format-dependent)."""
        pass

    def open(self) -> None:
        """Open file for writing."""
        self._file = self.filename.open("w")

    def close(self) -> None:
        """Close file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> TrajectoryWriter:
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    @property
    def n_frames(self) -> int:
        """Number of frames written."""
        return self._n_frames


class TrajectoryReader(ABC):
    """
    Abstract base class for trajectory readers.

    Trajectory readers deserialize simulation states from various file formats.
    They support random access (where possible) and iteration.

    Example:
        with XYZReader("trajectory.xyz") as reader:
            for frame in reader:
                analyze(frame)
    """

    def __init__(self, filename: str | Path) -> None:
        """
        Initialize trajectory reader.

        Args:
            filename: Input file path.
        """
        self.filename = Path(filename)
        self._file = None
        self._n_frames: int | None = None

    @abstractmethod
    def read_frame(self, index: int) -> dict:
        """
        Read a specific frame.

        Args:
            index: Frame index (0-based).

        Returns:
            Dictionary with frame data (positions, velocities, box, etc.).
        """
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[dict]:
        """Iterate over all frames."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return number of frames."""
        ...

    def read_header(self) -> dict:
        """Read file header (optional, format-dependent)."""
        return {}

    def open(self) -> None:
        """Open file for reading."""
        self._file = self.filename.open()

    def close(self) -> None:
        """Close file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> TrajectoryReader:
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    @property
    def n_frames(self) -> int:
        """Number of frames in trajectory."""
        if self._n_frames is None:
            self._n_frames = len(self)
        return self._n_frames


class BinaryTrajectoryWriter(TrajectoryWriter):
    """Base class for binary trajectory writers."""

    def open(self) -> None:
        """Open file for binary writing."""
        self._file = self.filename.open("wb")


class BinaryTrajectoryReader(TrajectoryReader):
    """Base class for binary trajectory readers."""

    def open(self) -> None:
        """Open file for binary reading."""
        self._file = self.filename.open("rb")
