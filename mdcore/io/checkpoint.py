"""Checkpointing system for deterministic restart."""

from __future__ import annotations

import gzip
import hashlib
import pickle
import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..system import MDState


# Checkpoint format version for compatibility checking
CHECKPOINT_VERSION = 1
CHECKPOINT_MAGIC = b"MDCK"  # Magic bytes for file identification


@dataclass
class Checkpoint:
    """
    Checkpoint data container.

    Stores all information needed for deterministic restart:
    - Full simulation state (positions, velocities, forces)
    - Integrator internal state
    - RNG state for reproducibility
    - Simulation metadata

    Attributes:
        version: Checkpoint format version.
        timestamp: When checkpoint was created.
        step: Simulation step number.
        time: Simulation time.
        positions: Atomic positions.
        velocities: Atomic velocities.
        forces: Atomic forces.
        masses: Atomic masses.
        box_vectors: Box vectors (3x3).
        integrator_state: Integrator-specific state dict.
        rng_state: NumPy RNG state for reproducibility.
        metadata: User-defined metadata.
    """

    version: int
    timestamp: str
    step: int
    time: float
    positions: NDArray[np.floating]
    velocities: NDArray[np.floating] | None
    forces: NDArray[np.floating]
    masses: NDArray[np.floating]
    box_vectors: NDArray[np.floating] | None
    integrator_state: dict[str, Any] = field(default_factory=dict)
    rng_state: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_state(
        cls,
        state: MDState,
        step: int = 0,
        integrator_state: dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """
        Create checkpoint from current simulation state.

        Args:
            state: Current MD state.
            step: Current step number.
            integrator_state: Integrator internal state.
            rng: NumPy random generator for reproducibility.
            metadata: Additional metadata to store.

        Returns:
            New Checkpoint instance.
        """
        # Capture RNG state if provided
        rng_state = None
        if rng is not None:
            bit_gen = rng.bit_generator
            rng_state = {
                "bit_generator": type(bit_gen).__name__,
                "state": bit_gen.state,
            }

        return cls(
            version=CHECKPOINT_VERSION,
            timestamp=datetime.now().isoformat(),
            step=step,
            time=state.time,
            positions=state.positions.copy(),
            velocities=state.velocities.copy()
            if state.velocities is not None
            else None,
            forces=state.forces.copy(),
            masses=state.masses.copy(),
            box_vectors=state.box.vectors.copy() if state.box is not None else None,
            integrator_state=integrator_state or {},
            rng_state=rng_state,
            metadata=metadata or {},
        )

    def to_state(self) -> dict[str, Any]:
        """
        Convert checkpoint to state dictionary.

        Returns:
            Dictionary suitable for creating MDState.
        """
        return {
            "positions": self.positions,
            "velocities": self.velocities,
            "forces": self.forces,
            "masses": self.masses,
            "box_vectors": self.box_vectors,
            "time": self.time,
            "step": self.step,
        }

    def restore_rng(self) -> np.random.Generator | None:
        """
        Restore RNG state from checkpoint.

        Returns:
            Restored NumPy Generator or None if no RNG state.
        """
        if self.rng_state is None:
            return None

        # Create generator with same bit generator type
        bit_gen_name = self.rng_state["bit_generator"]
        if bit_gen_name == "PCG64":
            bit_gen = np.random.PCG64()
        elif bit_gen_name == "MT19937":
            bit_gen = np.random.MT19937()
        elif bit_gen_name == "Philox":
            bit_gen = np.random.Philox()
        elif bit_gen_name == "SFC64":
            bit_gen = np.random.SFC64()
        else:
            # Default to PCG64
            bit_gen = np.random.PCG64()

        bit_gen.state = self.rng_state["state"]
        return np.random.Generator(bit_gen)


class CheckpointManager:
    """
    Manager for checkpoint I/O operations.

    Handles:
    - Binary serialization with compression
    - Version checking for compatibility
    - Automatic backup rotation
    - Integrity verification (checksums)

    Example:
        manager = CheckpointManager("checkpoints/")

        # Save checkpoint
        checkpoint = Checkpoint.from_state(state, step=1000)
        manager.save(checkpoint, "sim.chk")

        # Load checkpoint
        checkpoint = manager.load("sim.chk")
        state_dict = checkpoint.to_state()
    """

    def __init__(
        self,
        directory: str | Path = ".",
        compress: bool = True,
        max_backups: int = 3,
    ) -> None:
        """
        Initialize checkpoint manager.

        Args:
            directory: Directory for checkpoint files.
            compress: Whether to gzip compress checkpoints.
            max_backups: Maximum number of backup files to keep.
        """
        self.directory = Path(directory)
        self.compress = compress
        self.max_backups = max_backups

        # Create directory if needed
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        checkpoint: Checkpoint,
        filename: str = "checkpoint.chk",
        create_backup: bool = True,
    ) -> Path:
        """
        Save checkpoint to file.

        Args:
            checkpoint: Checkpoint to save.
            filename: Output filename.
            create_backup: Whether to backup existing file.

        Returns:
            Path to saved checkpoint.
        """
        filepath = self.directory / filename

        # Rotate backups if file exists
        if create_backup and filepath.exists():
            self._rotate_backups(filepath)

        # Serialize checkpoint data
        data = self._serialize(checkpoint)

        # Compute checksum
        checksum = hashlib.sha256(data).digest()

        # Write file
        open_func = gzip.open if self.compress else open
        mode = "wb"

        with open_func(filepath, mode) as f:
            # Write header: magic + version + checksum
            f.write(CHECKPOINT_MAGIC)
            f.write(struct.pack("<I", CHECKPOINT_VERSION))
            f.write(checksum)
            # Write data
            f.write(data)

        return filepath

    def load(self, filename: str = "checkpoint.chk") -> Checkpoint:
        """
        Load checkpoint from file.

        Args:
            filename: Input filename.

        Returns:
            Loaded Checkpoint.

        Raises:
            ValueError: If file is invalid or corrupted.
            FileNotFoundError: If file doesn't exist.
        """
        filepath = self.directory / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        # Try compressed first, then uncompressed
        try:
            open_func = gzip.open if self.compress else open
            with open_func(filepath, "rb") as f:
                content = f.read()
        except gzip.BadGzipFile:
            with open(filepath, "rb") as f:
                content = f.read()

        # Parse header
        if len(content) < 40:
            raise ValueError("Invalid checkpoint file (too small)")

        magic = content[:4]
        if magic != CHECKPOINT_MAGIC:
            raise ValueError("Invalid checkpoint file (bad magic)")

        version = struct.unpack("<I", content[4:8])[0]
        stored_checksum = content[8:40]
        data = content[40:]

        # Verify checksum
        computed_checksum = hashlib.sha256(data).digest()
        if computed_checksum != stored_checksum:
            raise ValueError("Checkpoint file corrupted (checksum mismatch)")

        # Check version compatibility
        if version > CHECKPOINT_VERSION:
            raise ValueError(
                f"Checkpoint version {version} not supported "
                f"(max supported: {CHECKPOINT_VERSION})"
            )

        # Deserialize
        checkpoint = self._deserialize(data, version)

        return checkpoint

    def list_checkpoints(self, pattern: str = "*.chk") -> list[Path]:
        """
        List available checkpoints.

        Args:
            pattern: Glob pattern for checkpoint files.

        Returns:
            List of checkpoint paths, sorted by modification time.
        """
        checkpoints = list(self.directory.glob(pattern))
        return sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)

    def _serialize(self, checkpoint: Checkpoint) -> bytes:
        """Serialize checkpoint to bytes."""
        # Convert numpy arrays to lists for JSON serialization
        # Use pickle for efficient numpy array storage
        data = {
            "version": checkpoint.version,
            "timestamp": checkpoint.timestamp,
            "step": checkpoint.step,
            "time": checkpoint.time,
            "positions": checkpoint.positions,
            "velocities": checkpoint.velocities,
            "forces": checkpoint.forces,
            "masses": checkpoint.masses,
            "box_vectors": checkpoint.box_vectors,
            "integrator_state": checkpoint.integrator_state,
            "rng_state": checkpoint.rng_state,
            "metadata": checkpoint.metadata,
        }

        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize(self, data: bytes, version: int) -> Checkpoint:
        """Deserialize checkpoint from bytes."""
        loaded = pickle.loads(data)

        return Checkpoint(
            version=loaded["version"],
            timestamp=loaded["timestamp"],
            step=loaded["step"],
            time=loaded["time"],
            positions=np.asarray(loaded["positions"]),
            velocities=(
                np.asarray(loaded["velocities"])
                if loaded["velocities"] is not None
                else None
            ),
            forces=np.asarray(loaded["forces"]),
            masses=np.asarray(loaded["masses"]),
            box_vectors=(
                np.asarray(loaded["box_vectors"])
                if loaded["box_vectors"] is not None
                else None
            ),
            integrator_state=loaded.get("integrator_state", {}),
            rng_state=loaded.get("rng_state"),
            metadata=loaded.get("metadata", {}),
        )

    def _rotate_backups(self, filepath: Path) -> None:
        """Rotate backup files."""
        # Remove oldest backup if at limit
        for i in range(self.max_backups - 1, 0, -1):
            old_backup = filepath.with_suffix(f".chk.{i}")
            new_backup = filepath.with_suffix(f".chk.{i + 1}")

            if old_backup.exists():
                if i == self.max_backups - 1:
                    old_backup.unlink()
                else:
                    old_backup.rename(new_backup)

        # Rename current to .1
        if filepath.exists():
            filepath.rename(filepath.with_suffix(".chk.1"))
