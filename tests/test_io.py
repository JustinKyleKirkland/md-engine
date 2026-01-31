"""Tests for I/O layer."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mdcore.io import (
    Checkpoint,
    CheckpointManager,
    PDBReader,
    PDBWriter,
    XYZReader,
    XYZWriter,
)
from mdcore.system import Box, MDState


@pytest.fixture
def simple_state():
    """Create a simple MD state for testing."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [1.5, 1.5, 0.0],
        ]
    )
    velocities = np.array(
        [
            [0.1, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, -0.1, 0.0],
        ]
    )
    forces = np.zeros((4, 3))
    masses = np.ones(4) * 12.0
    box = Box.cubic(10.0)

    return MDState(
        positions=positions,
        velocities=velocities,
        forces=forces,
        masses=masses,
        box=box,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestXYZFormat:
    """Tests for XYZ format I/O."""

    def test_xyz_writer_creation(self, temp_dir):
        """Test XYZ writer creation."""
        filepath = temp_dir / "test.xyz"
        writer = XYZWriter(filepath, elements=["C", "C", "C", "C"])

        assert writer.filename == filepath
        assert writer.elements == ["C", "C", "C", "C"]
        assert writer.n_frames == 0

    def test_xyz_write_single_frame(self, simple_state, temp_dir):
        """Test writing a single XYZ frame."""
        filepath = temp_dir / "single.xyz"

        with XYZWriter(filepath, elements=["C", "C", "C", "C"]) as writer:
            writer.write(simple_state)

        assert filepath.exists()
        assert writer.n_frames == 1

        # Read and verify content
        content = filepath.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "4"  # Number of atoms
        assert "C" in lines[2]  # First atom line

    def test_xyz_write_multiple_frames(self, simple_state, temp_dir):
        """Test writing multiple XYZ frames."""
        filepath = temp_dir / "multi.xyz"

        with XYZWriter(filepath, elements=["C", "C", "C", "C"]) as writer:
            for _ in range(5):
                writer.write(simple_state)

        assert writer.n_frames == 5

    def test_xyz_reader_single_frame(self, simple_state, temp_dir):
        """Test reading a single XYZ frame."""
        filepath = temp_dir / "read_single.xyz"

        # Write first
        with XYZWriter(filepath, elements=["C", "C", "C", "C"]) as writer:
            writer.write(simple_state)

        # Read back
        with XYZReader(filepath) as reader:
            assert len(reader) == 1
            frame = reader.read_frame(0)

            assert frame["n_atoms"] == 4
            assert len(frame["elements"]) == 4
            np.testing.assert_array_almost_equal(
                frame["positions"], simple_state.positions, decimal=6
            )

    def test_xyz_reader_multiple_frames(self, simple_state, temp_dir):
        """Test reading multiple XYZ frames."""
        filepath = temp_dir / "read_multi.xyz"

        # Write
        with XYZWriter(filepath, elements=["C", "C", "C", "C"]) as writer:
            for i in range(3):
                # Modify positions slightly
                state = MDState(
                    positions=simple_state.positions + i * 0.1,
                    velocities=simple_state.velocities,
                    forces=simple_state.forces,
                    masses=simple_state.masses,
                    box=simple_state.box,
                )
                writer.write(state)

        # Read
        with XYZReader(filepath) as reader:
            assert len(reader) == 3

            # Test iteration
            frames = list(reader)
            assert len(frames) == 3

            # Test indexing
            frame0 = reader[0]
            frame2 = reader[2]
            assert frame0["positions"][0, 0] < frame2["positions"][0, 0]

    def test_xyz_extended_format(self, simple_state, temp_dir):
        """Test extended XYZ with lattice info."""
        filepath = temp_dir / "extended.xyz"

        with XYZWriter(filepath, elements=["C", "C", "C", "C"]) as writer:
            writer.write(simple_state)

        with XYZReader(filepath) as reader:
            frame = reader.read_frame(0)

            # Should have box from extended XYZ
            assert "box" in frame
            np.testing.assert_array_almost_equal(
                frame["box"], simple_state.box.vectors, decimal=4
            )


class TestPDBFormat:
    """Tests for PDB format I/O."""

    def test_pdb_writer_creation(self, temp_dir):
        """Test PDB writer creation."""
        filepath = temp_dir / "test.pdb"
        writer = PDBWriter(filepath, elements=["C", "C", "C", "C"])

        assert writer.filename == filepath
        assert writer.n_frames == 0

    def test_pdb_write_single_frame(self, simple_state, temp_dir):
        """Test writing a single PDB frame."""
        filepath = temp_dir / "single.pdb"

        with PDBWriter(filepath, elements=["C", "C", "C", "C"]) as writer:
            writer.write(simple_state)
            writer.write_footer()

        assert filepath.exists()

        content = filepath.read_text()
        assert "ATOM" in content
        assert "MODEL" in content
        assert "CRYST1" in content  # Box info

    def test_pdb_reader_single_frame(self, simple_state, temp_dir):
        """Test reading a single PDB frame."""
        filepath = temp_dir / "read.pdb"

        with PDBWriter(filepath, elements=["C", "C", "C", "C"]) as writer:
            writer.write(simple_state, multiframe=False)

        with PDBReader(filepath) as reader:
            frame = reader.read_frame(0)

            assert frame["n_atoms"] == 4
            np.testing.assert_array_almost_equal(
                frame["positions"], simple_state.positions, decimal=2
            )


class TestCheckpoint:
    """Tests for checkpoint system."""

    def test_checkpoint_from_state(self, simple_state):
        """Test creating checkpoint from state."""
        checkpoint = Checkpoint.from_state(
            simple_state,
            step=100,
            metadata={"simulation": "test"},
        )

        assert checkpoint.step == 100
        assert checkpoint.time == simple_state.time
        np.testing.assert_array_equal(checkpoint.positions, simple_state.positions)
        np.testing.assert_array_equal(checkpoint.velocities, simple_state.velocities)
        assert checkpoint.metadata["simulation"] == "test"

    def test_checkpoint_to_state(self, simple_state):
        """Test converting checkpoint to state dict."""
        checkpoint = Checkpoint.from_state(simple_state, step=50)
        state_dict = checkpoint.to_state()

        assert "positions" in state_dict
        assert "velocities" in state_dict
        assert "forces" in state_dict
        assert state_dict["step"] == 50

    def test_checkpoint_rng_state(self, simple_state):
        """Test RNG state preservation."""
        rng = np.random.default_rng(12345)

        # Generate some random numbers
        _ = rng.random(10)

        checkpoint = Checkpoint.from_state(simple_state, step=0, rng=rng)

        # Restore RNG
        restored_rng = checkpoint.restore_rng()
        assert restored_rng is not None

        # Both should generate same sequence
        original_next = rng.random(5)
        restored_next = restored_rng.random(5)

        np.testing.assert_array_equal(original_next, restored_next)


class TestCheckpointManager:
    """Tests for checkpoint manager."""

    def test_manager_creation(self, temp_dir):
        """Test checkpoint manager creation."""
        manager = CheckpointManager(temp_dir)

        assert manager.directory == temp_dir
        assert manager.compress is True
        assert manager.max_backups == 3

    def test_save_and_load(self, simple_state, temp_dir):
        """Test saving and loading checkpoint."""
        manager = CheckpointManager(temp_dir)

        checkpoint = Checkpoint.from_state(simple_state, step=1000)
        saved_path = manager.save(checkpoint, "test.chk")

        assert saved_path.exists()

        loaded = manager.load("test.chk")

        assert loaded.step == 1000
        np.testing.assert_array_equal(loaded.positions, simple_state.positions)
        np.testing.assert_array_equal(loaded.velocities, simple_state.velocities)

    def test_checkpoint_integrity(self, simple_state, temp_dir):
        """Test that corrupted checkpoints are detected."""
        manager = CheckpointManager(temp_dir)

        checkpoint = Checkpoint.from_state(simple_state, step=0)
        manager.save(checkpoint, "corrupt.chk")

        # Corrupt the file
        filepath = temp_dir / "corrupt.chk"
        data = filepath.read_bytes()
        # Flip some bits in the data section (after header)
        corrupted = data[:50] + bytes([data[50] ^ 0xFF]) + data[51:]
        filepath.write_bytes(corrupted)

        # Corruption can cause various errors:
        # - ValueError for bad magic/checksum
        # - zlib.error/OSError for decompression failure
        with pytest.raises((ValueError, OSError, Exception)):
            manager.load("corrupt.chk")

    def test_backup_rotation(self, simple_state, temp_dir):
        """Test backup file rotation."""
        manager = CheckpointManager(temp_dir, max_backups=2)

        # Save multiple times
        for i in range(4):
            checkpoint = Checkpoint.from_state(simple_state, step=i)
            manager.save(checkpoint, "rotating.chk")

        # Check that backups exist
        assert (temp_dir / "rotating.chk").exists()
        assert (temp_dir / "rotating.chk.1").exists()
        # Max 2 backups, so .3 should not exist
        assert not (temp_dir / "rotating.chk.3").exists()

    def test_list_checkpoints(self, simple_state, temp_dir):
        """Test listing checkpoints."""
        manager = CheckpointManager(temp_dir)

        # Create several checkpoints
        for name in ["a.chk", "b.chk", "c.chk"]:
            checkpoint = Checkpoint.from_state(simple_state, step=0)
            manager.save(checkpoint, name, create_backup=False)

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3

    def test_uncompressed_checkpoint(self, simple_state, temp_dir):
        """Test uncompressed checkpoint."""
        manager = CheckpointManager(temp_dir, compress=False)

        checkpoint = Checkpoint.from_state(simple_state, step=0)
        manager.save(checkpoint, "uncompressed.chk")

        loaded = manager.load("uncompressed.chk")
        np.testing.assert_array_equal(loaded.positions, simple_state.positions)


class TestTrajectoryInterface:
    """Test trajectory reader/writer interface compliance."""

    def test_writer_is_context_manager(self, temp_dir):
        """Test writer context manager protocol."""
        filepath = temp_dir / "context.xyz"

        with XYZWriter(filepath) as writer:
            assert writer._file is not None

        assert writer._file is None

    def test_reader_is_context_manager(self, simple_state, temp_dir):
        """Test reader context manager protocol."""
        filepath = temp_dir / "context.xyz"

        # Create file first
        with XYZWriter(filepath, elements=["X"] * 4) as writer:
            writer.write(simple_state)

        with XYZReader(filepath) as reader:
            assert reader._file is not None

        assert reader._file is None

    def test_reader_is_iterable(self, simple_state, temp_dir):
        """Test reader iteration."""
        filepath = temp_dir / "iter.xyz"

        with XYZWriter(filepath, elements=["X"] * 4) as writer:
            for _ in range(5):
                writer.write(simple_state)

        with XYZReader(filepath) as reader:
            count = 0
            for frame in reader:
                count += 1
                assert "positions" in frame

            assert count == 5
