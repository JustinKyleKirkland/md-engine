"""Simulation box representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Box:
    """
    Simulation box representation.

    Supports orthorhombic and triclinic boxes via a 3x3 matrix representation.
    For orthorhombic boxes, the matrix is diagonal with box lengths on the diagonal.

    Attributes:
        vectors: 3x3 array where rows are box vectors [a, b, c].
    """

    vectors: NDArray[np.floating]

    def __post_init__(self) -> None:
        """Validate and convert vectors to proper shape."""
        vectors = np.asarray(self.vectors, dtype=np.float64)
        if vectors.shape == (3,):
            # Orthorhombic box specified by lengths
            vectors = np.diag(vectors)
        if vectors.shape != (3, 3):
            raise ValueError(f"Box vectors must be (3,) or (3, 3), got {vectors.shape}")
        # Use object.__setattr__ since dataclass is frozen
        object.__setattr__(self, "vectors", vectors)

    @classmethod
    def orthorhombic(cls, lx: float, ly: float, lz: float) -> Box:
        """Create an orthorhombic box with given side lengths."""
        return cls(np.array([lx, ly, lz]))

    @classmethod
    def cubic(cls, length: float) -> Box:
        """Create a cubic box with given side length."""
        return cls.orthorhombic(length, length, length)

    @classmethod
    def triclinic(cls, vectors: ArrayLike) -> Box:
        """Create a triclinic box from 3x3 matrix of box vectors."""
        return cls(np.asarray(vectors))

    @property
    def lengths(self) -> NDArray[np.floating]:
        """Return box vector lengths [|a|, |b|, |c|]."""
        return np.linalg.norm(self.vectors, axis=1)

    @property
    def volume(self) -> float:
        """Return box volume."""
        return float(np.abs(np.linalg.det(self.vectors)))

    @property
    def is_orthorhombic(self) -> bool:
        """Check if box is orthorhombic (diagonal matrix)."""
        off_diag = self.vectors.copy()
        np.fill_diagonal(off_diag, 0)
        return np.allclose(off_diag, 0)

    def wrap_positions(self, positions: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Wrap positions into the primary box using periodic boundary conditions.

        Args:
            positions: Positions array of shape (N, 3).

        Returns:
            Wrapped positions array of shape (N, 3).
        """
        positions = np.asarray(positions)
        if self.is_orthorhombic:
            # Fast path for orthorhombic boxes
            lengths = np.diag(self.vectors)
            return positions - lengths * np.floor(positions / lengths)
        else:
            # General triclinic case: convert to fractional, wrap, convert back
            inv_vectors = np.linalg.inv(self.vectors)
            fractional = positions @ inv_vectors.T
            fractional = fractional - np.floor(fractional)
            return fractional @ self.vectors

    def minimum_image(
        self, r1: NDArray[np.floating], r2: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Compute minimum image displacement vector r2 - r1.

        Args:
            r1: First position(s), shape (3,) or (N, 3).
            r2: Second position(s), shape (3,) or (N, 3).

        Returns:
            Displacement vector(s) under minimum image convention.
        """
        dr = np.asarray(r2) - np.asarray(r1)
        if self.is_orthorhombic:
            lengths = np.diag(self.vectors)
            return dr - lengths * np.round(dr / lengths)
        else:
            inv_vectors = np.linalg.inv(self.vectors)
            fractional = dr @ inv_vectors.T
            fractional = fractional - np.round(fractional)
            return fractional @ self.vectors

    def minimum_image_distance(
        self, r1: NDArray[np.floating], r2: NDArray[np.floating]
    ) -> float | NDArray[np.floating]:
        """
        Compute minimum image distance between positions.

        Args:
            r1: First position(s), shape (3,) or (N, 3).
            r2: Second position(s), shape (3,) or (N, 3).

        Returns:
            Distance(s) under minimum image convention.
        """
        dr = self.minimum_image(r1, r2)
        return np.linalg.norm(dr, axis=-1)
