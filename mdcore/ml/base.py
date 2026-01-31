"""Base classes for ML components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


class Descriptor(ABC):
    """
    Abstract base class for atomic environment descriptors.

    Descriptors transform atomic configurations into feature vectors
    suitable for ML models. Examples include:
    - Behler-Parrinello symmetry functions
    - SOAP (Smooth Overlap of Atomic Positions)
    - ACE (Atomic Cluster Expansion)
    """

    @abstractmethod
    def compute(
        self,
        positions: NDArray[np.floating],
        species: NDArray[np.integer],
        cell: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """
        Compute descriptors for all atoms.

        Args:
            positions: Atomic positions, shape (N, 3).
            species: Atomic species indices, shape (N,).
            cell: Unit cell vectors, shape (3, 3), or None for non-periodic.

        Returns:
            Descriptor array, shape (N, n_features).
        """
        ...

    @abstractmethod
    def compute_with_derivatives(
        self,
        positions: NDArray[np.floating],
        species: NDArray[np.integer],
        cell: NDArray[np.floating] | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute descriptors and their derivatives.

        Args:
            positions: Atomic positions, shape (N, 3).
            species: Atomic species indices, shape (N,).
            cell: Unit cell vectors, shape (3, 3), or None.

        Returns:
            Tuple of:
            - Descriptors, shape (N, n_features)
            - Derivatives, shape (N, n_features, N, 3) or sparse equivalent
        """
        ...

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of descriptor features per atom."""
        ...


class MLModel(ABC):
    """
    Abstract base class for ML energy/force models.

    Models take descriptors (or raw positions) and predict
    energies and forces.
    """

    @abstractmethod
    def predict_energy(
        self,
        descriptors: NDArray[np.floating],
        species: NDArray[np.integer],
    ) -> float:
        """
        Predict total energy from descriptors.

        Args:
            descriptors: Atomic descriptors, shape (N, n_features).
            species: Atomic species indices, shape (N,).

        Returns:
            Predicted total energy.
        """
        ...

    @abstractmethod
    def predict_forces(
        self,
        descriptors: NDArray[np.floating],
        descriptor_derivatives: NDArray[np.floating],
        species: NDArray[np.integer],
    ) -> NDArray[np.floating]:
        """
        Predict forces from descriptors and their derivatives.

        Args:
            descriptors: Atomic descriptors, shape (N, n_features).
            descriptor_derivatives: Descriptor derivatives.
            species: Atomic species indices, shape (N,).

        Returns:
            Predicted forces, shape (N, 3).
        """
        ...

    @abstractmethod
    def predict(
        self,
        descriptors: NDArray[np.floating],
        descriptor_derivatives: NDArray[np.floating],
        species: NDArray[np.integer],
    ) -> tuple[float, NDArray[np.floating]]:
        """
        Predict energy and forces together.

        Args:
            descriptors: Atomic descriptors, shape (N, n_features).
            descriptor_derivatives: Descriptor derivatives.
            species: Atomic species indices, shape (N,).

        Returns:
            Tuple of (energy, forces).
        """
        ...

    def save(self, path: str) -> None:
        """Save model to file."""
        raise NotImplementedError("Model saving not implemented")

    @classmethod
    def load(cls, path: str) -> MLModel:
        """Load model from file."""
        raise NotImplementedError("Model loading not implemented")


class EquivariantModel(MLModel):
    """
    Base class for equivariant neural network models.

    Equivariant models (like NequIP, MACE) directly operate on
    positions and produce forces without explicit descriptor derivatives,
    using equivariant message passing.
    """

    @abstractmethod
    def forward(
        self,
        positions: NDArray[np.floating],
        species: NDArray[np.integer],
        cell: NDArray[np.floating] | None = None,
    ) -> tuple[float, NDArray[np.floating]]:
        """
        Forward pass computing energy and forces.

        Args:
            positions: Atomic positions, shape (N, 3).
            species: Atomic species indices, shape (N,).
            cell: Unit cell vectors or None.

        Returns:
            Tuple of (energy, forces).
        """
        ...

    def predict_energy(
        self,
        descriptors: NDArray[np.floating],
        species: NDArray[np.integer],
    ) -> float:
        """Not used for equivariant models."""
        raise NotImplementedError("Use forward() for equivariant models")

    def predict_forces(
        self,
        descriptors: NDArray[np.floating],
        descriptor_derivatives: NDArray[np.floating],
        species: NDArray[np.integer],
    ) -> NDArray[np.floating]:
        """Not used for equivariant models."""
        raise NotImplementedError("Use forward() for equivariant models")

    def predict(
        self,
        descriptors: NDArray[np.floating],
        descriptor_derivatives: NDArray[np.floating],
        species: NDArray[np.integer],
    ) -> tuple[float, NDArray[np.floating]]:
        """Not used for equivariant models."""
        raise NotImplementedError("Use forward() for equivariant models")
