"""Uncertainty estimation for ML potentials."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..system import MDState
    from .base import MLModel


class UncertaintyEstimator(ABC):
    """
    Abstract base class for uncertainty estimation.

    Uncertainty estimation is critical for:
    - Active learning (identifying when to add training data)
    - Knowing when ML predictions are unreliable
    - Triggering fallback to reference calculations
    """

    @abstractmethod
    def estimate(
        self,
        state: MDState,
    ) -> float:
        """
        Estimate uncertainty for current state.

        Args:
            state: Current MD state.

        Returns:
            Scalar uncertainty value (higher = less confident).
        """
        ...

    @abstractmethod
    def estimate_per_atom(
        self,
        state: MDState,
    ) -> NDArray[np.floating]:
        """
        Estimate per-atom uncertainties.

        Args:
            state: Current MD state.

        Returns:
            Per-atom uncertainties, shape (N,).
        """
        ...


class EnsembleUncertainty(UncertaintyEstimator):
    """
    Uncertainty from ensemble of models.

    Uses the variance across an ensemble of independently trained
    models as an uncertainty measure.
    """

    def __init__(self, models: list[MLModel]) -> None:
        """
        Initialize ensemble uncertainty estimator.

        Args:
            models: List of trained models (should be diverse).
        """
        if len(models) < 2:
            raise ValueError("Ensemble requires at least 2 models")
        self.models = models

    def estimate(self, state: MDState) -> float:
        """Estimate uncertainty as energy variance across ensemble."""
        # This is a placeholder - actual implementation would compute
        # energies from each model and return variance
        per_atom = self.estimate_per_atom(state)
        return float(np.mean(per_atom))

    def estimate_per_atom(self, state: MDState) -> NDArray[np.floating]:
        """Estimate per-atom uncertainty as force variance."""
        # Placeholder - would compute forces from each model
        # and return per-atom standard deviation
        return np.zeros(state.n_atoms)


class DropoutUncertainty(UncertaintyEstimator):
    """
    Uncertainty from MC Dropout.

    Uses Monte Carlo dropout to estimate uncertainty without
    requiring an ensemble of models.
    """

    def __init__(
        self,
        model: MLModel,
        n_samples: int = 10,
    ) -> None:
        """
        Initialize dropout uncertainty estimator.

        Args:
            model: Model with dropout layers.
            n_samples: Number of forward passes for MC estimation.
        """
        self.model = model
        self.n_samples = n_samples

    def estimate(self, state: MDState) -> float:
        """Estimate uncertainty via MC dropout."""
        per_atom = self.estimate_per_atom(state)
        return float(np.mean(per_atom))

    def estimate_per_atom(self, state: MDState) -> NDArray[np.floating]:
        """Estimate per-atom uncertainty via MC dropout."""
        # Placeholder - would run n_samples forward passes with
        # dropout enabled and compute variance
        return np.zeros(state.n_atoms)


class DistanceBasedUncertainty(UncertaintyEstimator):
    """
    Uncertainty based on distance to training data.

    Simple approach: if the current configuration is far from
    any training configuration in descriptor space, it's uncertain.
    """

    def __init__(
        self,
        training_descriptors: NDArray[np.floating],
        threshold: float = 1.0,
    ) -> None:
        """
        Initialize distance-based uncertainty.

        Args:
            training_descriptors: Descriptors from training set.
            threshold: Distance threshold for "uncertain".
        """
        self.training_descriptors = training_descriptors
        self.threshold = threshold

    def estimate(self, state: MDState) -> float:
        """Estimate uncertainty as min distance to training data."""
        per_atom = self.estimate_per_atom(state)
        return float(np.max(per_atom))  # Most uncertain atom

    def estimate_per_atom(self, state: MDState) -> NDArray[np.floating]:
        """Estimate per-atom uncertainty as distance to training data."""
        # Placeholder - would compute descriptors for current state
        # and find minimum distance to training descriptors
        return np.zeros(state.n_atoms)


class ActiveLearningTrigger:
    """
    Trigger for active learning based on uncertainty.

    Monitors uncertainty during simulation and triggers
    retraining when threshold is exceeded.
    """

    def __init__(
        self,
        uncertainty_estimator: UncertaintyEstimator,
        threshold: float,
        cooldown_steps: int = 100,
    ) -> None:
        """
        Initialize active learning trigger.

        Args:
            uncertainty_estimator: Uncertainty estimation method.
            threshold: Uncertainty threshold for triggering.
            cooldown_steps: Minimum steps between triggers.
        """
        self.estimator = uncertainty_estimator
        self.threshold = threshold
        self.cooldown_steps = cooldown_steps

        self._steps_since_trigger = cooldown_steps
        self._triggered_states: list[MDState] = []

    def check(self, state: MDState) -> bool:
        """
        Check if active learning should be triggered.

        Args:
            state: Current MD state.

        Returns:
            True if retraining should be triggered.
        """
        self._steps_since_trigger += 1

        if self._steps_since_trigger < self.cooldown_steps:
            return False

        uncertainty = self.estimator.estimate(state)

        if uncertainty > self.threshold:
            self._triggered_states.append(state)
            self._steps_since_trigger = 0
            return True

        return False

    def get_triggered_states(self) -> list[MDState]:
        """Get states that triggered active learning."""
        return self._triggered_states

    def clear_triggered_states(self) -> None:
        """Clear the list of triggered states."""
        self._triggered_states = []
