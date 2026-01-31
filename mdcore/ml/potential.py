"""ML potential implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..forcefields.base import ForceProvider
from .base import Descriptor, EquivariantModel, MLModel

if TYPE_CHECKING:
    from ..neighborlists import NeighborList
    from ..system import MDState


class MLPotential(ForceProvider):
    """
    Machine learning potential as a ForceProvider.

    Wraps an ML model and descriptor to provide forces and energies
    compatible with the MD engine.

    Supports two modes:
    1. Descriptor-based: Compute descriptors, then predict
    2. Equivariant: Direct position -> force mapping

    Example:
        descriptor = SymmetryFunctions(...)
        model = NeuralNetwork.load("model.pt")
        potential = MLPotential(model, descriptor)

        engine = MDEngine(state, integrator, potential)
    """

    def __init__(
        self,
        model: MLModel | EquivariantModel,
        descriptor: Descriptor | None = None,
        species_map: dict[str, int] | None = None,
    ) -> None:
        """
        Initialize ML potential.

        Args:
            model: Trained ML model.
            descriptor: Atomic descriptor (not needed for equivariant models).
            species_map: Mapping from element symbols to species indices.
        """
        self.model = model
        self.descriptor = descriptor
        self.species_map = species_map or {}

        self._is_equivariant = isinstance(model, EquivariantModel)

        if not self._is_equivariant and descriptor is None:
            raise ValueError("Descriptor required for non-equivariant models")

    def compute(
        self,
        state: MDState,
        neighbors: NeighborList | None = None,
    ) -> NDArray[np.floating]:
        """
        Compute ML forces.

        Args:
            state: Current MD state.
            neighbors: Optional neighbor list (may be used by descriptor).

        Returns:
            Forces array, shape (N, 3).
        """
        _, forces = self.compute_with_energy(state, neighbors)
        return forces

    def compute_with_energy(
        self,
        state: MDState,
        neighbors: NeighborList | None = None,
    ) -> tuple[NDArray[np.floating], float]:
        """
        Compute ML forces and energy.

        Args:
            state: Current MD state.
            neighbors: Optional neighbor list.

        Returns:
            Tuple of (forces, energy).
        """
        positions = state.positions
        species = self._get_species(state)
        cell = state.box.vectors if state.box else None

        if self._is_equivariant:
            # Equivariant model: direct prediction
            assert isinstance(self.model, EquivariantModel)
            energy, forces = self.model.forward(positions, species, cell)
        else:
            # Descriptor-based model
            assert self.descriptor is not None
            desc, desc_deriv = self.descriptor.compute_with_derivatives(
                positions, species, cell
            )
            energy, forces = self.model.predict(desc, desc_deriv, species)

        return forces, energy

    def _get_species(self, state: MDState) -> NDArray[np.integer]:
        """Get species indices from state."""
        # If state has species info, use it
        if hasattr(state, "species") and state.species is not None:
            return state.species

        # Otherwise assume single species
        return np.zeros(state.n_atoms, dtype=np.int64)


class DeltaLearningPotential(ForceProvider):
    """
    Delta-learning potential combining classical FF with ML correction.

    The ML model learns the difference between a reference (e.g., DFT)
    and a baseline classical force field:

        E_total = E_classical + E_ML_correction
        F_total = F_classical + F_ML_correction

    This often requires less training data and generalizes better
    than pure ML potentials.

    Example:
        baseline_ff = ForceField([LennardJones(), Coulomb()])
        ml_correction = MLPotential(model, descriptor)

        potential = DeltaLearningPotential(baseline_ff, ml_correction)
    """

    def __init__(
        self,
        baseline: ForceProvider,
        correction: ForceProvider,
        correction_weight: float = 1.0,
    ) -> None:
        """
        Initialize delta-learning potential.

        Args:
            baseline: Classical force field baseline.
            correction: ML correction term.
            correction_weight: Weight for ML correction (for blending).
        """
        self.baseline = baseline
        self.correction = correction
        self.correction_weight = correction_weight

    def compute(
        self,
        state: MDState,
        neighbors: NeighborList | None = None,
    ) -> NDArray[np.floating]:
        """Compute delta-learning forces."""
        f_baseline = self.baseline.compute(state, neighbors)
        f_correction = self.correction.compute(state, neighbors)

        return f_baseline + self.correction_weight * f_correction

    def compute_with_energy(
        self,
        state: MDState,
        neighbors: NeighborList | None = None,
    ) -> tuple[NDArray[np.floating], float]:
        """Compute delta-learning forces and energy."""
        f_baseline, e_baseline = self.baseline.compute_with_energy(state, neighbors)
        f_correction, e_correction = self.correction.compute_with_energy(
            state, neighbors
        )

        forces = f_baseline + self.correction_weight * f_correction
        energy = e_baseline + self.correction_weight * e_correction

        return forces, energy
