"""MD system state representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .box import Box


@dataclass
class MDState:
    """
    Single source of truth for MD system state.

    This is a pure data container with no logic. All arrays are NumPy arrays
    but can be created from JAX/PyTorch arrays via np.asarray().

    Attributes:
        positions: Atomic positions, shape (N, 3).
        velocities: Atomic velocities, shape (N, 3).
        forces: Atomic forces, shape (N, 3).
        masses: Atomic masses, shape (N,).
        box: Simulation box.
        time: Current simulation time.
        step: Current step number.
    """

    positions: NDArray[np.floating]
    velocities: NDArray[np.floating]
    forces: NDArray[np.floating]
    masses: NDArray[np.floating]
    box: Box
    time: float = 0.0
    step: int = 0

    def __post_init__(self) -> None:
        """Validate and convert arrays."""
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.velocities = np.asarray(self.velocities, dtype=np.float64)
        self.forces = np.asarray(self.forces, dtype=np.float64)
        self.masses = np.asarray(self.masses, dtype=np.float64)

        n_atoms = len(self.masses)
        if self.positions.shape != (n_atoms, 3):
            raise ValueError(
                f"positions shape {self.positions.shape} incompatible with "
                f"{n_atoms} atoms"
            )
        if self.velocities.shape != (n_atoms, 3):
            raise ValueError(
                f"velocities shape {self.velocities.shape} incompatible with "
                f"{n_atoms} atoms"
            )
        if self.forces.shape != (n_atoms, 3):
            raise ValueError(
                f"forces shape {self.forces.shape} incompatible with {n_atoms} atoms"
            )

    @property
    def n_atoms(self) -> int:
        """Return number of atoms."""
        return len(self.masses)

    @classmethod
    def create(
        cls,
        positions: ArrayLike,
        masses: ArrayLike,
        box: Box,
        velocities: ArrayLike | None = None,
        forces: ArrayLike | None = None,
        time: float = 0.0,
        step: int = 0,
    ) -> MDState:
        """
        Create an MDState with optional velocity/force initialization.

        Args:
            positions: Atomic positions, shape (N, 3).
            masses: Atomic masses, shape (N,).
            box: Simulation box.
            velocities: Atomic velocities, shape (N, 3). Defaults to zeros.
            forces: Atomic forces, shape (N, 3). Defaults to zeros.
            time: Current simulation time.
            step: Current step number.

        Returns:
            New MDState instance.
        """
        positions = np.asarray(positions, dtype=np.float64)
        masses = np.asarray(masses, dtype=np.float64)
        n_atoms = len(masses)

        if velocities is None:
            velocities = np.zeros((n_atoms, 3), dtype=np.float64)
        if forces is None:
            forces = np.zeros((n_atoms, 3), dtype=np.float64)

        return cls(
            positions=positions,
            velocities=velocities,
            forces=forces,
            masses=masses,
            box=box,
            time=time,
            step=step,
        )

    def copy(self) -> MDState:
        """Create a deep copy of this state."""
        return MDState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            forces=self.forces.copy(),
            masses=self.masses.copy(),
            box=self.box,  # Box is immutable
            time=self.time,
            step=self.step,
        )

    def freeze(self) -> FrozenMDState:
        """Create an immutable snapshot of this state."""
        return FrozenMDState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            forces=self.forces.copy(),
            masses=self.masses.copy(),
            box=self.box,
            time=self.time,
            step=self.step,
        )

    @property
    def kinetic_energy(self) -> float:
        """Compute total kinetic energy: sum(0.5 * m * v^2)."""
        return 0.5 * np.sum(self.masses[:, np.newaxis] * self.velocities**2)

    @property
    def temperature(self) -> float:
        """
        Compute instantaneous temperature from kinetic energy.

        Uses T = 2 * KE / (N_dof * k_B) where N_dof = 3*N - 3.
        Returns 0 if N <= 1.
        """
        if self.n_atoms <= 1:
            return 0.0
        n_dof = 3 * self.n_atoms - 3  # Remove center of mass motion
        # k_B in appropriate units (kJ/mol/K for typical MD)
        k_B = 8.314462618e-3  # kJ/mol/K
        return 2.0 * self.kinetic_energy / (n_dof * k_B)

    @property
    def center_of_mass(self) -> NDArray[np.floating]:
        """Compute center of mass position."""
        total_mass = np.sum(self.masses)
        return np.sum(self.masses[:, np.newaxis] * self.positions, axis=0) / total_mass

    @property
    def center_of_mass_velocity(self) -> NDArray[np.floating]:
        """Compute center of mass velocity."""
        total_mass = np.sum(self.masses)
        return np.sum(self.masses[:, np.newaxis] * self.velocities, axis=0) / total_mass


@dataclass(frozen=True)
class FrozenMDState:
    """
    Immutable snapshot of MD system state.

    Used for checkpointing and analysis where state should not be modified.
    """

    positions: NDArray[np.floating]
    velocities: NDArray[np.floating]
    forces: NDArray[np.floating]
    masses: NDArray[np.floating]
    box: Box
    time: float
    step: int

    def __post_init__(self) -> None:
        """Make arrays read-only."""
        self.positions.flags.writeable = False
        self.velocities.flags.writeable = False
        self.forces.flags.writeable = False
        self.masses.flags.writeable = False

    @property
    def n_atoms(self) -> int:
        """Return number of atoms."""
        return len(self.masses)

    def thaw(self) -> MDState:
        """Create a mutable copy of this frozen state."""
        return MDState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            forces=self.forces.copy(),
            masses=self.masses.copy(),
            box=self.box,
            time=self.time,
            step=self.step,
        )
