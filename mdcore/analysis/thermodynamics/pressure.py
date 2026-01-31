"""Pressure tensor analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..base import StreamingAnalyzer

if TYPE_CHECKING:
    from ...system import MDState


class PressureTensor(StreamingAnalyzer):
    """
    Pressure tensor and scalar pressure analyzer.

    Computes the pressure tensor from the virial:
        P_αβ = (1/V) * [sum_i m_i v_iα v_iβ + sum_i r_iα F_iβ]

    where the first term is the kinetic contribution and the
    second is the virial (potential) contribution.

    The scalar pressure is: P = (1/3) * Tr(P_tensor)
    """

    def __init__(self) -> None:
        """Initialize pressure analyzer."""
        self.reset()

    @property
    def name(self) -> str:
        """Analyzer name."""
        return "pressure"

    def reset(self) -> None:
        """Reset statistics."""
        self._n_frames = 0
        self._pressure_tensor: list[NDArray[np.floating]] = []
        self._pressure_scalar: list[float] = []
        self._times: list[float] = []

    def update(self, state: MDState, **kwargs: Any) -> None:
        """
        Update pressure statistics.

        Args:
            state: Current simulation state.
            **kwargs: Should include 'forces' for virial calculation.
        """
        forces = kwargs.get("forces")
        if forces is None:
            # Can't compute virial without forces
            return

        volume = state.box.volume if state.box else 1.0
        positions = state.positions
        velocities = state.velocities
        masses = state.masses

        # Kinetic contribution: sum_i m_i * v_i ⊗ v_i
        if velocities is not None:
            kinetic_tensor = np.zeros((3, 3))
            for i in range(state.n_atoms):
                v = velocities[i]
                kinetic_tensor += masses[i] * np.outer(v, v)
        else:
            kinetic_tensor = np.zeros((3, 3))

        # Virial contribution: sum_i r_i ⊗ F_i
        virial_tensor = np.zeros((3, 3))
        for i in range(state.n_atoms):
            virial_tensor += np.outer(positions[i], forces[i])

        # Total pressure tensor
        pressure_tensor = (kinetic_tensor + virial_tensor) / volume

        # Scalar pressure
        pressure_scalar = np.trace(pressure_tensor) / 3.0

        # Store
        time = kwargs.get("time", float(self._n_frames))
        self._pressure_tensor.append(pressure_tensor)
        self._pressure_scalar.append(pressure_scalar)
        self._times.append(time)

        self._n_frames += 1

    def result(self) -> dict[str, Any]:
        """
        Get pressure statistics.

        Returns:
            Dictionary with pressure tensors and statistics.
        """
        if self._n_frames == 0:
            return {
                "n_frames": 0,
                "pressure_tensor": np.zeros((3, 3)),
                "pressure": 0.0,
            }

        tensor_array = np.array(self._pressure_tensor)
        scalar_array = np.array(self._pressure_scalar)
        times = np.array(self._times)

        results = {
            "time": times,
            "pressure_tensor_history": tensor_array,
            "pressure_history": scalar_array,
            "n_frames": self._n_frames,
            # Averages
            "pressure_tensor": np.mean(tensor_array, axis=0),
            "pressure": float(np.mean(scalar_array)),
            "pressure_std": float(np.std(scalar_array)),
        }

        # Stress tensor components
        avg_tensor = results["pressure_tensor"]
        results["Pxx"] = float(avg_tensor[0, 0])
        results["Pyy"] = float(avg_tensor[1, 1])
        results["Pzz"] = float(avg_tensor[2, 2])
        results["Pxy"] = float(avg_tensor[0, 1])
        results["Pxz"] = float(avg_tensor[0, 2])
        results["Pyz"] = float(avg_tensor[1, 2])

        return results

    @property
    def pressure(self) -> float:
        """Average scalar pressure."""
        if self._n_frames == 0:
            return 0.0
        return float(np.mean(self._pressure_scalar))

    @property
    def pressure_tensor(self) -> NDArray[np.floating]:
        """Average pressure tensor."""
        if self._n_frames == 0:
            return np.zeros((3, 3))
        return np.mean(self._pressure_tensor, axis=0)
