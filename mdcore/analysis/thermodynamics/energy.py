"""Energy analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..base import StreamingAnalyzer

if TYPE_CHECKING:
    from ...system import MDState


class EnergyAnalyzer(StreamingAnalyzer):
    """
    Energy and temperature analyzer.

    Computes running statistics of:
    - Kinetic energy
    - Potential energy
    - Total energy
    - Temperature
    - Energy conservation (drift)
    """

    def __init__(self) -> None:
        """Initialize energy analyzer."""
        self.reset()

    @property
    def name(self) -> str:
        """Analyzer name."""
        return "energy"

    def reset(self) -> None:
        """Reset statistics."""
        self._n_frames = 0
        self._kinetic: list[float] = []
        self._potential: list[float] = []
        self._total: list[float] = []
        self._temperature: list[float] = []
        self._times: list[float] = []

    def update(self, state: MDState, **kwargs: Any) -> None:
        """
        Update energy statistics.

        Args:
            state: Current simulation state.
            **kwargs: Should include 'potential_energy' and optionally 'time'.
        """
        # Kinetic energy
        ke = state.kinetic_energy

        # Potential energy from kwargs
        pe = kwargs.get("potential_energy", 0.0)

        # Temperature
        temp = state.temperature

        # Time
        time = kwargs.get("time", float(self._n_frames))

        self._kinetic.append(ke)
        self._potential.append(pe)
        self._total.append(ke + pe)
        self._temperature.append(temp)
        self._times.append(time)

        self._n_frames += 1

    def result(self) -> dict[str, Any]:
        """
        Get energy statistics.

        Returns:
            Dictionary with energy arrays and statistics.
        """
        kinetic = np.array(self._kinetic)
        potential = np.array(self._potential)
        total = np.array(self._total)
        temperature = np.array(self._temperature)
        times = np.array(self._times)

        # Compute statistics
        results = {
            "time": times,
            "kinetic": kinetic,
            "potential": potential,
            "total": total,
            "temperature": temperature,
            "n_frames": self._n_frames,
        }

        if self._n_frames > 0:
            results.update(
                {
                    "kinetic_mean": float(np.mean(kinetic)),
                    "kinetic_std": float(np.std(kinetic)),
                    "potential_mean": float(np.mean(potential)),
                    "potential_std": float(np.std(potential)),
                    "total_mean": float(np.mean(total)),
                    "total_std": float(np.std(total)),
                    "temperature_mean": float(np.mean(temperature)),
                    "temperature_std": float(np.std(temperature)),
                }
            )

            # Energy drift (conservation check)
            if self._n_frames > 1:
                # Linear fit to total energy
                slope, _ = np.polyfit(times, total, 1)
                results["energy_drift_per_time"] = float(slope)

                # Relative drift
                e_mean = np.mean(np.abs(total))
                if e_mean > 0:
                    results["relative_drift"] = float(
                        np.abs(slope) * (times[-1] - times[0]) / e_mean
                    )

        return results

    @property
    def total_energy(self) -> NDArray[np.floating]:
        """Total energy array."""
        return np.array(self._total)

    @property
    def temperature_history(self) -> NDArray[np.floating]:
        """Temperature array."""
        return np.array(self._temperature)
