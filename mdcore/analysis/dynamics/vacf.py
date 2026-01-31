"""Velocity autocorrelation function analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..base import StreamingAnalyzer

if TYPE_CHECKING:
    from ...system import MDState


class VelocityAutocorrelation(StreamingAnalyzer):
    """
    Velocity autocorrelation function (VACF) calculator.

    VACF(t) = <v(0) · v(t)>

    The VACF is related to the diffusion coefficient via:
        D = (1/3) * integral_0^inf VACF(t) dt

    The Fourier transform of VACF gives the vibrational density of states.
    """

    def __init__(
        self,
        max_lag: int = 1000,
    ) -> None:
        """
        Initialize VACF calculator.

        Args:
            max_lag: Maximum lag time in frames.
        """
        self.max_lag = max_lag
        self.reset()

    @property
    def name(self) -> str:
        """Analyzer name."""
        return "vacf"

    def reset(self) -> None:
        """Reset VACF state."""
        self._n_frames = 0
        self._velocities_history: list[NDArray[np.floating]] = []
        self._vacf_sum = np.zeros(self.max_lag, dtype=np.float64)
        self._vacf_count = np.zeros(self.max_lag, dtype=np.int64)

    def update(self, state: MDState, **kwargs: Any) -> None:
        """
        Update VACF with new frame.

        Args:
            state: Current simulation state (must have velocities).
        """
        if state.velocities is None:
            raise ValueError("State must have velocities for VACF")

        velocities = state.velocities.copy()

        # Store velocities
        self._velocities_history.append(velocities)

        # Keep only max_lag frames
        if len(self._velocities_history) > self.max_lag:
            self._velocities_history.pop(0)

        # Compute VACF for all stored origins
        n_stored = len(self._velocities_history)
        current_vel = velocities

        for lag in range(min(n_stored, self.max_lag)):
            origin_vel = self._velocities_history[n_stored - lag - 1]

            # VACF = <v(0) · v(t)> averaged over all atoms
            vacf = np.mean(np.sum(origin_vel * current_vel, axis=1))

            self._vacf_sum[lag] += vacf
            self._vacf_count[lag] += 1

        self._n_frames += 1

    def result(self) -> dict[str, Any]:
        """
        Get VACF result.

        Returns:
            Dictionary with 'lag' and 'vacf' values.
        """
        vacf = np.zeros(self.max_lag, dtype=np.float64)
        mask = self._vacf_count > 0
        vacf[mask] = self._vacf_sum[mask] / self._vacf_count[mask]

        # Normalize by VACF(0)
        if vacf[0] > 0:
            vacf_normalized = vacf / vacf[0]
        else:
            vacf_normalized = vacf

        return {
            "lag": np.arange(self.max_lag),
            "vacf": vacf,
            "vacf_normalized": vacf_normalized,
            "n_frames": self._n_frames,
            "count": self._vacf_count.copy(),
        }

    def compute_diffusion_coefficient(self, dt: float = 1.0) -> float:
        """
        Compute diffusion coefficient from VACF integral.

        D = (1/3) * integral_0^inf VACF(t) dt

        Args:
            dt: Time between frames.

        Returns:
            Diffusion coefficient D.
        """
        result = self.result()
        vacf = result["vacf"]
        count = result["count"]

        # Integrate using trapezoidal rule
        mask = count > 0
        if np.sum(mask) < 2:
            return 0.0

        valid_vacf = vacf[mask]
        D = np.trapezoid(valid_vacf, dx=dt) / 3.0

        return D

    def compute_vibrational_dos(
        self,
        dt: float = 1.0,
        n_freq: int | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute vibrational density of states from VACF.

        The VDOS is the Fourier transform of the VACF.

        Args:
            dt: Time between frames.
            n_freq: Number of frequency points.

        Returns:
            Tuple of (frequencies, dos).
        """
        result = self.result()
        vacf = result["vacf"]

        # Zero-pad and FFT
        n = len(vacf) if n_freq is None else n_freq
        vacf_padded = np.zeros(2 * n)
        vacf_padded[: len(vacf)] = vacf

        # FFT
        dos = np.abs(np.fft.rfft(vacf_padded))
        freq = np.fft.rfftfreq(2 * n, d=dt)

        return freq, dos

    @property
    def vacf(self) -> NDArray[np.floating]:
        """Current VACF estimate."""
        return self.result()["vacf"]
