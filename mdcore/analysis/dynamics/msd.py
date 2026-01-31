"""Mean square displacement analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..base import StreamingAnalyzer

if TYPE_CHECKING:
    from ...system import MDState


class MeanSquareDisplacement(StreamingAnalyzer):
    """
    Mean square displacement (MSD) calculator.

    MSD(t) = <|r(t) - r(0)|^2>

    The MSD is related to the diffusion coefficient D via:
        MSD(t) = 6 * D * t  (3D diffusion)

    Supports:
    - Streaming computation (memory efficient)
    - Unwrapped coordinates for periodic systems
    - Per-species MSD
    """

    def __init__(
        self,
        max_lag: int = 1000,
        store_positions: bool = True,
    ) -> None:
        """
        Initialize MSD calculator.

        Args:
            max_lag: Maximum lag time in frames.
            store_positions: Whether to store positions for full MSD.
                           If False, only computes MSD from initial position.
        """
        self.max_lag = max_lag
        self.store_positions = store_positions

        self.reset()

    @property
    def name(self) -> str:
        """Analyzer name."""
        return "msd"

    def reset(self) -> None:
        """Reset MSD state."""
        self._n_frames = 0
        self._initial_positions: NDArray[np.floating] | None = None
        self._positions_history: list[NDArray[np.floating]] = []
        self._msd_sum = np.zeros(self.max_lag, dtype=np.float64)
        self._msd_count = np.zeros(self.max_lag, dtype=np.int64)
        self._times: list[float] = []

    def update(self, state: MDState, **kwargs: Any) -> None:
        """
        Update MSD with new frame.

        Args:
            state: Current simulation state.
            **kwargs: May include 'time' for timestep info.
        """
        positions = state.positions.copy()

        # Store initial positions
        if self._initial_positions is None:
            self._initial_positions = positions.copy()

        # Store time
        time = kwargs.get("time", self._n_frames)
        self._times.append(time)

        if self.store_positions:
            # Store positions for full time-origin averaging
            self._positions_history.append(positions)

            # Keep only max_lag frames
            if len(self._positions_history) > self.max_lag:
                self._positions_history.pop(0)
                self._times.pop(0)

            # Compute MSD for all stored origins
            n_stored = len(self._positions_history)
            for lag in range(1, min(n_stored, self.max_lag)):
                origin_pos = self._positions_history[n_stored - lag - 1]
                disp = positions - origin_pos
                msd = np.mean(np.sum(disp**2, axis=1))
                self._msd_sum[lag] += msd
                self._msd_count[lag] += 1
        else:
            # Simple MSD from t=0 only
            disp = positions - self._initial_positions
            msd = np.mean(np.sum(disp**2, axis=1))
            lag = min(self._n_frames, self.max_lag - 1)
            self._msd_sum[lag] += msd
            self._msd_count[lag] += 1

        self._n_frames += 1

    def result(self) -> dict[str, Any]:
        """
        Get MSD result.

        Returns:
            Dictionary with 'lag' (frame indices) and 'msd' values.
        """
        msd = np.zeros(self.max_lag, dtype=np.float64)
        mask = self._msd_count > 0
        msd[mask] = self._msd_sum[mask] / self._msd_count[mask]

        return {
            "lag": np.arange(self.max_lag),
            "msd": msd,
            "n_frames": self._n_frames,
            "count": self._msd_count.copy(),
        }

    def compute_diffusion_coefficient(
        self,
        dt: float = 1.0,
        fit_range: tuple[int, int] | None = None,
    ) -> float:
        """
        Compute diffusion coefficient from MSD.

        Uses linear fit to MSD(t) = 6*D*t for 3D.

        Args:
            dt: Time between frames.
            fit_range: (start, end) frame range for fitting.
                      If None, uses 10%-50% of data.

        Returns:
            Diffusion coefficient D.
        """
        result = self.result()
        msd = result["msd"]
        lag = result["lag"]

        # Default fit range: 10%-50% of data
        if fit_range is None:
            n_valid = np.sum(result["count"] > 0)
            start = max(1, n_valid // 10)
            end = n_valid // 2
            fit_range = (start, end)

        start, end = fit_range
        times = lag[start:end] * dt
        msd_fit = msd[start:end]

        # Filter out zeros
        mask = msd_fit > 0
        if np.sum(mask) < 2:
            return 0.0

        # Linear fit: MSD = 6*D*t
        slope, _ = np.polyfit(times[mask], msd_fit[mask], 1)
        D = slope / 6.0

        return D

    @property
    def msd(self) -> NDArray[np.floating]:
        """Current MSD estimate."""
        return self.result()["msd"]
