"""Radial distribution function analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..base import StreamingAnalyzer

if TYPE_CHECKING:
    from ...system import MDState


class RadialDistributionFunction(StreamingAnalyzer):
    """
    Radial distribution function g(r) calculator.

    The RDF measures the probability of finding a particle at
    distance r from another particle, relative to an ideal gas.

    g(r) = (V / N^2) * <sum_i sum_{j!=i} delta(r - r_ij)> / (4 * pi * r^2 * dr)

    Supports:
    - Single species
    - Multiple species (partial RDFs)
    - Periodic boundary conditions
    """

    def __init__(
        self,
        r_max: float,
        n_bins: int = 100,
        species_pairs: list[tuple[int, int]] | None = None,
    ) -> None:
        """
        Initialize RDF calculator.

        Args:
            r_max: Maximum distance for RDF.
            n_bins: Number of histogram bins.
            species_pairs: List of (species_i, species_j) pairs to compute.
                          If None, computes total RDF.
        """
        self.r_max = r_max
        self.n_bins = n_bins
        self.species_pairs = species_pairs

        self._dr = r_max / n_bins
        self._r_edges = np.linspace(0, r_max, n_bins + 1)
        self._r_centers = 0.5 * (self._r_edges[:-1] + self._r_edges[1:])

        self.reset()

    @property
    def name(self) -> str:
        """Analyzer name."""
        return "rdf"

    def reset(self) -> None:
        """Reset histogram."""
        self._histogram = np.zeros(self.n_bins, dtype=np.float64)
        self._n_frames = 0
        self._volume_sum = 0.0
        self._n_pairs_sum = 0

    def update(self, state: MDState, **kwargs: Any) -> None:
        """
        Update RDF with new frame.

        Args:
            state: Current simulation state.
        """
        positions = state.positions
        n_atoms = state.n_atoms
        box = state.box

        # Compute pairwise distances
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Minimum image distance
                if box is not None:
                    rij = box.minimum_image(positions[i], positions[j])
                else:
                    rij = positions[j] - positions[i]

                r = np.linalg.norm(rij)

                if r < self.r_max:
                    bin_idx = int(r / self._dr)
                    if bin_idx < self.n_bins:
                        self._histogram[bin_idx] += 2  # Count i-j and j-i

        self._n_frames += 1
        if box is not None:
            self._volume_sum += box.volume
        self._n_pairs_sum += n_atoms * (n_atoms - 1)

    def result(self) -> dict[str, Any]:
        """
        Get RDF result.

        Returns:
            Dictionary with 'r' (bin centers) and 'g_r' (RDF values).
        """
        if self._n_frames == 0:
            return {
                "r": self._r_centers,
                "g_r": np.zeros(self.n_bins),
            }

        # Average volume and number of pairs
        avg_volume = self._volume_sum / self._n_frames
        avg_n_pairs = self._n_pairs_sum / self._n_frames

        # Normalize histogram to g(r)
        # Shell volume = 4 * pi * r^2 * dr
        shell_volumes = 4 * np.pi * self._r_centers**2 * self._dr

        # Ideal gas density
        rho = avg_n_pairs / avg_volume if avg_volume > 0 else 1.0

        # g(r) = histogram / (n_frames * shell_volume * rho)
        g_r = self._histogram / (self._n_frames * shell_volumes * rho + 1e-10)

        return {
            "r": self._r_centers.copy(),
            "g_r": g_r,
            "n_frames": self._n_frames,
        }

    @property
    def r(self) -> NDArray[np.floating]:
        """Bin centers."""
        return self._r_centers

    @property
    def g_r(self) -> NDArray[np.floating]:
        """Current g(r) estimate."""
        return self.result()["g_r"]
