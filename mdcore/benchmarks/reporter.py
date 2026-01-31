"""Standardized benchmark reporting."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    """
    Standardized benchmark result container.

    All benchmarks should produce results in this format for
    consistent reporting and comparison.

    Example output:
        {
            "test": "lj_nve_256",
            "energy_drift": 1.2e-6,
            "rms_force_error": 3.1e-7,
            "ns_per_day": 120.4,
            "n_ranks": 8,
            "backend": "mpi"
        }
    """

    # Required fields
    test: str  # Benchmark identifier

    # Performance metrics (optional)
    ns_per_day: float | None = None
    steps_per_second: float | None = None
    time_per_step_ms: float | None = None
    total_time_s: float | None = None

    # Accuracy metrics (optional)
    energy_drift: float | None = None
    energy_drift_per_step: float | None = None
    rms_force_error: float | None = None
    max_force_error: float | None = None
    mae_forces: float | None = None
    mae_energy: float | None = None

    # System info
    n_atoms: int | None = None
    n_steps: int | None = None
    n_ranks: int = 1
    backend: str = "serial"

    # Validation
    passed: bool = True
    failure_reason: str | None = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None and key != "extra":
                result[key] = value
        # Merge extra fields
        result.update(self.extra)
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class BenchmarkReporter:
    """
    Reporter for collecting and outputting benchmark results.

    Supports multiple output formats:
    - JSON (file or stdout)
    - Summary table
    - CI-friendly format

    Example:
        reporter = BenchmarkReporter()

        result = BenchmarkResult(
            test="lj_nve_256",
            energy_drift=1.2e-6,
            ns_per_day=120.4,
        )
        reporter.add_result(result)

        reporter.save("benchmarks.json")
        reporter.print_summary()
    """

    def __init__(self, name: str = "benchmark_run") -> None:
        """
        Initialize benchmark reporter.

        Args:
            name: Name for this benchmark run.
        """
        self.name = name
        self.results: list[BenchmarkResult] = []
        self.start_time = datetime.now()

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)

    def create_result(self, test: str, **kwargs) -> BenchmarkResult:
        """
        Create and add a benchmark result.

        Args:
            test: Test identifier.
            **kwargs: Additional result fields.

        Returns:
            The created BenchmarkResult.
        """
        result = BenchmarkResult(test=test, **kwargs)
        self.add_result(result)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert all results to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.start_time.isoformat(),
            "n_benchmarks": len(self.results),
            "n_passed": sum(1 for r in self.results if r.passed),
            "n_failed": sum(1 for r in self.results if not r.passed),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str | Path) -> None:
        """
        Save results to JSON file.

        Args:
            filepath: Output file path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.to_json())

    def print_summary(self) -> None:
        """Print summary table to stdout."""
        print(f"\n{'=' * 60}")
        print(f"Benchmark Report: {self.name}")
        print(f"{'=' * 60}")
        print(f"Total: {len(self.results)} benchmarks")
        print(f"Passed: {sum(1 for r in self.results if r.passed)}")
        print(f"Failed: {sum(1 for r in self.results if not r.passed)}")
        print(f"{'-' * 60}")

        # Table header
        print(f"{'Test':<30} {'Status':<8} {'Key Metric':<20}")
        print(f"{'-' * 60}")

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"

            # Pick most relevant metric to display
            if result.ns_per_day is not None:
                metric = f"{result.ns_per_day:.2f} ns/day"
            elif result.steps_per_second is not None:
                metric = f"{result.steps_per_second:.1f} steps/s"
            elif result.energy_drift is not None:
                metric = f"drift: {result.energy_drift:.2e}"
            elif result.rms_force_error is not None:
                metric = f"RMS: {result.rms_force_error:.2e}"
            else:
                metric = "-"

            print(f"{result.test:<30} {status:<8} {metric:<20}")

        print(f"{'=' * 60}\n")

    def get_failed(self) -> list[BenchmarkResult]:
        """Get list of failed benchmarks."""
        return [r for r in self.results if not r.passed]

    def get_passed(self) -> list[BenchmarkResult]:
        """Get list of passed benchmarks."""
        return [r for r in self.results if r.passed]

    def all_passed(self) -> bool:
        """Check if all benchmarks passed."""
        return all(r.passed for r in self.results)


class BenchmarkTimer:
    """Context manager for timing benchmarks."""

    def __init__(self) -> None:
        """Initialize timer."""
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> BenchmarkTimer:
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing."""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000
