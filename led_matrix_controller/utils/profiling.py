"""Profiling utilities for performance analysis."""

from __future__ import annotations

import cProfile
import json
import pstats
import time
from contextlib import contextmanager
from os import getenv
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wg_utilities.loggers import get_streaming_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

LOGGER = get_streaming_logger(__name__)

PROFILE_ENABLED: bool = True
# Use /tmp for profiling output - this is safe for profiling data
PROFILE_OUTPUT_DIR: Path = Path(
    getenv("PROFILE_OUTPUT_DIR", "/tmp/led_matrix_profiles"),  # noqa: S108
)


class TimingStats:
    """Collect timing statistics for operations."""

    def __init__(self) -> None:
        """Initialize timing statistics."""
        self.timings: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}

    def record(self, operation: str, duration: float) -> None:
        """Record a timing for an operation."""
        if operation not in self.timings:
            self.timings[operation] = []
            self.counts[operation] = 0
        self.timings[operation].append(duration)
        self.counts[operation] += 1

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all operations."""
        stats = {}
        for operation, durations in self.timings.items():
            if durations:
                stats[operation] = {
                    "count": self.counts[operation],
                    "total": sum(durations),
                    "mean": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "p50": sorted(durations)[len(durations) // 2] if durations else 0,
                    "p95": sorted(durations)[int(len(durations) * 0.95)]
                    if durations
                    else 0,
                    "p99": sorted(durations)[int(len(durations) * 0.99)]
                    if durations
                    else 0,
                }
        return stats

    def save(self, filepath: Path) -> None:
        """Save statistics to a JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.get_stats(), f, indent=2)


# Global timing stats instance
_timing_stats = TimingStats()

# Global profiler instance - using module-level variable to avoid global statement warnings
_profiler: cProfile.Profile | None = None


@contextmanager
def time_operation(operation: str) -> Iterator[None]:
    """Context manager to time an operation."""
    if not PROFILE_ENABLED:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        _timing_stats.record(operation, duration)


def save_profiling_results() -> None:
    """Save profiling results to files."""
    if not PROFILE_ENABLED:
        return

    PROFILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save timing statistics
    timing_file = PROFILE_OUTPUT_DIR / "timing_stats.json"
    _timing_stats.save(timing_file)
    LOGGER.info("Timing statistics saved to: %s", timing_file)

    # Save cProfile statistics if available
    if _profiler is not None:
        stats_file = PROFILE_OUTPUT_DIR / "cprofile_stats.txt"
        with stats_file.open("w", encoding="utf-8") as f:
            stats = pstats.Stats(_profiler, stream=f)
            stats.sort_stats("cumulative")
            stats.print_stats(50)  # Top 50 functions
        LOGGER.info("cProfile statistics saved to: %s", stats_file)


def get_profiler() -> cProfile.Profile | None:
    """Get a cProfile profiler if profiling is enabled."""
    if not PROFILE_ENABLED:
        return None
    # Use module-level variable to store profiler
    # pylint: disable=global-statement
    global _profiler  # noqa: PLW0603
    _profiler = cProfile.Profile()
    return _profiler
