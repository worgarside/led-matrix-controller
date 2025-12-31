"""Rain simulation."""

from __future__ import annotations

import atexit
import signal
import sys
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from content import (
    AudioVisualiser,
    Clock,
    Combination,
    GifViewer,
    ImageViewer,
    NowPlaying,
    RainingGrid,
    Sorter,
)
from models import Matrix
from utils import MqttClient
from utils.profiling import PROFILE_ENABLED, get_profiler, save_profiling_results
from wg_utilities.loggers import get_streaming_logger

LOGGER = get_streaming_logger(__name__)

# Setup profiling if enabled
if PROFILE_ENABLED:
    profiler = get_profiler()
    if profiler:
        profiler.enable()
        atexit.register(save_profiling_results)

        def _signal_handler(signum: int, _frame: Any) -> None:
            """Handle shutdown signals to ensure profiling results are saved."""
            LOGGER.info("Received signal %s, saving profiling results...", signum)
            save_profiling_results()
            sys.exit(0)

        # Register signal handlers to ensure profiling results are saved on service stop
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        LOGGER.info("Profiling enabled - results will be saved on exit")

if TYPE_CHECKING:
    from content.base import ContentBase


def get_library() -> tuple[ContentBase[Any], ...]:
    """Get the library of content."""
    return (
        Clock(),
        Combination(),
        GifViewer(path=Path("door/animated.gif"), frame_multiplier=2),
        GifViewer(path=Path("alert/bell.gif"), frame_multiplier=10),
        ImageViewer(path=Path("door/closed.bmp"), display_seconds=5),
        NowPlaying(),
        RainingGrid(),
        Sorter(),
        AudioVisualiser(),
    )


def main() -> None:
    """Run the rain simulation."""
    # Needs to be before the content library, idk why :(
    mqtt_client = MqttClient(connect=True)

    library = get_library()

    works_with: dict[type[ContentBase[Any]], set[type[ContentBase[Any]]]] = {
        Clock: {RainingGrid, Sorter, NowPlaying, AudioVisualiser},
        GifViewer: set(),
        ImageViewer: set(),
        NowPlaying: {Clock},
        RainingGrid: {Clock},
        Sorter: {Clock},
        AudioVisualiser: {Clock},
    }

    Matrix(
        mqtt_client=mqtt_client,
        content_works_with=works_with,
    ).register_content(
        *library,
    )

    mqtt_client.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        LOGGER.critical("Unhandled %s in main", type(err).__name__, exc_info=True)

        with suppress(Exception):
            Matrix(
                mqtt_client=MqttClient(connect=False),
                content_works_with={},
            ).clear_matrix()
        raise
