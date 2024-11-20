"""Rain simulation."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from content import (
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

if TYPE_CHECKING:
    from content.base import ContentBase

# Needs to be before the content library, idk why :(
MQTT_CLIENT = MqttClient(connect=True)

LIBRARY = (
    Clock(persistent=True),
    Combination(),
    GifViewer(path=Path("door/animated.gif")),
    ImageViewer(path=Path("door/closed.bmp"), display_seconds=5),
    NowPlaying(persistent=True),
    RainingGrid(persistent=True),
    Sorter(),
)

WORKS_WITH: dict[type[ContentBase[Any]], set[type[ContentBase[Any]]]] = {
    Clock: {RainingGrid, Sorter, NowPlaying},
    GifViewer: set(),
    ImageViewer: set(),
    NowPlaying: {Clock},
    RainingGrid: {Clock},
    Sorter: {Clock},
}


def main() -> None:
    """Run the rain simulation."""
    Matrix(mqtt_client=MQTT_CLIENT, content_works_with=WORKS_WITH).register_content(
        *LIBRARY,
    )

    MQTT_CLIENT.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with suppress(Exception):
            Matrix(
                mqtt_client=MqttClient(connect=False),
                content_works_with={},
            ).clear_matrix()
        raise
