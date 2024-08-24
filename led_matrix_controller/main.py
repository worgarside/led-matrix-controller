"""Rain simulation."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

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


def main() -> None:
    """Run the rain simulation."""
    Matrix(mqtt_client=MQTT_CLIENT).register_content(*LIBRARY)

    MQTT_CLIENT.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with suppress(Exception):
            Matrix(mqtt_client=MqttClient(connect=False)).clear_matrix()
        raise
