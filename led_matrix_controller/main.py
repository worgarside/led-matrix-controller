"""Rain simulation."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

from content import Clock, GifViewer, ImageViewer, NowPlaying, RainingGrid, Sorter
from content.combination import Combination
from models import Matrix
from utils import MqttClient


def main() -> None:
    """Run the rain simulation."""
    mqtt_client = MqttClient(connect=True)

    matrix = Matrix(mqtt_client=mqtt_client)

    matrix.register_content(
        Clock(persistent=True),
        Combination(),
        GifViewer(path=Path("door/animated.gif")),
        ImageViewer(path=Path("door/closed.bmp"), display_seconds=5),
        NowPlaying(persistent=True),
        RainingGrid(persistent=True),
        Sorter(),
    )

    mqtt_client.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with suppress(Exception):
            Matrix(mqtt_client=MqttClient(connect=False)).clear_matrix()
        raise
