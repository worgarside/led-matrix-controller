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

    clock = Clock(**matrix.dimensions, persistent=True)
    rain = RainingGrid(**matrix.dimensions, persistent=True)
    sorter = Sorter(**matrix.dimensions)

    matrix.register_content(
        ImageViewer(path=Path("door/closed.bmp"), **matrix.dimensions, display_seconds=5),
        GifViewer(path=Path("door/animated.gif"), **matrix.dimensions),
        rain,
        NowPlaying(**matrix.dimensions, persistent=True),
        sorter,
        clock,
        Combination(content=(rain, clock), **matrix.dimensions),
        Combination(content=(sorter, clock), **matrix.dimensions),
    )

    mqtt_client.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with suppress(Exception):
            Matrix(mqtt_client=MqttClient(connect=False)).clear_matrix()
        raise
