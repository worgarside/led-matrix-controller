"""Rain simulation."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

from content import GifViewer, ImageViewer, NowPlaying, RainingGrid
from models import Matrix
from utils import MqttClient


def main() -> None:
    """Run the rain simulation."""
    mqtt_client = MqttClient(connect=True)

    matrix = Matrix(mqtt_client=mqtt_client)

    matrix.register_content(
        RainingGrid(**matrix.dimensions, persistent=True),
        ImageViewer(path=Path("door/closed.bmp"), **matrix.dimensions, display_seconds=5),
        GifViewer(path=Path("door/animated.gif"), **matrix.dimensions),
        NowPlaying(**matrix.dimensions),
    )

    mqtt_client.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with suppress(Exception):
            Matrix(mqtt_client=MqttClient(connect=False)).clear_matrix()
        raise
