"""Rain simulation."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

from models.content import RainingGrid
from models.matrix import Matrix
from utils import ImageViewer, MqttClient
from utils.gif.viewer import GifViewer


def main() -> None:
    """Run the rain simulation."""

    mqtt_client = MqttClient(connect=True)

    matrix = Matrix(mqtt_client=mqtt_client)

    matrix.register_content(
        RainingGrid(**matrix.dimensions, mqtt_client=mqtt_client, persistent=True),
        ImageViewer(path=Path("door/closed.bmp"), **matrix.dimensions, display_seconds=5),
        GifViewer(path=Path("door/animated.gif"), **matrix.dimensions),
    )

    mqtt_client.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with suppress(Exception):
            Matrix(mqtt_client=MqttClient(connect=False)).clear_matrix()
        raise
