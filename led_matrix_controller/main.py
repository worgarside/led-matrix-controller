"""Rain simulation."""

from __future__ import annotations

from pathlib import Path

from models.content import RainingGrid
from models.matrix import Matrix
from utils import ImageViewer, MqttClient


def main() -> None:
    """Run the rain simulation."""

    mqtt_client = MqttClient(connect=True)

    matrix = Matrix(mqtt_client=mqtt_client)

    matrix.register_content(
        RainingGrid(**matrix.dimensions, mqtt_client=mqtt_client, persistent=True),
        ImageViewer(path=Path("door/closed.bmp"), **matrix.dimensions, display_seconds=5),
    )

    mqtt_client.loop_forever()


if __name__ == "__main__":
    main()
