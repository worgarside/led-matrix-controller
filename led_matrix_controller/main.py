"""Rain simulation."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

from models.content import ContentTag, RainingGrid
from models.matrix import Matrix
from utils.image import ImageViewer
from utils.mqtt import MQTT_CLIENT


def main() -> None:
    """Run the rain simulation."""

    matrix = Matrix()

    grid = RainingGrid(height=matrix.height, width=matrix.width)

    matrix.add_content(
        ImageViewer(Path("door/closed.bmp"), height=matrix.height, width=matrix.width),
        tag=ContentTag.IDLE,
    )

    matrix.add_content(grid, tag=ContentTag.IDLE)

    MQTT_CLIENT.loop_start()
    matrix.mainloop()
    MQTT_CLIENT.loop_stop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with suppress(Exception):
            Matrix().clear_matrix()
        raise
