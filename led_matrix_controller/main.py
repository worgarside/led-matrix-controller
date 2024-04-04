"""Rain simulation."""

from __future__ import annotations

from models.content import RainingGrid
from models.content.raining_grid import State
from models.matrix import Matrix
from utils.mqtt import MQTT_CLIENT


def main() -> None:
    """Run the rain simulation."""

    matrix = Matrix(colormap=State.colormap())

    grid = RainingGrid(height=matrix.height, width=matrix.width)

    MQTT_CLIENT.loop_start()

    for frame in grid.frames:
        matrix.render_array(frame)

    MQTT_CLIENT.loop_stop()


if __name__ == "__main__":
    main()
