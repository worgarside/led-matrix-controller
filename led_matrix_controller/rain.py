"""Rain simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from models import RGBMatrix, RGBMatrixOptions
from PIL import Image
from utils import const
from utils.cellular_automata import RainingGrid
from utils.cellular_automata.raining_grid import State
from utils.mqtt import MQTT_CLIENT

if TYPE_CHECKING:
    from models.matrix import LedMatrixOptions
    from numpy.typing import NDArray


class Matrix:
    """Class for displaying track information on an RGB LED Matrix."""

    OPTIONS: ClassVar[LedMatrixOptions] = {
        "cols": 64,
        "rows": 64,
        "brightness": 100,
        "gpio_slowdown": 4,
        "hardware_mapping": "adafruit-hat-pwm",
        "show_refresh_rate": const.DEBUG_MODE,
        "limit_refresh_rate_hz": 125,
        "pwm_lsb_nanoseconds": 60,
        # "pwm_dither_bits": 1,  # noqa: ERA001
    }

    def __init__(self, colormap: NDArray[np.int_]) -> None:
        options = RGBMatrixOptions()

        for name, value in self.OPTIONS.items():
            if getattr(options, name, None) != value:
                setattr(options, name, value)

        self.matrix = RGBMatrix(options=options)
        self.canvas = self.matrix.CreateFrameCanvas()

        self.colormap = colormap

    def render_array(self, array: NDArray[np.int_]) -> None:
        """Render the array to the LED matrix."""

        pixels = self.colormap[array]

        image = Image.fromarray(pixels.astype(np.uint8), "RGB")

        self.canvas.SetImage(image)
        self.canvas = self.matrix.SwapOnVSync(self.canvas)

    @property
    def height(self) -> int:
        """Return the height of the matrix."""
        return int(self.matrix.height)

    @property
    def width(self) -> int:
        """Return the width of the matrix."""
        return int(self.matrix.width)


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
