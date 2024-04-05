"""Constants and class for managing the RGB LED Matrix."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from logging import DEBUG, getLogger
from typing import TYPE_CHECKING, ClassVar

from models.content import ContentTag
from PIL import Image
from utils import const
from wg_utilities.loggers import add_stream_handler

from ._rgbmatrix import RGBMatrix, RGBMatrixOptions

if TYPE_CHECKING:
    import numpy as np
    from models.content.base import ContentBase
    from numpy.typing import NDArray
    from utils.cellular_automata.automaton import GridView

    from .led_matrix_options import LedMatrixOptions

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
add_stream_handler(LOGGER)


class Matrix:
    """Class for displaying track information on an RGB LED Matrix."""

    OPTIONS: ClassVar[LedMatrixOptions] = {
        "cols": 64,
        "rows": 64,
        "brightness": 100,
        "gpio_slowdown": 4,
        "hardware_mapping": "adafruit-hat-pwm",
        "show_refresh_rate": const.DEBUG_MODE,
        "limit_refresh_rate_hz": 100,
        "pwm_lsb_nanoseconds": 80,
        # "pwm_dither_bits": 1,  # noqa: ERA001
    }

    def __init__(self, options: LedMatrixOptions = OPTIONS) -> None:
        all_options = RGBMatrixOptions()

        for name, value in options.items():
            setattr(all_options, name, value)

        self.matrix = RGBMatrix(options=all_options)
        self.canvas = self.matrix.CreateFrameCanvas()

        self._content: dict[ContentTag, list[ContentBase]] = defaultdict(list)

    def add_content(self, content: ContentBase, tag: ContentTag) -> None:
        """Add content to the matrix."""
        self._content[tag].append(content)

    @staticmethod
    def _get_image(colormap: NDArray[np.uint8], grid: GridView) -> Image.Image:
        return Image.fromarray(
            colormap[grid],
            "RGB",
        )

    def mainloop(self) -> None:
        """Run the main loop for the matrix."""

        # Placeholders for future instance attributes(?)
        current_tag = ContentTag.IDLE
        current_content_index = 0

        content = self._content[current_tag][current_content_index]
        get_image = partial(self._get_image, content.STATE.colormap(), content.grid)

        for _ in content:
            self.canvas.SetImage(get_image())
            self.canvas = self.matrix.SwapOnVSync(self.canvas)

    @property
    def height(self) -> int:
        """Return the height of the matrix."""
        return int(self.matrix.height)

    @property
    def width(self) -> int:
        """Return the width of the matrix."""
        return int(self.matrix.width)
