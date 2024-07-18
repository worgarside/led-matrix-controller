"""Class for the creation, caching, and management of artwork images."""

from __future__ import annotations

from colorsys import hls_to_rgb
from dataclasses import dataclass
from random import randint, shuffle, uniform
from typing import Generator

import numpy as np
from content.base import StopType
from wg_utilities.loggers import get_streaming_logger

from .dynamic_content import DynamicContent

LOGGER = get_streaming_logger(__name__)


def bubblesort(list_: list[int], /) -> Generator[None, None, None]:
    """Swap the elements to arrange in order."""
    for iter_num in range(len(list_) - 1, 0, -1):
        for idx in range(iter_num):
            if list_[idx] < list_[idx + 1]:
                list_[idx], list_[idx + 1] = list_[idx + 1], list_[idx]

                yield


@dataclass(kw_only=True, slots=True)
class Sorter(DynamicContent):
    """Display various sorting algorithms."""

    def __post_init__(self) -> None:
        """Initialize the image getter."""
        DynamicContent.__post_init__(self)

        self.pixels = self.zeros()
        self.update_colormap()

    def refresh_content(self) -> Generator[None, None, None]:
        """Refresh the content."""
        values = list(range(1, self.width + 1))
        shuffle(values)

        for idx, value in enumerate(values):
            self.pixels[value:, idx] = value
            self.pixels[:value, idx] = 0

        yield

        for _ in bubblesort(values):
            # Fill each column of the array with the corresponding value, to the height of that value
            for idx, value in enumerate(values):
                self.pixels[value:, idx] = value
                self.pixels[:value, idx] = 0

            yield

        self.stop(StopType.EXPIRED)

        del self._image_getter
        self.update_colormap()

    def update_colormap(self) -> None:
        """Randomly generate a new gradient colormap."""
        interval = 360.0 / self.width

        offset = randint(0, 360)  # noqa: S311

        lightness = uniform(0.25, 0.75)  # noqa: S311
        saturation = uniform(0.25, 1)  # noqa: S311

        LOGGER.debug(
            "Creating colormap with offset=%s, step=%s, lightness=%s, saturation=%s",
            offset,
            interval,
            lightness,
            saturation,
        )

        colors = [
            hls_to_rgb(
                (((offset + i) * interval) % 360) / 360,
                lightness,
                saturation,
            )
            for i in range(self.width)
        ]

        self.colormap = np.array(
            [(0, 0, 0)] + [tuple(int(c * 255) for c in color) for color in colors],
            dtype=np.uint8,
        )

    def teardown(self) -> Generator[None, None, None]:  # noqa: PLR6301
        """No teardown needed."""
        yield
