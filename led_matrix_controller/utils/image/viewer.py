"""Content for viewing static image."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generator

import numpy as np
from models.content.base import ContentBase
from PIL import Image
from utils import const, to_kebab_case

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from utils.cellular_automata.automaton import GridView


class ImageViewer(ContentBase):
    """Content for viewing static image."""

    BITMAP_DIRECTORY = const.REPO_PATH / "assets" / "images" / "64x64"

    colormap: NDArray[np.uint8]
    pixels: GridView

    def __init__(self, path: Path, height: int, width: int) -> None:
        if not path.is_absolute():
            path = self.BITMAP_DIRECTORY / path

        self._image = Image.open(path).convert("RGB")

        img_array = np.array(self._image)

        unique_colors, pixels = np.unique(
            img_array.reshape(-1, 3), axis=0, return_inverse=True
        )
        self.colormap = np.array(
            [tuple(color) for color in unique_colors], dtype=np.uint8
        )
        self.pixels = pixels.reshape(height, width)

        self.id = "image-" + to_kebab_case(
            path.relative_to(self.BITMAP_DIRECTORY).with_suffix("").as_posix()
        )

    def __iter__(self) -> Generator[None, None, None]:
        """Yield nothing; this is a static image."""
        yield
