"""Content for viewing static image."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import sleep
from typing import TYPE_CHECKING, Final, Generator

import numpy as np
from models.content.base import ContentBase
from PIL import Image
from utils import const, to_kebab_case

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(kw_only=True, slots=True)
class ImageViewer(ContentBase):
    """Content for viewing static image."""

    BITMAP_DIRECTORY: Final[Path] = const.REPO_PATH / "assets" / "images" / "64x64"

    display_seconds: int
    path: Path

    _image: Image.Image = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the image."""
        if not self.path.is_absolute():
            self.path = self.BITMAP_DIRECTORY / self.path

        self._image = Image.open(self.path).convert("RGB")

        img_array = np.array(self._image)

        unique_colors, pixels = np.unique(
            img_array.reshape(-1, 3), axis=0, return_inverse=True
        )
        self.colormap = np.array(
            [tuple(color) for color in unique_colors], dtype=np.uint8
        )
        self.pixels = pixels.reshape(self.height, self.width)

    @property
    def content_id(self) -> str:
        """Return the ID of the content."""
        return "image-" + to_kebab_case(
            self.path.relative_to(self.BITMAP_DIRECTORY).with_suffix("").as_posix()
        )

    def __iter__(self) -> Generator[None, None, None]:
        """Yield nothing; this is a static image."""
        yield
        sleep(max(0, self.display_seconds - (2 * const.FRAME_TIME)))
        yield
