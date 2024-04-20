"""Content for viewing static image."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import sleep
from typing import TYPE_CHECKING, Final, Generator, Literal

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

    has_teardown_sequence: Literal[False] = False

    _image: Image.Image = field(init=False, repr=False)

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

    def teardown(self) -> Generator[None, None, None]:
        """No teardown needed."""
        yield

    def __iter__(self) -> Generator[None, None, None]:
        """Yield nothing; this is a static image."""
        yield

        ticks_to_sleep = self.display_seconds / const.TICK_LENGTH

        while self._active and ticks_to_sleep > 0:
            sleep(const.TICK_LENGTH)
            ticks_to_sleep -= 1

        yield
