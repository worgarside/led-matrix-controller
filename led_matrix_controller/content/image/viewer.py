"""Content for viewing static image."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from logging import DEBUG, getLogger
from time import sleep
from typing import TYPE_CHECKING, Callable, ClassVar, Literal

from content.base import PreDefinedContent
from PIL import Image
from utils import const, to_kebab_case
from wg_utilities.loggers import add_stream_handler

if TYPE_CHECKING:
    from pathlib import Path

    from utils import mtrx

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
add_stream_handler(LOGGER)


@dataclass(kw_only=True, slots=True)
class ImageViewer(PreDefinedContent):
    """Content for viewing static image."""

    IMAGE_DIRECTORY: ClassVar[Path] = const.ASSETS_DIRECTORY / "images" / "64x64"

    display_seconds: int
    path: Path

    canvas_count: Literal[1] = field(init=False, default=1)
    image: Image.Image = field(init=False, repr=False)

    ticks_to_sleep: int = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the image."""
        if not self.path.is_absolute():
            self.path = self.IMAGE_DIRECTORY / self.path

        self.image = Image.open(self.path).convert("RGB")

        self.ticks_to_sleep = int(self.display_seconds / const.TICK_LENGTH)

        super(ImageViewer, self).__post_init__()

    def generate_canvases(
        self,
        new_canvas: Callable[[Image.Image | None], mtrx.Canvas],
    ) -> None:
        """Generate the canvas for the content ahead of time.

        This is a static image, so we only need to generate the canvas once.
        """
        self.canvases = (new_canvas(self.image),)

    @property
    def id(self) -> str:
        """Return the ID of the content."""
        return "image-" + to_kebab_case(
            self.path.relative_to(self.IMAGE_DIRECTORY).with_suffix("").as_posix(),
        )

    def __iter__(self) -> Generator[None, None, None]:
        """Yield nothing; this is a static image."""
        tts = self.ticks_to_sleep

        yield

        self.is_sleeping = True

        LOGGER.debug(
            "Sleeping for %d ticks whilst displaying %s",
            tts,
            self.path,
        )

        while tts > 0:
            sleep(const.TICK_LENGTH)
            tts -= 1

            # Allow `stop` override
            if not self.active:
                break

        self.is_sleeping = False
