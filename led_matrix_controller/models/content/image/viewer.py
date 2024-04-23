"""Content for viewing static image."""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import DEBUG, getLogger
from time import sleep
from typing import TYPE_CHECKING, Callable, Generator, Literal

from models.content.base import PreDefinedContent
from PIL import Image
from utils import const, to_kebab_case
from wg_utilities.loggers import add_stream_handler

if TYPE_CHECKING:
    from pathlib import Path

    from models import Canvas

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
add_stream_handler(LOGGER)


@dataclass(kw_only=True, slots=True)
class ImageViewer(PreDefinedContent):
    """Content for viewing static image."""

    display_seconds: int
    path: Path

    canvas_count: Literal[2] = field(init=False, default=2)
    image: Image.Image = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the image."""
        if not self.path.is_absolute():
            self.path = const.IMAGE_DIRECTORY / self.path

        self.image = Image.open(self.path).convert("RGB")

    def generate_canvases(
        self,
        new_canvas: Callable[[Image.Image | None], Canvas],
    ) -> None:
        """Generate the canvases for the content ahead of time.

        This is a static image, so we only need to generate the canvas once. It is included twice to
        account for the two `yield` statements in ImageContent.__iter__.
        """
        self.canvases = (new_canvas(self.image), new_canvas(self.image))

    @property
    def content_id(self) -> str:
        """Return the ID of the content."""
        return "image-" + to_kebab_case(
            self.path.relative_to(const.IMAGE_DIRECTORY).with_suffix("").as_posix(),
        )

    def teardown(self) -> Generator[None, None, None]:
        """No teardown needed."""
        yield

    def __iter__(self) -> Generator[None, None, None]:
        """Yield nothing; this is a static image."""
        yield

        ticks_to_sleep = self.display_seconds / const.TICK_LENGTH

        LOGGER.debug(
            "Sleeping for %d ticks whilst displaying %s",
            ticks_to_sleep,
            self.path,
        )

        self._active = True
        while ticks_to_sleep > 0:
            sleep(const.TICK_LENGTH)
            ticks_to_sleep -= 1

            # Allow `stop` override
            if not self._active:
                break

        self._active = False

        yield
