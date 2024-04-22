"""Content for viewing static image."""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import DEBUG, getLogger
from typing import TYPE_CHECKING, Callable, Generator

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
class GifViewer(PreDefinedContent):
    """Content for playing an animated GIF."""

    path: Path

    image: Image.Image = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the image."""
        if not self.path.is_absolute():
            self.path = const.IMAGE_DIRECTORY / self.path

        self.image = Image.open(self.path)

        self.canvas_count = self.image.n_frames

    def generate_canvases(
        self,
        new_canvas: Callable[[Image.Image | None], Canvas],
    ) -> None:
        """Generate the canvases for the content.

        https://github.com/hzeller/rpi-rgb-led-matrix/blob/master/bindings/python/samples/gif-viewer.py
        """
        canvases = []
        for frame_index in range(self.canvas_count):
            self.image.seek(frame_index)

            canvases.append(new_canvas(self.image))

        self.canvases = tuple(canvases)

    def teardown(self) -> Generator[None, None, None]:
        """No teardown needed."""
        yield

    @property
    def content_id(self) -> str:
        """Return the ID of the content."""
        return "gif-" + to_kebab_case(
            self.path.relative_to(const.IMAGE_DIRECTORY).with_suffix("").as_posix(),
        )

    def __iter__(self) -> Generator[None, None, None]:
        """Iterate once per each frame of the GIF."""
        for _ in range(self.canvas_count):
            yield
