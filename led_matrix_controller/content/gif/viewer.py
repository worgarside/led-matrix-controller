"""Content for viewing static image."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from content.base import PreDefinedContent, StopType
from PIL import Image
from utils import const, to_kebab_case
from wg_utilities.loggers import get_streaming_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from utils import mtrx

LOGGER = get_streaming_logger(__name__)


@dataclass(kw_only=True, slots=True)
class GifViewer(PreDefinedContent):
    """Content for playing an animated GIF."""

    GIF_DIRECTORY: ClassVar[Path] = const.ASSETS_DIRECTORY / "gifs" / "64x64"

    path: Path

    image: Image.Image = field(init=False, repr=False)

    frame_multiplier: int = 1
    """The number of ticks to display each frame for."""

    def __post_init__(self) -> None:
        """Initialize the image."""
        if self.frame_multiplier < 1:
            LOGGER.warning("Frame multiplier is less than 1, setting to 1")
            self.frame_multiplier = 1

        if not self.path.is_absolute():
            self.path = self.GIF_DIRECTORY / self.path

        self.image = Image.open(self.path)

        self.canvas_count = int(self.image.n_frames) * self.frame_multiplier

        LOGGER.info("Canvas count for %s: %s", self.path, self.canvas_count)

        super(GifViewer, self).__post_init__()

    def generate_canvases(
        self,
        new_canvas: Callable[[Image.Image | None], mtrx.Canvas],
    ) -> None:
        """Generate the canvases for the content.

        https://github.com/hzeller/rpi-rgb-led-matrix/blob/master/bindings/python/samples/gif-viewer.py
        """
        canvases: list[mtrx.Canvas] = []
        for frame_index in range(self.image.n_frames):
            self.image.seek(frame_index)

            canvas = new_canvas(self.image)

            canvases.extend(canvas for _ in range(self.frame_multiplier))

        self.canvases = tuple(canvases)

    @property
    def id(self) -> str:
        """Return the ID of the content."""
        return "gif-" + to_kebab_case(
            self.path.relative_to(self.GIF_DIRECTORY).with_suffix("").as_posix(),
        )

    def __iter__(self) -> Generator[None, None, None]:
        """Iterate once per each frame of the GIF."""
        for _ in range(self.canvas_count):
            yield

        self.stop(StopType.EXPIRED)
