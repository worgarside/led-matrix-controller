"""Content for viewing static image."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, ClassVar

from content.base import PreDefinedContent, StopType
from PIL import Image
from utils import const, to_kebab_case
from wg_utilities.loggers import get_streaming_logger

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from utils import mtrx

LOGGER = get_streaming_logger(__name__)


@dataclass(kw_only=True, slots=True)
class GifViewer(PreDefinedContent):
    """Content for playing an animated GIF."""

    GIF_DIRECTORY: ClassVar[Path] = const.ASSETS_DIRECTORY / "gifs" / "64x64"

    path: Path

    image: Image.Image = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the image."""
        if not self.path.is_absolute():
            self.path = self.GIF_DIRECTORY / self.path

        self.image = Image.open(self.path)

        self.canvas_count = self.image.n_frames

        super(GifViewer, self).__post_init__()

    def generate_canvases(
        self,
        new_canvas: Callable[[Image.Image | None], mtrx.Canvas],
    ) -> None:
        """Generate the canvases for the content.

        https://github.com/hzeller/rpi-rgb-led-matrix/blob/master/bindings/python/samples/gif-viewer.py
        """
        canvases = []
        for frame_index in range(self.canvas_count):
            self.image.seek(frame_index)

            canvases.append(new_canvas(self.image))

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
