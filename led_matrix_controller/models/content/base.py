"""Base class for content models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
from logging import DEBUG, getLogger
from typing import (
    Callable,
    ClassVar,
    Generator,
    Iterator,
    final,
)

import numpy as np
from models import Canvas
from numpy.typing import NDArray
from PIL import Image
from wg_utilities.loggers import add_stream_handler

_BY_VALUE: dict[int, StateBase] = {}

GridView = NDArray[np.int_]


LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
add_stream_handler(LOGGER)


class StateBase(Enum):
    """Base class for the states of a cell."""

    def __init__(
        self,
        value: int,
        char: str,
        color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self._value_ = value
        self.state = value  # This is only really for type checkers, _value_ is the same but has a different type
        self.char = char
        self.color = color

        _BY_VALUE[value] = self

    @classmethod
    def colormap(cls) -> NDArray[np.uint8]:
        """Return the color map of the states."""
        return np.array([state.color for state in cls], dtype=np.uint8)

    @staticmethod
    def by_value(value: int | np.int_) -> StateBase:
        """Return the state by its value."""
        return _BY_VALUE[int(value)]

    def __hash__(self) -> int:
        """Return the hash of the value of the state."""
        return hash(self.value)


def _get_image(colormap: NDArray[np.uint8], grid: GridView) -> Image.Image:
    return Image.fromarray(colormap[grid], "RGB")


CanvasGetter = partial[Canvas]
ImageGetter = partial[Image.Image]


class StopType(Enum):
    """Type of stop for content."""

    CANCEL = auto()
    PRIORITY = auto()


@dataclass(kw_only=True, slots=True)
class ContentBase(ABC):
    """Base class for content models."""

    HAS_TEARDOWN_SEQUENCE: ClassVar[bool] = False

    height: int
    width: int

    instance_id: str | None = None
    persistent: bool = field(default=False)

    _active: bool = field(init=False, default=False)
    _image_getter: ImageGetter = field(init=False, repr=False)
    colormap: NDArray[np.uint8] = field(init=False, repr=False)
    pixels: GridView = field(init=False, repr=False)

    stop_reason: StopType | None = field(init=False, repr=False)

    @abstractmethod
    def teardown(self) -> Generator[None, None, None]:
        """Perform any necessary cleanup."""

    @property
    @abstractmethod
    def content_id(self) -> str:
        """Return the ID of the content."""

    @property
    @abstractmethod
    def content_getter(self) -> CanvasGetter | ImageGetter:
        """Return the image representation of the content."""

    @abstractmethod
    def __iter__(self) -> Generator[None, None, None]:
        """Iterate over the frames."""

    @property
    def active(self) -> bool:
        """Return whether the content is active."""
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        """Set the active state of the content."""
        self._active = value

    @final
    def stop(self, stop_type: StopType, /) -> None:
        """Stop the content immediately."""
        self.active = False
        self.stop_reason = stop_type

        LOGGER.info("Stopped content with ID `%s`: %r", self.content_id, stop_type)

    @final
    @property
    def id(self) -> str:
        """Return the ID of the content."""
        return self.instance_id or self.content_id

    def __gt__(self, other: ContentBase) -> bool:
        """Return whether this content should be de-prioritized over another."""
        if type(self) != type(other):
            # If the other content is pre-defined (finite) and this one is not, the other should be prioritized
            return isinstance(other, PreDefinedContent) and isinstance(
                self,
                DynamicContent,
            )

        if isinstance(other, PreDefinedContent) and isinstance(self, PreDefinedContent):
            return self.canvas_count > other.canvas_count

        # Not sure of the best way to compare dynamic content yet
        return False

    def __lt__(self, other: ContentBase) -> bool:
        """Return whether this content should be prioritized over another."""
        if type(self) != type(other):
            # If the other content is pre-defined (finite) and this one is not, the other should be prioritized
            return isinstance(other, PreDefinedContent) and isinstance(
                self,
                DynamicContent,
            )

        if isinstance(other, PreDefinedContent) and isinstance(self, PreDefinedContent):
            return self.canvas_count < other.canvas_count

        # Not sure of the best way to compare dynamic content yet
        return False


@dataclass(kw_only=True, slots=True)
class DynamicContent(ContentBase, ABC):
    """Base class for content which is dynamically created."""

    @final
    @property
    def content_getter(self) -> ImageGetter:
        """Return the image representation of the content."""
        if not hasattr(self, "_image_getter"):
            self._image_getter = partial(_get_image, self.colormap, self.pixels)

        return self._image_getter


@dataclass(kw_only=True, slots=True)
class PreDefinedContent(ContentBase, ABC):
    """Base class for content for which all frames are already known."""

    canvas_count: int = field(init=False)
    canvases: tuple[Canvas, ...] = field(init=False, repr=False)

    _iter_canvases: Iterator[Canvas] = field(init=False, repr=False)

    @abstractmethod
    def generate_canvases(
        self,
        new_canvas: Callable[[Image.Image | None], Canvas],
    ) -> None:
        """Generate the canvases for the content."""

    @final
    @property
    def content_getter(self) -> CanvasGetter:
        """Return the image representation of the content."""

        return partial(next, iter(self.canvases))


__all__ = ["StateBase", "ContentBase", "DynamicContent", "PreDefinedContent", "GridView"]
