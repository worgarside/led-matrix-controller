"""Base class for content models."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass, field, is_dataclass
from enum import Enum, auto
from functools import partial
from json import dumps
from logging import DEBUG, getLogger
from os import PathLike
from typing import (
    Any,
    Callable,
    ClassVar,
    Generator,
    Iterator,
    final,
)

import numpy as np
from httpx import URL
from numpy.typing import NDArray
from PIL import Image
from utils import const, mtrx
from utils.helpers import camel_to_kebab_case
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

    def __json__(self) -> dict[str, Any]:
        """Return the JSON representation of the state."""
        return {
            "state": self.state,
            "char": self.char,
            "color": self.color,
        }


def _get_image(colormap: NDArray[np.uint8], grid: GridView) -> Image.Image:
    return Image.fromarray(colormap[grid], "RGB")


CanvasGetter = partial[mtrx.Canvas]
ImageGetter = partial[Image.Image]


class StopType(Enum):
    """Type of stop for content."""

    CANCEL = auto()
    """Content was stopped via an MQTT message."""

    PRIORITY = auto()
    """Higher priority content was started."""

    EXPIRED = auto()
    """Content expired naturally.

    e.g. rain chance becoming 0 or music stopping
    """


@dataclass(kw_only=True, slots=True)
class ContentBase(ABC):
    """Base class for content models."""

    HAS_TEARDOWN_SEQUENCE: ClassVar[bool] = False

    height: int
    width: int

    instance_id: str | None = None
    persistent: bool = field(default=False)
    is_sleeping: bool = field(default=False, init=False, repr=False)
    canvas_count: int | None = field(init=False)

    active: bool = field(init=False, default=False)
    _image_getter: ImageGetter = field(init=False, repr=False)
    colormap: NDArray[np.uint8] = field(init=False, repr=False)
    pixels: GridView = field(init=False, repr=False)

    stop_reason: StopType | None = field(init=False, repr=False)

    @abstractmethod
    def teardown(self) -> Generator[None, None, None]:
        """Perform any necessary cleanup."""

    @property
    def content_id(self) -> str:
        """Return the ID of the content."""
        return camel_to_kebab_case(self.__class__.__name__)

    @property
    @abstractmethod
    def content_getter(self) -> CanvasGetter | ImageGetter:
        """Return the image representation of the content."""

    @property
    def mqtt_attributes(self) -> str:
        """Return extra attributes for the MQTT message."""

        return dumps(self, default=self._json_encode)

    @abstractmethod
    def __iter__(self) -> Iterator[None]:
        """Iterate over the frames."""

    @final
    @staticmethod
    def _json_encode(obj: Any) -> Any:  # noqa: PLR0911
        if hasattr(obj, "__json__"):
            with suppress(TypeError):
                return obj.__json__()

        if isinstance(obj, PathLike | URL):
            return str(obj)

        if isinstance(obj, re.Pattern):
            return obj.pattern

        with suppress(TypeError):
            if issubclass(obj, Enum):
                return obj.__qualname__

        if is_dataclass(obj):
            return {
                key: getattr(obj, key)
                for key, dc_field in obj.__dataclass_fields__.items()
                if hasattr(obj, key) and dc_field.repr
            }

        if isinstance(obj, slice):
            return f"[{obj.start or ''}:{obj.stop or ''}:{obj.step or ''}]"

        try:
            return dumps(obj)
        except TypeError:
            if not callable(obj):
                LOGGER.error("Could not serialize object (%s): %r", obj, obj)  # noqa: TRY400
                if not const.IS_PI:
                    raise
            return None

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
        return (self.canvas_count or math.inf) > (other.canvas_count or math.inf)

    def __lt__(self, other: ContentBase) -> bool:
        """Return whether this content should be prioritized over another."""
        return (self.canvas_count or math.inf) < (other.canvas_count or math.inf)


@dataclass(kw_only=True, slots=True)
class PreDefinedContent(ContentBase, ABC):
    """Base class for content for which all frames are already known."""

    canvas_count: int = field(init=False)
    canvases: tuple[mtrx.Canvas, ...] = field(init=False, repr=False)

    @abstractmethod
    def generate_canvases(
        self,
        new_canvas: Callable[[Image.Image | None], mtrx.Canvas],
    ) -> None:
        """Generate the canvases for the content."""

    @final
    @property
    def content_getter(self) -> CanvasGetter:
        """Return the image representation of the content."""

        return partial(next, iter(self.canvases))


__all__ = ["StateBase", "ContentBase", "PreDefinedContent", "GridView"]
