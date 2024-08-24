"""Base class for content models."""

from __future__ import annotations

import itertools
import math
import re
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass, field, is_dataclass
from enum import Enum, auto
from functools import cached_property, partial
from json import dumps
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    ClassVar,
    Generator,
    Generic,
    Iterator,
    Literal,
    final,
)

import numpy as np
from httpx import URL
from numpy.typing import NDArray
from typing_extensions import TypeVar
from utils import const, mtrx
from utils.helpers import camel_to_kebab_case
from wg_utilities.loggers import get_streaming_logger

from .setting import TransitionableParameterSetting  # noqa: TCH001

if TYPE_CHECKING:
    from PIL import Image

_BY_VALUE: dict[int, StateBase] = {}

GridView = NDArray[np.int_]


LOGGER = get_streaming_logger(__name__)


class StateBase(Enum):
    """Base class for the states of a cell."""

    def __init__(
        self,
        value: int,
        char: str,
        color: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        self._value_ = value
        self.state = value  # This is only really for type checkers, _value_ is the same but has a different type
        self.char = char
        self.color = color

        _BY_VALUE[value] = self

    @classmethod
    def colormap(cls) -> GridView:
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


ContentType = TypeVar("ContentType", GridView, mtrx.Canvas)


def _limit_position(
    payload: int,
    content: ContentBase[ContentType],
    *,
    limit: int,
    attr: Literal["height", "width"],
) -> int:
    """Helper function for limiting the position of content.

    This isn't implemented as a lambda within the `TransitionableParameterSetting` definition
    to avoid namespace issues.
    """
    return min(
        payload,
        limit - int(getattr(content, attr)),
    )


_CONTENT_STORE: dict[str, ContentBase[Any]] = {}


@dataclass(kw_only=True, slots=True)
class ContentBase(ABC, Generic[ContentType]):
    """Base class for content models."""

    IS_OPAQUE: ClassVar[bool] = False

    height: int = const.MATRIX_HEIGHT
    width: int = const.MATRIX_WIDTH

    x_pos: Annotated[
        int,
        TransitionableParameterSetting(
            minimum=0,
            maximum=const.MATRIX_WIDTH - 1,
            transition_rate=0.01,
            payload_modifier=partial(
                _limit_position,
                limit=const.MATRIX_WIDTH,
                attr="width",
            ),
        ),
    ] = 0

    y_pos: Annotated[
        int,
        TransitionableParameterSetting(
            minimum=0,
            maximum=const.MATRIX_HEIGHT - 1,
            transition_rate=0.01,
            payload_modifier=partial(
                _limit_position,
                limit=const.MATRIX_HEIGHT,
                attr="height",
            ),
        ),
    ] = 0

    id_override: str | None = None
    persistent: bool = field(default=False)
    is_sleeping: bool = field(default=False, init=False, repr=False)
    canvas_count: int | None = field(init=False)

    active: bool = field(init=False, default=False)
    colormap: GridView = field(init=False, repr=False)
    pixels: GridView = field(init=False, repr=False)

    stop_reason: StopType | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Add the content to the registry."""
        if self.id in _CONTENT_STORE:
            raise ValueError(f"Content with ID `{self.id}` already exists")

        _CONTENT_STORE[self.id] = self

    @classmethod
    def get(cls, content_id: str) -> ContentBase[ContentType]:
        """Get a content model by its ID."""
        return _CONTENT_STORE[content_id]

    @abstractmethod
    def get_content(self) -> ContentType:
        """Convert the array to an image."""

    @final
    def chain_generators(self) -> itertools.chain[None]:
        """Chain the generators of the content."""
        chain: list[Iterator[None]] = []

        if (setup := self.setup()) is not None:
            chain.append(setup)

        chain.append(iter(self))

        if (teardown := self.teardown()) is not None:
            chain.append(teardown)

        return itertools.chain(*chain)

    def setup(self) -> Generator[None, None, None] | None:  # noqa: PLR6301
        """Perform any necessary setup."""
        return None

    def teardown(self) -> Generator[None, None, None] | None:  # noqa: PLR6301
        """Perform any necessary cleanup."""
        return None

    @final
    def stop(self, stop_type: StopType, /) -> None:
        """Stop the content immediately."""
        self.active = False
        self.stop_reason = stop_type

        LOGGER.info("Stopped content with ID `%s`: %r", self.id, stop_type)

    @final
    def validate_setup(self) -> None:
        """Validate that the content has been set up correctly."""
        if self.id not in _CONTENT_STORE:
            raise ValueError(f"Content with ID `{self.id}` not found")

    @property
    def id(self) -> str:
        """Return the ID of the content."""
        return self.id_override or camel_to_kebab_case(self.__class__.__name__)

    @cached_property
    def is_small(self) -> bool:
        """Return whether the content is smaller than the matrix."""
        return self.shape < const.MATRIX_SHAPE

    @property
    def mqtt_attributes(self) -> str:
        """Return extra attributes for the MQTT message."""
        return dumps(self, default=self._json_encode)

    @property
    def position(self) -> tuple[int, int]:
        """Return the position of the content."""
        return self.x_pos, self.y_pos

    @cached_property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the content."""
        return self.height, self.width

    @abstractmethod
    def __iter__(self) -> Iterator[None]:
        """Iterate over the frames."""

    @final
    @staticmethod
    def _json_encode(obj: Any) -> Any:  # noqa: PLR0911,C901
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

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        try:
            return dumps(obj)
        except TypeError:
            if not callable(obj):
                LOGGER.error("Could not serialize object (%s): %r", obj, obj)  # noqa: TRY400
                if not const.IS_PI:
                    raise
            return None

    def __gt__(self, other: ContentBase[Any]) -> bool:
        """Return whether this content should be de-prioritized over another."""
        return (self.canvas_count or math.inf) > (other.canvas_count or math.inf)

    def __lt__(self, other: ContentBase[Any]) -> bool:
        """Return whether this content should be prioritized over another."""
        return (self.canvas_count or math.inf) < (other.canvas_count or math.inf)


@dataclass(kw_only=True, slots=True)
class PreDefinedContent(ContentBase[mtrx.Canvas], ABC):
    """Base class for content for which all frames are already known."""

    canvas_count: int = field(init=False)
    canvases: tuple[mtrx.Canvas, ...] = field(init=False, repr=False)

    canvas_iter: Iterator[mtrx.Canvas] = field(init=False, repr=False)

    @abstractmethod
    def generate_canvases(
        self,
        new_canvas: Callable[[Image.Image | None], mtrx.Canvas],
    ) -> None:
        """Generate the canvases for the content."""

    def get_content(self) -> mtrx.Canvas:
        """Return the image representation of the content."""
        try:
            return next(self.canvas_iter)
        except (AttributeError, StopIteration):
            self.canvas_iter = iter(self.canvases)
            return next(self.canvas_iter)


__all__ = ["ContentBase", "GridView", "PreDefinedContent", "StateBase"]
