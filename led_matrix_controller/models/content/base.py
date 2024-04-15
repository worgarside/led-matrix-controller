"""Base class for content models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import (
    ClassVar,
    Generator,
    final,
)

import numpy as np
from numpy.typing import NDArray
from PIL import Image

_BY_VALUE: dict[int, StateBase] = {}

GridView = NDArray[np.int_]


# TODO can this be an ABC?
class StateBase(Enum):
    """Base class for the states of a cell."""

    def __init__(
        self, value: int, char: str, color: tuple[int, int, int] = (0, 0, 0)
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


@dataclass(kw_only=True, slots=True)
class ContentBase(ABC):
    """Base class for content models."""

    STATE: ClassVar[type[StateBase]]
    HAS_TEARDOWN_SEQUENCE: ClassVar[bool] = False

    height: int
    width: int

    instance_id: str | None = None
    persistent: bool = field(default=False)

    _active: bool = field(init=False, default=False)
    _image_getter: partial[Image.Image] = field(init=False)
    colormap: NDArray[np.uint8] = field(init=False)
    pixels: GridView = field(init=False)

    @abstractmethod
    def teardown(self) -> Generator[None, None, None]:
        """Perform any necessary cleanup."""

    @final
    def stop(self) -> None:
        """Stop the content immediately."""
        self._active = False

    @property
    def image_getter(self) -> partial[Image.Image]:
        """Return the image representation of the content."""
        if not hasattr(self, "_image_getter"):
            self._image_getter = partial(_get_image, self.colormap, self.pixels)

        return self._image_getter

    @property
    @abstractmethod
    def content_id(self) -> str:
        """Return the ID of the content."""

    @property
    def id(self) -> str:
        """Return the ID of the content."""
        return self.instance_id or self.content_id

    @abstractmethod
    def __iter__(self) -> Generator[None, None, None]:
        """Iterate over the frames."""


__all__ = ["StateBase", "ContentBase", "GridView"]
