"""Base class for content models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from typing import (
    ClassVar,
    Generator,
)

import numpy as np
from numpy.typing import NDArray  # noqa: TCH002

_BY_VALUE: dict[int, StateBase] = {}


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
    @lru_cache(maxsize=1, typed=True)
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


class ContentBase(ABC):
    """Base class for content models."""

    STATE: ClassVar[type[StateBase]]

    colormap: NDArray[np.uint8]
    grid: NDArray[np.int_]

    @abstractmethod
    def __iter__(self) -> Generator[None, None, None]:
        """Iterate over the frames."""
