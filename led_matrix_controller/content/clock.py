"""Display the time."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Generator, cast

import numpy as np
from numpy.typing import NDArray
from wg_utilities.loggers import get_streaming_logger

from .dynamic_content import DynamicContent

LOGGER = get_streaming_logger(__name__)


class Symbol:
    """Symbols for the clock."""

    N0 = np.array([[0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0]])

    N1 = np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 1]])

    N2 = np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]])

    N3 = np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]])

    N4 = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]])

    N5 = np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]])

    N6 = np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]])

    N7 = np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]])

    N8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]])

    N9 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]])

    COLON = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]])

    PADDING = np.array([[0], [0], [0], [0], [0]])

    @staticmethod
    @lru_cache(maxsize=12)
    def get(_v: str, scale: tuple[int, int] = (2, 2)) -> NDArray[np.uint8]:
        """Gt an array representation of the symbol."""
        if _v == ":":
            symbol = Symbol.COLON
        elif _v == " ":
            symbol = Symbol.PADDING
        else:
            symbol = cast(NDArray[np.uint8], getattr(Symbol, f"N{_v}"))

        return np.kron(symbol, np.ones(scale))


@dataclass(kw_only=True, slots=True)
class Clock(DynamicContent):
    """Display the time."""

    colormap: NDArray[np.uint8] = field(
        default_factory=lambda: np.array([(0, 0, 0), (255, 255, 255)], dtype=np.uint8),
    )

    _values: list[int] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the image getter."""
        DynamicContent.__post_init__(self)

        self.pixels = self.zeros()

    def refresh_content(self) -> Generator[None, None, None]:
        """Refresh the content."""
        prev_str = ""
        while self.active:
            now_str = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005

            if now_str == prev_str:
                yield
                continue

            arrays = []
            for i, c in enumerate(now_str):
                arrays.append(Symbol.get(c))
                try:
                    int(c)
                except ValueError:
                    continue

                try:
                    nxt_chr = now_str[i + 1]
                except IndexError:
                    continue

                if nxt_chr != ":":
                    arrays.append(Symbol.get(" "))

            stacked = np.hstack(arrays)

            row_idx = 2
            col_idx = 5
            rows, cols = stacked.shape

            self.pixels[row_idx : row_idx + rows, col_idx : col_idx + cols] = stacked

            prev_str = now_str
            yield
