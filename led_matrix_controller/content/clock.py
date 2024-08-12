"""Display the time."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import sleep
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
    def get(_v: str) -> NDArray[np.uint8]:
        """Gt an array representation of the symbol."""
        if _v == ":":
            return Symbol.COLON

        if _v == " ":
            return Symbol.PADDING

        name = f"N{_v}"

        return cast(NDArray[np.uint8], getattr(Symbol, name))


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
        while self.active:
            now_str = datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005

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

            scaled = np.kron(stacked, np.ones((2, 2)))

            row_idx = 2
            col_idx = 5
            rows, cols = scaled.shape

            self.pixels[row_idx : row_idx + rows, col_idx : col_idx + cols] = scaled

            yield

            sleep(0.5)
