"""Display the time."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Final, cast

import numpy as np
from content.base import GridView  # noqa: TC002
from utils import const
from wg_utilities.loggers import get_streaming_logger

from .dynamic_content import DynamicContent
from .setting import ParameterSetting

if TYPE_CHECKING:
    from collections.abc import Generator

LOGGER = get_streaming_logger(__name__)


class Symbol:
    """Symbols for the clock."""

    PAD_PATTERN: Final = re.compile(r"[\w]{2}")
    WIDTH: Final[int] = 3
    HEIGHT: Final[int] = 5

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
    def get(v: str, scale: tuple[int, int] = (2, 2)) -> GridView:
        """Gt an array representation of the symbol."""
        if v == ":":
            symbol = Symbol.COLON
        elif v == " ":
            symbol = Symbol.PADDING
        else:
            symbol = cast("GridView", getattr(Symbol, f"N{v}"))

        return np.kron(symbol, np.ones(scale, dtype=np.int_))


@dataclass(kw_only=True, slots=True)
class Clock(DynamicContent):
    """Display the time."""

    colormap: GridView = field(
        default_factory=lambda: np.array(
            [(0, 0, 0, 0), (255, 255, 255, 255)],
            dtype=np.uint8,
        ),
        init=False,
        repr=False,
    )

    scale: Annotated[
        int,
        ParameterSetting(
            minimum=1,
            maximum=2,
            invoke_settings_callback=True,
            icon="mdi:relative-scale",
            unit_of_measurement="",
        ),
    ] = 2

    def __post_init__(self) -> None:
        """Initialize the clock."""
        DynamicContent.__post_init__(self)

        self.setting_update_callback()

        self.x_pos = int((const.MATRIX_WIDTH - self.width) / 2)
        self.y_pos = int((const.MATRIX_HEIGHT - self.height) / 2)

    def refresh_content(self) -> Generator[None, None, None]:
        """Refreshes the content."""
        prev_str = ""
        while self.active:
            now_str = self.now_str()
            now_str_len = len(now_str)

            if now_str == prev_str:
                yield
                continue

            arrays = []
            for i, c in enumerate(now_str):
                arrays.append(Symbol.get(c, (self.scale, self.scale)))

                if c.isnumeric() and now_str_len > i + 1 and now_str[i + 1].isnumeric():
                    arrays.append(Symbol.get(" ", (self.scale, self.scale)))

            self.pixels[:, :] = np.hstack(arrays)

            prev_str = now_str
            yield

    def setting_update_callback(self, update_setting: str | None = None) -> None:
        """Update the dimensions of the content."""
        _ = update_setting
        now_str = self.now_str()
        self.height = self.scale * Symbol.HEIGHT
        self.width = (
            self.scale * Symbol.WIDTH * len(now_str)
            + len(Symbol.PAD_PATTERN.findall(now_str)) * self.scale
        )
        self.pixels = self.zeros()

        LOGGER.debug("Updated dimensions to %dx%d", self.width, self.height)

    @staticmethod
    def now_str() -> str:
        """Return the current time as a string."""
        return datetime.now().strftime("%H:%M:%S")  # noqa: DTZ005
