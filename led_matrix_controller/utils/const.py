"""Constant values."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from os import environ, getenv
from pathlib import Path
from socket import gethostname
from sys import platform
from typing import Any, Final

import numpy as np
from PIL import Image

DEBUG_MODE: Final[bool] = bool(int(getenv("DEBUG_MODE", "0")))

MQTT_USERNAME: Final[str] = environ["MQTT_USERNAME"]
MQTT_PASSWORD: Final[str] = environ["MQTT_PASSWORD"]

MQTT_HOST: Final[str] = getenv("MQTT_HOST", "http://homeassistant.local")

HOSTNAME: Final[str] = getenv(
    "HOSTNAME_OVERRIDE",
    re.sub(r"[^a-z0-9]", "-", gethostname().lower()),
)

IS_PI = gethostname().lower() == "mtrxpi" and platform != "darwin"

FONT_WIDTH: Final[int] = 5
FONT_HEIGHT: Final[int] = 7
SCROLL_INCREMENT_DISTANCE: Final[int] = 2 * FONT_WIDTH

BOOLEANS: Final[np.typing.NDArray[np.bool_]] = np.array([False, True], dtype=np.bool_)
RNG = np.random.default_rng(int(getenv("RNG_SEED", datetime.now(UTC).timestamp())))

REPO_PATH: Final[Path] = Path(__file__).parents[2]
ASSETS_DIRECTORY: Final[Path] = REPO_PATH / "assets"

TICKS_PER_SECOND: Final[int] = int(getenv("TICKS_PER_SECOND", "100"))  # ticks
TICK_LENGTH: Final[float] = 1 / TICKS_PER_SECOND  # seconds

MATRIX_HEIGHT: Final[int] = 64
MATRIX_WIDTH: Final[int] = 64

MATRIX_SHAPE: Final[tuple[int, int]] = (MATRIX_HEIGHT, MATRIX_WIDTH)

EMPTY_IMAGE: Final[Image.Image] = Image.new("RGB", MATRIX_SHAPE, (0, 0, 0))

MAX_PRIORITY: Final[float] = 1e10


def seconds_to_ticks(seconds: float, *_: Any) -> int:
    """Convert seconds to ticks."""
    return int(seconds * TICKS_PER_SECOND)


def ticks_to_seconds(ticks: int) -> float:
    """Convert ticks to seconds."""
    return ticks * TICK_LENGTH
