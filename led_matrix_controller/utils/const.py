"""Constant values."""

from __future__ import annotations

import re
from os import environ, getenv
from pathlib import Path
from socket import gethostname
from sys import platform
from typing import Final

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

HA_LED_MATRIX_PAYLOAD_TOPIC: Final[str] = "/homeassistant/led_matrix/display"
HA_LED_MATRIX_BRIGHTNESS_TOPIC: Final[str] = "/homeassistant/led_matrix/brightness"
HA_LED_MATRIX_STATE_TOPIC: Final[str] = "/homeassistant/led_matrix/state"
HA_MTRXPI_CONTENT_TOPIC: Final[str] = "/homeassistant/mtrxpi/content"
HA_FORCE_UPDATE_TOPIC: Final[str] = "/home-assistant/script/mtrxpi_update_display/run"

FONT_WIDTH: Final[int] = 5
FONT_HEIGHT: Final[int] = 7
SCROLL_INCREMENT_DISTANCE: Final[int] = 2 * FONT_WIDTH


BOOLEANS: Final[np.typing.NDArray[np.bool_]] = np.array([False, True], dtype=np.bool_)
RNG = np.random.default_rng(830003040)

REPO_PATH: Final[Path] = Path(__file__).parents[2]
ASSETS_DIRECTORY: Final[Path] = REPO_PATH / "assets"

TICKS_PER_SECOND: Final[int] = 100
TICK_LENGTH: Final[float] = 1 / TICKS_PER_SECOND

EMPTY_IMAGE: Final[Image.Image] = Image.new("RGB", (64, 64), (0, 0, 0))
