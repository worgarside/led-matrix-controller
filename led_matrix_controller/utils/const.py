"""Constant values."""

from __future__ import annotations

import re
from os import environ, getenv
from pathlib import Path
from socket import gethostname
from typing import Final

import numpy as np

MQTT_USERNAME: Final[str] = environ["MQTT_USERNAME"]
MQTT_PASSWORD: Final[str] = environ["MQTT_PASSWORD"]

DEBUG_MODE: Final[bool] = bool(int(getenv("DEBUG_MODE", "0")))

MQTT_HOST: Final[str] = getenv("MQTT_HOST", "http://homeassistant.local")


HOSTNAME: Final[str] = re.sub(r"[^a-z0-9]", "-", gethostname().lower())

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

TICKS_PER_SECOND: Final[int] = 100
TICK_LENGTH: Final[float] = 1 / TICKS_PER_SECOND
