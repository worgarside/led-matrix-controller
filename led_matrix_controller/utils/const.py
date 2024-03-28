"""Constant values."""

from __future__ import annotations

from os import environ, getenv
from typing import Final, Literal

import numpy as np

MQTT_USERNAME: Final[str] = environ["MQTT_USERNAME"]
MQTT_PASSWORD: Final[str] = environ["MQTT_PASSWORD"]

MQTT_HOST: Final[str] = getenv("MQTT_HOST", "homeassistant.local")

HOSTNAME: Final[Literal["mtrxpi"]] = "mtrxpi"

HA_LED_MATRIX_PAYLOAD_TOPIC: Final[str] = "/homeassistant/led_matrix/display"
HA_LED_MATRIX_BRIGHTNESS_TOPIC: Final[str] = "/homeassistant/led_matrix/brightness"
HA_LED_MATRIX_STATE_TOPIC: Final[str] = "/homeassistant/led_matrix/state"
HA_MTRXPI_CONTENT_TOPIC: Final[str] = "/homeassistant/mtrxpi/content"
HA_FORCE_UPDATE_TOPIC: Final[str] = "/home-assistant/script/mtrxpi_update_display/run"

FONT_WIDTH: Final[int] = 5
FONT_HEIGHT: Final[int] = 7
SCROLL_INCREMENT_DISTANCE: Final[int] = 2 * FONT_WIDTH


RNG = np.random.default_rng(830003040)
