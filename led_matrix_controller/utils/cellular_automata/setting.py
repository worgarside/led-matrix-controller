"""Module for the Setting class."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from json import JSONDecodeError, loads
from logging import DEBUG, getLogger
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

from utils import const
from utils.mqtt import MQTT_CLIENT
from wg_utilities.loggers import add_stream_handler

if TYPE_CHECKING:
    from paho.mqtt.client import Client, MQTTMessage
    from utils.cellular_automata.ca import Grid

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
add_stream_handler(LOGGER)


T = TypeVar("T")


class SettingType(StrEnum):
    FREQUENCY = auto()
    PARAMETER = auto()


@dataclass(kw_only=True, slots=True)
class Setting(Generic[T]):
    """Class for a setting that can be controlled via MQTT."""

    setting_type: SettingType

    grid: Grid = field(init=False)
    settings_index: int = field(init=False)
    slug: str = field(init=False)
    topic: str = field(init=False)
    type_: type[T] = field(init=False)

    def callback(self, payload: T) -> None:
        setattr(self.grid, self.slug, payload)

    def get_value_from_grid(self) -> T:
        return cast(T, getattr(self.grid, self.slug))

    def setup(self, index: int, field_name: str, grid: Grid, type_: type[T]) -> None:
        """Set up the setting."""
        self.settings_index = index
        self.slug = field_name
        self.topic = f"/{const.HOSTNAME}/{grid.id}/{self.setting_type}/{self.slug.replace('_', '-')}"
        self.grid = grid
        self.type_ = type_

        MQTT_CLIENT.subscribe(self.topic)
        MQTT_CLIENT.message_callback_add(self.topic, self.on_message)
        LOGGER.info("Subscribed to topic: %s", self.topic)

    def on_message(self, _: Client, __: Any, message: MQTTMessage) -> None:
        """Handle an MQTT message.

        Args:
            _ (Client): the client instance for this callback
            __ (Any): the private user data as set in Client() or userdata_set()
            message (MQTTMessage): the message object from the MQTT subscription
        """
        try:
            payload = loads(message.payload.decode())
        except JSONDecodeError:
            LOGGER.exception("Failed to decode payload: %s", message.payload)
            return

        if not isinstance(payload, self.type_):
            LOGGER.error(
                "Payload %s is not of type %s",
                payload,
                self.type_,
            )
            return

        LOGGER.debug("Received payload: %s", payload)

        self.callback(payload)


@dataclass(slots=True)
class FrequencySetting(Setting[int]):
    """Set a rule's frequency."""

    setting_type: Literal[SettingType.FREQUENCY] = SettingType.FREQUENCY
    type_: type[int] = int

    def callback(self, payload: T) -> None:
        """Set the rule's frequency and re-generate the rules loop."""
        setattr(self.grid, self.slug, payload)

        self.grid.generate_rules_loop()


@dataclass(slots=True)
class ParameterSetting(Setting[T]):
    """Set a parameter for a rule."""

    setting_type: Literal[SettingType.PARAMETER] = SettingType.PARAMETER


__all__ = ["FrequencySetting", "ParameterSetting"]
