"""Module for the Setting class."""

from __future__ import annotations

from dataclasses import dataclass, field
from json import JSONDecodeError, loads
from logging import DEBUG, getLogger
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

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


@dataclass(kw_only=True)
class Setting(Generic[T]):
    """Class for a setting that can be controlled via MQTT."""

    type_: type[T] = field(init=False)
    topic: str = field(init=False)

    callback: Callable[[Any], None] = field(init=False)

    def setup(self, field_name: str, grid: Grid, type_: type[T]) -> None:
        """Set up the setting."""
        self.id = field_name.replace("_", "-")
        self.topic = f"/{const.HOSTNAME}/{grid.ID}/{self.id}"
        self.callback = lambda value: setattr(grid, field_name, value)
        self.type_ = type_

        MQTT_CLIENT.subscribe(self.topic)
        MQTT_CLIENT.message_callback_add(self.topic, self.on_message)

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
