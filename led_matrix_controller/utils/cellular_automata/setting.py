"""Module for the Setting class."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from json import JSONDecodeError, loads
from logging import DEBUG, getLogger
from threading import Thread
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, cast

from utils import const
from utils.mqtt import MQTT_CLIENT
from wg_utilities.loggers import add_stream_handler

if TYPE_CHECKING:
    from paho.mqtt.client import Client, MQTTMessage
    from utils.cellular_automata.ca import Grid

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
add_stream_handler(LOGGER)


S = TypeVar("S", int, float)


class SettingType(StrEnum):
    FREQUENCY = auto()
    PARAMETER = auto()


@dataclass(kw_only=True, slots=True)
class Setting(Generic[S]):
    """Class for a setting that can be controlled via MQTT."""

    setting_type: SettingType
    transition_rate: tuple[S, float] = field(default=None)  # type: ignore[assignment]
    """The rate at which the setting can be changed. Only applies to numerical settings.

    If None (or 0), the setting modification will be applied immediately. In the form (X, Y), the setting
    will be de/incremented by X every Y seconds until it reaches the target value.
    """

    callback: Callable[[S], None] = field(init=False)
    grid: Grid = field(init=False)
    settings_index: int = field(init=False)
    slug: str = field(init=False)
    topic: str = field(init=False)
    type_: type[S] = field(init=False)

    def get_value_from_grid(self) -> S:
        return cast(S, getattr(self.grid, self.slug))

    def set_value_in_grid(self, value: S) -> None:
        LOGGER.debug("Setting %s to %s", self.slug, value)
        setattr(self.grid, self.slug, value)

    def transition_value_in_grid(self, value: S) -> None:
        transition_thread = Thread(target=self._transition_value, args=(value,))
        transition_thread.start()

    def _transition_value(self, value: S) -> None:
        LOGGER.debug("Transitioning %s to %s", self.slug, value)
        target_value = self.type_(value)

        while (current_value := self.type_(self.get_value_from_grid())) != target_value:
            transition_amount = min(
                self.transition_rate[0], abs(current_value - target_value)
            )
            LOGGER.debug(
                "Transitioning %s from %s to %s by %s",
                self.slug,
                current_value,
                target_value,
                transition_amount,
            )

            self.set_value_in_grid(
                round(
                    current_value + transition_amount
                    if current_value < target_value
                    else current_value - transition_amount,
                    6,
                )
            )

            sleep(self.transition_rate[1])

    def setup(self, *, index: int, field_name: str, grid: Grid, type_: type[S]) -> None:
        """Set up the setting."""
        self.settings_index = index
        self.slug = field_name
        self.topic = f"/{const.HOSTNAME}/{grid.id}/{self.setting_type}/{self.slug.replace('_', '-')}"
        self.grid = grid
        self.type_ = type_

        if self.type_ in {int, float} and self.transition_rate:
            self.callback = self.transition_value_in_grid
        else:
            self.callback = self.set_value_in_grid

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

        LOGGER.debug("INCOMING ON `%s`: %s", self.topic, payload)

        self.callback(payload)


@dataclass(slots=True)
class FrequencySetting(Setting[int]):
    """Set a rule's frequency."""

    setting_type: Literal[SettingType.FREQUENCY] = SettingType.FREQUENCY
    type_: type[int] = int

    def set_value_in_grid(self, payload: S) -> None:
        """Set the rule's frequency and re-generate the rules loop."""
        setattr(self.grid, self.slug, payload)

        self.grid.generate_frame_rulesets()


@dataclass(slots=True)
class ParameterSetting(Setting[S]):
    """Set a parameter for a rule."""

    setting_type: Literal[SettingType.PARAMETER] = SettingType.PARAMETER


__all__ = ["FrequencySetting", "ParameterSetting"]
