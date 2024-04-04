"""Module for the Setting class."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from json import loads
from logging import DEBUG, getLogger
from threading import Thread
from time import sleep
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    TypeVar,
    cast,
)

from utils import const
from utils.mqtt import MQTT_CLIENT
from wg_utilities.loggers import add_stream_handler

if TYPE_CHECKING:
    from paho.mqtt.client import Client, MQTTMessage
    from utils.cellular_automata.grid import Grid

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
add_stream_handler(LOGGER)


S = TypeVar("S", int, float)


class SettingType(StrEnum):
    FREQUENCY = auto()
    PARAMETER = auto()


class InvalidPayloadError(ValueError):
    """Raised when an invalid MQTT payload is received."""

    def __init__(self, *, decoded: Any, strict: bool, coerced: Any | Exception) -> None:
        super().__init__(f"Invalid payload with {strict=}: {decoded=}, {coerced=}")


class InvalidSettingError(ValueError):
    """Raised when an invalid setting is created."""


@dataclass(kw_only=True, slots=True)
class Setting(Generic[S]):
    """Class for a setting that can be controlled via MQTT."""

    setting_type: SettingType
    """The type of setting, i.e. frequency or parameter."""

    transition_rate: tuple[S, float] = field(default=None)  # type: ignore[assignment]
    """The rate at which the setting can be changed. Only applies to numerical settings.

    If None (or 0), the setting modification will be applied immediately. In the form (X, Y), the setting
    will be de/incremented by X every Y seconds until it reaches the target value.
    """

    callback: Callable[[S], None] = field(init=False)
    """Callback called with any values received via MQTT.

    This isn't directly defined as a method because it can vary by instance (e.g. transitioning) as well as
    per subclass.
    """

    grid: Grid = field(init=False)
    """The grid to which the setting applies."""

    slug: str = field(init=False)
    """The field name/slug of the setting."""

    target_value: S = field(init=False)
    """The target value for a setting.

    Used to enable transition functionality.
    """

    transition_thread: Thread = field(init=False)
    """Thread for transitioning values over a non-zero period of time."""

    type_: type[S] = field(init=False)
    """The type of the setting's value (e.g. int, float, bool)."""

    min: int | float | None = None
    """Inclusive lower bound for this setting."""

    max: int | float | None = None
    """Inclusive upper bound for this setting."""

    strict: bool = True
    """Apply strict validation to incoming payloads.

    Setting this to False will allow type coercion, "soft" bounds
    """

    fp_precision: int = 6
    """The number of decimal places to round floats to during processing."""

    def __post_init__(self) -> None:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise InvalidSettingError(
                f"The 'min' value ({self.min}) cannot be greater than the 'max' value ({self.max})."
            )

        if isinstance(self.max, float):
            self.max = round(self.max, self.fp_precision)

        if isinstance(self.min, float):
            self.min = round(self.min, self.fp_precision)

    def get_value_from_grid(self) -> S:
        """Get the setting's value from the grid's attribute."""
        return cast(S, getattr(self.grid, self.slug))

    def set_value_in_grid(self, value: S) -> None:
        setattr(self.grid, self.slug, value)

    def transition_value_in_grid(self, value: S) -> None:
        self.target_value = value

        if not (hasattr(self, "transition_thread") and self.transition_thread.is_alive()):
            self.transition_thread = Thread(target=self._transition_value)
            self.transition_thread.start()

    def _transition_value(self) -> None:
        LOGGER.debug("Transitioning %s to %s", self.slug, self.target_value)

        while (
            current_value := self.type_(self.get_value_from_grid())
        ) != self.target_value:
            transition_amount = round(
                min(self.transition_rate[0], abs(current_value - self.target_value)),
                self.fp_precision,
            )
            LOGGER.debug(
                "Transitioning %s from %s to %s by %s",
                self.slug,
                current_value,
                self.target_value,
                transition_amount,
            )

            self.set_value_in_grid(
                round(
                    (
                        current_value + transition_amount
                        if current_value < self.target_value
                        else current_value - transition_amount
                    ),
                    self.fp_precision,
                )
            )

            sleep(self.transition_rate[1])

    def setup(
        self,
        *,
        field_name: str,
        grid: Grid,
        type_: type[S],
    ) -> None:
        """Set up the setting."""
        self.slug = field_name
        self.grid = grid
        self.type_ = type_
        self.callback = self.set_value_in_grid

        if self.type_ in {int, float}:
            if self.transition_rate:
                self.callback = self.transition_value_in_grid

            if self.type_ is int:
                self.fp_precision = 0

            if self.max:
                self.max = round(self.max, self.fp_precision)
            if self.min:
                self.min = round(self.min, self.fp_precision)

        MQTT_CLIENT.subscribe(self.mqtt_topic)
        MQTT_CLIENT.message_callback_add(self.mqtt_topic, self.on_message)
        LOGGER.info("Subscribed to topic: %s", self.mqtt_topic)

    def on_message(self, _: Client, __: Any, message: MQTTMessage) -> None:
        """Handle an MQTT message.

        Args:
            _ (Client): the client instance for this callback
            __ (Any): the private user data as set in Client() or userdata_set()
            message (MQTTMessage): the message object from the MQTT subscription
        """
        try:
            payload = self.validate(message.payload)
        except InvalidPayloadError:
            LOGGER.exception("Invalid payload")
            return
        except Exception:
            LOGGER.exception("An unexpected error occurred while validating the payload")
            return

        self.callback(payload)

    def _coerce_and_format(self, payload: Any) -> S:
        coerced: S = payload if isinstance(payload, self.type_) else self.type_(payload)

        # Check not out of bounds
        if isinstance(coerced, float | int):
            if self.max is not None and coerced > self.max:
                coerced = self.type_(self.max)

            if self.min is not None and coerced < self.min:
                coerced = self.type_(self.min)

        if isinstance(coerced, float):
            coerced = round(coerced, self.fp_precision)

        return coerced

    def validate(self, raw_payload: bytes | bytearray) -> S:
        """Check that the incoming payload is a valid value for this Setting."""
        decoded = loads(raw_payload.decode())

        if isinstance(decoded, self.type_) and self.type_ not in {int, float}:
            # Non-numeric type, decoded and parsed correctly
            return decoded

        try:
            coerced_and_formatted = self._coerce_and_format(decoded)
        except Exception as err:
            raise InvalidPayloadError(
                decoded=decoded,
                strict=self.strict,
                coerced=err,
            ) from err

        if self.strict and coerced_and_formatted != decoded:
            raise InvalidPayloadError(
                decoded=decoded,
                strict=self.strict,
                coerced=coerced_and_formatted,
            )

        return coerced_and_formatted

    @property
    def mqtt_topic(self) -> str:
        """The MQTT topic that this setting is subscribed to.

        Auto-generated, in the form `/<hostname>/<grid ID>/<setting type>/<kebab-case-slug>`

        e.g. /mtrxpi/raining-grid/frequency/rain-speed
        """
        return f"/{const.HOSTNAME}/{self.grid.id}/{self.setting_type}/{self.slug.replace('_', '-')}"


@dataclass(slots=True)
class FrequencySetting(Setting[int]):
    """Set a rule's frequency."""

    setting_type: Literal[SettingType.FREQUENCY] = SettingType.FREQUENCY
    type_: type[int] = int
    min: Literal[0] = 0

    def set_value_in_grid(self, payload: S) -> None:
        """Set the rule's frequency and re-generate the rules loop."""
        setattr(self.grid, self.slug, payload)

        self.grid.generate_frame_rulesets()


@dataclass(slots=True)
class ParameterSetting(Setting[S]):
    """Set a parameter for a rule."""

    setting_type: Literal[SettingType.PARAMETER] = SettingType.PARAMETER

    def set_value_in_grid(self, value: S) -> None:
        """Set the parameter and re-generate the rules loop."""
        LOGGER.debug("Setting %s to %s", self.slug, value)
        setattr(self.grid, self.slug, value)

        self.grid.generate_frame_rulesets(update_parameter=self.slug)


__all__ = ["FrequencySetting", "ParameterSetting"]
