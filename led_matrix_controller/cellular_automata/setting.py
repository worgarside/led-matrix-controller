"""Module for the Setting class."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from logging import DEBUG, getLogger
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    cast,
)

from utils import const
from utils.helpers import to_kebab_case
from utils.mqtt import MqttClient
from wg_utilities.loggers import add_stream_handler

if TYPE_CHECKING:
    from cellular_automata.automaton import Automaton
    from models.matrix import Matrix

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
add_stream_handler(LOGGER)


S = TypeVar("S", int, float)


class SettingType(StrEnum):
    FREQUENCY = auto()
    PARAMETER = auto()


class InvalidPayloadError(ValueError):
    """Raised when an invalid MQTT payload is received."""

    def __init__(
        self, *, raw_payload: Any, strict: bool, coerced: Any | Exception
    ) -> None:
        super().__init__(f"Invalid payload with {strict=}: {raw_payload=}, {coerced=}")


class InvalidSettingError(ValueError):
    """Raised when an invalid setting is created."""


@dataclass(kw_only=True, slots=True)
class Setting(Generic[S]):
    """Class for a setting that can be controlled via MQTT."""

    setting_type: SettingType
    """The type of setting, i.e. frequency or parameter."""

    transition_rate: S = field(default=None)  # type: ignore[assignment]
    """The rate at which the setting will be changed per tick/frame.

    If None (or 0), the setting modification will be applied immediately.
    """

    automaton: Automaton = field(init=False)
    """The automaton to which the setting applies."""

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

    requires_rule_regeneration: bool = True
    """Whether changing this setting requires the rules to be regenerated."""

    strict: bool = True
    """Apply strict validation to incoming payloads.

    Setting this to False will allow type coercion, "soft" bounds
    """

    fp_precision: int = 6
    """The number of decimal places to round floats to during processing."""

    matrix: Matrix = field(init=False)

    _disable_outgoing_mqtt_updates: bool = False
    """Flag to disable outgoing MQTT updates when transitioning."""

    _mqtt_client: ClassVar[MqttClient]
    """The MQTT client to use for this setting."""

    _has_received_first_message: bool = False
    """Flag to indicate that the first message has been received.

    This is used to prevent the setting from being updated with the default
    value before the first message is received.
    """

    def __post_init__(self) -> None:
        if not hasattr(self.__class__, "_mqtt_client"):
            self.__class__._mqtt_client = MqttClient.CLIENT

        if self.min is not None and self.max is not None and self.min > self.max:
            raise InvalidSettingError(
                f"The 'min' value ({self.min}) cannot be greater than the 'max' value ({self.max})."
            )

        if isinstance(self.max, float):
            self.max = round(self.max, self.fp_precision)

        if isinstance(self.min, float):
            self.min = round(self.min, self.fp_precision)

    def _transition_worker(self) -> None:
        LOGGER.info("Transitioning %s to %s", self.slug, self.target_value)

        tick_condition = self.matrix.tick_condition
        tick_condition.acquire()

        self._disable_outgoing_mqtt_updates = True
        while (current_value := self.type_(self.value)) != self.target_value:
            transition_amount = round(
                min(self.transition_rate, abs(current_value - self.target_value)),
                self.fp_precision,
            )
            LOGGER.debug(
                "%s: %s + %s => %s",
                self.slug,
                current_value,
                transition_amount,
                self.target_value,
            )

            self.value = round(
                (
                    current_value + transition_amount
                    if current_value < self.target_value
                    else current_value - transition_amount
                ),
                self.fp_precision,
            )

            tick_condition.wait()

        tick_condition.release()
        self._disable_outgoing_mqtt_updates = False

    def setup(
        self,
        *,
        field_name: str,
        automaton: Automaton,
        type_: type[S],
    ) -> None:
        """Set up the setting."""
        self.slug = field_name
        self.automaton = automaton
        self.type_ = type_

        if self.type_ in {int, float}:
            if self.type_ is int:
                self.fp_precision = 0

            if self.max:
                self.max = round(self.max, self.fp_precision)
            if self.min:
                self.min = round(self.min, self.fp_precision)

        self.automaton.mqtt_client.add_topic_callback(self.mqtt_topic, self.on_message)

    def on_message(self, raw_payload: S) -> None:
        """Handle an MQTT message.

        Args:
            raw_payload: The decoded payload from the MQTT message
        """
        try:
            payload = self.validate(raw_payload)
        except InvalidPayloadError:
            LOGGER.exception("Invalid payload")
            return
        except Exception:
            LOGGER.exception("An unexpected error occurred while validating the payload")
            return

        if (
            self.automaton._active
            and self._has_received_first_message
            and (self.transition_rate or 0) > 0
        ):
            self.target_value = payload

            if not (
                hasattr(self, "transition_thread") and self.transition_thread.is_alive()
            ):
                self.transition_thread = Thread(target=self._transition_worker)
                self.transition_thread.start()
        elif payload != self.value:
            self.value = payload

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

    def send_value_update_message(self) -> None:
        """Send a message with the current value of the setting.

        Disabled if outgoing updates are disabled or if the first message has not been received (i.e. the setting
        is still at its default value). This is to prevent the default value from being sent to the MQTT broker.
        """
        if self._disable_outgoing_mqtt_updates or not self._has_received_first_message:
            self._has_received_first_message = True
            return

        self.matrix.mqtt_client.publish(
            self.mqtt_topic,
            self.value,
        )

    def validate(self, raw_payload: S) -> S:
        """Check that the incoming payload is a valid value for this Setting."""
        if isinstance(raw_payload, self.type_) and self.type_ not in {int, float}:
            # Non-numeric type, decoded and parsed correctly
            return raw_payload

        try:
            coerced_and_formatted = self._coerce_and_format(raw_payload)
        except Exception as err:
            raise InvalidPayloadError(
                raw_payload=raw_payload,
                strict=self.strict,
                coerced=err,
            ) from err

        if self.strict and coerced_and_formatted != raw_payload:
            raise InvalidPayloadError(
                raw_payload=raw_payload,
                strict=self.strict,
                coerced=coerced_and_formatted,
            )

        return coerced_and_formatted

    _mqtt_topic: str = field(init=False)

    @property
    def mqtt_topic(self) -> str:
        """The MQTT topic that this setting is subscribed to.

        Auto-generated, in the form `/<hostname>/<automaton ID>/<setting type>/<kebab-case-slug>`

        e.g. /mtrxpi/raining-grid/frequency/rain-speed
        """

        if not hasattr(self, "_mqtt_topic"):
            self._mqtt_topic = "/" + "/".join(
                to_kebab_case(
                    const.HOSTNAME,
                    self.automaton.id,
                    self.setting_type,
                    self.slug,
                )
            )

        return self._mqtt_topic

    @property
    def value(self) -> S:
        """Get the setting's value from the automaton's attribute."""
        return cast(S, getattr(self.automaton, self.slug))

    @value.setter
    def value(self, value: S) -> None:
        """Set the setting's value in the automaton's attribute."""
        setattr(self.automaton, self.slug, value)
        LOGGER.info("Setting %s:%s to %s", self.automaton.id, self.slug, value)

        if self.requires_rule_regeneration:
            # TODO could go in a separate thread?
            self.automaton.generate_frame_rulesets(update_setting=self.slug)

        self.send_value_update_message()


@dataclass(kw_only=True, slots=True)
class FrequencySetting(Setting[int]):
    """Set a rule's frequency."""

    setting_type: Literal[SettingType.FREQUENCY] = SettingType.FREQUENCY
    type_: type[int] = int
    min: Literal[0] = 0


@dataclass(kw_only=True, slots=True)
class ParameterSetting(Setting[S]):
    """Set a parameter for a rule."""

    setting_type: Literal[SettingType.PARAMETER] = SettingType.PARAMETER


__all__ = ["FrequencySetting", "ParameterSetting"]
