"""Module for the Setting class."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from logging import DEBUG, getLogger
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Self,
    TypeVar,
    cast,
)

from utils import const
from utils.helpers import to_kebab_case
from utils.mqtt import MqttClient
from wg_utilities.loggers import add_stream_handler

if TYPE_CHECKING:
    from .automaton import Automaton
    from .matrix import Matrix

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
        self,
        *,
        raw_payload: Any,
        strict: bool,
        coerced: Any | Exception,
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

    instance: Automaton | Matrix = field(init=False, repr=False)
    """The Automaton or Matrix instance to which the setting applies."""

    slug: str = field(init=False, repr=False)
    """The field name/slug of the setting."""

    payload_modifier: Callable[[S], S] | None = field(repr=False, default=None)
    """Function to modify the payload before it is validated.

    Allows for scaling, conversion, etc. relative to Home Assistant's inputs.
    """

    target_value: S = field(init=False, repr=False)
    """The target value for a setting.

    Used to enable transition functionality.
    """

    transition_thread: Thread = field(init=False, repr=False)
    """Thread for transitioning values over a non-zero period of time."""

    type_: type[S] = field(init=False, repr=False)
    """The type of the setting's value (e.g. int, float, bool)."""

    minimum: int | float | None = None
    """Inclusive lower bound for this setting."""

    maximum: int | float | None = None
    """Inclusive upper bound for this setting."""

    requires_rule_regeneration: bool = True
    """Whether changing this setting requires the rules to be regenerated."""

    strict: bool = True
    """Apply strict validation to incoming payloads.

    Setting this to False will allow type coercion, "soft" bounds
    """

    fp_precision: int = 6
    """The number of decimal places to round floats to during processing."""

    matrix: Matrix = field(init=False, repr=False)

    _mqtt_client: ClassVar[MqttClient]
    """The MQTT client to use for this setting."""

    def __post_init__(self) -> None:
        if not hasattr(self.__class__, "_mqtt_client"):
            self.__class__._mqtt_client = MqttClient()

        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise InvalidSettingError(
                f"The 'min' value ({self.minimum}) cannot be greater than the 'max' value ({self.maximum}).",
            )

        if isinstance(self.maximum, float):
            self.maximum = round(self.maximum, self.fp_precision)

        if isinstance(self.minimum, float):
            self.minimum = round(self.minimum, self.fp_precision)

    def setup(
        self,
        *,
        field_name: str,
        automaton: Automaton | Matrix,
        type_: type[S],
    ) -> Self:
        """Set up the setting.

        Args:
            field_name: The field name/slug of the setting.
            automaton: The automaton to which the setting applies.
            type_: The type of the setting's value (e.g. int, float, bool).
        """
        self.slug = field_name
        self.instance = automaton
        self.type_ = type_

        if self.type_ in {int, float}:
            if self.type_ is int:
                self.fp_precision = 0

            if self.maximum:
                self.maximum = round(self.maximum, self.fp_precision)
            if self.minimum:
                self.minimum = round(self.minimum, self.fp_precision)

        self.instance.mqtt_client.add_topic_callback(self.mqtt_topic, self.on_message)

        return self

    def on_message(self, raw_payload: S) -> None:
        """Handle an MQTT message.

        Args:
            raw_payload: The decoded payload from the MQTT message
        """
        try:
            payload = self.validate_payload(raw_payload)
        except InvalidPayloadError:
            LOGGER.exception("Invalid payload: %r", raw_payload)
            return
        except Exception:
            LOGGER.exception("An unexpected error occurred while validating the payload")
            return

        if self.payload_modifier:
            LOGGER.debug("Applying payload modifier to %r", payload)

            try:
                payload = self.validate_payload(self.payload_modifier(payload))
            except InvalidPayloadError:
                LOGGER.exception("Invalid payload modifier output: %r", raw_payload)
                return
            except Exception:
                LOGGER.exception(
                    "An unexpected error occurred while modifying/validating the payload",
                )
                return

        # Only transition if the automaton is currently displaying and a transition rate is set
        if self.instance.active and (self.transition_rate or 0) > 0:
            LOGGER.debug("Set target value to %r", payload)
            self.target_value = payload

            if not (
                hasattr(self, "transition_thread") and self.transition_thread.is_alive()
            ):
                self.transition_thread = Thread(target=self._transition_worker)
                self.transition_thread.start()
                LOGGER.debug("Started transition thread")
            else:
                LOGGER.debug("Transition thread already running")
        elif payload != self.value:
            LOGGER.info("Set value to %r", payload)
            self.value = payload
        else:
            LOGGER.debug("Value unchanged: %r", payload)

    def validate_payload(self, raw_payload: S) -> S:
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

    def _coerce_and_format(self, payload: Any) -> S:
        coerced: S = payload if isinstance(payload, self.type_) else self.type_(payload)

        # Check not out of bounds
        if isinstance(coerced, float | int):
            if self.maximum is not None and coerced > self.maximum:
                coerced = self.type_(self.maximum)

            if self.minimum is not None and coerced < self.minimum:
                coerced = self.type_(self.minimum)

        if isinstance(coerced, float):
            coerced = round(coerced, self.fp_precision)

        return coerced

    def _transition_worker(self) -> None:
        LOGGER.info("Transitioning %s to %s", self.slug, self.target_value)

        tick_condition = self.matrix.tick_condition
        tick_condition.acquire()

        cc = self.instance.current_content

        while (current_value := self.type_(self.value)) != self.target_value:
            transition_amount = round(
                min(self.transition_rate, abs(current_value - self.target_value)),
                self.fp_precision,
            ) * (1 if self.target_value > current_value else -1)

            self.value = round(current_value + transition_amount, self.fp_precision)

            if not (cc and cc.is_sleeping):
                # If the content is sleeping, then it isn't yielding, so the canvas isn't being
                # swapped, and that means that no ticks are happening...
                # So only wait for a tick notification if the content is not sleeping!
                tick_condition.wait()

        tick_condition.release()

        LOGGER.info(
            'Transition complete: %s("%s").%s = %s',
            self.instance.__class__.__name__,
            self.instance.id,
            self.slug,
            self.value,
        )

    _mqtt_topic: str = field(init=False, repr=False)

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
                    self.instance.id,
                    self.setting_type,
                    self.slug,
                ),
            )

        return self._mqtt_topic

    @property
    def value(self) -> S:
        """Get the setting's value from the automaton's attribute."""
        return cast(S, getattr(self.instance, self.slug))

    @value.setter
    def value(self, value: S) -> None:
        """Set the setting's value in the automaton's attribute."""
        setattr(self.instance, self.slug, value)

        if self.requires_rule_regeneration:
            # TODO could go in a separate thread?
            self.instance.generate_frame_rulesets(update_setting=self.slug)


@dataclass(kw_only=True, slots=True)
class FrequencySetting(Setting[int]):
    """Set a rule's frequency."""

    setting_type: Literal[SettingType.FREQUENCY] = field(
        init=False,
        default=SettingType.FREQUENCY,
    )
    type_: type[int] = field(init=False, repr=False, default=int)
    min: Literal[0] = field(init=False, default=0)


@dataclass(kw_only=True, slots=True)
class ParameterSetting(Setting[S]):
    """Set a parameter for a rule."""

    setting_type: Literal[SettingType.PARAMETER] = field(
        init=False,
        default=SettingType.PARAMETER,
    )


__all__ = ["FrequencySetting", "ParameterSetting"]
