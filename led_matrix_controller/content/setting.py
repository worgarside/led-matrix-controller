"""Module for the Setting class."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum, auto
from functools import cached_property
from json import dumps
from threading import Thread
from time import sleep
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Mapping,
    Self,
    TypeVar,
    cast,
)

from utils import const
from utils.helpers import to_kebab_case
from utils.mqtt import MqttClient
from wg_utilities.loggers import get_streaming_logger

if TYPE_CHECKING:
    from models.matrix import Matrix

    from .dynamic_content import DynamicContent

LOGGER = get_streaming_logger(__name__)


S = TypeVar("S", dict[str, Any], list[Any], str, int, float, bool)


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


@dataclass(kw_only=True)
class Setting(Generic[S]):
    """Class for a setting that can be controlled via MQTT."""

    setting_type: SettingType
    """The type of setting, i.e. frequency or parameter."""

    instance: DynamicContent | Matrix = field(init=False, repr=False)
    """The Automaton or Matrix instance to which the setting applies."""

    slug: str = field(init=False, repr=False)
    """The field name/slug of the setting."""

    payload_modifier: Callable[[S, DynamicContent | Matrix], S] | None = field(
        repr=False,
        default=None,
    )
    """Function to modify the payload before it is validated.

    Allows for scaling, conversion, etc. relative to Home Assistant's inputs.
    """

    type_: type[S] = field(init=False, repr=False)
    """The type of the setting's value (e.g. int, float, bool)."""

    minimum: int | float | None = None
    """Inclusive lower bound for this setting."""

    maximum: int | float | None = None
    """Inclusive upper bound for this setting."""

    invoke_settings_callback: bool = False
    """Whether changing this setting should invoke a callback on the content instance."""

    strict: bool = True
    """Apply strict validation to incoming payloads.

    Setting this to False will allow type coercion, "soft" bounds
    """

    fp_precision: int = 6
    """The number of decimal places to round floats to during processing."""

    value_callbacks: Mapping[S, Callable[..., Any]] | None = field(
        default=None,
        repr=False,
    )
    """Mapping of values to callbacks, which are invoked when the setting is set to that value."""

    icon: str
    unit_of_measurement: str | None = None
    display_mode: Literal["box", "slider"] = "box"

    matrix: Matrix = field(init=False, repr=False)

    _MQTT_CLIENT: ClassVar[MqttClient]
    """The MQTT client to use for this setting."""

    def __post_init__(self) -> None:
        if not hasattr(self.__class__, "_MQTT_CLIENT"):
            self.__class__._MQTT_CLIENT = MqttClient()  # noqa: SLF001

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

    def _coerce_and_format(self, payload: Any) -> S:
        coerced: S = payload if isinstance(payload, self.type_) else self.type_(payload)

        # Check not out of bounds
        if isinstance(coerced, float | int) and not isinstance(coerced, bool):
            if self.maximum is not None and coerced > self.maximum:
                coerced = self.type_(self.maximum)

            if self.minimum is not None and coerced < self.minimum:
                coerced = self.type_(self.minimum)

        if isinstance(coerced, float):
            coerced = round(coerced, self.fp_precision)  # type: ignore[assignment]

        return coerced

    def _set_value_from_payload(self, payload: S) -> None:
        """Set the value of the setting from the payload."""
        LOGGER.info("Set `%s` value to %s", self.slug, str(payload)[:100])
        self.value = payload

        self.matrix.publish_attributes()

    def on_message(self, raw_payload: S) -> None:  # noqa: C901
        """Handle an MQTT message.

        Args:
            raw_payload: The decoded payload from the MQTT message
        """
        try:
            payload = self.validate_payload(raw_payload)
        except InvalidPayloadError:
            LOGGER.exception("Invalid %s payload: %r", type(raw_payload), raw_payload)
            return
        except Exception:
            LOGGER.exception(
                "An unexpected error occurred while validating the %s payload: %r",
                type(raw_payload),
                raw_payload,
            )
            return

        if self.payload_modifier:
            LOGGER.debug("Applying payload modifier to %r", payload)

            try:
                payload = self._coerce_and_format(
                    self.payload_modifier(payload, self.instance),
                )
            except Exception:
                LOGGER.exception(
                    "An unexpected error occurred while modifying the %s payload %r",
                    type(payload).__name__,
                    payload,
                )
                return

            try:
                payload = self.validate_payload(payload)
            except InvalidPayloadError:
                LOGGER.exception(
                    "Invalid %s payload modifier output: %r",
                    type(payload).__name__,
                    payload,
                )
                return
            except Exception:
                LOGGER.exception(
                    "An unexpected error occurred while validating the modified %s payload: %r",
                    type(payload),
                    payload,
                )
                return

        if payload != self.value:
            self._set_value_from_payload(payload)
        elif payload != raw_payload:
            # e.g. out of bounds value was coerced to min/max
            try:
                mqtt_payload = dumps(payload)
            except TypeError:
                if not (isinstance(payload, list | tuple) and self.slug == "content"):
                    raise

                if (payload_ids := [c.id for c in payload]) == raw_payload:
                    return

                mqtt_payload = dumps(payload_ids)

            self.mqtt_client.publish(
                self.mqtt_topic,
                mqtt_payload,
                retain=True,
            )
        else:
            # Exact same message retrieved
            LOGGER.debug("Value unchanged: %r", payload)

    def setup(
        self,
        *,
        field_name: str,
        instance: DynamicContent | Matrix,
        type_: type[S],
    ) -> Self:
        """Set up the setting.

        Args:
            field_name: The field name/slug of the setting.
            instance: The automaton to which the setting applies.
            type_: The type of the setting's value (e.g. int, float, bool).
        """
        if hasattr(self, "slug"):
            raise InvalidSettingError(
                f"Setting `{self.slug}` already set up for {self.instance}",
            )

        acceptable = {dict, list, str, int, float, bool, tuple}

        if not (type_ in acceptable or issubclass(type_, StrEnum)):
            if hasattr(type_, "__total__"):
                # TypedDict
                type_ = dict  # type: ignore[assignment]
            elif hasattr(type_, "__origin__") and type_.__origin__ in acceptable:
                # Union, List, Tuple, etc.
                type_ = type_.__origin__
            else:
                raise InvalidSettingError(f"Invalid setting type: {type_}")

        self.slug = field_name
        self.instance = instance
        self.type_ = type_

        if self.type_ in {int, float}:
            if self.unit_of_measurement is None:
                raise InvalidSettingError(
                    f"Setting {self.slug!r} must have a unit of measurement with type {self.type_!r}",
                )

            if self.maximum:
                self.maximum = round(self.maximum, self.fp_precision)

            if self.minimum:
                self.minimum = round(self.minimum, self.fp_precision)

        return self

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

    @property
    def mqtt_client(self) -> MqttClient:
        """The MQTT client to use for this setting."""
        return self.__class__._MQTT_CLIENT  # noqa: SLF001

    @cached_property
    def mqtt_topic(self) -> str:
        """The MQTT topic that this setting is subscribed to.

        Auto-generated, in the form `/<hostname>/<automaton ID>/<setting type>/<kebab-case-slug>`

        e.g. /mtrxpi/raining-grid/frequency/rain-speed
        """
        return "/" + "/".join(
            to_kebab_case(
                const.HOSTNAME,
                self.instance.id,
                self.setting_type,
                self.slug,
            ),
        )

    @property
    def value(self) -> S:
        """Get the setting's value from the automaton's attribute."""
        return cast(S, getattr(self.instance, self.slug))

    @value.setter
    def value(self, value: S) -> None:
        """Set the setting's value in the automaton's attribute."""
        if (
            self.type_ is float
            and isinstance(value, float)
            and not isinstance(value, bool)
        ):
            value = round(value, self.fp_precision)

        setattr(self.instance, self.slug, value)

        if self.invoke_settings_callback and hasattr(
            self.instance,
            "setting_update_callback",
        ):
            self.instance.setting_update_callback(update_setting=self.slug)

        if self.value_callbacks and (value_cb := self.value_callbacks.get(value)):
            value_cb(self.instance)

    def __json__(self) -> dict[str, Any]:
        """Return the setting as a JSON-serializable dictionary."""
        return {
            "slug": self.slug,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "type": self.type_.__name__,
            "mqtt_topic": self.mqtt_topic,
        }


N = TypeVar("N", int, float)


@dataclass
class TransitionableSettingMixin(Setting[N]):
    transition_rate: N
    """The amount by which the setting will be changed per tick."""

    target_value: N = field(init=False, repr=False)
    """The target value for a setting.

    Used to enable transition functionality.
    """

    transition_thread: Thread = field(init=False, repr=False)
    """Thread for transitioning values over a non-zero period of time."""

    def __post_init__(self) -> None:
        self.transition_rate = abs(self.transition_rate)

        return super().__post_init__()

    def _set_value_from_payload(self, payload: N) -> None:
        """Set the value of the setting from the payload."""
        # Only transition if the automaton is currently displaying and a transition rate is set
        if self.instance.active and (self.transition_rate or 0) > 0:
            LOGGER.debug("Set target value to %r", payload)
            self.target_value = payload

            if not (
                hasattr(self, "transition_thread") and self.transition_thread.is_alive()
            ):
                self.transition_thread = Thread(
                    target=self._transition_worker,
                    name=f"{self.slug}_transition",
                )
                self.transition_thread.start()
                LOGGER.debug("Started transition thread")
            else:
                LOGGER.debug("Transition thread already running")
        else:
            Setting._set_value_from_payload(self, payload)  # noqa: SLF001

    def _transition_worker(self) -> None:
        LOGGER.info("Transitioning %s to %s", self.slug, self.target_value)

        tick_condition = self.matrix.tick_condition
        tick_condition.acquire()

        cc = self.matrix.current_content

        if self.type_ is int:
            # Wait until accumulated change reaches 1
            ticks_between_transitions = math.ceil(1 / self.transition_rate)

            transition_amount: N = 1
        else:
            # Wait until accumulated change reaches the specified precision
            precision_value = 10**-self.fp_precision
            ticks_between_transitions = math.ceil(precision_value / self.transition_rate)

            transition_amount = ticks_between_transitions * self.transition_rate

        direction = 1 if self.target_value > self.value else -1
        while self.value != self.target_value:
            if self.matrix.tick % ticks_between_transitions == 0:
                self.value = round(
                    self.value
                    + min(transition_amount, abs(self.value - self.target_value))
                    * direction,
                    self.fp_precision,
                )

            if not (cc and cc.is_sleeping):
                # If the content is sleeping, then it isn't yielding, so the canvas isn't being
                # swapped, and that means that no ticks are happening...
                # So only wait for a tick notification if the content is not sleeping!
                tick_condition.wait()
            else:
                sleep(const.TICK_LENGTH)

        tick_condition.release()

        LOGGER.info(
            'Transition complete: %s("%s").%s = %s',
            self.instance.__class__.__name__,
            self.instance.id,
            self.slug,
            self.value,
        )

        self.matrix.publish_attributes()

        if self.value != self.target_value:
            self.mqtt_client.publish(
                self.mqtt_topic,
                self.value,
                retain=True,
            )


@dataclass(kw_only=True)
class FrequencySetting(Setting[int]):
    """Set a rule's frequency."""

    setting_type: Literal[SettingType.FREQUENCY] = field(
        init=False,
        default=SettingType.FREQUENCY,
    )
    type_: type[int] = field(init=False, repr=False, default=int)
    min: Literal[0] = field(init=False, default=0)


@dataclass(kw_only=True, slots=True)
class TransitionableFrequencySetting(FrequencySetting, TransitionableSettingMixin[int]):
    """Set a rule's frequency with transition functionality."""


@dataclass(kw_only=True)
class ParameterSetting(Setting[S]):
    """Set a parameter for a rule."""

    setting_type: Literal[SettingType.PARAMETER] = field(
        init=False,
        default=SettingType.PARAMETER,
    )


@dataclass(kw_only=True, slots=True)
class TransitionableParameterSetting(
    ParameterSetting[N],
    TransitionableSettingMixin[N],
):
    """Set a parameter for a rule with transition functionality."""


__all__ = ["FrequencySetting", "ParameterSetting"]
