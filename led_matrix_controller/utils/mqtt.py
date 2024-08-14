"""Module for holding the main controller function(s) for controlling the GUI."""

from __future__ import annotations

import sys
from json import JSONDecodeError, loads
from logging import DEBUG, getLogger
from typing import TYPE_CHECKING, Any, Callable, ClassVar, TypeVar

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from utils import const
from wg_utilities.loggers import add_stream_handler

if TYPE_CHECKING:
    from paho.mqtt.properties import Properties
    from paho.mqtt.reasoncodes import ReasonCode

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
add_stream_handler(LOGGER)


class Singleton(type):
    """Singleton metaclass."""

    _instances: ClassVar[dict[type[Any], Any]] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Return the instance if it already exists, otherwise create it."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]


T = TypeVar("T", bound=object)


class MqttClient(metaclass=Singleton):
    """MQTT Client wrapper class."""

    CONNECTION_RETRY_LIMIT: ClassVar[int] = 3

    def __init__(
        self,
        *,
        connect: bool = False,
        userdata: Any = None,
        retain: bool = False,
    ) -> None:
        self._client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            userdata=userdata,
            protocol=mqtt.MQTTv5,
        )
        self._client.username_pw_set(
            username=const.MQTT_USERNAME,
            password=const.MQTT_PASSWORD,
        )
        self._client.on_connect = self._on_connect
        self._client.on_connect_fail = self._on_connect_fail
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.on_subscribe = self._on_subscribe

        self._retain = retain

        if connect and not self._client.is_connected():
            LOGGER.debug(
                "Connecting to MQTT broker at %s. This device's hostname is %s",
                const.MQTT_HOST,
                const.HOSTNAME,
            )
            self._client.connect(const.MQTT_HOST)

        self._connection_failures = 0

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: mqtt.ConnectFlags,
        rc: ReasonCode,
        properties: Properties | None,
    ) -> None:
        _ = client, userdata, flags, properties
        LOGGER.info("Connected with result code: %s", rc)
        self._connection_failures = 0

    def _on_connect_fail(
        self,
        client: mqtt.Client,
        userdata: Any,
    ) -> None:
        _ = client, userdata
        LOGGER.error("Failed to connect to MQTT broker")
        self._connection_failures += 1

        if self._connection_failures >= self.CONNECTION_RETRY_LIMIT:
            sys.exit(
                f"Failed to connect to MQTT broker after {self.CONNECTION_RETRY_LIMIT} attempts. Exiting.",
            )

    def _on_disconnect(  # noqa: PLR6301
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: mqtt.DisconnectFlags,
        rc: ReasonCode,
        properties: Properties | None,
    ) -> None:
        _ = client, userdata, flags, properties
        LOGGER.info("Disconnected with reason code: %s", rc)

    def _on_message(  # noqa: PLR6301
        self,
        client: mqtt.Client,
        userdata: Any,
        message: mqtt.MQTTMessage,
    ) -> None:
        _ = client, userdata
        LOGGER.info("Received message: %s", message.payload)

    def _on_subscribe(  # noqa: PLR6301
        self,
        client: mqtt.Client,
        userdata: Any,
        mid: int,
        rc: list[ReasonCode],
        properties: Properties,
    ) -> None:
        _ = client, userdata, rc, properties
        LOGGER.debug("Subscribed with message ID: %s", mid)

    def add_topic_callback(
        self,
        topic: str,
        callback: Callable[[T], None],
    ) -> None:
        """Subscribe to the topic and add a callback for the payload."""

        def _cb(_: mqtt.Client, __: Any, msg: mqtt.MQTTMessage) -> None:
            LOGGER.debug(
                "Received message on topic %s with payload %r",
                topic,
                msg.payload,
            )

            try:
                payload = loads(msg.payload)
            except JSONDecodeError:
                if not msg.payload and msg.retain:
                    LOGGER.debug("Message received to clear topic %s", msg.topic)
                    return

                LOGGER.exception("Failed to decode payload: %r", msg.payload)
                return

            callback(payload)

        self._client.subscribe(topic)
        self._client.message_callback_add(topic, _cb)

        LOGGER.info("Added callback `%s` for topic: %s", callback.__qualname__, topic)

    def loop_forever(self) -> None:
        """Start the MQTT loop."""
        LOGGER.info("Starting MQTT loop")
        self._client.loop_forever()

    def publish(
        self,
        topic: str,
        payload: mqtt.PayloadType,
        *,
        qos: int = 0,
        retain: bool | None = None,
        properties: Properties | None = None,
    ) -> None:
        """Publish a message to the MQTT broker.

        Args:
            topic: The topic to publish the message to
            payload: The message to publish
            qos: The quality of service to use
            retain: Whether to retain the message on the topic
            properties: The properties to include in the message
        """
        self._client.publish(
            topic,
            payload,
            qos=qos,
            retain=retain or self._retain,
            properties=properties,
        )
        LOGGER.debug(
            "Published message to topic %s%s: %r",
            topic,
            " with retain flag set" if retain or self._retain else "",
            payload,
        )
