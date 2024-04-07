"""Module for holding the main controller function(s) for controlling the GUI."""

from __future__ import annotations

from json import loads
from logging import DEBUG, getLogger
from typing import TYPE_CHECKING, Any, Callable, TypeVar

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

T = TypeVar("T", bound=object)


class MqttClient:
    """MQTT Client wrapper class."""

    def __init__(self, *, connect: bool = True, userdata: Any = None) -> None:
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

        if connect:
            LOGGER.debug("Connecting to MQTT broker")
            self._client.connect(const.MQTT_HOST)

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

    def _on_connect_fail(
        self,
        client: mqtt.Client,
        userdata: Any,
    ) -> None:
        _ = client, userdata
        LOGGER.error("Failed to connect to MQTT broker")

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: mqtt.DisconnectFlags,
        rc: ReasonCode,
        properties: Properties | None,
    ) -> None:
        _ = client, userdata, flags, properties
        LOGGER.info("Disconnected with reason code: %s", rc)

    def _on_message(
        self, client: mqtt.Client, userdata: Any, message: mqtt.MQTTMessage
    ) -> None:
        _ = client, userdata
        LOGGER.info("Received message: %s", message.payload)

    def _on_subscribe(
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

            callback(loads(msg.payload))

        self._client.subscribe(topic)
        self._client.message_callback_add(topic, _cb)

        LOGGER.info("Added callback `%s` for topic: %s", callback.__qualname__, topic)

    def loop_forever(self) -> None:
        """Start the MQTT loop."""
        LOGGER.info("Starting MQTT loop")
        self._client.loop_forever()
