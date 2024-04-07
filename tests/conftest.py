"""Conftest file for the tests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from utils.mqtt import MqttClient


@pytest.fixture(name="mqtt_client")
def mqtt_client_() -> MqttClient:
    """MQTT Client fixture."""
    return Mock(spec=MqttClient)
