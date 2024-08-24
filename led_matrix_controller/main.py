"""Rain simulation."""

from __future__ import annotations

from contextlib import suppress

from content import LIBRARY
from models import Matrix
from utils import MqttClient


def main() -> None:
    """Run the rain simulation."""
    mqtt_client = MqttClient(connect=True)

    Matrix(mqtt_client=mqtt_client).register_content(*LIBRARY)

    mqtt_client.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with suppress(Exception):
            Matrix(mqtt_client=MqttClient(connect=False)).clear_matrix()
        raise
