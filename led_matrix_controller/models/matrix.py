"""Constants and class for managing the RGB LED Matrix."""

from __future__ import annotations

from logging import DEBUG, getLogger
from threading import Thread
from typing import TYPE_CHECKING, ClassVar, TypedDict

from utils import const
from wg_utilities.loggers import add_stream_handler

from ._rgbmatrix import RGBMatrix, RGBMatrixOptions

if TYPE_CHECKING:
    from models.content.base import ContentBase
    from utils.mqtt import MqttClient

    from .led_matrix_options import LedMatrixOptions

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
add_stream_handler(LOGGER)


class ContentPayload(TypedDict, total=False):
    """Object sent via MQTT to display content on the matrix."""

    id: str
    priority: int


class Dimensions(TypedDict):
    """Dimensions of the matrix."""

    height: int
    width: int


class Matrix:
    """Class for displaying track information on an RGB LED Matrix."""

    content_thread: Thread

    OPTIONS: ClassVar[LedMatrixOptions] = {
        "cols": 64,
        "rows": 64,
        "brightness": 100,
        "gpio_slowdown": 4,
        "hardware_mapping": "adafruit-hat-pwm",
        "show_refresh_rate": const.DEBUG_MODE,
        "limit_refresh_rate_hz": 100,
        "pwm_lsb_nanoseconds": 80,
        # "pwm_dither_bits": 1,  # noqa: ERA001
    }

    def __init__(
        self,
        *,
        mqtt_client: MqttClient,
        options: LedMatrixOptions = OPTIONS,
    ) -> None:
        self.mqtt_client = mqtt_client

        all_options = RGBMatrixOptions()

        for name, value in options.items():
            setattr(all_options, name, value)

        self.matrix = RGBMatrix(options=all_options)
        self.canvas = self.matrix.CreateFrameCanvas()

        self._content: dict[str, ContentBase] = {}

        self.content_topic = (
            f"/{const.HOSTNAME}/{self.__class__.__name__.lower()}/content"
        )

        self.mqtt_client.add_topic_callback(
            self.content_topic,
            self._on_content_message,
        )

        self._content_thread = Thread(target=self._content_loop)
        self._pending_content: list[ContentBase] = []

    def _on_content_message(
        self,
        payload: ContentPayload,
    ) -> None:
        """Callback for when a message is received on the content topic."""
        self._pending_content.append(self._content[payload["id"]])

        if not self._content_thread.is_alive():
            try:
                self._content_thread.start()
            except RuntimeError as err:
                if str(err) != "threads can only be started once":
                    raise

                self._content_thread = Thread(target=self._content_loop)
                self._content_thread.start()

    def _content_loop(self) -> None:
        while self._pending_content:
            content = self._pending_content.pop(0)

            get_image = content.image_getter

            LOGGER.info("Displaying content with ID `%s`", content.id)

            for _ in content:
                self.canvas.SetImage(get_image())
                self.canvas = self.matrix.SwapOnVSync(self.canvas)

            LOGGER.debug("Content `%s` complete", content.id)

    def register_content(self, *content: ContentBase) -> None:
        """Add content to the matrix."""
        for c in content:
            self._content[c.id] = c

            LOGGER.info("Added content with ID `%s`", c.id)

    @property
    def dimensions(self) -> Dimensions:
        """Return the dimensions of the matrix."""
        return {
            "height": self.height,
            "width": self.width,
        }

    @property
    def height(self) -> int:
        """Return the height of the matrix."""
        return int(self.matrix.height)

    @property
    def width(self) -> int:
        """Return the width of the matrix."""
        return int(self.matrix.width)
