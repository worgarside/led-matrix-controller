"""Constants and class for managing the RGB LED Matrix."""

from __future__ import annotations

from logging import DEBUG, getLogger
from queue import PriorityQueue
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


class ContentPayload(TypedDict):
    """Object sent via MQTT to display content on the matrix."""

    id: str
    priority: float | None


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
        "limit_refresh_rate_hz": const.FRAMES_PER_SECOND,
        "pwm_lsb_nanoseconds": 80,
        # "pwm_dither_bits": 1,  # noqa: ERA001
    }

    MAX_PRIORITY: ClassVar[float] = 1e10

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
            f"/{const.HOSTNAME}/{self.__class__.__name__}/content".lower()
        )

        self.mqtt_client.add_topic_callback(
            self.content_topic,
            self._on_content_message,
        )

        self._content_queue: PriorityQueue[tuple[float, ContentBase]] = PriorityQueue()
        self._content_thread = Thread(target=self._content_loop)

        self.now_playing: str | None = None

        self.next_priority = self.MAX_PRIORITY

    def _on_content_message(
        self,
        payload: ContentPayload,
    ) -> None:
        """Callback for when a message is received on the content topic."""

        if payload["priority"] is None:
            for p, c in self._content_queue.queue:
                if c.content_id == payload["id"]:
                    self._content_queue.queue.remove((p, c))
                    LOGGER.info("Removed content with ID `%s` from queue", payload["id"])
                    break

            if self.now_playing == payload["id"]:
                self.reset_now_playing()
                LOGGER.info(
                    "Removed content with ID `%s` from now playing", payload["id"]
                )

            return

        priority = max(
            min(float(payload["priority"]), self.MAX_PRIORITY),
            -self.MAX_PRIORITY,
        )

        self.next_priority = min(self.next_priority, priority)

        self._content_queue.put((priority, self._content[payload["id"]]))

        if not self._content_thread.is_alive():
            try:
                self._content_thread.start()
            except RuntimeError as err:
                if str(err) != "threads can only be started once":
                    raise

                self._content_thread = Thread(target=self._content_loop)
                self._content_thread.start()

    def _content_loop(self) -> None:
        while not self._content_queue.empty():
            current_content_priority, content = self._content_queue.get()

            self.now_playing = content.id

            LOGGER.debug(
                "Content with ID `%s` has priority %s",
                content.content_id,
                current_content_priority,
            )

            get_image = content.image_getter

            LOGGER.info("Displaying content with ID `%s`", content.content_id)

            for _ in content:
                self.canvas.SetImage(get_image())
                self.canvas = self.matrix.SwapOnVSync(self.canvas)

                if current_content_priority > self.next_priority or not self.now_playing:
                    break

            LOGGER.debug("Content `%s` complete", content.content_id)

            if content.persistent and content.id == self.now_playing:
                self._content_queue.put((current_content_priority, content))
                LOGGER.info(
                    "Content `%s` is persistent with priority %s",
                    content.content_id,
                    current_content_priority,
                )

            self.reset_now_playing()

    def register_content(self, *content: ContentBase) -> None:
        """Add content to the matrix."""
        for c in content:
            self._content[c.content_id] = c

            LOGGER.info("Added content with ID `%s`", c.content_id)

    def reset_now_playing(self) -> None:
        """Reset the now playing content."""
        self.now_playing = None
        self.next_priority = self.MAX_PRIORITY

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
