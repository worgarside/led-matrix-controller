"""Constants and class for managing the RGB LED Matrix."""

from __future__ import annotations

from functools import partial
from logging import DEBUG, getLogger
from queue import PriorityQueue
from threading import Condition, Thread
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, cast

from models.content.base import (
    CanvasGetter,
    ContentBase,
    DynamicContent,
    ImageGetter,
    PreDefinedContent,
)
from utils import const
from utils.helpers import to_kebab_case
from wg_utilities.loggers import add_stream_handler

from ._rgbmatrix import Canvas, RGBMatrix, RGBMatrixOptions

if TYPE_CHECKING:
    from PIL import Image
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

    OPTIONS: ClassVar[LedMatrixOptions] = {
        "cols": 64,
        "rows": 64,
        "brightness": 100,
        "gpio_slowdown": 4,
        "hardware_mapping": "adafruit-hat-pwm",
        "show_refresh_rate": const.DEBUG_MODE,
        "limit_refresh_rate_hz": const.TICKS_PER_SECOND,
        "pwm_lsb_nanoseconds": 80,
        # "pwm_dither_bits": 1,  # noqa: ERA001
    }

    MAX_PRIORITY: ClassVar[float] = 1e10

    canvas: Canvas

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

        self.queue_content_topic = self._mqtt_topic("queue-content")
        self.mqtt_client.add_topic_callback(
            self.queue_content_topic,
            self._on_content_message,
        )

        self.current_content_topic = self._mqtt_topic("current-content")

        self._content_queue: PriorityQueue[tuple[float, ContentBase]] = PriorityQueue()

        self._content_thread = Thread(target=self._content_loop)

        self._current_content: ContentBase | None = None

        self.current_priority: float = self.MAX_PRIORITY  # Lower value = higher priority

        self.tick = 0
        self.tick_condition = Condition()

    def get_canvas_swap_canvas(self, get_canvas: CanvasGetter) -> None:
        """Get the canvas and swap it."""
        self.swap_canvas(get_canvas())

    def set_image_swap_canvas(self, get_image: ImageGetter) -> None:
        """Set the image and swap the canvas."""
        self.canvas.SetImage(get_image())

        self.swap_canvas(self.canvas)

    def swap_canvas(self, content: Canvas | None = None, /) -> None:
        """Update the content of the canvas and increment the tick count."""
        self.canvas = self.matrix.SwapOnVSync(content or self.canvas)

        self.tick += 1

        self.tick_condition.acquire(timeout=const.TICK_LENGTH)
        self.tick_condition.notify_all()
        self.tick_condition.release()

    def _content_loop(self) -> None:
        """Loop through the content queue."""
        while not self._content_queue.empty():
            self.current_priority, self.current_content = self._content_queue.get()

            LOGGER.info(
                "Displaying content with ID `%s` at priority %s",
                self.current_content.id,
                self.current_priority,
            )

            if isinstance(self.current_content, DynamicContent):
                set_content = partial(
                    self.set_image_swap_canvas,
                    self.current_content.content_getter,
                )
            else:
                set_content = partial(
                    self.get_canvas_swap_canvas,
                    self.current_content.content_getter,
                )

            # Actual loop through individual content instances:
            for _ in self.current_content:
                set_content()

            if self.current_content.HAS_TEARDOWN_SEQUENCE:
                # Only run teardown if the stop isn't due to higher priority content
                LOGGER.info("Running teardown sequence for %s", self.current_content.id)

                for _ in self.current_content.teardown():
                    set_content()

            LOGGER.info("Content `%s` complete", self.current_content.id)

            if self.current_content.persistent:
                self._content_queue.put((self.current_priority, self.current_content))
                LOGGER.debug(
                    "Content `%s` is persistent with priority %s",
                    self.current_content.id,
                    self.current_priority,
                )

        self.clear_matrix()

    def _mqtt_topic(self, suffix: str) -> str:
        """Create an MQTT topic with the given suffix."""
        return "/" + "/".join(
            to_kebab_case(const.HOSTNAME, self.__class__.__name__, *suffix.split("/")),
        )

    def _on_content_message(
        self,
        payload: ContentPayload,
    ) -> None:
        """Add/remove content to/from the queue."""
        if payload["priority"] is None:
            return self._remove_content_from_queue(payload["id"])

        priority = max(
            min(float(payload["priority"]), self.MAX_PRIORITY),
            -self.MAX_PRIORITY,
        )

        if (curr_cont := self.is_active()) and curr_cont.id == payload["id"]:
            LOGGER.debug("Updating %s priority to %s", payload["id"], priority)
            self.current_priority = priority
            return None

        try:
            self._content_queue.put((priority, self._content[payload["id"]]))
        except KeyError:
            LOGGER.exception("Content with ID `%s` not found", payload["id"])
            return None

        LOGGER.info(
            "Added content with ID `%s` to queue with priority %s",
            payload["id"],
            priority,
        )

        # Lower value = higher priority
        if (curr_cont := self.is_active()) and priority < self.current_priority:
            curr_cont.stop()

        self._start_content_thread()

        return None

    def _remove_content_from_queue(self, content_id: str) -> None:
        """Remove content from the queue."""
        for p, c in self._content_queue.queue:
            if c.content_id == content_id:
                self._content_queue.queue.remove((p, c))
                LOGGER.info("Removed content with ID `%s` from queue", content_id)
                break

        if (curr_cont := self.is_active()) and curr_cont.content_id == content_id:
            curr_cont.stop()

    def _start_content_thread(self) -> None:
        """Start the content thread. If it has already been started, do nothing."""
        if not self._content_thread.is_alive():
            try:
                self._content_thread.start()
            except RuntimeError as err:
                if str(err) != "threads can only be started once":
                    raise

                self._content_thread = Thread(target=self._content_loop)
                self._content_thread.start()

    def clear_matrix(self) -> None:
        """Clear the matrix."""
        self.current_content = None
        self.current_priority = self.MAX_PRIORITY

        self.canvas.Clear()
        self.swap_canvas()

        LOGGER.info("Matrix cleared")

    def new_canvas(self, image: Image.Image | None = None) -> Canvas:
        """Return a new canvas, optionally with an image."""
        canvas = cast(Canvas, self.matrix.CreateFrameCanvas())

        if image is not None:
            canvas.SetImage(image.convert("RGB"))

        return canvas

    def register_content(self, *content: ContentBase) -> None:
        """Add content to the matrix."""

        content_ids = []
        for c in content:
            self._content[c.content_id] = c
            content_ids.append(c.content_id)

            for setting in getattr(c, "settings", {}).values():
                setting.matrix = self

            LOGGER.info("Added content with ID `%s`", c.content_id)

            if isinstance(c, PreDefinedContent):
                c.generate_canvases(self.new_canvas)

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

    @property
    def current_content(self) -> ContentBase | None:
        """Return the currently displaying content."""
        return self._current_content

    @current_content.setter
    def current_content(self, value: ContentBase | None) -> None:
        self._current_content = value

        self.mqtt_client.publish(
            topic=self.current_content_topic,
            payload=value.content_id if value is not None else None,
            retain=True,
        )
        LOGGER.info("Now playing: %s", value.id if value is not None else None)

    def is_active(self) -> ContentBase | Literal[False]:
        """Return whether content is currently playing."""
        if (curr_cont := self.current_content) is not None:
            return curr_cont

        return False

    def __del__(self) -> None:
        """Clear the matrix when the object is deleted."""
        self.clear_matrix()
