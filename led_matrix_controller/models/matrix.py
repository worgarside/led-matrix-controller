"""Constants and class for managing the RGB LED Matrix."""

from __future__ import annotations

from queue import PriorityQueue
from threading import Condition, Thread
from typing import TYPE_CHECKING, Any, ClassVar, Final, TypedDict, cast

from content.base import (
    ContentBase,
    GridView,
    PreDefinedContent,
    StopType,
)
from models.setting import Setting, TransitionableParameterSetting
from PIL import Image
from utils import const, mtrx
from utils.helpers import to_kebab_case
from wg_utilities.decorators import process_exception
from wg_utilities.loggers import get_streaming_logger

if TYPE_CHECKING:
    from utils.mqtt import MqttClient

    from .led_matrix_options import LedMatrixOptions

LOGGER = get_streaming_logger(__name__)


class ContentPayload(TypedDict):
    """Object sent via MQTT to display content on the matrix."""

    id: str
    priority: float | None
    parameters: dict[str, Any]


class Dimensions(TypedDict):
    """Dimensions of the matrix."""

    height: int
    width: int


class Matrix:
    """Class for displaying track information on an RGB LED Matrix."""

    OPTIONS: ClassVar[LedMatrixOptions] = {
        "cols": const.MATRIX_HEIGHT,
        "rows": const.MATRIX_WIDTH,
        "brightness": 100,
        "gpio_slowdown": 4,
        "hardware_mapping": "adafruit-hat-pwm",
        "show_refresh_rate": const.DEBUG_MODE,
        "limit_refresh_rate_hz": const.TICKS_PER_SECOND,
        "pwm_lsb_nanoseconds": 80,
        # "pwm_dither_bits": 1,  # noqa: ERA001
    }

    MAX_PRIORITY: ClassVar[float] = 1e10

    canvas: mtrx.Canvas
    tick_condition: Condition

    id: Final = "matrix"

    def __init__(
        self,
        *,
        mqtt_client: MqttClient,
        options: LedMatrixOptions = OPTIONS,
    ) -> None:
        self.mqtt_client = mqtt_client

        self._brightness_setting = TransitionableParameterSetting(
            minimum=0,
            maximum=100,
            transition_rate=1,
            fp_precision=0,
        ).setup(
            field_name="brightness",
            instance=self,
            type_=int,
        )
        self.mqtt_client.add_topic_callback(
            self._brightness_setting.mqtt_topic,
            self._brightness_setting.on_message,
        )

        self._brightness_setting.matrix = self
        self._brightness = options["brightness"]

        all_options = mtrx.RGBMatrixOptions()
        for name, value in options.items():
            setattr(all_options, name, value)

        self.matrix = mtrx.RGBMatrix(options=all_options)
        self.canvas = self.matrix.CreateFrameCanvas()

        self._content: dict[str, ContentBase[Any]] = {}

        self.queue_content_topic = self._mqtt_topic("queue-content")
        self.mqtt_client.add_topic_callback(
            self.queue_content_topic,
            self._on_content_message,
        )

        self.current_content_topic = self._mqtt_topic("current-content")

        self._content_queue: PriorityQueue[
            tuple[
                float,
                ContentBase[Any],
                dict[str, Any],
            ]
        ] = PriorityQueue()

        self._content_thread = Thread(target=self._content_loop)

        # Setting via property to trigger MQTT update
        self.current_content: ContentBase[GridView] | ContentBase[mtrx.Canvas] | None = (
            None
        )

        self.current_priority: float = self.MAX_PRIORITY  # Lower value = higher priority

        self.tick = 0
        self.tick_condition = Condition()

    def get_canvas_swap_canvas(self) -> None:
        """Get the canvas and swap it."""
        self.swap_canvas(self.current_content.get_content())  # type: ignore[union-attr]

    def set_image_swap_canvas(self) -> None:
        """Set the image and swap the canvas."""
        image = Image.fromarray(self.current_content.get_content(), "RGB")  # type: ignore[union-attr]

        self.canvas.SetImage(image)

        self.swap_canvas(self.canvas)

    def swap_canvas(self, content: mtrx.Canvas | None = None, /) -> None:
        """Update the content of the canvas and increment the tick count."""
        self.canvas = self.matrix.SwapOnVSync(content or self.canvas)

        self.tick += 1

        self.tick_condition.acquire(timeout=const.TICK_LENGTH)
        self.tick_condition.notify_all()
        self.tick_condition.release()

    def _content_loop(self) -> None:
        """Loop through the content queue."""
        while not self._content_queue.empty():
            (
                self.current_priority,
                self.current_content,
                parameters,
            ) = self._content_queue.get()

            LOGGER.info(
                "Displaying content with ID `%s` at priority %s",
                self.current_content.id,
                self.current_priority,
            )

            if self.current_content.canvas_count is None:
                # DynamicContent
                set_content = self.set_image_swap_canvas
            else:
                # PreDefinedContent
                set_content = self.get_canvas_swap_canvas

            self.current_content.active = True

            if (setup_gen := self.current_content.setup()) is not None:
                # Only run setup if it returns a generator
                LOGGER.info("Running setup sequence for %s", self.current_content.id)

                for _ in setup_gen:
                    set_content()

            # Actual loop through individual content instances:
            for _ in self.current_content:
                set_content()

            if self.current_content.stop_reason != StopType.PRIORITY and (
                teardown_gen := self.current_content.teardown()
            ):
                # Only run teardown if the stop isn't due to higher priority content
                LOGGER.info("Running teardown sequence for %s", self.current_content.id)

                for _ in teardown_gen:
                    set_content()

            self.current_content.active = False

            LOGGER.info("Content `%s` complete", self.current_content.id)

            if (
                self.current_content.persistent
                and self.current_content.stop_reason
                not in {StopType.CANCEL, StopType.EXPIRED}
            ):
                self._content_queue.put(
                    (self.current_priority, self.current_content, parameters),
                )
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

    @process_exception(logger=LOGGER, raise_after_processing=False)
    def _on_content_message(
        self,
        payload: ContentPayload,
    ) -> None:
        """Add/remove content to/from the queue."""
        target_content = self._content[payload["id"]]
        parameters = payload.get("parameters", {})

        if (priority := payload["priority"]) is None:
            for prio, ctnt, prms in self._content_queue.queue:
                if ctnt is target_content:
                    self._content_queue.queue.remove((prio, ctnt, prms))
                    LOGGER.info(
                        "Removed content with ID `%s` from queue",
                        target_content.id,
                    )
                    break

            if self.current_content is target_content:
                self.current_content.stop(StopType.CANCEL)

            return

        priority = max(
            min(float(priority), self.MAX_PRIORITY),
            -self.MAX_PRIORITY,
        )

        if self.current_content is target_content:
            LOGGER.debug("Updating %s priority to %s", target_content, priority)
            self.current_priority = priority
            return

        self._content_queue.put((priority, target_content, parameters))

        LOGGER.info(
            "Added content with ID `%s` to queue with priority %s",
            target_content.id,
            priority,
        )

        # Lower value = higher priority
        if self.current_content and priority < self.current_priority:
            self.current_content.stop(StopType.PRIORITY)

        self._start_content_thread()

        return

    def _start_content_thread(self) -> None:
        """Start the content thread. If it has already been started, do nothing."""
        if self._content_thread.is_alive():
            LOGGER.debug("Content thread already running")
        else:
            try:
                self._content_thread.start()
                LOGGER.info("Started existing content thread")
            except RuntimeError as err:
                if str(err) != "threads can only be started once":
                    raise

                self._content_thread = Thread(target=self._content_loop)
                self._content_thread.start()

                LOGGER.info("Started new content thread")

    def clear_matrix(self) -> None:
        """Clear the matrix."""
        self.current_content = None
        self.current_priority = self.MAX_PRIORITY

        self.canvas.Clear()
        self.swap_canvas()

        LOGGER.info("Matrix cleared")

    def new_canvas(self, image: Image.Image | None = None) -> mtrx.Canvas:
        """Return a new canvas, optionally with an image."""
        canvas = cast(mtrx.Canvas, self.matrix.CreateFrameCanvas())

        if image is not None:
            canvas.SetImage(image.convert("RGB"))

        return canvas

    def register_content(self, *content: ContentBase[Any]) -> None:
        """Add content to the matrix."""
        content_ids = []
        for c in content:
            self._content[c.content_id] = c
            content_ids.append(c.content_id)

            setting: Setting[Any]
            for setting in getattr(c, "settings", {}).values():
                setting.matrix = self

            LOGGER.info("Added content with ID `%s`", c.content_id)

            if isinstance(c, PreDefinedContent):
                c.generate_canvases(self.new_canvas)

            if hasattr(c, "settings"):
                for setting in c.settings.values():
                    self.mqtt_client.add_topic_callback(
                        setting.mqtt_topic,
                        setting.on_message,
                    )

    @property
    def active(self) -> bool:
        """Return whether content is currently playing."""
        return self.current_content is not None and self.current_content.active

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
    def brightness(self) -> int:
        """Return the brightness of the matrix."""
        return self._brightness

    @brightness.setter
    def brightness(self, value: int) -> None:
        """Set the brightness of the matrix."""
        self._brightness = value
        self.matrix.brightness = value

        LOGGER.debug("Set brightness to %s (%i)", value, self.tick)

        # If there is no content playing, actively swap the canvas to update the brightness
        # Otherwise the brightness will be updated on each tick anyway
        if self.current_content is None or self.current_content.is_sleeping:
            self.swap_canvas()

    @property
    def current_content(self) -> ContentBase[Any] | None:
        """Return the currently displaying content."""
        return self._current_content

    @current_content.setter
    def current_content(self, value: ContentBase[Any] | None) -> None:
        self._current_content = value

        self.mqtt_client.publish(
            topic=self.current_content_topic,
            payload=value.content_id if value is not None else None,
            retain=True,
        )
        LOGGER.info("Current Content: %s", value.id if value is not None else None)

        self.publish_attributes()

    def publish_attributes(self) -> None:
        """Publish the attributes of the current content."""
        self.mqtt_client.publish(
            topic=f"{self.current_content_topic}/attributes",
            payload=self.current_content.mqtt_attributes
            if self.current_content is not None
            else "{}",
        )

    def __del__(self) -> None:
        """Clear the matrix when the object is deleted."""
        self.clear_matrix()

    def setting_update_callback(self, *_: Any, **__: Any) -> None:
        """Generate the frame rulesets for the matrix."""
        raise NotImplementedError
