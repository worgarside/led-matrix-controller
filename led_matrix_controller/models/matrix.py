"""Constants and class for managing the RGB LED Matrix."""

from __future__ import annotations

from json import dumps
from queue import PriorityQueue
from threading import Condition, Thread
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Iterator,
    Sequence,
    TypedDict,
    cast,
)

import numpy as np
from content import Combination
from content.base import (
    ContentBase,
    GridView,
    PreDefinedContent,
    StopType,
)
from content.dynamic_content import DynamicContent
from content.setting import Setting, TransitionableParameterSetting
from PIL import Image
from utils import const, mtrx
from utils.helpers import to_kebab_case
from wg_utilities.decorators import process_exception
from wg_utilities.loggers import get_streaming_logger

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray
    from utils.mqtt import MqttClient

    from .led_matrix_options import LedMatrixOptions

LOGGER = get_streaming_logger(__name__)


ContentParameters = MappingProxyType[str, str]


class ContentPayload(TypedDict):
    """Object sent via MQTT to display content on the matrix."""

    id: str
    priority: float | None
    parameters: ContentParameters


class Dimensions(TypedDict):
    """Dimensions of the matrix."""

    height: int
    width: int


class ContentQueue(
    PriorityQueue[
        tuple[
            float,
            ContentBase[Any],
            ContentParameters,
        ]
    ],
):
    """Priority queue for content to be displayed on the matrix.

    Each item is a tuple of the content's priority, the content itself, and any parameters.
    The tuple contains the priority as well as the content to enable simple priority-based
    sorting.
    """

    class MqttMeta(TypedDict):
        """Metadata for MQTT messages."""

        id: str
        parameters: ContentParameters

    def __init__(self, mqtt_client: MqttClient, topic_root: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.mqtt_client = mqtt_client

        self.state_topic = f"/{topic_root.strip('/')}/state"
        self.attrs_topic = f"/{topic_root.strip('/')}/attributes"

        self.mqtt_client.add_topic_callback(
            self.attrs_topic,
            self.validate_queue_content,
        )

    def get(
        self,
        block: bool = True,  # noqa: FBT001,FBT002
        timeout: float | None = None,
    ) -> tuple[float, ContentBase[Any], ContentParameters]:
        """Get the next item from the queue and send MQTT messages."""
        next_content = super().get(block, timeout)

        self.send_mqtt_messages()

        return next_content

    def add(
        self,
        content: ContentBase[Any],
        parameters: ContentParameters,
    ) -> None:
        """Put an item into the queue and send MQTT messages."""
        super().put((round(content.priority, 3), content, parameters))

        self.send_mqtt_messages()

    def remove(
        self,
        content: ContentBase[Any],
        parameters: ContentParameters,
    ) -> None:
        """Remove an item from the queue and send MQTT messages."""
        for prio, ctnt, prms in self.queue:
            if ctnt is content and prms == parameters:
                queued_priority = prio
                break
        else:
            LOGGER.warning(
                "Content with ID `%s` not found in queue",
                content.id,
            )
            return

        try:
            self.queue.remove((queued_priority, content, parameters))
        except ValueError as err:
            if str(err) != "list.remove(x): x not in list":
                raise

        self.send_mqtt_messages()

    def send_mqtt_messages(self) -> None:
        """Send MQTT messages."""
        attrs = self.mqtt_attributes

        self.mqtt_client.publish(
            topic=self.attrs_topic,
            payload=dumps(attrs),
            retain=True,
        )
        self.mqtt_client.publish(
            topic=self.state_topic,
            payload=len(attrs),
            retain=True,
        )

    def validate_queue_content(self, payload: dict[str, MqttMeta]) -> None:
        """Ensure the (usually retained) MQTT messages match the queue."""
        if payload != self.mqtt_attributes:
            self.send_mqtt_messages()

    def __contains__(
        self,
        content_tuple: tuple[ContentBase[Any], ContentParameters],
    ) -> bool:
        """Check if the queue contains the content."""
        return any((c, p) == content_tuple for _, c, p in self)

    def __iter__(self) -> Iterator[tuple[float, ContentBase[Any], ContentParameters]]:
        """Iterate over the queue."""
        return iter(self.queue)

    @property
    def mqtt_attributes(self) -> dict[str, MqttMeta]:
        """Return the MQTT attributes of the queue."""
        return {
            f"{item[1].priority:.3f}": {
                "id": item[1].id,
                "parameters": item[2],
            }
            for item in self.queue
        }


class Matrix:
    """Class for displaying track information on an RGB LED Matrix."""

    COMBINATION_OVERRIDES: ClassVar[dict[tuple[str, ...], tuple[str, ...]]] = {
        ("clock", "raining-grid"): (
            "clock",
            "raining-grid",
        ),  # Rainfall in front of clock
    }

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

    canvas: mtrx.Canvas
    tick_condition: Condition

    id: Final = "matrix"

    def __init__(
        self,
        *,
        mqtt_client: MqttClient,
        content_works_with: dict[type[ContentBase[Any]], set[type[ContentBase[Any]]]],
        options: LedMatrixOptions = OPTIONS,
    ) -> None:
        self.mqtt_client = mqtt_client
        self.content_works_with = content_works_with

        self._brightness_setting = TransitionableParameterSetting(
            minimum=0,
            maximum=100,
            transition_rate=0.1,
            fp_precision=0,
            icon="mdi:brightness-percent",
            unit_of_measurement="%",
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

        self._content_queue = ContentQueue(
            mqtt_client=self.mqtt_client,
            topic_root=self._mqtt_topic("content-queue"),
        )

        self._content_thread = Thread(target=self._content_loop)

        # Setting via property to trigger MQTT update
        self.current_content: ContentBase[GridView] | ContentBase[mtrx.Canvas] | None = (
            None
        )

        self.tick: int = 0
        self.tick_condition = Condition()

        self.array = self.zeros(dtype=np.uint8)

    def _content_works_with(
        self,
        cls_a: type[ContentBase[Any]] | ContentBase[Any],
        cls_b: type[ContentBase[Any]] | ContentBase[Any],
    ) -> bool:
        """Check if the classes are compatible."""
        if not isinstance(cls_a, type):
            cls_a = cls_a.__class__

        if not isinstance(cls_b, type):
            cls_b = cls_b.__class__

        return cls_b in self.content_works_with.get(cls_a, set())

    def get_canvas_swap_canvas(self) -> None:
        """Get the canvas and swap it."""
        self.swap_canvas(self.current_content.get_content())  # type: ignore[union-attr]

    def set_image_swap_canvas(self) -> None:
        """Set the image and swap the canvas."""
        content_array = self.current_content.get_content()  # type: ignore[union-attr]

        self.array.fill(0)

        x, y = self.current_content.position  # type: ignore[union-attr]

        self.array[
            y : y + self.current_content.height,  # type: ignore[union-attr]
            x : x + self.current_content.width,  # type: ignore[union-attr]
        ] = content_array

        image = Image.fromarray(self.array, "RGBA").convert("RGB")

        self.canvas.SetImage(image)

        self.swap_canvas(self.canvas)

    def swap_canvas(self, content: mtrx.Canvas | None = None, /) -> None:
        """Update the content of the canvas and increment the tick count."""
        self.canvas = self.matrix.SwapOnVSync(content or self.canvas)

        self.tick += 1

        self.tick_condition.acquire(timeout=const.TICK_LENGTH)
        self.tick_condition.notify_all()
        self.tick_condition.release()

    def _attempt_combination(
        self,
        target_content: DynamicContent,
    ) -> DynamicContent | Combination:
        """Attempt to combine the current content with the target content.

        If a combination can't be made, the target content is returned as-is.
        """
        # Doesn't matter anyway
        if not isinstance(self.current_content, DynamicContent):
            LOGGER.debug(
                "Skipping combination attempt between current %r and target %r",
                self.current_content.id if self.current_content is not None else None,
                target_content.id,
            )
            return target_content

        if isinstance(self.current_content, Combination):
            # If it's already a Combination, check if the target content can be added
            if all(
                self._content_works_with(target_content, c)
                for c in self.current_content.content
            ):
                LOGGER.debug(
                    "Added %r to existing Combination(content=(%s))",
                    target_content.id,
                    ", ".join(c.id for c in self.current_content.content),
                )
                return self.current_content.update_setting(
                    "content",
                    value=sorted(
                        (*self.current_content.content, target_content),
                        key=lambda c: (not c.IS_OPAQUE, -c.priority),
                    ),
                    invoke_callback=True,
                )

            LOGGER.debug(
                "Unable to add %r to existing Combination(content=(%s))",
                target_content.id,
                ", ".join(c.id for c in self.current_content.content),
            )

            # Combination can't be made
            return target_content

        if self._content_works_with(self.current_content, target_content):
            # Not a combination, but can be combined
            combo_content = cast(Combination, ContentBase.get("combination"))
            combo_content.priority = min(
                self.current_content.priority,
                target_content.priority,
            )
            if override_ids := self.COMBINATION_OVERRIDES.get(
                tuple(sorted((self.current_content.id, target_content.id))),
            ):
                combined_content: Sequence[ContentBase[Any]] = ContentBase.get_many(
                    override_ids,
                )
            else:
                combined_content = sorted(
                    (self.current_content, target_content),
                    key=lambda c: (not c.IS_OPAQUE, -c.priority),
                )

            LOGGER.debug(
                "Created new Combination(content=(%s)) with priority %.3f",
                ", ".join(c.id for c in combined_content),
                combo_content.priority,
            )

            # Force-stop the current content, it will be replaced by the combination
            self.current_content.stop(StopType.CANCEL)

            return combo_content.update_setting(
                "content",
                value=combined_content,
                invoke_callback=True,
            )

        LOGGER.debug(
            "Unable to combine %r with %r",
            self.current_content,
            target_content.id,
        )

        return target_content

    def _content_loop(self) -> None:
        """Loop through the content queue."""
        while not self._content_queue.empty():
            (
                _,
                self.current_content,
                parameters,
            ) = self._content_queue.get()

            LOGGER.info(
                "Displaying content with ID `%s` at priority %s",
                self.current_content.id,
                self.current_content.priority,
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
                LOGGER.info(
                    "Running setup sequence for %s",
                    self.current_content.id,
                )

                for _ in setup_gen:
                    set_content()

            # Actual loop through individual content instances:
            for _ in self.current_content:
                set_content()

            if self.current_content.stop_reason != StopType.PRIORITY and (
                teardown_gen := self.current_content.teardown()
            ):
                # Only run teardown if the stop isn't due to higher priority content
                LOGGER.info(
                    "Running teardown sequence for %s",
                    self.current_content.id,
                )

                for _ in teardown_gen:
                    set_content()

            self.current_content.active = False

            LOGGER.info("Content `%s` complete", self.current_content.id)

            if (
                self.current_content.persistent
                and self.current_content.priority != const.MAX_PRIORITY
                and self.current_content.stop_reason
                not in {StopType.CANCEL, StopType.EXPIRED}
            ):
                self._content_queue.add(
                    self.current_content,
                    parameters,
                )
                LOGGER.debug(
                    "Content `%s` is persistent with priority %s",
                    self.current_content.id,
                    self.current_content.priority,
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
        target_content.priority = payload["priority"] or const.MAX_PRIORITY
        parameters = cast(ContentParameters, payload.get("parameters", {}))

        if target_content.priority == const.MAX_PRIORITY:
            for _, ctnt, prms in self._content_queue:
                if ctnt is target_content:
                    self._content_queue.remove(ctnt, prms)
                    LOGGER.info(
                        "Removed content with ID `%s` from queue",
                        target_content.id,
                    )
                    break

            if self.current_content is target_content:
                self.current_content.stop(StopType.CANCEL)

            return

        target_content.priority = round(
            max(
                min(float(target_content.priority), const.MAX_PRIORITY),
                -const.MAX_PRIORITY,
            ),
            3,
        )

        if self.current_content is target_content:
            LOGGER.debug(
                "Updating %s priority to %s",
                target_content.id,
                target_content.priority,
            )
            self.current_content.priority = target_content.priority
            return

        # If they're both dynamic, they could be combined
        if isinstance(target_content, DynamicContent):
            target_content = self._attempt_combination(target_content)

        # Add it to the queue, this will get picked up within the _content_loop
        self._content_queue.add(target_content, parameters)

        LOGGER.info(
            "Added content with ID `%s` to queue with priority %s",
            target_content.id,
            target_content.priority,
        )

        # Lower value = higher priority
        if (
            self.current_content is not None
            and self.current_content.priority is not None
            and target_content.priority is not None
            and target_content.priority < self.current_content.priority
        ):
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
        if self.current_content is not None:
            self.current_content.priority = const.MAX_PRIORITY
        self.current_content = None

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
        for c in content:
            c.validate_setup()

            self._content[c.id] = c

            setting: Setting[Any]
            for setting in getattr(c, "settings", {}).values():
                setting.matrix = self

            LOGGER.info("Added content with ID `%s`", c.id)

            if isinstance(c, PreDefinedContent):
                c.generate_canvases(self.new_canvas)

            if hasattr(c, "settings"):
                for setting in c.settings.values():
                    self.mqtt_client.add_topic_callback(
                        setting.mqtt_topic,
                        setting.on_message,
                    )

    def zeros(self, *, dtype: DTypeLike = np.int_) -> NDArray[Any]:
        """Return a grid of zeros."""
        return np.zeros((self.height, self.width, 4), dtype=dtype)

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
            payload=value.id if value is not None else None,
            retain=True,
        )
        LOGGER.info(
            "Current Content: %s",
            value.id if value is not None else None,
        )

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
