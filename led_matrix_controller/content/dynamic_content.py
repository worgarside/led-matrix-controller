"""Module for dynamic content classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from json import dumps
from typing import TYPE_CHECKING, Any, Self, get_type_hints

import numpy as np
from wg_utilities.loggers import get_streaming_logger

from .base import ContentBase, GridView
from .setting import Setting

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import DTypeLike, NDArray

LOGGER = get_streaming_logger(__name__)


@dataclass(kw_only=True, slots=True)
class DynamicContent(ContentBase[GridView], ABC):
    """Base class for content which is dynamically created."""

    canvas_count: None = field(init=False, default=None)
    """None == Infinity. Right?"""

    settings: dict[str, Setting[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compile settings from type hints."""
        super(DynamicContent, self).__post_init__()
        self.pixels = self.zeros()

        try:
            type_hints = get_type_hints(
                self.__class__,
                include_extras=True,
            )
        except TypeError:
            LOGGER.error("Failed to get type hints for %s", self.__class__.__name__)  # noqa: TRY400
            raise

        for field_name, field_type in type_hints.items():
            for annotation in getattr(field_type, "__metadata__", ()):
                if isinstance(annotation, Setting):
                    annotation.setup(
                        field_name=field_name,
                        instance=self,
                        type_=field_type.__origin__,
                    )

                    self.settings[annotation.slug] = annotation

    def get_content(self) -> GridView:
        """Converts the pixel array to an image using the colormap.

        If pixel values exceed the colormap bounds, rescales them to fit within the valid range before conversion.
        """
        try:
            return self.colormap[self.pixels]
        except IndexError as err:
            # This is usually due to an increase in magnitude, which then causes the pixels values
            # to exceed the colormap bounds
            LOGGER.warning(
                "IndexError in get_content for %s. Pixel values exceed colormap bounds. Scaling down. %s",
                self.__class__.__name__,
                err,
            )
            colormap_size = self.colormap.shape[0]

            # Scale values to fit within the colormap range [0, colormap_size - 1]
            scaled_pixels = (
                self.pixels / np.max(self.pixels) * (colormap_size - 1)
            ).astype(self.pixels.dtype)  # Preserving original dtype, ensure it's int

            return self.colormap[scaled_pixels]

    def zeros(self, *, dtype: DTypeLike = np.int_) -> NDArray[Any]:
        """Returns a zero-filled grid array matching the content's height and width.

        Args:
                dtype: Data type of the returned array (default: int).

        Returns:
                A NumPy array of zeros with shape (height, width).
        """
        return np.zeros((self.height, self.width), dtype=dtype)

    @abstractmethod
    def refresh_content(self) -> Generator[None, None, None]:
        """Refresh the content."""

    def update_setting(
        self,
        slug: str,
        value: Any,
        *,
        invoke_callback: bool = False,
    ) -> Self:
        """Update a setting."""
        setting = self.settings[slug]

        setting.value = value

        payload = dumps(
            setting.value,
            default=lambda x: x.id
            if isinstance(x, ContentBase)
            else self._json_encode(x),
        )

        LOGGER.debug(
            "Sending payload %r (raw: %r) to topic %r",
            payload,
            setting.value,
            setting.mqtt_topic,
        )

        setting.mqtt_client.publish(
            setting.mqtt_topic,
            payload,
            retain=True,
        )

        if invoke_callback:
            self.setting_update_callback(update_setting=slug)

        return self

    def __iter__(self) -> Generator[None, None, None]:
        """Iterate over the frames."""
        self.active = True

        while self.active:
            yield from self.refresh_content()
