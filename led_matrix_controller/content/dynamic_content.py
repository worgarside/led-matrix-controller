"""Module for dynamic content classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from json import dumps
from typing import TYPE_CHECKING, Any, Generator, final, get_type_hints

import numpy as np
from wg_utilities.loggers import get_streaming_logger

from .base import ContentBase, GridView
from .setting import Setting

if TYPE_CHECKING:
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
        """Convert the pixels to an image."""
        return self.colormap[self.pixels]

    def zeros(self, *, dtype: DTypeLike = np.int_) -> NDArray[Any]:
        """Return a grid of zeros."""
        return np.zeros((self.height, self.width), dtype=dtype)

    @abstractmethod
    def refresh_content(self) -> Generator[None, None, None]:
        """Refresh the content."""

    def update_setting(self, slug: str, value: Any) -> None:
        """Update a setting."""
        setting = self.settings[slug]

        setting.value = value

        payload = dumps(setting.value)

        LOGGER.debug("Sending payload %r to topic %r", payload, setting.mqtt_topic)

        setting.mqtt_client.publish(
            setting.mqtt_topic,
            payload,
            retain=True,
        )

    @final
    def __iter__(self) -> Generator[None, None, None]:
        """Iterate over the frames."""
        self.active = True

        while self.active:
            yield from self.refresh_content()
