"""Module for dynamic content classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Generator, final, get_type_hints

from models.setting import Setting

from .base import ContentBase, ImageGetter, _get_image


@dataclass(kw_only=True, slots=True)
class DynamicContent(ContentBase, ABC):
    """Base class for content which is dynamically created."""

    canvas_count: None = field(init=False, default=None)
    """None == Infinity. Right?"""

    settings: dict[str, Setting[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compile settings from type hints."""
        for field_name, field_type in get_type_hints(
            self.__class__,
            include_extras=True,
        ).items():
            for annotation in getattr(field_type, "__metadata__", ()):
                if isinstance(annotation, Setting):
                    annotation.setup(
                        field_name=field_name,
                        instance=self,
                        type_=field_type.__origin__,
                    )

                    self.settings[annotation.slug] = annotation

    @final
    @property
    def content_getter(self) -> ImageGetter:
        """Return the image representation of the content."""
        if not hasattr(self, "_image_getter"):
            self._image_getter = partial(_get_image, self.colormap, self.pixels)

        return self._image_getter

    @abstractmethod
    def refresh_content(self) -> Generator[None, None, None]:
        """Refresh the content."""

    @final
    def __iter__(self) -> Generator[None, None, None]:
        """Iterate over the frames."""
        self.active = True

        while self.active:
            yield from self.refresh_content()
