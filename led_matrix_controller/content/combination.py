"""Display the time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator

import numpy as np
from wg_utilities.loggers import get_streaming_logger

from .dynamic_content import DynamicContent

if TYPE_CHECKING:
    from content.base import GridView
    from numpy.typing import DTypeLike, NDArray

LOGGER = get_streaming_logger(__name__)


@dataclass(kw_only=True, slots=True)
class Combination(DynamicContent):
    """Combination of multiple content models."""

    content: tuple[DynamicContent, ...]

    def __post_init__(self) -> None:
        """Derive the instance id."""
        self.instance_id = "combo-" + "-".join(content.id for content in self.content)

        DynamicContent.__post_init__(self)

    def get_content(self) -> GridView:
        """Get the content."""
        return self.pixels

    def refresh_content(self) -> Generator[None, None, None]:
        """Refresh the content."""
        content_chains = {}
        for content in self.content:
            content.active = True
            content_chains[content.id] = content.chain_generators()

        while self.active:
            for content in self.content:
                try:
                    next(content_chains[content.id])
                except StopIteration:
                    if content.active:
                        LOGGER.error("Content %s stopped unexpectedly", content.id)  # noqa: TRY400
                        raise

                new_pixels = content.get_content()

                mask = np.any(new_pixels != (0, 0, 0, 0), axis=-1)

                self.pixels[
                    content.y_pos : content.y_pos + content.height,
                    content.x_pos : content.x_pos + content.width,
                ][mask] = new_pixels[mask]

            yield

    def zeros(self, *, dtype: DTypeLike = np.int_) -> NDArray[Any]:
        """Return a grid of zeros."""
        return np.zeros((self.height, self.width, 4), dtype=dtype)

    @property
    def content_id(self) -> str:
        """Return the ID of the content."""
        return "combo-" + "-".join(content.content_id for content in self.content)
