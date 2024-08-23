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

    multiple_opaque: bool = False
    """Whether there are multiple opaque content models.

    If more than 1 content model is opaque, the mask to ignore the BG color can't be applied.
    Instead, `self.pixels` must be reset to `self.zeros()` before each tick.

    Normally, the single opaque model will be the first in the list and thus its background
    will overwrite the previous array.
    """

    def __post_init__(self) -> None:
        """Derive the instance id."""
        self.instance_id = "combo-" + "-".join(content.id for content in self.content)

        DynamicContent.__post_init__(self)

        opaque = [c for c in self.content if c.IS_OPAQUE]

        # If there is more than one opaque content model or the single opaque model is not
        # the first one (and would thus overwrite the previous array)
        if len(opaque) > 1 or (len(opaque) == 1 and opaque[0] != self.content[0]):
            self.multiple_opaque = True

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
            if self.multiple_opaque:
                self.pixels[:, :] = self.zeros()

            for content in self.content:
                try:
                    next(content_chains[content.id])
                except StopIteration:
                    if content.active:
                        LOGGER.error("Content %s stopped unexpectedly", content.id)  # noqa: TRY400
                        raise

                new_pixels = content.get_content()

                if not self.multiple_opaque and content.IS_OPAQUE:
                    # i.e. don't apply a mask
                    self.pixels[
                        content.y_pos : content.y_pos + content.height,
                        content.x_pos : content.x_pos + content.width,
                    ] = new_pixels
                else:
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
