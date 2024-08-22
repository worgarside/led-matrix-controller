"""Display the time."""

from __future__ import annotations

import itertools
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

            chain = []

            if (setup := content.setup()) is not None:
                chain.append(setup)

            chain.append(iter(content))

            if (teardown := content.teardown()) is not None:
                chain.append(teardown)

            content_chains[content.id] = itertools.chain(*chain)

        while self.active:
            for content in self.content:
                next(content_chains[content.id])

                self.pixels[
                    content.y_pos : content.y_pos + content.height,
                    content.x_pos : content.x_pos + content.width,
                ] = content.get_content()

            yield

    def zeros(self, *, dtype: DTypeLike = np.int_) -> NDArray[Any]:
        """Return a grid of zeros."""
        return np.zeros((self.height, self.width, 3), dtype=dtype)
