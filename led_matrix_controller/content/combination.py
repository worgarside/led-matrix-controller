"""Display the time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Generator

import numpy as np
from wg_utilities.loggers import get_streaming_logger

from .base import StopType
from .dynamic_content import DynamicContent
from .setting import ParameterSetting  # noqa: TCH001

if TYPE_CHECKING:
    import itertools

    from content.base import GridView
    from numpy.typing import DTypeLike, NDArray

LOGGER = get_streaming_logger(__name__)


@dataclass(kw_only=True, slots=True)
class Combination(DynamicContent):
    """Combination of multiple content models."""

    content: Annotated[
        tuple[DynamicContent, ...],
        ParameterSetting(
            payload_modifier=DynamicContent.get_many,
            strict=False,
            invoke_settings_callback=True,
            icon="mdi:vector-combine",
        ),
    ] = ()

    multiple_opaque: bool = False
    """Whether there are multiple opaque content models.

    If more than 1 content model is opaque, the mask to ignore the BG color can't be applied.
    Instead, `self.pixels` must be reset to `self.zeros()` before each tick.

    Normally, the single opaque model will be the first in the list and thus its background
    will overwrite the previous array.
    """

    def __post_init__(self) -> None:
        """Derive the instance id."""
        DynamicContent.__post_init__(self)

        for c in self.content:
            c.active = True

        self.setting_update_callback(update_setting="content")

    def get_content(self) -> GridView:
        """Get the content."""
        return self.pixels

    def get_content_chains(
        self,
        *,
        force_active: bool = False,
    ) -> dict[str, itertools.chain[None]]:
        """Get the content chains."""
        content_chains = {}
        for content in self.content:
            content.active = force_active or content.active
            content_chains[content.id] = content.chain_generators()

        LOGGER.debug("Content chains: %s", ", ".join(content_chains.keys()))

        return content_chains

    def refresh_content(self) -> Generator[None, None, None]:
        """Refresh the content."""
        content_chains = self.get_content_chains(force_active=True)

        while self.active:
            if self.multiple_opaque:
                self.pixels[:, :] = self.zeros()

            for content in self.content:
                try:
                    # Trigger content refresh
                    next(content_chains[content.id])
                except StopIteration:
                    if content.active:
                        LOGGER.error(  # noqa: TRY400
                            "Content %s stopped unexpectedly",
                            content.id,
                        )
                        raise

                    # Remove inactive content from the list
                    self.update_setting(
                        "content",
                        tuple(c for c in self.content if c.active),
                        invoke_callback=True,
                    )

                    content_chains = self.get_content_chains()
                except KeyError as err:
                    # Double check that it's not the generator that's thrown the KeyError
                    if (
                        str(err).strip("\"'") == content.id
                        and content.id not in content_chains
                    ):
                        content_chains = self.get_content_chains()
                        break  # Soft reset - go back to the while loop
                    else:
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

    def setting_update_callback(self, update_setting: str | None = None) -> None:
        """Change the opaque flag if necessary."""
        if update_setting == "content":
            if len(self.content) <= 1:
                self.stop(StopType.TRANSFORM_REQUIRED)
                return

            opaque = [c for c in self.content if c.IS_OPAQUE]

            # If there is more than one opaque content model or the single opaque model is not
            # the first one (and would thus overwrite the previous array)
            self.multiple_opaque = len(opaque) > 1 or (
                len(opaque) == 1 and opaque[0] != self.content[0]
            )

            LOGGER.debug(
                "Setting %r changed to %r, updated `multiple_opaque` to %s",
                update_setting,
                self.content_ids,
                self.multiple_opaque,
            )

            try:
                # Force an update for the "current content" topics
                self.settings[update_setting].matrix.publish_attributes()
            except AttributeError:
                LOGGER.debug(
                    "Setting %r does not have a `matrix` attribute",
                    update_setting,
                )

    def zeros(self, *, dtype: DTypeLike = np.int_) -> NDArray[Any]:
        """Return a grid of zeros."""
        return np.zeros((self.height, self.width, 4), dtype=dtype)

    @property
    def content_ids(self) -> tuple[str, ...]:
        """Get the content ids."""
        return tuple(c.id for c in self.content)
