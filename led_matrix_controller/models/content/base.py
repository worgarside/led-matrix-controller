"""Base class for content models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from PIL import Image


class ContentBase(ABC):
    """Base class for content models."""

    @property
    @abstractmethod
    def frames(self) -> Generator[Image.Image, None, None]:
        """Generator of frames to display."""
