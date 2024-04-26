from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

try:
    from rgbmatrix import RGBMatrix, RGBMatrixOptions  # type: ignore[import-not-found]
    from rgbmatrix.graphics import (  # type: ignore[import-not-found]
        Color,
        DrawText,
        Font,
    )
except ImportError:
    from socket import gethostname

    if gethostname() == "mtrxpi":
        raise

    from RGBMatrixEmulator import (  # type: ignore[import-untyped]
        RGBMatrix,
        RGBMatrixOptions,
    )
    from RGBMatrixEmulator.graphics import (  # type: ignore[import-untyped]
        Color,
        DrawText,
        Font,
    )


class Canvas(ABC):
    """Helper for typing/instance checks on canvases.

    Based on https://github.com/hzeller/rpi-rgb-led-matrix/blob/master/bindings/python/rgbmatrix/core.pyx
    """

    @abstractmethod
    def Clear(self) -> None:  # noqa: N802
        """Clear the canvas."""

    @abstractmethod
    def Fill(self, red: int, green: int, blue: int) -> None:  # noqa: N802
        """Fill the canvas with a color."""

    @abstractmethod
    def SetPixel(self, x: int, y: int, red: int, green: int, blue: int) -> None:  # noqa: N802
        """Set the color of a pixel."""

    @abstractmethod
    def SetImage(  # noqa: N802
        self,
        image: Image.Image,
        offset_x: int = 0,
        offset_y: int = 0,
        unsafe: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Set the image on the canvas."""


__all__ = [
    "Color",
    "Font",
    "RGBMatrix",
    "RGBMatrixOptions",
    "DrawText",
    "Canvas",
]
