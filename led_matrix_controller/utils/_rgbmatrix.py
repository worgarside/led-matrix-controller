from __future__ import annotations

from abc import ABC, abstractmethod

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

    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from PIL import Image

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
    def Clear(self) -> None:  # ruff:ignore[invalid-function-name]
        """Clear the canvas."""

    @abstractmethod
    def Fill(self, red: int, green: int, blue: int) -> None:  # ruff:ignore[invalid-function-name]
        """Fill the canvas with a color."""

    @abstractmethod
    def SetPixel(self, x: int, y: int, red: int, green: int, blue: int) -> None:  # ruff:ignore[invalid-function-name]
        """Set the color of a pixel."""

    @abstractmethod
    def SetImage(  # ruff:ignore[invalid-function-name]
        self,
        image: Image.Image,
        offset_x: int = 0,
        offset_y: int = 0,
        unsafe: bool = True,  # ruff:ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
    ) -> None:
        """Set the image on the canvas."""


__all__ = [
    "Canvas",
    "Color",
    "DrawText",
    "Font",
    "RGBMatrix",
    "RGBMatrixOptions",
]
