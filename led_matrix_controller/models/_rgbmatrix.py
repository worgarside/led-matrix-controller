from __future__ import annotations

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


__all__ = ["Color", "Font", "RGBMatrix", "RGBMatrixOptions", "DrawText"]
