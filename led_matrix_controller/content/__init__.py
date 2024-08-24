from __future__ import annotations

from pathlib import Path

from .clock import Clock
from .combination import Combination
from .gif import GifViewer
from .image import ImageViewer
from .now_playing import NowPlaying
from .raining_grid import RainingGrid
from .sorting_algorithms import Sorter

LIBRARY = (
    Clock(persistent=True),
    Combination(),
    GifViewer(path=Path("door/animated.gif")),
    ImageViewer(path=Path("door/closed.bmp"), display_seconds=5),
    NowPlaying(persistent=True),
    RainingGrid(persistent=True),
    Sorter(),
)


__all__ = [
    "LIBRARY",
    "Clock",
    "Combination",
    "GifViewer",
    "ImageViewer",
    "NowPlaying",
    "RainingGrid",
    "Sorter",
]
