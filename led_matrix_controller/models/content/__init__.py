from __future__ import annotations

from enum import Enum, auto

from .raining_grid import RainingGrid


class ContentTag(Enum):
    IDLE = auto()


__all__ = [
    "RainingGrid",
]
