"""Rain simulation cellular automaton."""

from __future__ import annotations

from dataclasses import dataclass
from enum import unique
from typing import Annotated, Literal

import numpy as np
from utils import const

from .ca import (
    Direction,
    Grid,
    Mask,
    MaskGen,
    StateBase,
    TargetSlice,
)
from .setting import FrequencySetting, ParameterSetting  # noqa: TCH001


@unique
class State(StateBase):
    """Enum representing the state of a cell."""

    NULL = 0, " "
    RAINDROP = 1, "O", (13, 94, 255)
    SPLASHDROP = 2, "o", (107, 155, 250)
    SPLASH_LEFT = 3, "*", (170, 197, 250)
    SPLASH_RIGHT = 4, "*", (170, 197, 250)


@dataclass(slots=True)
class RainingGrid(Grid):
    """Basic rain simulation."""

    rain_chance: Annotated[float, ParameterSetting()] = 0.025
    rain_speed: Annotated[int, FrequencySetting()] = 1
    splash_speed: Annotated[int, FrequencySetting()] = 3

    id: str = "raining-grid"


@RainingGrid.rule(State.RAINDROP, target_slice=0)
def generate_raindrops(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Generate raindrops at the top of the grid."""
    shape = ca._grid[target_slice].shape

    def _mask() -> Mask:
        return const.RNG.random(shape) < ca.rain_chance

    return _mask


@RainingGrid.rule(State.RAINDROP, target_slice=(slice(1, None), slice(None)))
def move_rain_down(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Move raindrops down one cell."""
    lower_slice = ca._grid[target_slice]
    upper_slice = ca._grid[ca.translate_slice(target_slice, vrt=Direction.UP)]
    raindrop = State.RAINDROP.state
    null = State.NULL.state

    def _mask() -> Mask:
        return (upper_slice == raindrop) & (lower_slice == null)  # type: ignore[no-any-return]

    return _mask


@RainingGrid.rule(State.NULL)
def top_of_rain_down(ca: RainingGrid, _: TargetSlice) -> MaskGen:
    """Move the top of a raindrop down."""
    above_slice = ca._grid[slice(None, -2), slice(None)]
    middle_slice = ca._grid[slice(1, -1), slice(None)]
    below_slice = ca._grid[slice(2, None), slice(None)]
    top_row = ca._grid[0]
    second_row = ca._grid[1]
    penultimate_row = ca._grid[-2]
    last_row = ca._grid[-1]

    raindrop = State.RAINDROP.state

    def _mask() -> Mask:
        return np.vstack(
            (
                (top_row == raindrop) & (second_row == raindrop),
                (
                    (above_slice != raindrop)
                    & (middle_slice == raindrop)
                    & (below_slice == raindrop)
                ),
                (last_row == raindrop) & (penultimate_row != raindrop),
            )
        )

    return _mask


def _splash(
    ca: RainingGrid,
    target_slice: TargetSlice,
    *,
    source_slice_direction: Literal[Direction.LEFT, Direction.RIGHT],
) -> MaskGen:
    # TODO this would be better as "will be NULL", instead of "is NULL"
    source_slice = ca._grid[
        ca.translate_slice(
            target_slice,
            vrt=Direction.DOWN,
            hrz=source_slice_direction,
        )
    ]
    splash_spots = ca._grid[target_slice]
    below_slice = ca._grid[ca.translate_slice(target_slice, vrt=Direction.DOWN)]

    raindrop = State.RAINDROP.state
    null = State.NULL.state

    def _mask() -> Mask:
        return (  # type: ignore[no-any-return]
            (source_slice == raindrop) & (splash_spots == null) & (below_slice == null)
        )

    return _mask


@RainingGrid.rule(
    State.SPLASH_LEFT,
    target_slice=(-2, slice(None, -1)),
    frequency="splash_speed",
)
def splash_left(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Create a splash to the left."""
    return _splash(ca, target_slice, source_slice_direction=Direction.RIGHT)


@RainingGrid.rule(
    State.SPLASH_RIGHT,
    target_slice=(-2, slice(1, None)),
    frequency="splash_speed",
)
def splash_right(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Create a splash to the right."""
    return _splash(ca, target_slice, source_slice_direction=Direction.LEFT)


def _splash_high(
    ca: RainingGrid,
    target_slice: TargetSlice,
    *,
    splash_state: State,
    source_slice_direction: Literal[Direction.LEFT, Direction.RIGHT],
) -> MaskGen:
    source_slice = ca._grid[
        ca.translate_slice(
            target_slice,
            vrt=Direction.DOWN,
            hrz=source_slice_direction,
        )
    ]

    state = splash_state.state

    def _mask() -> Mask:
        return (  # type: ignore[no-any-return]
            source_slice == state
        )  # & ca._grid[target_slice] will be NULL

    return _mask


@RainingGrid.rule(
    State.SPLASH_LEFT,
    target_slice=(-3, slice(None, -1)),
    frequency="splash_speed",
)
def splash_left_high(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Continue the splash to the left."""
    return _splash_high(
        ca,
        target_slice,
        splash_state=State.SPLASH_LEFT,
        source_slice_direction=Direction.RIGHT,
    )


@RainingGrid.rule(
    State.SPLASH_RIGHT,
    target_slice=(-3, slice(1, None)),
    frequency="splash_speed",
)
def splash_right_high(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Continue the splash to the right."""
    return _splash_high(
        ca,
        target_slice,
        splash_state=State.SPLASH_RIGHT,
        source_slice_direction=Direction.LEFT,
    )


@RainingGrid.rule(
    State.NULL,
    target_slice=(slice(-3, None), slice(None)),
    frequency="splash_speed",
)
def remove_splashes(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Remove any splashes - they only last one frame."""
    any_splash = (
        State.SPLASH_LEFT.state,
        State.SPLASH_RIGHT.state,
        State.SPLASHDROP.state,
    )
    view = ca._grid[target_slice]

    def _mask() -> Mask:
        return np.isin(view, any_splash)

    return _mask


@RainingGrid.rule(State.SPLASHDROP, target_slice=-3, frequency="splash_speed")
def create_splashdrop(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Convert a splash to a splashdrop."""
    active_splashes = State.SPLASH_LEFT.state, State.SPLASH_RIGHT.state
    view = ca._grid[target_slice]

    def _mask() -> Mask:
        return np.isin(view, active_splashes)

    return _mask


@RainingGrid.rule(
    State.SPLASHDROP,
    target_slice=(slice(-3, None)),
    frequency="splash_speed",
)
def move_splashdrop_down(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Move the splashdrop down."""
    source_slice = ca._grid[ca.translate_slice(target_slice, vrt=Direction.UP)]
    splashdrop = State.SPLASHDROP.state

    def _mask() -> Mask:
        return source_slice == splashdrop  # type: ignore[no-any-return]
        # & ca._grid[target_slice] will be State.NULL

    return _mask


__all__ = ["RainingGrid", "State"]
