"""Rain simulation cellular automaton."""

from __future__ import annotations

from dataclasses import dataclass
from enum import unique
from functools import partial
from typing import Annotated, ClassVar, Literal

import numpy as np
from utils import const
from utils.cellular_automata.automaton import (
    Automaton,
    Direction,
    GridView,
    Mask,
    MaskGen,
    TargetSlice,
)
from utils.cellular_automata.setting import (  # noqa: TCH002
    FrequencySetting,
    ParameterSetting,
)

from .base import StateBase


@unique
class State(StateBase):
    """Enum representing the state of a cell."""

    NULL = 0, " "
    RAINDROP = 1, "O", (13, 94, 255)
    SPLASHDROP = 2, "o", (107, 155, 250)
    SPLASH_LEFT = 3, "*", (170, 197, 250)
    SPLASH_RIGHT = 4, "*", (170, 197, 250)


@dataclass(slots=True)
class RainingGrid(Automaton):
    """Basic rain simulation."""

    STATE: ClassVar[type[StateBase]] = State

    colormap = State.colormap()

    rain_chance: Annotated[
        float,
        ParameterSetting(min=0, max=1, transition_rate=(0.002, 0.2), fp_precision=3),
    ] = 0.025
    rain_speed: Annotated[int, FrequencySetting()] = 1
    splash_speed: Annotated[int, FrequencySetting()] = 8

    id: str = "raining-grid"


def generate_raindrops_mask(shape: tuple[int, int], chance: float) -> Mask:
    return const.RNG.random(shape) < chance


@RainingGrid.rule(State.RAINDROP, target_slice=0, frequency="rain_speed")
def generate_raindrops(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Generate raindrops at the top of the grid."""

    return partial(
        generate_raindrops_mask,
        shape=ca.grid[target_slice].shape,
        chance=ca.rain_chance,
    )


def move_rain_down_mask(
    upper_slice: GridView,
    raindrop: int,
    lower_slice: GridView,
    null: int,
) -> Mask:
    return (upper_slice == raindrop) & (lower_slice == null)  # type: ignore[no-any-return]


@RainingGrid.rule(
    State.RAINDROP,
    target_slice=(slice(1, None)),
    frequency="rain_speed",
)
def move_rain_down(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Move raindrops down one cell."""
    lower_slice = ca.grid[target_slice]
    upper_slice = ca.grid[ca.translate_slice(target_slice, vrt=Direction.UP)]
    raindrop = State.RAINDROP.state
    null = State.NULL.state

    return partial(move_rain_down_mask, upper_slice, raindrop, lower_slice, null)


def top_of_rain_down_mask(
    top_row: GridView,
    raindrop: int,
    second_row: GridView,
    above_slice: GridView,
    middle_slice: GridView,
    below_slice: GridView,
    last_row: GridView,
    penultimate_row: GridView,
) -> Mask:
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


@RainingGrid.rule(State.NULL, frequency="rain_speed")
def top_of_rain_down(ca: RainingGrid, _: TargetSlice) -> MaskGen:
    """Move the top of a raindrop down."""

    return partial(
        top_of_rain_down_mask,
        top_row=ca.grid[0],
        raindrop=State.RAINDROP.state,
        second_row=ca.grid[1],
        above_slice=ca.grid[slice(None, -2), slice(None)],
        middle_slice=ca.grid[slice(1, -1), slice(None)],
        below_slice=ca.grid[slice(2, None), slice(None)],
        last_row=ca.grid[-1],
        penultimate_row=ca.grid[-2],
    )


def _splash_mask(
    source_slice: GridView,
    raindrop: int,
    splash_spots: GridView,
    null: int,
    below_slice: GridView,
) -> Mask:
    return (  # type: ignore[no-any-return]
        (source_slice == raindrop) & (splash_spots == null) & (below_slice == null)
    )


def _splash(
    ca: RainingGrid,
    target_slice: TargetSlice,
    *,
    source_slice_direction: Literal[Direction.LEFT, Direction.RIGHT],
) -> MaskGen:
    # TODO this would be better as "will be NULL", instead of "is NULL"
    source_slice = ca.grid[
        ca.translate_slice(
            target_slice,
            vrt=Direction.DOWN,
            hrz=source_slice_direction,
        )
    ]
    splash_spots = ca.grid[target_slice]
    below_slice = ca.grid[ca.translate_slice(target_slice, vrt=Direction.DOWN)]

    raindrop = State.RAINDROP.state
    null = State.NULL.state

    return partial(_splash_mask, source_slice, raindrop, splash_spots, null, below_slice)


@RainingGrid.rule(
    State.SPLASH_LEFT,
    target_slice=(-2, slice(None, -1)),
    frequency="rain_speed",
)
def splash_left(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Create a splash to the left."""
    return _splash(ca, target_slice, source_slice_direction=Direction.RIGHT)


@RainingGrid.rule(
    State.SPLASH_RIGHT,
    target_slice=(-2, slice(1, None)),
    frequency="rain_speed",
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
    source_slice = ca.grid[
        ca.translate_slice(
            target_slice,
            vrt=Direction.DOWN,
            hrz=source_slice_direction,
        )
    ]

    state = splash_state.state

    return partial(np.equal, source_slice, state)  # & ca._grid[target_slice] will be NULL


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
    target_slice=(slice(-3, None)),
    frequency="splash_speed",
)
def remove_splashes(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Remove any splashes - they only last one frame."""
    any_splash = (
        State.SPLASH_LEFT.state,
        State.SPLASH_RIGHT.state,
        State.SPLASHDROP.state,
    )
    view = ca.grid[target_slice]

    return partial(np.isin, view, any_splash)


@RainingGrid.rule(State.SPLASHDROP, target_slice=-3, frequency="splash_speed")
def create_splashdrop(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Convert a splash to a splashdrop."""
    active_splashes = State.SPLASH_LEFT.state, State.SPLASH_RIGHT.state
    view = ca.grid[target_slice]

    return partial(np.isin, view, active_splashes)


@RainingGrid.rule(
    State.SPLASHDROP,
    target_slice=(slice(-3, None)),
    frequency="splash_speed",
)
def move_splashdrop_down(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Move the splashdrop down."""
    source_slice = ca.grid[ca.translate_slice(target_slice, vrt=Direction.UP)]
    splashdrop = State.SPLASHDROP.state

    return partial(np.equal, source_slice, splashdrop)


__all__ = ["RainingGrid", "State"]