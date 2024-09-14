"""Rain simulation cellular automaton."""

from __future__ import annotations

from dataclasses import dataclass
from enum import unique
from itertools import islice
from typing import TYPE_CHECKING, Annotated, ClassVar, Generator, Literal, cast

import numpy as np
from content.automaton import (
    Automaton,
    Direction,
    Mask,
    MaskGen,
    TargetSlice,
)
from utils import const
from wg_utilities.loggers import get_streaming_logger

from .base import GridView, StateBase, StopType
from .setting import (
    FrequencySetting,
    TransitionableParameterSetting,
)

if TYPE_CHECKING:
    from content.base import GridView

LOGGER = get_streaming_logger(__name__)


@unique
class State(StateBase):
    """Enum representing the state of a cell."""

    NULL = 0, " "
    RAINDROP = 1, "O", (13, 94, 255, 255)
    SPLASHDROP = 2, "o", (107, 155, 250, 255)
    SPLASH_LEFT = 3, "*", (170, 197, 250, 255)
    SPLASH_RIGHT = 4, "*", (170, 197, 250, 255)


@dataclass(kw_only=True, slots=True)
class RainingGrid(Automaton):
    """Basic rain simulation."""

    TRACK_STATES_DURATION: ClassVar[tuple[int, ...]] = ()  # (State.OLD_PLANT.state,)

    IS_OPAQUE: ClassVar[bool] = True
    # i.e. has a full background to overwrite previous content

    STATE = State

    def stop_when_no_rain(self) -> None:
        """Stop the simulation when there is no rain."""
        if self.active:
            self.stop(StopType.EXPIRED)

    rain_chance: Annotated[
        float,
        TransitionableParameterSetting(
            minimum=0,
            maximum=100,
            transition_rate=0.0001,
            fp_precision=4,
            # Anything above 0.1 is too much rain!
            payload_modifier=lambda x, _: x / 1000,
            icon="mdi:cloud-percent-outline",
            unit_of_measurement="%",
            value_callbacks={0: stop_when_no_rain},
        ),
    ] = 0.025

    rain_speed: Annotated[
        int,
        FrequencySetting(
            minimum=1,
            maximum=1000,
            invoke_settings_callback=True,
            icon="mdi:speedometer",
            unit_of_measurement="ticks",
        ),
    ] = 1

    splash_speed: Annotated[
        int,
        FrequencySetting(
            minimum=1,
            maximum=1000,
            invoke_settings_callback=True,
            icon="mdi:speedometer",
            unit_of_measurement="ticks",
        ),
    ] = 8

    def teardown(self) -> Generator[None, None, None]:
        """Transition the rain chance to 0 then run the simulation until the grid is clear."""
        rain_chance_setting = cast(
            TransitionableParameterSetting[float],
            self.settings["rain_chance"],
        )

        total_change = original_rain_chance = rain_chance_setting.value
        original_transition_rate = rain_chance_setting.transition_rate

        ticks = 0.5 * const.TICKS_PER_SECOND

        rain_chance_setting.transition_rate = max(
            round(
                total_change / ticks,
                rain_chance_setting.fp_precision,
            ),
            0,
        )

        LOGGER.debug(
            "Modified `rain_chance` transition rate from %f to %f",
            original_transition_rate,
            rain_chance_setting.transition_rate,
        )

        rain_chance_setting.value = 0

        for _ in islice(self, const.TICKS_PER_SECOND * 5):
            yield

            if np.all(self.pixels == 0):
                break

        rain_chance_setting.value = original_rain_chance
        rain_chance_setting.transition_rate = original_transition_rate

        LOGGER.debug(
            "Reset `rain_chance` to %f and transition rate to %f",
            original_rain_chance,
            original_transition_rate,
        )

        self._stop_rules_thread()


@RainingGrid.rule(State.RAINDROP, target_slice=0, frequency="rain_speed")
def generate_raindrops(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Generate raindrops at the top of the grid."""
    shape = ca.pixels[target_slice].shape

    def mask_gen(_: GridView) -> Mask:
        return const.RNG.random(shape) < ca.rain_chance

    return mask_gen


@RainingGrid.rule(
    State.RAINDROP,
    target_slice=(slice(1, None)),
    frequency="rain_speed",
)
def move_rain_down(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Move raindrops down one cell."""
    above_slice = ca.translate_slice(target_slice, vrt=Direction.UP)

    def mask_gen(pixels: GridView) -> Mask:
        return (pixels[above_slice] == State.RAINDROP.state) & (  # type: ignore[no-any-return]
            pixels[target_slice] == State.NULL.state
        )

    return mask_gen


@RainingGrid.rule(State.NULL, frequency="rain_speed")
def top_of_rain_down(_: RainingGrid, __: TargetSlice) -> MaskGen:
    """Move the top of a raindrop down."""
    raindrop = State.RAINDROP.state
    above_slice = slice(None, -2), slice(None)
    middle_slice = slice(1, -1), slice(None)
    below_slice = slice(2, None), slice(None)

    def mask_gen(pixels: GridView) -> Mask:
        top_row = pixels[0]
        second_row = pixels[1]
        above_pixels = pixels[above_slice]
        middle_pixels = pixels[middle_slice]
        below_pixels = pixels[below_slice]
        last_row = pixels[-1]
        penultimate_row = pixels[-2]

        return np.vstack(
            (
                (top_row == raindrop) & (second_row == raindrop),
                (
                    (above_pixels != raindrop)
                    & (middle_pixels == raindrop)
                    & (below_pixels == raindrop)
                ),
                (last_row == raindrop) & (penultimate_row != raindrop),
            ),
        )

    return mask_gen


def _splash(
    ca: RainingGrid,
    target_slice: TargetSlice,
    *,
    source_slice_direction: Literal[Direction.LEFT, Direction.RIGHT],
) -> MaskGen:
    source_slice = ca.translate_slice(
        target_slice,
        vrt=Direction.DOWN,
        hrz=source_slice_direction,
    )
    below_slice = ca.translate_slice(target_slice, vrt=Direction.DOWN)

    def mask_gen(pixels: GridView) -> Mask:
        return (  # type: ignore[no-any-return]
            (pixels[source_slice] == State.RAINDROP.state)
            & (pixels[target_slice] == State.NULL.state)
            & (pixels[below_slice] == State.NULL.state)
        )

    return mask_gen


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
    state = splash_state.state
    below_slice = ca.translate_slice(
        target_slice,
        vrt=Direction.DOWN,
        hrz=source_slice_direction,
    )

    def mask_gen(pixels: GridView) -> Mask:
        return (pixels[below_slice] == state) & (pixels[target_slice] == 0)  # type: ignore[no-any-return]

    return mask_gen


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
def remove_splashes(_: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Remove any splashes - they only last one frame."""
    any_splash = (
        State.SPLASH_LEFT.state,
        State.SPLASH_RIGHT.state,
        State.SPLASHDROP.state,
    )

    def mask_gen(pixels: GridView) -> Mask:
        return np.isin(pixels[target_slice], any_splash)

    return mask_gen


@RainingGrid.rule(State.SPLASHDROP, target_slice=-3, frequency="splash_speed")
def create_splashdrop(_: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Convert a splash to a splashdrop."""
    active_splashes = State.SPLASH_LEFT.state, State.SPLASH_RIGHT.state

    def mask_gen(pixels: GridView) -> Mask:
        return np.isin(
            pixels[target_slice],
            active_splashes,
        )  # & np.equal(pixels[target_slice], State.NULL.state)

    return mask_gen


@RainingGrid.rule(
    State.SPLASHDROP,
    target_slice=(slice(-3, None)),
    frequency="splash_speed",
)
def move_splashdrop_down(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Move the splashdrop down."""

    def mask_gen(pixels: GridView) -> Mask:
        source_pixels = pixels[ca.translate_slice(target_slice, vrt=Direction.UP)]
        return (source_pixels == State.SPLASHDROP.state) & (  # type: ignore[no-any-return]
            pixels[target_slice] == State.NULL.state
        )

    return mask_gen


__all__ = ["RainingGrid", "State"]
