"""Rain simulation cellular automaton."""

from __future__ import annotations

from dataclasses import dataclass
from enum import unique
from itertools import islice
from typing import Annotated, ClassVar, Generator, Literal, cast

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
    ParameterSetting,
    TransitionableParameterSetting,
)

LOGGER = get_streaming_logger(__name__)


@unique
class State(StateBase):
    """Enum representing the state of a cell."""

    NULL = 0, " "
    RAINDROP = 1, "O", (13, 94, 255, 255)
    SPLASHDROP = 2, "o", (107, 155, 250, 255)
    SPLASH_LEFT = 3, "*", (170, 197, 250, 255)
    SPLASH_RIGHT = 4, "*", (170, 197, 250, 255)

    GROWABLE_PLANT = 5, "P", (0, 192, 0, 255)
    NEW_PLANT = 6, "P", (0, 255, 0, 255)
    OLD_PLANT = 7, "P", (0, 128, 0, 255)

    LEAF_STEM_1 = 8, "-", (53, 143, 57, 255)
    LEAF_STEM_2 = 9, "-", (106, 143, 57, 255)
    LEAF_STEM_3A = 10, "-", (106, 143, 57, 255)
    LEAF_STEM_3B = 11, "-", (106, 143, 57, 255)


@dataclass(kw_only=True, slots=True)
class RainingGrid(Automaton):
    """Basic rain simulation."""

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

    plant_count: int = 0

    plant_limit: Annotated[
        int,
        ParameterSetting(
            minimum=0,
            maximum=const.MATRIX_WIDTH,
            icon="mdi:flower",
            unit_of_measurement="plants",
        ),
    ] = const.MATRIX_WIDTH // 20

    plant_growth_chance: Annotated[
        float,
        ParameterSetting(
            minimum=0,
            maximum=100,
            icon="mdi:flower",
            unit_of_measurement="%",
            payload_modifier=lambda x, _: x / 100,
        ),
    ] = 0.01

    distance_between_plants: Annotated[
        int,
        ParameterSetting(
            minimum=0,
            maximum=const.MATRIX_WIDTH - 3,  # -1 for each side, -1 for the plant itself
            icon="mdi:flower",
            unit_of_measurement="pixels",
        ),
    ] = 4

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


@RainingGrid.rule(
    State.GROWABLE_PLANT,
    target_slice=(slice(-1, None), slice(1, -1)),
    frequency="splash_speed",
    predicate=lambda ca: ca.plant_count < ca.plant_limit,
)
def start_plant(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    def mask_gen(pixels: GridView) -> Mask:
        above_pixels = pixels[ca.translate_slice(target_slice, vrt=Direction.UP)]
        left_pixels = pixels[ca.translate_slice(target_slice, hrz=Direction.LEFT)]
        right_pixels = pixels[ca.translate_slice(target_slice, hrz=Direction.RIGHT)]
        target_pixels = pixels[target_slice]

        mask = (
            (above_pixels == State.SPLASHDROP.state)
            & (target_pixels == State.NULL.state)
            & (left_pixels == State.NULL.state)
            & (right_pixels == State.NULL.state)
        ) & (const.RNG.random(above_pixels.shape) < ca.plant_growth_chance)

        if len(valid_indices := np.where(mask[0])[0]) == 0:
            return np.zeros_like(mask)

        plant_related_mask = (
            (target_pixels == State.OLD_PLANT.state)
            | (target_pixels == State.NEW_PLANT.state)
            | (target_pixels == State.GROWABLE_PLANT.state)
        )

        plant_indices = np.where(plant_related_mask[0])[0]

        if len(plant_indices) > 0:
            # Filter out valid indices that are within N cells of an existing plant
            for plant_idx in plant_indices:
                valid_indices = valid_indices[
                    (valid_indices < plant_idx - ca.distance_between_plants)
                    | (valid_indices > plant_idx + ca.distance_between_plants)
                ]

            if len(valid_indices) == 0:
                LOGGER.debug("No valid locations for plant growth")
                return np.zeros_like(mask)

        mask[0, :] = False
        mask[0, const.RNG.choice(valid_indices, size=1)] = True

        ca.plant_count += 1

        LOGGER.debug("Plant count: %d", ca.plant_count)

        return mask  # type: ignore[no-any-return]

    return mask_gen


@RainingGrid.rule(
    State.NULL,
    target_slice=(slice(None, -1)),
    frequency="rain_speed",
)
def remove_rain_on_plant(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Remove raindrops that are sitting on top of a plant."""
    plant_states = (
        State.GROWABLE_PLANT.state,
        State.NEW_PLANT.state,
        State.OLD_PLANT.state,
        State.LEAF_STEM_1.state,
        State.LEAF_STEM_2.state,
        State.LEAF_STEM_3A.state,
    )

    def mask_gen(pixels: GridView) -> Mask:
        source_pixels = pixels[target_slice]
        below_pixels = pixels[ca.translate_slice(target_slice, vrt=Direction.DOWN)]

        return (source_pixels == State.RAINDROP.state) & np.isin(  # type: ignore[no-any-return]
            below_pixels,
            plant_states,
        )

    return mask_gen


@RainingGrid.rule(
    State.NEW_PLANT,
    target_slice=(slice(1, None)),
    frequency="rain_speed",
)
def plant_growth(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    def mask_gen(pixels: GridView) -> Mask:
        source_pixels = pixels[target_slice]
        above_pixels = pixels[ca.translate_slice(target_slice, vrt=Direction.UP)]
        left_pixels = pixels[ca.translate_slice(target_slice, hrz=Direction.LEFT)]
        right_pixels = pixels[ca.translate_slice(target_slice, hrz=Direction.RIGHT)]

        plant_mask = source_pixels == State.GROWABLE_PLANT.state
        raindrop_mask = above_pixels == State.RAINDROP.state

        eligible_mask = plant_mask & raindrop_mask

        random_directions = const.RNG.choice(
            (0, 1, 2),
            size=source_pixels.shape,
            p=[0.2, 0.2, 0.6],
        )

        left_mask = ((random_directions == 0) & eligible_mask)[:, 1:]
        right_mask = ((random_directions == 1) & eligible_mask)[:, :-1]
        up_mask = ((random_directions == 2) & eligible_mask)[1:, :]  # noqa: PLR2004

        change_mask = np.zeros_like(source_pixels, dtype=bool)

        change_mask[:, :-1][left_mask & (left_pixels == State.NULL.state)] = True
        change_mask[:, 1:][right_mask & (right_pixels == State.NULL.state)] = True

        change_mask[:-1, :][up_mask] = True

        return change_mask

    return mask_gen


@RainingGrid.rule(
    State.OLD_PLANT,
    target_slice=(slice(None, 2)),
    frequency="rain_speed",
)
def halt_plant_growth(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    def mask_gen(pixels: GridView) -> Mask:
        source_pixels = pixels[target_slice]
        below_pixels = pixels[ca.translate_slice(target_slice, vrt=Direction.DOWN)]

        return (source_pixels == State.OLD_PLANT.state) | (  # type: ignore[no-any-return]
            below_pixels == State.OLD_PLANT.state
        )

    return mask_gen


@RainingGrid.rule(
    State.OLD_PLANT,
    target_slice=(slice(1, None), slice(1, -1)),
    frequency="rain_speed",
)
def plant_aging(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    def mask_gen(pixels: GridView) -> Mask:
        source_pixels = pixels[target_slice]

        left_pixels = pixels[ca.translate_slice(target_slice, hrz=Direction.LEFT)]
        right_pixels = pixels[ca.translate_slice(target_slice, hrz=Direction.RIGHT)]
        above_pixels = pixels[ca.translate_slice(target_slice, vrt=Direction.UP)]

        return (source_pixels == State.GROWABLE_PLANT.state) & (  # type: ignore[no-any-return]
            (left_pixels == State.NEW_PLANT.state)
            | (right_pixels == State.NEW_PLANT.state)
            | (above_pixels == State.NEW_PLANT.state)
        )

    return mask_gen


@RainingGrid.rule(
    State.GROWABLE_PLANT,
    target_slice=(slice(1, None), slice(1, -1)),
    frequency="rain_speed",
)
def plant_aging_2(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    def mask_gen(pixels: GridView) -> Mask:
        source_pixels = pixels[target_slice]
        above_pixels = pixels[ca.translate_slice(target_slice, vrt=Direction.UP)]
        return (source_pixels == State.NEW_PLANT.state) & (  # type: ignore[no-any-return]
            above_pixels == State.RAINDROP.state
        )

    return mask_gen


@RainingGrid.rule(
    State.LEAF_STEM_1,
    target_slice=(slice(5, -2), slice(5, -5)),
    frequency="rain_speed",
)
def leaf_growth_1(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    def mask_gen(pixels: GridView) -> Mask:
        source_pixels = pixels[target_slice]
        above_pixels = pixels[ca.translate_slice(target_slice, vrt=Direction.UP)]
        above2_pixels = pixels[ca.translate_slice(target_slice, vrt=Direction.UP * 2)]
        below_pixels = pixels[ca.translate_slice(target_slice, vrt=Direction.DOWN)]
        below2_pixels = pixels[ca.translate_slice(target_slice, vrt=Direction.DOWN * 2)]
        left_pixels = pixels[ca.translate_slice(target_slice, hrz=Direction.LEFT)]
        left_above_pixels = pixels[
            ca.translate_slice(target_slice, hrz=Direction.LEFT, vrt=Direction.UP)
        ]
        right_pixels = pixels[ca.translate_slice(target_slice, hrz=Direction.RIGHT)]
        right_above_pixels = pixels[
            ca.translate_slice(target_slice, hrz=Direction.RIGHT, vrt=Direction.UP)
        ]

        return (  # type: ignore[no-any-return]
            (source_pixels == State.NULL.state)
            & (above_pixels == State.NULL.state)
            & (above2_pixels == State.NULL.state)
            & (below_pixels == State.NULL.state)
            & (below2_pixels == State.NULL.state)
            & (
                (
                    (left_pixels == State.OLD_PLANT.state)
                    & (left_above_pixels == State.OLD_PLANT.state)
                )
                | (
                    (right_pixels == State.OLD_PLANT.state)
                    & (right_above_pixels == State.OLD_PLANT.state)
                )
            )
        )

    return mask_gen


@RainingGrid.rule(
    State.LEAF_STEM_2,
    target_slice=(slice(4, -2), slice(4, -4)),
    frequency="rain_speed",
)
def leaf_growth_2(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    def mask_gen(pixels: GridView) -> Mask:
        source_pixels = pixels[target_slice]
        left_pixels = pixels[ca.translate_slice(target_slice, hrz=Direction.LEFT)]
        left_2_pixels = pixels[ca.translate_slice(target_slice, hrz=Direction.LEFT * 2)]
        right_pixels = pixels[ca.translate_slice(target_slice, hrz=Direction.RIGHT)]
        right_2_pixels = pixels[ca.translate_slice(target_slice, hrz=Direction.RIGHT * 2)]
        return (source_pixels == State.NULL.state) & (  # type: ignore[no-any-return]
            (left_pixels == State.LEAF_STEM_1.state)
            & (left_2_pixels == State.OLD_PLANT.state)
            | (right_pixels == State.LEAF_STEM_1.state)
            & (right_2_pixels == State.OLD_PLANT.state)
        )

    return mask_gen


__all__ = ["RainingGrid", "State"]
