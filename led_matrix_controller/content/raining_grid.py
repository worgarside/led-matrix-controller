"""Rain simulation cellular automaton."""

from __future__ import annotations

from dataclasses import dataclass
from enum import unique
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, cast

import numpy as np
from content.automaton import (
    Automaton,
    BooleanMask,
    Direction,
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

if TYPE_CHECKING:
    from collections.abc import Generator

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
    NEW_PLANT = 6, "P", (0, 192, 0, 255)
    MATURE_PLANT = 7, "P", (0, 128, 0, 255)

    LEAF_STEM_1 = 8, "-", (53, 143, 57, 255)
    LEAF_STEM_2 = 9, "-", (106, 194, 57, 255)
    LEAF_STEM_3A = 10, "-", (66, 245, 81, 255)
    LEAF_STEM_3B = 11, "-", (99, 142, 81, 255)

    LEAF_1 = 12, "L", (33, 191, 40, 255)
    LEAF_2 = 13, "L", (48, 145, 53, 255)
    LEAF_3 = 14, "L", (82, 171, 67, 255)
    LEAF_4 = 15, "L", (82, 171, 67, 255)

    DYING_PLANT = 16, "P", (140, 191, 31, 255)
    DEAD_PLANT = 17, "P", (135, 118, 20, 255)


@dataclass(kw_only=True, slots=True)
class RainingGrid(Automaton):
    """Basic rain simulation."""

    TRACK_STATES_DURATION: ClassVar[tuple[int, ...]] = (
        State.NEW_PLANT.state,
        State.GROWABLE_PLANT.state,
        State.DYING_PLANT.state,
        State.DEAD_PLANT.state,
    )

    IS_OPAQUE: ClassVar[bool] = True
    # i.e. has a full background to overwrite previous content

    STATE: ClassVar[type[StateBase]] = State

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

    plant_decay_propagation_speed: Annotated[
        int,
        ParameterSetting(
            minimum=1,
            maximum=const.seconds_to_ticks(300),  # 5 minutes
            icon="mdi:flower",
            unit_of_measurement="ticks",
            payload_modifier=const.seconds_to_ticks,
        ),
    ] = const.seconds_to_ticks(10)

    leaf_growth_chance: Annotated[
        float,
        ParameterSetting(
            minimum=0,
            maximum=100,
            icon="mdi:flower",
            unit_of_measurement="%",
            payload_modifier=lambda x, _: x / 100,
            invoke_settings_callback=True,
        ),
    ] = 0.005

    def teardown(self) -> Generator[None, None, None]:
        """Transition the rain chance to 0 then run the simulation until the grid is clear."""
        rain_chance_setting = cast(
            "TransitionableParameterSetting[float]",
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

        for _ in self:
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

        yield from self._stop_rules_thread()

    def __hash__(self) -> int:
        """Return the hash of the object."""
        return hash(self.id)


@RainingGrid.rule(State.RAINDROP, target_slice=0, frequency="rain_speed")
def generate_raindrops(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Generate raindrops at the top of the grid."""
    shape = ca.pixels[target_slice].shape

    def mask_gen(pixels: GridView) -> BooleanMask:
        return (pixels[target_slice] == State.NULL.state) & (  # type: ignore[no-any-return]
            const.RNG.random(shape) < ca.rain_chance
        )

    return mask_gen


@RainingGrid.rule(
    State.RAINDROP,
    target_slice=(slice(1, None)),
    frequency="rain_speed",
)
def move_rain_down(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Move raindrops down one cell."""
    above_slice = ca.translate_slice(target_slice, vrt=Direction.UP)

    def mask_gen(pixels: GridView) -> BooleanMask:
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

    def mask_gen(pixels: GridView) -> BooleanMask:
        return np.vstack(
            (
                (pixels[0] == raindrop) & (pixels[1] == raindrop),
                (
                    (pixels[above_slice] != raindrop)
                    & (pixels[middle_slice] == raindrop)
                    & (pixels[below_slice] == raindrop)
                ),
                (pixels[-1] == raindrop) & (pixels[-2] != raindrop),
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

    def mask_gen(pixels: GridView) -> BooleanMask:
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

    def mask_gen(pixels: GridView) -> BooleanMask:
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

    def mask_gen(pixels: GridView) -> BooleanMask:
        return np.isin(pixels[target_slice], any_splash)

    return mask_gen


@RainingGrid.rule(State.SPLASHDROP, target_slice=-3, frequency="splash_speed")
def create_splashdrop(_: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Convert a splash to a splashdrop."""
    active_splashes = State.SPLASH_LEFT.state, State.SPLASH_RIGHT.state

    def mask_gen(pixels: GridView) -> BooleanMask:
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
    above_slice = ca.translate_slice(target_slice, vrt=Direction.UP)

    def mask_gen(pixels: GridView) -> BooleanMask:
        return (pixels[above_slice] == State.SPLASHDROP.state) & (  # type: ignore[no-any-return]
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
    above_slice = ca.translate_slice(target_slice, vrt=Direction.UP)
    left_slice = ca.translate_slice(target_slice, hrz=Direction.LEFT)
    right_slice = ca.translate_slice(target_slice, hrz=Direction.RIGHT)

    null_state = State.NULL.state

    plant_states = (
        State.MATURE_PLANT.state,
        State.NEW_PLANT.state,
        State.GROWABLE_PLANT.state,
    )

    def mask_gen(pixels: GridView) -> BooleanMask:
        target_pixels = pixels[target_slice]

        mask = np.logical_and.reduce((
            pixels[above_slice] == State.SPLASHDROP.state,
            target_pixels == null_state,
            pixels[left_slice] == null_state,
            pixels[right_slice] == null_state,
            const.RNG.random(target_pixels.shape) < ca.plant_growth_chance,
        ))

        if len(valid_indices := np.where(mask[0])[0]) == 0:
            return np.zeros_like(mask)

        plant_indices = np.where(np.isin(target_pixels, plant_states))[0]

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
        State.MATURE_PLANT.state,
        State.LEAF_STEM_1.state,
        State.LEAF_STEM_2.state,
        State.LEAF_STEM_3A.state,
        State.LEAF_STEM_3B.state,
        State.LEAF_1.state,
        State.LEAF_2.state,
        State.LEAF_3.state,
        State.LEAF_4.state,
    )

    below_slice = ca.translate_slice(target_slice, vrt=Direction.DOWN)

    def mask_gen(pixels: GridView) -> BooleanMask:
        return (pixels[target_slice] == State.RAINDROP.state) & np.isin(  # type: ignore[no-any-return]
            pixels[below_slice],
            plant_states,
        )

    return mask_gen


@RainingGrid.rule(
    State.NEW_PLANT,
    target_slice=(slice(1, None)),
    frequency="rain_speed",
)
def plant_growth(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    above_slice = ca.translate_slice(target_slice, vrt=Direction.UP)
    left_slice = ca.translate_slice(target_slice, hrz=Direction.LEFT)
    right_slice = ca.translate_slice(target_slice, hrz=Direction.RIGHT)

    growable_state = State.GROWABLE_PLANT.state
    raindrop_state = State.RAINDROP.state

    def mask_gen(pixels: GridView) -> BooleanMask:
        source_pixels = pixels[target_slice]

        eligible_mask = (source_pixels == growable_state) & (
            pixels[above_slice] == raindrop_state
        )

        random_directions = const.RNG.choice(
            (0, 1, 2),
            size=source_pixels.shape,
            p=[0.2, 0.2, 0.6],
        )

        left_mask = ((random_directions == 0) & eligible_mask)[:, 1:]
        right_mask = ((random_directions == 1) & eligible_mask)[:, :-1]
        up_mask = ((random_directions == 2) & eligible_mask)[1:, :]  # noqa: PLR2004

        change_mask = np.zeros_like(source_pixels, dtype=bool)

        change_mask[:, :-1][left_mask & (pixels[left_slice] == State.NULL.state)] = True
        change_mask[:, 1:][right_mask & (pixels[right_slice] == State.NULL.state)] = True

        change_mask[:-1, :][up_mask] = True

        return change_mask

    return mask_gen


@RainingGrid.rule(
    State.MATURE_PLANT,
    target_slice=(slice(None, 2)),
    frequency="rain_speed",
)
def halt_plant_growth(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    below_slice = ca.translate_slice(target_slice, vrt=Direction.DOWN)

    mature_plant = State.MATURE_PLANT.state

    def mask_gen(pixels: GridView) -> BooleanMask:
        return (  # type: ignore[no-any-return]
            (pixels[target_slice] == State.NULL.state)
            | (pixels[target_slice] == State.NEW_PLANT.state)
            | (pixels[target_slice] == State.GROWABLE_PLANT.state)
        ) & (pixels[below_slice] == mature_plant)

    return mask_gen


@RainingGrid.rule(
    State.MATURE_PLANT,
    target_slice=(slice(1, None), slice(1, -1)),
    frequency="rain_speed",
)
def plant_aging(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    left_slice = ca.translate_slice(target_slice, hrz=Direction.LEFT)
    right_slice = ca.translate_slice(target_slice, hrz=Direction.RIGHT)
    above_slice = ca.translate_slice(target_slice, vrt=Direction.UP)

    new_plant_state = State.NEW_PLANT.state

    def mask_gen(pixels: GridView) -> BooleanMask:
        return (  # type: ignore[no-any-return]
            pixels[target_slice] == State.GROWABLE_PLANT.state
        ) & np.logical_or.reduce((
            pixels[left_slice] == new_plant_state,
            pixels[right_slice] == new_plant_state,
            pixels[above_slice] == new_plant_state,
        ))

    return mask_gen


@RainingGrid.rule(
    State.GROWABLE_PLANT,
    target_slice=(slice(1, None), slice(1, -1)),
    frequency="rain_speed",
)
def plant_aging_2(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    above_slice = ca.translate_slice(target_slice, vrt=Direction.UP)

    def mask_gen(pixels: GridView) -> BooleanMask:
        return (pixels[target_slice] == State.NEW_PLANT.state) & (  # type: ignore[no-any-return]
            pixels[above_slice] == State.RAINDROP.state
        )

    return mask_gen


@RainingGrid.rule(
    State.LEAF_STEM_1,
    target_slice=(slice(5, -3), slice(5, -5)),
    frequency="rain_speed",
    random_multiplier="leaf_growth_chance",
)
def leaf_growth_1(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    above_slice = ca.translate_slice(target_slice, vrt=Direction.UP)
    above2_slice = ca.translate_slice(target_slice, vrt=Direction.UP * 2)
    above3_slice = ca.translate_slice(target_slice, vrt=Direction.UP * 3)

    below_slice = ca.translate_slice(target_slice, vrt=Direction.DOWN)
    below2_slice = ca.translate_slice(target_slice, vrt=Direction.DOWN * 2)
    below3_slice = ca.translate_slice(target_slice, vrt=Direction.DOWN * 3)

    left_slice = ca.translate_slice(target_slice, hrz=Direction.LEFT)
    left_above_slice = ca.translate_slice(
        target_slice,
        hrz=Direction.LEFT,
        vrt=Direction.UP,
    )

    right_slice = ca.translate_slice(target_slice, hrz=Direction.RIGHT)
    right_above_slice = ca.translate_slice(
        target_slice,
        hrz=Direction.RIGHT,
        vrt=Direction.UP,
    )

    null_state = State.NULL.state

    def mask_gen(pixels: GridView) -> BooleanMask:
        return (  # type: ignore[no-any-return]
            np.logical_and.reduce((
                pixels[target_slice] == null_state,
                pixels[above_slice] == null_state,
                pixels[above2_slice] == null_state,
                pixels[above3_slice] == null_state,
                pixels[below_slice] == null_state,
                pixels[below2_slice] == null_state,
                pixels[below3_slice] == null_state,
            ))
            & (
                (
                    (pixels[left_slice] == State.MATURE_PLANT.state)
                    & (pixels[left_above_slice] == State.MATURE_PLANT.state)
                )
                | (
                    (pixels[right_slice] == State.MATURE_PLANT.state)
                    & (pixels[right_above_slice] == State.MATURE_PLANT.state)
                )
            )
        )

    return mask_gen


@RainingGrid.rule(
    State.LEAF_STEM_2,
    target_slice=(slice(4, -2), slice(4, -4)),
    frequency="rain_speed",
    random_multiplier="leaf_growth_chance",
)
def leaf_growth_2(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    left_slice = ca.translate_slice(target_slice, hrz=Direction.LEFT)
    left_2_slice = ca.translate_slice(target_slice, hrz=Direction.LEFT * 2)
    right_slice = ca.translate_slice(target_slice, hrz=Direction.RIGHT)
    right_2_slice = ca.translate_slice(target_slice, hrz=Direction.RIGHT * 2)

    def mask_gen(pixels: GridView) -> BooleanMask:
        return (pixels[target_slice] == State.NULL.state) & (  # type: ignore[no-any-return]
            (
                (pixels[left_slice] == State.LEAF_STEM_1.state)
                & (pixels[left_2_slice] == State.MATURE_PLANT.state)
            )
            | (
                (pixels[right_slice] == State.LEAF_STEM_1.state)
                & (pixels[right_2_slice] == State.MATURE_PLANT.state)
            )
        )

    return mask_gen


@RainingGrid.rule(
    (State.LEAF_STEM_3A, State.LEAF_STEM_3B),
    target_slice=(slice(2, -2), slice(2, -2)),
    frequency="rain_speed",
    random_multiplier="leaf_growth_chance",
)
def leaf_growth_3(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    left_slice = ca.translate_slice(target_slice, hrz=Direction.LEFT)
    right_slice = ca.translate_slice(target_slice, hrz=Direction.RIGHT)

    leaf_stem_2 = State.LEAF_STEM_2.state

    def mask_gen(pixels: GridView) -> BooleanMask:
        source_pixels = pixels[target_slice]

        return (  # type: ignore[no-any-return]
            (source_pixels == State.NULL.state)
            & ((pixels[left_slice] == leaf_stem_2) | (pixels[right_slice] == leaf_stem_2))
            # & (const.RNG.random(source_pixels.shape) < 0.5)
        )

    return mask_gen


@RainingGrid.rule(
    (State.LEAF_1, State.LEAF_2, State.LEAF_3, State.LEAF_4),
    target_slice=(slice(1, -2), slice(1, -2)),
    frequency="rain_speed",
    random_multiplier="leaf_growth_chance",
)
def leaf_growth_a(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    left_slice = ca.translate_slice(target_slice, hrz=Direction.LEFT)
    right_slice = ca.translate_slice(target_slice, hrz=Direction.RIGHT)
    below_slice = ca.translate_slice(target_slice, vrt=Direction.DOWN)
    above_right_slice = ca.translate_slice(
        target_slice,
        vrt=Direction.UP,
        hrz=Direction.RIGHT,
    )
    above_left_slice = ca.translate_slice(
        target_slice,
        vrt=Direction.UP,
        hrz=Direction.LEFT,
    )
    above_slice = ca.translate_slice(target_slice, vrt=Direction.UP)

    leaf_stem_2 = State.LEAF_STEM_2.state
    leaf_stem_3a = State.LEAF_STEM_3A.state

    def mask_gen(pixels: GridView) -> BooleanMask:
        source_pixels = pixels[target_slice]

        above_is_not_leaf_stem_2 = pixels[above_slice] != leaf_stem_2

        return (source_pixels == State.NULL.state) & np.logical_or.reduce((  # type: ignore[no-any-return]
            pixels[below_slice] == leaf_stem_3a,
            pixels[left_slice] == leaf_stem_3a,
            pixels[right_slice] == leaf_stem_3a,
            pixels[above_slice] == leaf_stem_3a,
            np.logical_or(
                np.logical_and(
                    pixels[above_right_slice] == leaf_stem_3a,
                    above_is_not_leaf_stem_2,
                ),
                np.logical_and(
                    pixels[above_left_slice] == leaf_stem_3a,
                    above_is_not_leaf_stem_2,
                ),
            ),
        ))

    return mask_gen


@RainingGrid.rule(
    (State.LEAF_1, State.LEAF_2, State.LEAF_3, State.LEAF_4),
    target_slice=(slice(1, -2), slice(1, -2)),
    frequency="rain_speed",
    random_multiplier="leaf_growth_chance",
)
def leaf_growth_b(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    left_slice = ca.translate_slice(target_slice, hrz=Direction.LEFT)
    right_slice = ca.translate_slice(target_slice, hrz=Direction.RIGHT)
    below_slice = ca.translate_slice(target_slice, vrt=Direction.DOWN)
    below_left_slice = ca.translate_slice(
        target_slice,
        vrt=Direction.DOWN,
        hrz=Direction.LEFT,
    )
    below_right_slice = ca.translate_slice(
        target_slice,
        vrt=Direction.DOWN,
        hrz=Direction.RIGHT,
    )
    above_right_slice = ca.translate_slice(
        target_slice,
        vrt=Direction.UP,
        hrz=Direction.RIGHT,
    )
    above_left_slice = ca.translate_slice(
        target_slice,
        vrt=Direction.UP,
        hrz=Direction.LEFT,
    )
    above_slice = ca.translate_slice(target_slice, vrt=Direction.UP)

    leaf_stem_2 = State.LEAF_STEM_2.state
    leaf_stem_3b = State.LEAF_STEM_3B.state

    def mask_gen(pixels: GridView) -> BooleanMask:
        source_pixels = pixels[target_slice]

        above_is_not_leaf_stem_2 = pixels[above_slice] != leaf_stem_2
        below_is_leaf_stem_2 = pixels[below_slice] == leaf_stem_2

        return (source_pixels == State.NULL.state) & np.logical_or.reduce((  # type: ignore[no-any-return]
            pixels[below_slice] == leaf_stem_3b,
            pixels[left_slice] == leaf_stem_3b,
            pixels[right_slice] == leaf_stem_3b,
            pixels[above_slice] == leaf_stem_3b,
            np.logical_or.reduce((
                np.logical_and(
                    pixels[above_right_slice] == leaf_stem_3b,
                    above_is_not_leaf_stem_2,
                ),
                np.logical_and(
                    pixels[above_left_slice] == leaf_stem_3b,
                    above_is_not_leaf_stem_2,
                ),
                np.logical_and(
                    pixels[below_right_slice] == leaf_stem_3b,
                    below_is_leaf_stem_2,
                ),
                np.logical_and(
                    pixels[below_left_slice] == leaf_stem_3b,
                    below_is_leaf_stem_2,
                ),
            )),
        ))

    return mask_gen


@RainingGrid.rule(
    State.DYING_PLANT,
    frequency=const.TICKS_PER_SECOND,
    predicate=lambda ca: ca.rain_chance < 0.002,  # noqa: PLR2004
    random_multiplier=0.25,
)
def kill_stagnant_plant(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Kill off plants when the rain level drops below a threshold.

    0.002 (i.e. 2 / 1000) is used above because the threshold is 2% rain intensity, and the payload modifier
    for `rain_chance` divides the value by 1000.
    """
    durations = ca.durations[target_slice]

    def mask_gen(pixels: GridView) -> BooleanMask:
        target_pixels = pixels[target_slice]
        return (  # type: ignore[no-any-return]
            (target_pixels == State.NEW_PLANT.state)
            | (target_pixels == State.GROWABLE_PLANT.state)
        ) & (durations >= ca.plant_decay_propagation_speed)

    return mask_gen


@RainingGrid.rule(
    State.DYING_PLANT,
    frequency=const.TICKS_PER_SECOND,
    random_multiplier=0.005,
)
def kill_really_stagnant_plant(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Kill off super-stagnant plants - usually when another plant has grown over them."""
    durations = ca.durations[target_slice]
    five_minutes = const.seconds_to_ticks(300)

    def mask_gen(pixels: GridView) -> BooleanMask:
        target_pixels = pixels[target_slice]
        return (  # type: ignore[no-any-return]
            (target_pixels == State.NEW_PLANT.state)
            | (target_pixels == State.GROWABLE_PLANT.state)
        ) & (durations >= five_minutes)

    return mask_gen


@RainingGrid.rule(
    State.DEAD_PLANT,
    frequency=const.TICKS_PER_SECOND,
    random_multiplier=0.25,
)
def kill_stagnant_plant_2(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    durations = ca.durations[target_slice]

    def mask_gen(pixels: GridView) -> BooleanMask:
        return (pixels[target_slice] == State.DYING_PLANT.state) & (  # type: ignore[no-any-return]
            durations >= ca.plant_decay_propagation_speed
        )

    return mask_gen


@RainingGrid.rule(State.NULL, frequency=const.TICKS_PER_SECOND, random_multiplier=0.1)
def kill_stagnant_plant_3(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    durations = ca.durations[target_slice]

    def mask_gen(pixels: GridView) -> BooleanMask:
        return (pixels[target_slice] == State.DEAD_PLANT.state) & (  # type: ignore[no-any-return]
            durations >= ca.plant_decay_propagation_speed
        )

    return mask_gen


@RainingGrid.rule(
    State.DYING_PLANT,
    target_slice=(slice(1, -1), slice(1, -1)),
    frequency=const.TICKS_PER_SECOND,
)
def propagate_plant_decay(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    above_slice = ca.translate_slice(target_slice, vrt=Direction.UP)
    left_slice = ca.translate_slice(target_slice, hrz=Direction.LEFT)
    right_slice = ca.translate_slice(target_slice, hrz=Direction.RIGHT)
    below_slice = ca.translate_slice(target_slice, vrt=Direction.DOWN)

    dead_state = State.DEAD_PLANT.state

    propagate_to = (
        State.MATURE_PLANT.state,
        State.LEAF_STEM_1.state,
        State.LEAF_STEM_2.state,
        State.LEAF_STEM_3A.state,
        State.LEAF_STEM_3B.state,
        State.LEAF_1.state,
        State.LEAF_2.state,
        State.LEAF_3.state,
        State.LEAF_4.state,
    )

    def mask_gen(pixels: GridView) -> BooleanMask:
        return np.isin(pixels[target_slice], propagate_to) & np.logical_or.reduce((  # type: ignore[no-any-return]
            pixels[above_slice] == dead_state,
            pixels[left_slice] == dead_state,
            pixels[right_slice] == dead_state,
            pixels[below_slice] == dead_state,
        ))

    return mask_gen


@RainingGrid.rule(
    State.NULL,
    target_slice=(slice(-1, None)),
    frequency=const.seconds_to_ticks(60),
)
def trim_plant_bases(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    """Remove mature plant cells with nothing above them.

    This makes it easier to track plant counts etc.
    """
    above_slice = ca.translate_slice(target_slice, vrt=Direction.UP)

    def mask_gen(pixels: GridView) -> BooleanMask:
        bottom_row = pixels[target_slice]
        above_row = pixels[above_slice]

        bottom_row_mature = bottom_row == State.MATURE_PLANT.state

        ca.plant_count = np.count_nonzero(
            (bottom_row_mature & (above_row == State.MATURE_PLANT.state))
            | (bottom_row == State.NEW_PLANT.state)
            | (bottom_row == State.GROWABLE_PLANT.state),
        )

        return bottom_row_mature & (above_row == State.NULL.state)  # type: ignore[no-any-return]

    return mask_gen


@RainingGrid.rule(
    State.NULL,
    target_slice=(slice(None, 1)),
    frequency=const.seconds_to_ticks(60),
)
def trim_plant_tops(ca: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    below_slice = ca.translate_slice(target_slice, vrt=Direction.DOWN)

    def mask_gen(pixels: GridView) -> BooleanMask:
        return (pixels[target_slice] == State.MATURE_PLANT.state) & (  # type: ignore[no-any-return]
            pixels[below_slice] == State.NULL.state
        )

    return mask_gen


@RainingGrid.rule(
    State.DYING_PLANT,
    target_slice=(slice(None, 1)),
    frequency=const.seconds_to_ticks(60),
    predicate=lambda ca: ca.rain_chance <= 0.0005,  # noqa: PLR2004  (0.5% rain intensity)
    random_multiplier=0.2,
)
def kill_off_full_height_plants(_: RainingGrid, target_slice: TargetSlice) -> MaskGen:
    def mask_gen(pixels: GridView) -> BooleanMask:
        return pixels[target_slice] == State.MATURE_PLANT.state  # type: ignore[no-any-return]

    return mask_gen


__all__ = ["RainingGrid", "State"]
