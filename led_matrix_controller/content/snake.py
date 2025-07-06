"""Snake game."""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, ClassVar

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
)

if TYPE_CHECKING:
    from collections.abc import Generator

LOGGER = get_streaming_logger(__name__)


@enum.unique
class State(StateBase):
    """Enum representing the state of a cell."""

    NULL = 0, " "
    HEAD = 1, "O", (255, 94, 13, 255)
    BODY = 2, "o", (107, 155, 250, 255)
    FOOD = 3, "*", (10, 230, 190, 255)


@enum.unique
class SnakeDirection(enum.Enum):
    """Enum representing the direction of the snake."""

    UP = enum.auto()
    DOWN = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()

    @property
    def is_horizontal(self) -> bool:
        """Return True if the direction is horizontal."""
        return self in {SnakeDirection.LEFT, SnakeDirection.RIGHT}

    @property
    def is_vertical(self) -> bool:
        """Return True if the direction is vertical."""
        return self in {SnakeDirection.UP, SnakeDirection.DOWN}

    @property
    def translation_direction(self) -> Direction:
        """Return the translation direction of the snake."""
        return Direction[self.name]


@dataclass(kw_only=True, slots=True)
class Snake(Automaton):
    """Basic Snake game simulation."""

    QUEUE_SIZE: ClassVar[int] = 100

    TRACK_STATES_DURATION: ClassVar[tuple[int, ...]] = (
        State.HEAD.state,
        State.BODY.state,
    )

    IS_OPAQUE: ClassVar[bool] = True
    # i.e. has a full background to overwrite previous content

    STATE: ClassVar[type[StateBase]] = State

    turn_chance: Annotated[
        float,
        ParameterSetting(
            minimum=0,
            maximum=1,
            fp_precision=4,
            icon="mdi:dice-multiple",
            unit_of_measurement="%",
        ),
    ] = 0.05
    """Chance of the snake turning left or right on each tick."""

    turn_cooldown: Annotated[
        int,
        ParameterSetting(
            minimum=0,
            maximum=const.MATRIX_WIDTH,
            icon="mdi:timer-sand",
            unit_of_measurement="ticks",
        ),
    ] = 16
    """Number of ticks before the snake can turn again.

    Prevents the snake from turning into itself too often.
    """

    last_turn_tick: int = field(init=False, repr=False, default=0)

    snake_speed: Annotated[
        int,
        FrequencySetting(
            minimum=1,
            maximum=1000,
            invoke_settings_callback=True,
            icon="mdi:speedometer",
            unit_of_measurement="ticks",
        ),
    ] = 10
    """Speed of the snake."""

    snake_length: Annotated[
        int,
        ParameterSetting(
            minimum=1,
            maximum=const.MATRIX_WIDTH * const.MATRIX_HEIGHT,
            icon="mdi:snake",
            unit_of_measurement="cells",
            ha_read_only=True,
        ),
    ] = 1

    current_direction: SnakeDirection = field(
        init=False,
        repr=False,
        default=SnakeDirection.UP,
    )

    head_location: tuple[int, int] = field(
        init=False,
        repr=False,
    )

    food_generation_freq: Annotated[
        int,
        FrequencySetting(
            minimum=100,
            maximum=10000,
            icon="mdi:food-apple",
            unit_of_measurement="ticks",
        ),
    ] = 100
    """Frequency of food generation."""

    food_count: Annotated[
        int,
        ParameterSetting(
            minimum=0,
            maximum=const.MATRIX_WIDTH * const.MATRIX_HEIGHT,
            icon="mdi:food",
            unit_of_measurement="bits",
            ha_read_only=True,
        ),
    ] = 0
    """Number of food cells."""

    high_score: Annotated[
        int,
        ParameterSetting(
            minimum=0,
            maximum=const.MATRIX_WIDTH * const.MATRIX_HEIGHT,
            icon="mdi:trophy",
            unit_of_measurement="bits",
            ha_read_only=True,
        ),
    ] = 0

    def roll_direction_dice(  # noqa: C901
        self,
        *,
        force: bool = False,
        current_edge: SnakeDirection | None = None,
    ) -> bool:
        """Change the direction to a random other direction. Chance of change is configurable."""
        if force is False and random.random() >= self.turn_chance:  # noqa: S311
            return False

        match current_edge, self.current_direction:
            case SnakeDirection.LEFT, SnakeDirection.UP:
                self.current_direction = SnakeDirection.RIGHT
            case SnakeDirection.LEFT, SnakeDirection.DOWN:
                self.current_direction = SnakeDirection.RIGHT
            case SnakeDirection.RIGHT, SnakeDirection.UP:
                self.current_direction = SnakeDirection.LEFT
            case SnakeDirection.RIGHT, SnakeDirection.DOWN:
                self.current_direction = SnakeDirection.LEFT
            case SnakeDirection.UP, SnakeDirection.LEFT:
                self.current_direction = SnakeDirection.DOWN
            case SnakeDirection.UP, SnakeDirection.RIGHT:
                self.current_direction = SnakeDirection.DOWN
            case SnakeDirection.DOWN, SnakeDirection.LEFT:
                self.current_direction = SnakeDirection.UP
            case SnakeDirection.DOWN, SnakeDirection.RIGHT:
                self.current_direction = SnakeDirection.UP
            case None, dirn if dirn.is_vertical:
                self.current_direction = random.choice((  # noqa: S311
                    SnakeDirection.LEFT,
                    SnakeDirection.RIGHT,
                ))
            case None, dirn if dirn.is_horizontal:
                self.current_direction = random.choice((  # noqa: S311
                    SnakeDirection.UP,
                    SnakeDirection.DOWN,
                ))
            case _:
                raise ValueError(
                    f"Invalid edge/direction: {current_edge}/{self.current_direction}",
                )

        self.last_turn_tick = self.frame_index

        LOGGER.info("New direction: %s", self.current_direction)

        return True

    def setup(self) -> Generator[None, None, None]:
        """Setup the snake."""
        # Snake has its head in the middle and one body cell either above/below/next to it
        self.head_location = (self.width // 2, self.height // 2)
        self.pixels[self.head_location] = State.HEAD.state

        self.roll_direction_dice(force=True)

        LOGGER.info("Initial direction: %s", self.current_direction)

        if self.current_direction.is_vertical:
            body_location = (
                self.head_location[0] - self.current_direction.translation_direction,
                self.head_location[1],
            )
        elif self.current_direction.is_horizontal:
            body_location = (
                self.head_location[0],
                self.head_location[1] - self.current_direction.translation_direction,
            )

        self.pixels[body_location] = State.BODY.state

        yield

    def __hash__(self) -> int:
        """Return the hash of the automaton."""
        return hash(self.id)


@Snake.rule(
    State.HEAD,
    frequency="snake_speed",
    target_slice=(slice(None, -1), slice(None)),
    predicate=lambda ca: ca.current_direction == SnakeDirection.UP,
)
def move_snake_head_up(ca: Snake, target_slice: TargetSlice) -> MaskGen:
    """Move the snake's head up."""
    prev_slice = ca.translate_slice(
        target_slice,
        vrt=SnakeDirection.DOWN.translation_direction,
    )

    def mask_gen(pixels: GridView) -> BooleanMask:
        ca.head_location = (
            ca.head_location[0] + SnakeDirection.UP.translation_direction,
            ca.head_location[1],
        )

        return (pixels[prev_slice] == State.HEAD.state) & (  # type: ignore[no-any-return]
            (pixels[target_slice] == State.NULL.state)
            | (pixels[target_slice] == State.FOOD.state)
        )

    return mask_gen


@Snake.rule(
    State.HEAD,
    frequency="snake_speed",
    target_slice=(slice(1, None), slice(None)),
    predicate=lambda ca: ca.current_direction == SnakeDirection.DOWN,
)
def move_snake_head_down(ca: Snake, target_slice: TargetSlice) -> MaskGen:
    """Move the snake's head down."""
    prev_slice = ca.translate_slice(
        target_slice,
        vrt=SnakeDirection.UP.translation_direction,
    )

    def mask_gen(pixels: GridView) -> BooleanMask:
        ca.head_location = (
            ca.head_location[0] + SnakeDirection.DOWN.translation_direction,
            ca.head_location[1],
        )

        return (pixels[prev_slice] == State.HEAD.state) & (  # type: ignore[no-any-return]
            (pixels[target_slice] == State.NULL.state)
            | (pixels[target_slice] == State.FOOD.state)
        )

    return mask_gen


@Snake.rule(
    State.HEAD,
    frequency="snake_speed",
    target_slice=(slice(None), slice(None, -1)),
    predicate=lambda ca: ca.current_direction == SnakeDirection.LEFT,
)
def move_snake_head_left(ca: Snake, target_slice: TargetSlice) -> MaskGen:
    """Move the snake's head left."""
    right_slice = ca.translate_slice(
        target_slice,
        hrz=SnakeDirection.RIGHT.translation_direction,
    )

    def mask_gen(pixels: GridView) -> BooleanMask:
        ca.head_location = (
            ca.head_location[0],
            ca.head_location[1] + SnakeDirection.LEFT.translation_direction,
        )

        return (pixels[right_slice] == State.HEAD.state) & (  # type: ignore[no-any-return]
            (pixels[target_slice] == State.NULL.state)
            | (pixels[target_slice] == State.FOOD.state)
        )

    return mask_gen


@Snake.rule(
    State.HEAD,
    frequency="snake_speed",
    target_slice=(slice(None), slice(1, None)),
    predicate=lambda ca: ca.current_direction == SnakeDirection.RIGHT,
)
def move_snake_head_right(ca: Snake, target_slice: TargetSlice) -> MaskGen:
    """Move the snake's head right."""
    left_slice = ca.translate_slice(
        target_slice,
        hrz=SnakeDirection.LEFT.translation_direction,
    )

    def mask_gen(pixels: GridView) -> BooleanMask:
        ca.head_location = (
            ca.head_location[0],
            ca.head_location[1] + SnakeDirection.RIGHT.translation_direction,
        )

        return (pixels[left_slice] == State.HEAD.state) & (  # type: ignore[no-any-return]
            (pixels[target_slice] == State.NULL.state)
            | (pixels[target_slice] == State.FOOD.state)
        )

    return mask_gen


@Snake.rule(State.BODY, frequency="snake_speed")
def follow_head_with_body(ca: Snake, target_slice: TargetSlice) -> MaskGen:
    """Follow the snake's head up with the body."""
    durations = ca.durations[target_slice]

    def mask_gen(pixels: GridView) -> BooleanMask:
        return (pixels[target_slice] == State.HEAD.state) & (durations >= 1)  # type: ignore[no-any-return]

    return mask_gen


def update_snake_direction(ca: Snake) -> None:  # noqa: PLR0915, PLR0912, C901
    """Update the snake's direction based on its location."""
    prev_direction = ca.current_direction
    edge = None
    cooled_down = ca.frame_index - ca.last_turn_tick >= (
        ca.turn_cooldown * ca.snake_speed
    )

    match ca.head_location:
        case (0, 0):  # Top-left corner
            if ca.current_direction == SnakeDirection.UP:
                ca.current_direction = SnakeDirection.RIGHT
                LOGGER.debug(
                    "Forced turn to RIGHT from %s in top-left corner",
                    prev_direction,
                )
            elif ca.current_direction == SnakeDirection.LEFT:
                ca.current_direction = SnakeDirection.DOWN
                LOGGER.debug(
                    "Forced turn to DOWN from %s in top-left corner",
                    prev_direction,
                )

        case (0, 63):  # Top-right corner
            if ca.current_direction == SnakeDirection.UP:
                ca.current_direction = SnakeDirection.LEFT
                LOGGER.debug(
                    "Forced turn to LEFT from %s in top-right corner",
                    prev_direction,
                )
            elif ca.current_direction == SnakeDirection.RIGHT:
                ca.current_direction = SnakeDirection.DOWN
                LOGGER.debug(
                    "Forced turn to DOWN from %s in top-right corner",
                    prev_direction,
                )

        case (63, 0):  # Bottom-left corner
            if ca.current_direction == SnakeDirection.DOWN:
                ca.current_direction = SnakeDirection.RIGHT
                LOGGER.debug(
                    "Forced turn to RIGHT from %s in bottom-left corner",
                    prev_direction,
                )
            elif ca.current_direction == SnakeDirection.LEFT:
                ca.current_direction = SnakeDirection.UP
                LOGGER.debug(
                    "Forced turn to UP from %s in bottom-left corner",
                    prev_direction,
                )

        case (63, 63):  # Bottom-right corner
            if ca.current_direction == SnakeDirection.DOWN:
                ca.current_direction = SnakeDirection.LEFT
                LOGGER.debug(
                    "Forced turn to LEFT from %s in bottom-right corner",
                    prev_direction,
                )
            elif ca.current_direction == SnakeDirection.RIGHT:
                ca.current_direction = SnakeDirection.UP
                LOGGER.debug(
                    "Forced turn to UP from %s in bottom-right corner",
                    prev_direction,
                )

        case (_, 0):  # Left edge
            if ca.current_direction == SnakeDirection.LEFT:
                ca.roll_direction_dice(force=True)
                LOGGER.debug(
                    "Turned to %s from %s on left edge",
                    ca.current_direction,
                    prev_direction,
                )
            else:
                edge = SnakeDirection.LEFT

        case (_, 63):  # Right edge
            if ca.current_direction == SnakeDirection.RIGHT:
                ca.roll_direction_dice(force=True)
                LOGGER.debug(
                    "Turned to %s from %s on right edge",
                    ca.current_direction,
                    prev_direction,
                )
            else:
                edge = SnakeDirection.RIGHT

        case (0, _):  # Top edge
            if ca.current_direction == SnakeDirection.UP:
                ca.roll_direction_dice(force=True)
                LOGGER.debug(
                    "Turned to %s from %s on top edge",
                    ca.current_direction,
                    prev_direction,
                )
            else:
                edge = SnakeDirection.UP

        case (63, _):  # Bottom edge
            if ca.current_direction == SnakeDirection.DOWN:
                ca.roll_direction_dice(force=True)
                LOGGER.debug(
                    "Turned to %s from %s on bottom edge",
                    ca.current_direction,
                    prev_direction,
                )
            else:
                edge = SnakeDirection.DOWN

        case _ if cooled_down:
            ca.roll_direction_dice()

    if cooled_down and edge and ca.roll_direction_dice(current_edge=edge):
        LOGGER.debug(
            "Randomly turned to %s from %s on edge %s",
            ca.current_direction,
            prev_direction,
            edge,
        )


def check_food_consumption(ca: Snake, actual_food_count: int) -> None:
    """Check if the snake has consumed food."""
    if actual_food_count < ca.food_count:
        delta = ca.food_count - actual_food_count
        ca.food_count = actual_food_count

        ca.snake_length += delta

        ca.high_score = max(ca.snake_length, ca.high_score)

        LOGGER.info(
            "Snake length increased by %d bits (high score: %d)",
            delta,
            ca.high_score,
        )


@Snake.rule(State.NULL, frequency="snake_speed")
def move_snake_tail(ca: Snake, target_slice: TargetSlice) -> MaskGen:
    """Move the snake's tail."""
    durations = ca.durations[target_slice]

    def mask_gen(pixels: GridView) -> BooleanMask:
        sliced_pixels = pixels[target_slice]

        update_snake_direction(ca)
        check_food_consumption(
            ca,
            actual_food_count=(sliced_pixels == State.FOOD.state).sum(),
        )

        if (sliced_pixels == State.BODY.state).sum() == 0:
            ca.stop(StopType.EXPIRED, reset_priority=False)

        return (pixels[target_slice] == State.BODY.state) & (  # type: ignore[no-any-return]
            durations >= ca.snake_length * ca.snake_speed
        )

    return mask_gen


@Snake.rule(State.FOOD, frequency="food_generation_freq")
def generate_food(ca: Snake, target_slice: TargetSlice) -> MaskGen:
    """Generate a single food cell at a random location."""

    def mask_gen(pixels: GridView) -> BooleanMask:
        # Find all NULL cells in the target_slice
        target_pixels = pixels[target_slice]
        null_indices = np.argwhere(target_pixels == State.NULL.state)
        mask = np.zeros_like(target_pixels, dtype=bool)

        # Pick one at random
        mask[tuple(null_indices[const.RNG.choice(null_indices.shape[0])])] = True

        ca.food_count += 1

        return mask

    return mask_gen


__all__ = ["Snake", "State"]
