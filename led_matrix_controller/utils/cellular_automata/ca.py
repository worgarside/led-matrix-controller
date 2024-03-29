"""Cellular Automata module."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from functools import lru_cache, wraps
from itertools import islice
from logging import DEBUG, getLogger
from typing import Any, Callable, ClassVar, Generator, Self, get_type_hints

import numpy as np
from numpy.typing import DTypeLike, NDArray
from utils import const
from wg_utilities.loggers import add_stream_handler

from .setting import FrequencySetting, Setting

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
add_stream_handler(LOGGER)


_BY_VALUE: dict[int, StateBase] = {}
EVERYWHERE = (slice(None), slice(None))


class Direction(IntEnum):
    """Enum representing the direction of a cell."""

    LEFT = -1
    RIGHT = 1
    UP = -1
    DOWN = 1


# TODO can this be an ABC?
class StateBase(Enum):
    """Base class for the states of a cell."""

    def __init__(
        self, value: int, char: str, color: tuple[int, int, int] = (0, 0, 0)
    ) -> None:
        self._value_ = value
        self.state = value  # This is only really for type checkers, _value_ is the same but has a different type
        self.char = char
        self.color = color

        _BY_VALUE[value] = self

    @classmethod
    def colormap(cls) -> NDArray[np.int_]:
        """Return the color map of the states."""
        return np.array([state.color for state in cls])

    @staticmethod
    def by_value(value: int | np.int_) -> StateBase:
        """Return the state by its value."""
        return _BY_VALUE[int(value)]

    def __hash__(self) -> int:
        """Return the hash of the value of the state."""
        return hash(self.value)


TargetSliceDecVal = slice | int | tuple[int | slice, int | slice]
TargetSlice = tuple[slice, slice]
Mask = NDArray[np.bool_]
View = NDArray[np.int_]
MaskGen = Callable[[], Mask]
RuleFunc = Callable[["Grid", TargetSlice], MaskGen]
MaskGenLoop = tuple[tuple[View, MaskGen, int], ...]
CAMEL_CASE = re.compile(r"(?<!^)(?=[A-Z])")


@dataclass(slots=True)
class Rule:
    """Class for a rule to update cells."""

    target_slice: TargetSlice
    rule_func: RuleFunc
    to_state: StateBase
    frequency: int | str

    _frequency_setting: FrequencySetting = field(init=False)

    def active_on_frame(self, i: int, /) -> bool:
        """Return whether the rule is active on the given frame."""
        if isinstance(self.frequency, int):
            return i % self.frequency == 0

        return i % self._frequency_setting.get_value_from_grid() == 0


@dataclass(slots=True)
class Grid:
    """Base class for a grid of cells."""

    RULES: ClassVar[list[Rule]] = []
    _RULE_FUNCTIONS: ClassVar[list[Callable[..., MaskGen]]] = []

    height: int
    width: int
    id: str

    frame_index: int = -1

    _grid: NDArray[np.int_] = field(init=False)
    settings: dict[str, Setting[Any]] = field(init=False)
    mask_generator_loops: tuple[MaskGenLoop, ...] = field(init=False)

    class OutOfBoundsError(ValueError):
        """Error for when a slice goes out of bounds."""

        def __init__(self, current: int | None, delta: int, limit: int) -> None:
            """Initialize the OutOfBoundsError."""

            self.current = current
            self.delta = delta
            self.limit = limit

            super().__init__(f"Out of bounds: {current} + {delta} > {limit}")

    def __post_init__(self) -> None:
        """Set the calculated attributes of the Grid."""
        self._grid = self.zeros()

        settings: dict[str, Setting[object]] = {}
        for field_name, field_type in get_type_hints(
            self.__class__, include_extras=True
        ).items():
            if hasattr(field_type, "__metadata__"):
                for annotation in field_type.__metadata__:
                    if isinstance(annotation, Setting):
                        annotation.setup(
                            index=len(settings),
                            field_name=field_name,
                            grid=self,
                            type_=field_type.__origin__,
                        )

                        settings[field_name] = annotation

                        if isinstance(annotation, FrequencySetting):
                            for rule in self.RULES:
                                if (
                                    isinstance(rule.frequency, str)
                                    and rule.frequency == field_name
                                ):
                                    rule._frequency_setting = annotation

        self.settings = settings

        self.generate_rules_loop()

    def generate_rules_loop(self) -> None:
        """Generate the rules loop."""

        largest_frequency = math.lcm(
            *(
                setting.get_value_from_grid()
                for setting in self.settings.values()
                if isinstance(setting, FrequencySetting)
            )
        )

        self.mask_generator_loops = tuple(
            tuple(
                (
                    self._grid[rule.target_slice],
                    rule.rule_func(self, rule.target_slice),
                    rule.to_state.state,
                )
                for rule in self.RULES
                if rule.active_on_frame(i)
            )
            for i in range(largest_frequency)
        )

        LOGGER.info(
            "Mask Generator loop re-generated. New length: %i; largest ruleset: %i",
            len(self.mask_generator_loops),
            max(len(loop) for loop in self.mask_generator_loops),
        )

    @classmethod
    def rule(
        cls,
        to_state: StateBase,
        *,
        target_slice: TargetSliceDecVal = EVERYWHERE,
        frequency: int | str = 1,
    ) -> Callable[[Callable[[Any, TargetSlice], MaskGen]], Callable[[Self], MaskGen]]:
        """Decorator to add a rule to the grid.

        Args:
            to_state (StateBase): The state to change to.
            target_slice (TargetSliceDecVal | None, optional): The slice to target. Defaults to entire grid.
            frequency (int, optional): The frequency of the rule (in frames). Defaults to 1 (i.e. every frame).
        """
        match target_slice:
            case int(n):
                actual_slice = (slice(n, n + 1), slice(None))
            case slice(start=x_start, stop=x_stop, step=x_step):
                actual_slice = (slice(x_start, x_stop, x_step), slice(None))
            case (int(n), slice(start=y_start, stop=y_stop, step=y_step)):
                actual_slice = (slice(n, n + 1), slice(y_start, y_stop, y_step))
            case (slice(start=x_start, stop=x_stop, step=x_step), int(y)):
                actual_slice = (slice(x_start, x_stop, x_step), slice(y, y + 1))
            case (
                slice(start=x_start, stop=x_stop, step=x_step),
                slice(start=y_start, stop=y_stop, step=y_step),
            ):
                actual_slice = (
                    slice(x_start, x_stop, x_step),
                    slice(y_start, y_stop, y_step),
                )
            case _:
                raise ValueError(f"Invalid target_slice: {target_slice}")

        del target_slice

        def decorator(
            rule_func: Callable[[Grid, TargetSlice], MaskGen],
        ) -> Callable[[Grid], MaskGen]:
            # This is the only bit that matters here
            cls.RULES.append(
                Rule(
                    target_slice=actual_slice,
                    rule_func=rule_func,
                    to_state=to_state,
                    frequency=frequency,
                )
            )

            if const.DEBUG_MODE:
                # This is just to enable testing/debugging/validation/etc.
                @wraps(rule_func)
                def wrapper(grid: Grid) -> MaskGen:
                    return rule_func(grid, actual_slice)

                cls._RULE_FUNCTIONS.append(wrapper)

                return wrapper

            return None  # The function is never called explicitly

        return decorator

    def run(self, limit: int) -> Generator[NDArray[np.int_], None, None]:
        """Run the simulation for a given number of frames."""
        yield from islice(self.frames, limit)

    def fresh_mask(self) -> Mask:
        """Return a fresh mask."""
        return self.zeros(dtype=np.bool_)

    def zeros(self, *, dtype: DTypeLike = np.int_) -> NDArray[Any]:
        """Return a grid of zeros."""
        return np.zeros((self.height, self.width), dtype=dtype)

    @property
    def frames(self) -> Generator[View, None, None]:
        """Generate the frames of the grid."""
        while True:
            for mask_gen_loop in self.mask_generator_loops:
                masks = tuple(
                    (target_slice, mask_gen(), state)
                    for target_slice, mask_gen, state in mask_gen_loop
                )

                for target_view, mask, state in masks:
                    target_view[mask] = state

                self.frame_index += 1

                yield self._grid

    @property
    def str_repr(self) -> str:
        """Return a string representation of the grid."""
        return "\n".join(" ".join(state.char for state in row) for row in self._grid)

    def translate_slice(
        self,
        slice_: TargetSlice,
        /,
        *,
        vrt: int = 0,
        hrz: int = 0,
    ) -> TargetSlice:
        """Translate a slice in the vertical (down) and horizontal (right) directions.

        Args:
            slice_ (TargetSlice): The slice to translate.
            vrt (int, optional): The vertical translation: positive is down, negative is up. Defaults to 0.
            hrz (int, optional): The horizontal translation: positive is right, negative is left. Defaults to 0.
        """
        rows, cols = slice_
        return _translate_slice(
            rows_start=rows.start,
            rows_stop=rows.stop,
            rows_step=rows.step,
            cols_start=cols.start,
            cols_stop=cols.stop,
            cols_step=cols.step,
            vrt=vrt,
            hrz=hrz,
            height=self.height,
            width=self.width,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the grid."""
        return self._grid.shape  # type: ignore[return-value]

    def __getitem__(self, key: TargetSliceDecVal) -> NDArray[np.int_]:
        """Get an item from the grid."""
        return self._grid[key]


@lru_cache
def _translate_slice(
    *,
    rows_start: int | None,
    rows_stop: int | None,
    rows_step: int,
    cols_start: int | None,
    cols_stop: int | None,
    cols_step: int,
    vrt: int = 0,
    hrz: int = 0,
    height: int,
    width: int,
) -> TargetSlice:
    return (
        slice(
            _translate_slice_start(
                current=rows_start,
                delta=vrt,
                size=height,
            ),
            _translate_slice_stop(
                current=rows_stop,
                delta=vrt,
                size=height,
            ),
            rows_step,
        ),
        slice(
            _translate_slice_start(
                current=cols_start,
                delta=hrz,
                size=width,
            ),
            _translate_slice_stop(
                current=cols_stop,
                delta=hrz,
                size=width,
            ),
            cols_step,
        ),
    )


def _translate_slice_start(*, current: int | None, delta: int, size: int) -> int | None:
    """Translate the start of a slice by a given delta.

    Takes into account the limit of the grid: returns None if the slice goes out of bounds in a negative
    direction; raises an error if the slice goes out of bounds in a positive direction (because this means
    the entire slice has gone off the grid).

    Args:
        delta (int): The translation delta.
        current (int | None): The current start of the slice.
        size (int): The limit of the grid (either its height or width, depending on the slice direction)

    Returns:
        int | None: The new start of the slice.
    """
    if delta > 0:
        # Right/Down - can go OOB (trailing edge == off-grid in this direction)
        new_value = (current or 0) + delta

        upper_bound = (size - 1) if current is None or current >= 0 else -1
        if new_value > upper_bound:  # Gone off grid - not good!
            raise Grid.OutOfBoundsError(current, delta, size)
    elif delta < 0:  # Left/Up - can't go OOB
        if current is None:  # Immediately going off grid, but that's okay
            new_value = None
        else:
            lower_bound = 0 if current >= 0 else -size

            if (new_value := current + delta) < lower_bound:
                new_value = None
    elif delta == 0:  # No change
        new_value = current

    return new_value


def _translate_slice_stop(*, current: int | None, delta: int, size: int) -> int | None:
    """Translate the stop of a slice by a given delta.

    Takes into account the limit of the grid: returns None if the slice goes out of bounds in a positive
    direction; raises an error if the slice goes out of bounds in a negative direction (because this means
    the entire slice has gone off the grid).

    Args:
        delta (int): The translation delta.
        current (int | None): The current stop of the slice.
        size (int): The limit of the grid (either its height or width, depending on the slice direction)

    Returns:
        int | None: The new stop of the slice.
    """
    if delta > 0:
        # Right/Down - can't go OOB (leading edge beacomes open end in this direction)
        if current is None:
            new_value = None
        else:
            upper_bound = (size - 1) if current >= 0 else -1

            if (new_value := current + delta) > upper_bound:  # i.e. gone off grid
                new_value = None
    elif delta < 0:
        # Left/Up - can go OOB (trailing edge == off-grid in this direction)
        new_value = (current or 0) + delta  # Negative number!

        lower_bound = 0 if current is not None and current >= 0 else -size
        if new_value < lower_bound:
            raise Grid.OutOfBoundsError(current, delta, size)
    elif delta == 0:  # No change
        new_value = current

    return new_value
