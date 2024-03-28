"""Cellular Automata module."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import lru_cache, wraps
from itertools import islice
from typing import Any, Callable, ClassVar, Generator, get_type_hints

import numpy as np
from numpy.typing import DTypeLike, NDArray

from .setting import Setting

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
MaskGen = Callable[[], Mask]
Rule = Callable[["Grid"], MaskGen]

CAMEL_CASE = re.compile(r"(?<!^)(?=[A-Z])")


@dataclass
class Grid:
    """Base class for a grid of cells."""

    ID: ClassVar[str]
    _RULES: ClassVar[list[tuple[TargetSlice, Callable[..., MaskGen], StateBase]]]
    _RULE_METHODS: ClassVar[list[Callable[..., MaskGen]]]

    height: int
    width: int

    class OutOfBoundsError(ValueError):
        """Error for when a slice goes out of bounds."""

        def __init__(self, current: int | None, delta: int, limit: int) -> None:
            """Initialize the OutOfBoundsError."""

            self.current = current
            self.delta = delta
            self.limit = limit

            super().__init__(f"Out of bounds: {current} + {delta} > {limit}")

    def __init_subclass__(cls) -> None:
        """Initialize the subclass."""
        cls._RULES = []
        cls._RULE_METHODS = []

        cls.ID = CAMEL_CASE.sub("-", cls.__name__).lower()

    def __post_init__(self) -> None:
        """Set the calculated attributes of the Grid."""
        self.frame_index = -1
        self._grid: NDArray[np.int_] = self.zeros()

        self.frame_updates = []
        for target_slice, rule, to_state in self._RULES:
            mask_generator = rule(self, target_slice)
            self.frame_updates.append(
                (self._grid[target_slice], mask_generator, to_state.value)
            )

        settings = {}
        for field_name, field_type in get_type_hints(
            self.__class__, include_extras=True
        ).items():
            if hasattr(field_type, "__metadata__"):
                for annotation in field_type.__metadata__:
                    if isinstance(annotation, Setting):
                        annotation.setup(
                            field_name=field_name, grid=self, type_=field_type.__origin__
                        )

                        settings[field_name] = annotation

    @classmethod
    def rule(
        cls,
        to_state: StateBase,
        *,
        target_slice: TargetSliceDecVal = EVERYWHERE,
    ) -> Callable[[Callable[[Any, TargetSlice], MaskGen]], Callable[..., MaskGen]]:
        """Decorator to add a rule to the grid.

        Args:
            to_state (StateBase): The state to change to.
            target_slice (TargetSliceDecVal | None, optional): The slice to target. Defaults to entire grid.
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

        def decorator(rule_func: Callable[[Grid, TargetSlice], MaskGen]) -> Rule:
            @wraps(rule_func)
            def wrapper(grid: Grid) -> MaskGen:
                return rule_func(grid, actual_slice)

            cls._RULES.append((actual_slice, rule_func, to_state))
            cls._RULE_METHODS.append(wrapper)

            return wrapper

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
    def frames(self) -> Generator[NDArray[np.int_], None, None]:
        """Generate the frames of the grid."""
        frame_updates = self.frame_updates

        while True:
            masks = tuple(mask_generator() for _, mask_generator, _ in frame_updates)

            for mask, (target_view, _, state) in zip(masks, frame_updates):
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
