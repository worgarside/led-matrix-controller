"""Rain simulation."""

from __future__ import annotations

from enum import unique
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from models import RGBMatrix, RGBMatrixOptions
from PIL import Image
from utils import const
from utils.cellular_automata.ca import (
    Direction,
    Grid,
    Mask,
    MaskGen,
    StateBase,
    TargetSlice,
)

if TYPE_CHECKING:
    from models.matrix import LedMatrixOptionsInfo
    from numpy.typing import NDArray


@unique
class State(StateBase):
    """Enum representing the state of a cell."""

    NULL = 0, " "
    RAINDROP = 1, "O", (13, 94, 255)
    SPLASHDROP = 2, "o", (107, 155, 250)
    SPLASH_LEFT = 3, "*", (170, 197, 250)
    SPLASH_RIGHT = 4, "*", (170, 197, 250)


class RainingGrid(Grid):
    """Basic rain simulation."""

    @Grid.rule(State.RAINDROP, target_slice=0)
    def generate_raindrops(self, target_slice: TargetSlice) -> MaskGen:
        """Generate raindrops at the top of the grid."""
        shape = self._grid[target_slice].shape

        def _mask() -> Mask:
            return const.RNG.random(shape) < 0.025  # noqa: PLR2004

        return _mask

    @Grid.rule(State.RAINDROP, target_slice=(slice(1, None), slice(None)))
    def move_rain_down(self, target_slice: TargetSlice) -> MaskGen:
        """Move raindrops down one cell."""
        lower_slice = self._grid[target_slice]
        upper_slice = self._grid[self.translate_slice(target_slice, vrt=Direction.UP)]
        raindrop = State.RAINDROP.state
        null = State.NULL.state

        def _mask() -> Mask:
            return (upper_slice == raindrop) & (lower_slice == null)  # type: ignore[no-any-return]

        return _mask

    @Grid.rule(State.NULL)
    def top_of_rain_down(self, _: TargetSlice) -> MaskGen:
        """Move the top of a raindrop down."""
        above_slice = self._grid[slice(None, -2), slice(None)]
        middle_slice = self._grid[slice(1, -1), slice(None)]
        below_slice = self._grid[slice(2, None), slice(None)]
        top_row = self._grid[0]
        second_row = self._grid[1]
        penultimate_row = self._grid[-2]
        last_row = self._grid[-1]

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
        self,
        target_slice: TargetSlice,
        *,
        source_slice_direction: Literal[Direction.LEFT, Direction.RIGHT],
    ) -> MaskGen:
        # TODO this would be better as "will be NULL", instead of "is NULL"
        source_slice = self._grid[
            self.translate_slice(
                target_slice,
                vrt=Direction.DOWN,
                hrz=source_slice_direction,
            )
        ]
        splash_spots = self._grid[target_slice]
        below_slice = self._grid[self.translate_slice(target_slice, vrt=Direction.DOWN)]

        raindrop = State.RAINDROP.state
        null = State.NULL.state

        def _mask() -> Mask:
            return (  # type: ignore[no-any-return]
                (source_slice == raindrop)
                & (splash_spots == null)
                & (below_slice == null)
            )

        return _mask

    @Grid.rule(State.SPLASH_LEFT, target_slice=(-2, slice(None, -1)))
    def splash_left(self, target_slice: TargetSlice) -> MaskGen:
        """Create a splash to the left."""
        return self._splash(target_slice, source_slice_direction=Direction.RIGHT)

    @Grid.rule(State.SPLASH_RIGHT, target_slice=(-2, slice(1, None)))
    def splash_right(self, target_slice: TargetSlice) -> MaskGen:
        """Create a splash to the right."""
        return self._splash(target_slice, source_slice_direction=Direction.LEFT)

    def _splash_high(
        self,
        target_slice: TargetSlice,
        *,
        splash_state: State,
        source_slice_direction: Literal[Direction.LEFT, Direction.RIGHT],
    ) -> MaskGen:
        source_slice = self._grid[
            self.translate_slice(
                target_slice,
                vrt=Direction.DOWN,
                hrz=source_slice_direction,
            )
        ]

        state = splash_state.state

        def _mask() -> Mask:
            return (  # type: ignore[no-any-return]
                source_slice == state
            )  # & self._grid[target_slice] will be NULL

        return _mask

    @Grid.rule(State.SPLASH_LEFT, target_slice=(-3, slice(None, -1)))
    def splash_left_high(self, target_slice: TargetSlice) -> MaskGen:
        """Continue the splash to the left."""
        return self._splash_high(
            target_slice,
            splash_state=State.SPLASH_LEFT,
            source_slice_direction=Direction.RIGHT,
        )

    @Grid.rule(State.SPLASH_RIGHT, target_slice=(-3, slice(1, None)))
    def splash_right_high(self, target_slice: TargetSlice) -> MaskGen:
        """Continue the splash to the right."""
        return self._splash_high(
            target_slice,
            splash_state=State.SPLASH_RIGHT,
            source_slice_direction=Direction.LEFT,
        )

    @Grid.rule(State.NULL, target_slice=(slice(-3, None), slice(None)))
    def remove_splashes(self, target_slice: TargetSlice) -> MaskGen:
        """Remove any splashes - they only last one frame."""
        any_splash = (
            State.SPLASH_LEFT.state,
            State.SPLASH_RIGHT.state,
            State.SPLASHDROP.state,
        )
        view = self._grid[target_slice]

        def _mask() -> Mask:
            return np.isin(view, any_splash)

        return _mask

    @Grid.rule(State.SPLASHDROP, target_slice=-3)
    def create_splashdrop(self, target_slice: TargetSlice) -> MaskGen:
        """Convert a splash to a splashdrop."""
        active_splashes = State.SPLASH_LEFT.state, State.SPLASH_RIGHT.state
        view = self._grid[target_slice]

        def _mask() -> Mask:
            return np.isin(view, active_splashes)

        return _mask

    @Grid.rule(State.SPLASHDROP, target_slice=(slice(-3, None)))
    def move_splashdrop_down(self, target_slice: TargetSlice) -> MaskGen:
        """Move the splashdrop down."""
        source_slice = self._grid[self.translate_slice(target_slice, vrt=Direction.UP)]
        splashdrop = State.SPLASHDROP.state

        def _mask() -> Mask:
            return source_slice == splashdrop  # type: ignore[no-any-return]
            # & self._grid[target_slice] will be State.NULL

        return _mask


class Matrix:
    """Class for displaying track information on an RGB LED Matrix."""

    OPTIONS: ClassVar[LedMatrixOptionsInfo] = {
        "cols": 64,
        "rows": 64,
        "brightness": 80,
        "gpio_slowdown": 4,
        "hardware_mapping": "adafruit-hat-pwm",
        "inverse_colors": False,
        "led_rgb_sequence": "RGB",
        "show_refresh_rate": False,
    }

    COLORMAP: NDArray[np.int_] = State.colormap()

    def __init__(self) -> None:
        options = RGBMatrixOptions()

        for name, value in self.OPTIONS.items():
            if getattr(options, name, None) != value:
                setattr(options, name, value)

        self.matrix = RGBMatrix(options=options)
        self.canvas = self.matrix.CreateFrameCanvas()

    def render_array(self, array: NDArray[np.int_]) -> None:
        """Render the array to the LED matrix."""

        pixels = self.COLORMAP[array]

        image = Image.fromarray(pixels.astype(np.uint8), "RGB")

        self.canvas.SetImage(image)
        self.canvas = self.matrix.SwapOnVSync(self.canvas)

    @property
    def height(self) -> int:
        """Return the height of the matrix."""
        return int(self.matrix.height)

    @property
    def width(self) -> int:
        """Return the width of the matrix."""
        return int(self.matrix.width)


def main() -> None:
    """Run the rain simulation."""
    matrix = Matrix()

    grid = RainingGrid(height=matrix.height, width=matrix.width)

    for frame in grid.frames:
        matrix.render_array(frame)


if __name__ == "__main__":
    main()
