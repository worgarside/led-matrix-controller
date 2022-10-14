"""Constants and class for text labels on the RGB LED Matrix"""
from __future__ import annotations

from logging import DEBUG, getLogger
from pathlib import Path

from wg_utilities.loggers import add_stream_handler

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
add_stream_handler(LOGGER)

try:
    from rgbmatrix.graphics import Color, Font
except ImportError as exc:
    LOGGER.warning(
        "Could not import `rgbmatrix`, using emulator instead: %s", repr(exc)
    )

    from RGBMatrixEmulator.graphics import Color, Font

FONT_WIDTH = 5
FONT_HEIGHT = 7
SCROLL_INCREMENT_DISTANCE = 2 * FONT_WIDTH
FONT = Font()
FONT.LoadFont(
    str(
        Path(__file__).absolute().parents[3]
        / "assets"
        / "fonts"
        / f"{FONT_WIDTH}x{FONT_HEIGHT}.bdf"
    )
)


class Text:
    """Class for displaying text on the matrix"""

    DEFAULT_TEXT_COLOR = Color(255, 255, 255)  # white
    CLEAR_TEXT_COLOR = Color()  # black

    def __init__(
        self,
        content: str,
        y_pos: int,
        color: Color | None = None,
        *,
        matrix_width: int | None = None,
    ):
        self.color = color or self.DEFAULT_TEXT_COLOR
        self.y_pos = y_pos

        self.original_content = content
        self._matrix_width = matrix_width
        self.scrollable: bool = False

        self._current_x_pos = self.original_x_pos

    def get_next_x_pos(self, reference_only: bool = False) -> int:
        """Gets the next X position for the text (i.e. the same if it's a short label,
        but incremented for a scroll effect if it's a longer label)

        Args:
            reference_only (bool): If True, the current x position will not be updated.
                Defaults to False.

        Returns:
            int: the next x position of the text
        """

        if not self.scrollable:
            return self.original_x_pos

        next_x_pos = self._current_x_pos - SCROLL_INCREMENT_DISTANCE

        if next_x_pos <= -2 / 3 * self.label_len:
            # If 2/3 of the label has scrolled off the screen, reset to the original X
            # position to give a clean wrap effect
            next_x_pos = self.original_x_pos

        if not reference_only:
            self._current_x_pos = next_x_pos

        return next_x_pos

    def reset_x_pos(self) -> None:
        """Resets the x position of the text to the original position"""

        self._current_x_pos = self.original_x_pos

    @property
    def display_content(self) -> str:
        """
        Returns:
            str: the display_content of the text
        """
        if self.scrollable:
            return ("   " + self.original_content + "   ") * 3

        return self.original_content

    @display_content.setter
    def display_content(self, value: str) -> None:
        """Sets the display_content of the text

        Args:
            value (str): the new display_content
        """

        self.original_content = value

        if self.matrix_width:
            # Set the `scrollable` attribute based on the new display_content's length
            self.scrollable = (
                len(self.original_content) * FONT_WIDTH > self.matrix_width
            )
        else:
            LOGGER.warning("Matrix width not set, defaulting scrollable to False")
            self.scrollable = False

    @property
    def label_len(self) -> int:
        """
        Returns:
            int: length of the text in pixels
        """

        return len(self.display_content) * FONT_WIDTH

    @property
    def matrix_width(self) -> int | None:
        """
        Returns:
            int: the width of the matrix
        """

        return self._matrix_width

    @matrix_width.setter
    def matrix_width(self, value: int | None) -> None:
        """
        Args:
            value (int, optional): the width of the matrix
        """

        self._matrix_width = value

    @property
    def original_x_pos(self) -> int:
        """Returns the x position of the text

        Returns:
            int: x position of the text
        """

        if self.scrollable:
            # There are 3 spaces before the text, so negate them here to get the string
            # to align with the left side of the screen. The SCROLL_INCREMENT_DISTANCE
            # is added to account for the first incremental movement of the text within
            # the `get_next_x_pos` method.
            return (-3 * FONT_WIDTH) + SCROLL_INCREMENT_DISTANCE

        if not self.matrix_width:
            LOGGER.warning("Matrix width not set, defaulting original x position to 10")
            return 10

        # Otherwise center the text on the screen
        return int((self.matrix_width - (len(self.original_content) * FONT_WIDTH)) / 2)

    def __len__(self) -> int:
        """
        Returns:
            int: the length of the text
        """

        return len(self.display_content)
