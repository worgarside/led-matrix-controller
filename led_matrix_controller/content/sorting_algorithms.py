"""Visualise various sorting algorithms."""

from __future__ import annotations

from colorsys import hls_to_rgb
from dataclasses import dataclass, field
from enum import StrEnum, auto
from random import choice, randint, shuffle, uniform
from typing import Annotated, ClassVar, Generator

import numpy as np
from utils import const
from wg_utilities.loggers import get_streaming_logger

from .base import StopType
from .dynamic_content import DynamicContent
from .setting import ParameterSetting  # noqa: TCH001

LOGGER = get_streaming_logger(__name__)


def _merge(list_: list[int], /, l: int, m: int, r: int) -> Generator[None, None, None]:  # noqa: E741
    """Merges two sorted subarrays of the array.

    The first subarray is array[l..m]
    The second subarray is array[m+1..r]

    Args:
        list_ (list[int]): The list to be sorted.
        l (int): The starting index of the first subarray.
        m (int): The ending index of the first subarray and the starting index of the second subarray.
        r (int): The ending index of the second subarray.
    """
    len1, len2 = m - l + 1, r - m
    left = [list_[l + i] for i in range(len1)]
    right = [list_[m + 1 + i] for i in range(len2)]

    i, j, k = 0, 0, l

    while i < len1 and j < len2:
        if left[i] >= right[j]:  # Swap the sign to change timsort order
            list_[k] = left[i]
            i += 1
        else:
            list_[k] = right[j]
            j += 1
        k += 1
        yield

    while i < len1:
        list_[k] = left[i]
        k += 1
        i += 1
        yield

    while j < len2:
        list_[k] = right[j]
        k += 1
        j += 1
        yield


# =============================================================================
# Algorithm Implementations


class SortingAlgorithm(StrEnum):
    """Enumeration of sorting algorithms."""

    BINARY_INSERTION_SORT = auto()
    BUBBLESORT = auto()
    COCKTAIL_SHAKER_SORT = auto()
    COMB_SORT = auto()
    GNOME_SORT = auto()
    ODD_EVEN_SORT = auto()
    PANCAKE_SORT = auto()
    SLOW_SORT = auto()
    STOOGE_SORT = auto()
    TIMSORT = auto()

    def __call__(self, list_: list[int], /) -> Generator[None, None, None]:
        """Return the sorting algorithm implementation."""
        return getattr(self, self)(list_)  # type: ignore[no-any-return]

    @staticmethod
    def binary_insertion_sort(list_: list[int]) -> Generator[None, None, None]:
        """Sorts the portion of the array from index left to right using binary insertion sort."""
        left = 0

        for i in range(1, len(list_)):
            key_item = list_[i]
            j = i - 1

            while j >= left and list_[j] < key_item:
                list_[j + 1] = list_[j]
                j -= 1
                yield

            list_[j + 1] = key_item
            yield

    @staticmethod
    def bubblesort(list_: list[int]) -> Generator[None, None, None]:
        """Swap the elements to arrange in order."""
        for iter_num in range(len(list_) - 1, 0, -1):
            for idx in range(iter_num):
                if list_[idx] < list_[idx + 1]:  # Swap the sign to change sort order
                    list_[idx], list_[idx + 1] = list_[idx + 1], list_[idx]

                    yield

    @staticmethod
    def cocktail_shaker_sort(list_: list[int]) -> Generator[None, None, None]:
        """Compare and swap adjacent elements in both directions."""
        n = len(list_)
        swapped = True
        start = 0
        end = n - 1

        while swapped:
            swapped = False
            for i in range(start, end):
                if list_[i] < list_[i + 1]:  # Swap the sign to change sort order
                    list_[i], list_[i + 1] = list_[i + 1], list_[i]
                    swapped = True
                    yield

            if not swapped:
                break

            swapped = False
            end -= 1
            for i in range(end - 1, start - 1, -1):
                if list_[i] < list_[i + 1]:  # Swap the sign to change sort order
                    list_[i], list_[i + 1] = list_[i + 1], list_[i]
                    swapped = True
                    yield

            start += 1

    @staticmethod
    def comb_sort(list_: list[int]) -> Generator[None, None, None]:
        """Compare and swap elements with a gap that decreases over time."""
        n = len(list_)
        gap = n
        shrink = 1.001
        is_sorted = False

        while not is_sorted:
            gap = int(gap / shrink)
            if gap <= 1:
                gap = 1
                is_sorted = True

            i = 0
            while i + gap < n:
                if list_[i] < list_[i + gap]:  # Swap the sign to change sort order
                    list_[i], list_[i + gap] = list_[i + gap], list_[i]
                    is_sorted = False
                    yield
                i += 1

    @staticmethod
    def gnome_sort(list_: list[int]) -> Generator[None, None, None]:
        """Sorts the array by comparing the current element with the previous one."""
        index = 0
        while index < len(list_):
            if (
                index == 0 or list_[index] <= list_[index - 1]
            ):  # Swap the sign to change sort order
                index += 1
            else:
                list_[index], list_[index - 1] = list_[index - 1], list_[index]
                index -= 1

            yield

    @staticmethod
    def odd_even_sort(list_: list[int]) -> Generator[None, None, None]:
        """Perform comparisons and swaps on alternating odd and even indexed elements."""
        n = len(list_)
        is_sorted = False
        while not is_sorted:
            is_sorted = True
            for i in range(1, n - 1, 2):
                if list_[i] < list_[i + 1]:  # Swap the sign to change sort order
                    list_[i], list_[i + 1] = list_[i + 1], list_[i]
                    is_sorted = False
                    yield

            for i in range(0, n - 1, 2):
                if list_[i] < list_[i + 1]:  # Swap the sign to change sort order
                    list_[i], list_[i + 1] = list_[i + 1], list_[i]
                    is_sorted = False
                    yield

    @staticmethod
    def _flip(list_: list[int], i: int) -> None:
        """Reverse the array from start to index i."""
        start = 0
        while start < i:
            list_[start], list_[i] = list_[i], list_[start]
            start += 1
            i -= 1

    def pancake_sort(self, list_: list[int]) -> Generator[None, None, None]:
        """Repeatedly flip subarrays to move the largest unsorted element to its correct position."""
        n = len(list_)
        for curr_size in range(n, 1, -1):
            min_idx = list_.index(min(list_[:curr_size]))
            if min_idx != curr_size - 1:
                if min_idx != 0:
                    self._flip(list_, min_idx)
                    yield

                self._flip(list_, curr_size - 1)
                yield

    def slow_sort(
        self,
        list_: list[int],
        i: int = 0,
        j: int | None = None,
    ) -> Generator[None, None, None]:
        """A deliberately inefficient sorting algorithm based on divide and conquer."""
        if j is None:
            j = len(list_) - 1

        if i >= j:
            return

        m = (i + j) // 2
        yield from self.slow_sort(list_, i, m)
        yield from self.slow_sort(list_, m + 1, j)

        if list_[m] < list_[j]:  # Swap the sign to change sort order
            list_[m], list_[j] = list_[j], list_[m]
            yield

        yield from self.slow_sort(list_, i, j - 1)

    def stooge_sort(
        self,
        list_: list[int],
        l: int = 0,  # noqa: E741
        h: int | None = None,
    ) -> Generator[None, None, None]:
        """Recursively sorts by sorting the first two-thirds, last two-thirds, and first two-thirds again."""
        if h is None:
            h = len(list_) - 1

        if l >= h:
            return

        if list_[l] < list_[h]:  # Swap the sign to change sort order
            list_[l], list_[h] = list_[h], list_[l]
            yield

        if h - l + 1 > 2:  # noqa: PLR2004
            t = (h - l + 1) // 3
            yield from self.stooge_sort(list_, l, h - t)
            yield from self.stooge_sort(list_, l + t, h)
            yield from self.stooge_sort(list_, l, h - t)

    @staticmethod
    def timsort(list_: list[int]) -> Generator[None, None, None]:
        """Sorts the array using Timsort algorithm."""
        n = len(list_)

        size = 1
        while size < n:
            for left in range(0, n, 2 * size):
                mid = min(n - 1, left + size - 1)
                right = min((left + 2 * size - 1), (n - 1))

                if mid < right:
                    yield from _merge(list_, left, mid, right)

            size *= 2


# =============================================================================
# Content


@dataclass(kw_only=True, slots=True)
class Sorter(DynamicContent):
    """Display various sorting algorithms."""

    BG_COLOR: ClassVar[tuple[int, int, int, int]] = (0, 0, 0, 255)

    algorithm: Annotated[
        SortingAlgorithm,
        ParameterSetting(),
    ] = SortingAlgorithm.BUBBLESORT

    completion_display_time: Annotated[
        float,
        ParameterSetting(
            minimum=0,
            maximum=30,
            fp_precision=2,
        ),
    ] = 5.0
    """Number of seconds to display the sorted array for."""

    iterations: Annotated[
        int,
        ParameterSetting(
            minimum=1,
            maximum=1000,
        ),
    ] = 1
    """Number of iterations to run the sorting algorithm per."""

    randomize_algorithm: Annotated[
        bool,
        ParameterSetting(),
    ] = False

    _values: list[int] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the image getter."""
        DynamicContent.__post_init__(self)

        self.pixels = self.zeros()
        self.update_colormap()

    def _set_pixels(self, offset: int = -1) -> None:
        """Fill each column of the array with the corresponding value, to the height of that value.

        Args:
            offset (int, optional): The vertical offset to start the fill from. Defaults to -1.
        """
        for idx, value in enumerate(self._values):
            self.pixels[value + offset :, idx] = value
            self.pixels[: value + offset, idx] = 0

    def refresh_content(self) -> Generator[None, None, None]:
        """Refresh the content."""
        for iter_num in range(self.iterations):
            if iter_num > 0:
                yield from self.setup()

            for _ in self.algorithm(self._values):
                self._set_pixels()
                yield

            if iter_num < self.iterations - 1:
                # Don't run teardown for the last iteration, it's run outside this method anyway
                yield from self.teardown()

        self.stop(StopType.EXPIRED)

    def setup(self) -> Generator[None, None, None]:
        """Animate the columns' arrival."""
        self._values = list(range(1, self.width + 1))
        shuffle(self._values)

        if self.randomize_algorithm:
            self.update_setting("algorithm", choice(list(SortingAlgorithm)))  # noqa: S311

        # Initial render
        for i in range(self.height, -2, -1):
            self._set_pixels(offset=i)
            yield
            yield

    def teardown(self) -> Generator[None, None, None]:
        """Display the sorted list for N seconds, then reset the colormap."""
        LOGGER.debug("Sleeping for %f seconds", self.completion_display_time)
        completion_ticks = self.completion_display_time * const.TICKS_PER_SECOND

        for _ in range(int(completion_ticks)):
            yield

        for i in range(-1, self.height + 1):
            self._set_pixels(offset=i)
            yield
            yield  # Second yield to slow down the wipe

        self.update_colormap()
        yield

    def update_colormap(self) -> None:
        """Randomly generate a new gradient colormap."""
        interval = 360.0 / self.width

        offset = randint(0, 360)  # noqa: S311

        lightness = uniform(0.25, 0.75)  # noqa: S311
        saturation = uniform(0.25, 1)  # noqa: S311

        LOGGER.debug(
            "Creating colormap with offset=%s, step=%s, lightness=%s, saturation=%s",
            offset,
            interval,
            lightness,
            saturation,
        )

        colors = [
            (
                *hls_to_rgb(
                    (((offset + i) * interval) % 360) / 360,
                    lightness,
                    saturation,
                ),
                1,  # Alpha channel
            )
            for i in range(self.width)
        ]

        self.colormap = np.array(
            [self.BG_COLOR, *(tuple(int(c * 255) for c in color) for color in colors)],
            dtype=np.uint8,
        )
