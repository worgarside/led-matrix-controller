"""Class for the creation, caching, and management of artwork images."""

from __future__ import annotations

from colorsys import hls_to_rgb
from dataclasses import dataclass
from enum import StrEnum, auto
from random import randint, shuffle, uniform
from typing import Annotated, Generator, Protocol

import numpy as np
from content.base import StopType
from models.setting import ParameterSetting  # noqa: TCH002
from wg_utilities.loggers import get_streaming_logger

from .dynamic_content import DynamicContent

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


class SortingAlgorithmImpl(Protocol):
    """Typing protocol for the sorting algorithm implementations."""

    def __call__(self, list_: list[int]) -> Generator[None, None, None]:
        """A sorting algorithm which yields every time the list is updated."""


# =============================================================================
# Algorithm Implementations


class SortingAlgorithm(StrEnum):
    """Enumeration of sorting algorithms."""

    BINARY_INSERTION_SORT = auto()
    BUBBLESORT = auto()
    GNOME_SORT = auto()
    SLOW_SORT = auto()
    STOOGE_SORT = auto()
    TIMSORT = auto()

    def __call__(self, list_: list[int]) -> Generator[None, None, None]:
        """Return the sorting algorithm implementation."""
        lookup: dict[SortingAlgorithm, SortingAlgorithmImpl] = {
            SortingAlgorithm.BINARY_INSERTION_SORT: SortingAlgorithm.binary_insertion_sort,
            SortingAlgorithm.BUBBLESORT: SortingAlgorithm.bubblesort,
            SortingAlgorithm.GNOME_SORT: SortingAlgorithm.gnome_sort,
            SortingAlgorithm.SLOW_SORT: SortingAlgorithm.slow_sort,
            SortingAlgorithm.STOOGE_SORT: SortingAlgorithm.stooge_sort,
            SortingAlgorithm.TIMSORT: SortingAlgorithm.timsort,
        }

        return lookup[self](list_)

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
                if list_[idx] < list_[idx + 1]:
                    list_[idx], list_[idx + 1] = list_[idx + 1], list_[idx]

                    yield

    @staticmethod
    def gnome_sort(list_: list[int]) -> Generator[None, None, None]:
        """Sorts the array by comparing the current element with the previous one."""
        index = 0
        while index < len(list_):
            if index == 0 or list_[index] >= list_[index - 1]:
                index += 1
            else:
                list_[index], list_[index - 1] = list_[index - 1], list_[index]
                index -= 1

            yield

    @staticmethod
    def slow_sort(
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
        yield from SortingAlgorithm.slow_sort(list_, i, m)
        yield from SortingAlgorithm.slow_sort(list_, m + 1, j)

        if list_[m] > list_[j]:
            list_[m], list_[j] = list_[j], list_[m]
            yield

        yield from SortingAlgorithm.slow_sort(list_, i, j - 1)

    @staticmethod
    def stooge_sort(
        list_: list[int],
        l: int = 0,  # noqa: E741
        h: int | None = None,
    ) -> Generator[None, None, None]:
        """Recursively sorts by sorting the first two-thirds, last two-thirds, and first two-thirds again."""
        if h is None:
            h = len(list_) - 1

        if l >= h:
            return
        if list_[l] > list_[h]:
            list_[l], list_[h] = list_[h], list_[l]
            yield

        if h - l + 1 > 2:  # noqa: PLR2004
            t = (h - l + 1) // 3
            yield from SortingAlgorithm.stooge_sort(list_, l, h - t)
            yield from SortingAlgorithm.stooge_sort(list_, l + t, h)
            yield from SortingAlgorithm.stooge_sort(list_, l, h - t)

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

    algorithm: Annotated[
        SortingAlgorithm,
        ParameterSetting(
            requires_rule_regeneration=True,
            payload_modifier=lambda x: x.casefold(),
            strict=False,
        ),
    ] = SortingAlgorithm.BUBBLESORT

    def __post_init__(self) -> None:
        """Initialize the image getter."""
        DynamicContent.__post_init__(self)

        self.pixels = self.zeros()
        self.update_colormap()

    def _set_pixels(self, values: list[int]) -> None:
        """Fill each column of the array with the corresponding value, to the height of that value."""
        for idx, value in enumerate(values):
            self.pixels[value - 1 :, idx] = value
            self.pixels[: value - 1, idx] = 0

    def refresh_content(self) -> Generator[None, None, None]:
        """Refresh the content."""
        values = list(range(1, self.width + 1))
        shuffle(values)

        # Initial render
        self._set_pixels(values)
        yield

        for _ in self.algorithm(values):
            self._set_pixels(values)
            yield

        self.stop(StopType.EXPIRED)

        del self._image_getter
        self.update_colormap()

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
            hls_to_rgb(
                (((offset + i) * interval) % 360) / 360,
                lightness,
                saturation,
            )
            for i in range(self.width)
        ]

        self.colormap = np.array(
            [(0, 0, 0)] + [tuple(int(c * 255) for c in color) for color in colors],
            dtype=np.uint8,
        )

    def teardown(self) -> Generator[None, None, None]:  # noqa: PLR6301
        """No teardown needed."""
        yield
