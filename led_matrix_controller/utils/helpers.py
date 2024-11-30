"""Helper functions."""

from __future__ import annotations

import atexit
import re
from enum import Enum
from functools import lru_cache
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Final, overload

from wg_utilities.loggers import get_streaming_logger

from .const import AUDIO_VISUALISER_SHM_NAME

if TYPE_CHECKING:
    from logging import Logger

LOGGER = get_streaming_logger(__name__)


class Patterns(Enum):
    """Regex patterns."""

    NOT_LETTERS: Final[re.Pattern[str]] = re.compile(r"[^a-z]", flags=re.IGNORECASE)
    CAMELCASE: Final[re.Pattern[str]] = re.compile(r"(?<!^)(?=[A-Z])")
    MULTIPLE_HYPHENS: Final[re.Pattern[str]] = re.compile(r"-+")

    def sub(self, repl: str, string: str) -> str:
        """Substitute pattern with replacement."""
        return self.value.sub(repl, string)


@overload
def to_kebab_case(string: str) -> str: ...


@overload
def to_kebab_case(*string: str) -> tuple[str, ...]: ...


def to_kebab_case(*string: str) -> str | tuple[str, ...]:  # type: ignore[misc]
    """Convert string to kebab case."""
    if len(string) == 1:
        return _remove_multiple_hyphens(
            Patterns.NOT_LETTERS.sub("-", string[0]).casefold(),
        )

    return tuple(
        _remove_multiple_hyphens(
            *(Patterns.NOT_LETTERS.sub("-", s).casefold() for s in string),
        ),
    )


@overload
def camel_to_kebab_case(string: str) -> str: ...


@overload
def camel_to_kebab_case(*string: str) -> tuple[str, ...]: ...


def camel_to_kebab_case(*string: str) -> str | tuple[str, ...]:  # type: ignore[misc]
    """Convert camel case string to kebab case."""
    if len(string) == 1:
        return _remove_multiple_hyphens(Patterns.CAMELCASE.sub("-", string[0]).casefold())

    return tuple(
        _remove_multiple_hyphens(
            *(Patterns.CAMELCASE.sub("-", s).casefold() for s in string),
        ),
    )


@overload
def _remove_multiple_hyphens(string: str) -> str: ...


@overload
def _remove_multiple_hyphens(*string: str) -> tuple[str, ...]: ...


def _remove_multiple_hyphens(*string: str) -> str | tuple[str, ...]:  # type: ignore[misc]
    """Remove multiple hyphens."""
    if len(string) == 1:
        return Patterns.MULTIPLE_HYPHENS.sub("-", string[0]).strip("-")

    return tuple(Patterns.MULTIPLE_HYPHENS.sub("-", s).strip("-") for s in string)


@lru_cache(maxsize=128)
def hex_to_rgba(hex_code: str) -> tuple[int, int, int, int]:
    """Convert hex code to rgba tuple."""
    hex_code = hex_code.lstrip("#")

    if len(hex_code) == 6:  # noqa: PLR2004
        r, g, b = int(hex_code[0:2], 16), int(hex_code[2:4], 16), int(hex_code[4:6], 16)
        a = 255
    else:
        r, g, b, a = (int(hex_code[i : i + 2], 16) for i in (0, 2, 4, 6))

    return r, g, b, a


def get_shared_memory(
    *,
    logger: Logger = LOGGER,
    size: int = 2048,
    close_at_exit: bool = True,
) -> shared_memory.SharedMemory:
    """Get a shared memory object for use in the AudioVisualiser."""
    try:
        shm = shared_memory.SharedMemory(
            name=AUDIO_VISUALISER_SHM_NAME,
            create=True,
            size=size,
        )
        logger.info(
            "Created shared memory %r with size %s",
            shm.name,
            shm.size,
        )
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=AUDIO_VISUALISER_SHM_NAME)
        logger.info("Opened existing shared memory %r with size %s", shm.name, shm.size)

    if close_at_exit:
        atexit.register(shm.close)
        atexit.register(shm.unlink)

    return shm
