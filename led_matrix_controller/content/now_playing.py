"""Class for the creation, caching, and management of artwork images."""

from __future__ import annotations

import ssl
from contextlib import suppress
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from re import Pattern
from re import compile as compile_regex
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Final, TypedDict

# The httpcore import is needed because httpx throws an error here for some reason
# https://github.com/encode/httpx/blob/master/httpx/_transports/default.py#L150
import httpcore  # noqa: F401
import httpx
import numpy as np
from content.base import GridView, StopType
from PIL import Image
from utils import const
from wg_utilities.functions import backoff, force_mkdir
from wg_utilities.loggers import get_streaming_logger

from .dynamic_content import DynamicContent
from .setting import ParameterSetting

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import DTypeLike, NDArray

LOGGER = get_streaming_logger(__name__)

SSL_CONTEXT = ssl.create_default_context()


class TrackMeta(TypedDict):
    """Type definition for the track metadata."""

    title: str | None
    album: str | None
    artist: str | None
    album_artwork_url: httpx.URL | None


_INITIAL_TRACK_META: Final[TrackMeta] = {
    "title": "",
    "album": "",
    "artist": "",
    "album_artwork_url": httpx.URL(""),
}


@dataclass(kw_only=True, slots=True)
class NowPlaying(DynamicContent):
    """Class for the creation, caching, and management of artwork images."""

    ARTWORK_DIRECTORY: ClassVar[Path] = force_mkdir(
        (
            Path("/var/cache")  # Script is run as root so this needs to be hardcoded
            if const.IS_PI
            else Path.home()
        )
        .joinpath(
            "led-matrix-controller",
            "artwork",
        )
        .resolve(),
    )

    ALPHANUM_PATTERN: ClassVar[Pattern[str]] = compile_regex(r"[\W_]+")

    IS_OPAQUE: ClassVar[bool] = True

    track_metadata: Annotated[TrackMeta, ParameterSetting(icon="")] = field(
        default_factory=lambda: _INITIAL_TRACK_META,
    )

    _current_image: tuple[Path, GridView] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    """Cache of current file path and image array."""

    def get_content(self) -> GridView:
        """Get the Image of the artwork image from the local file/remote URL.

        Returns:
            Image: Image instance of the artwork image
        """
        if not self.file_path:
            return self.zeros()

        if hasattr(self, "_current_image") and self._current_image[0] == self.file_path:
            return self._current_image[1]

        if self.file_path.is_file():
            LOGGER.debug(
                "Opening image from path %s for %s",
                self.file_path,
                self.album,
            )
            with suppress(FileNotFoundError):
                self._current_image = (self.file_path, np.load(self.file_path))

                return self._current_image[1]

        LOGGER.debug("Image not found at %s for %s", self.file_path, self.album)

        try:
            return self.download()
        except httpx.HTTPStatusError as err:
            LOGGER.exception(
                "HTTP error (%s %s) downloading artwork from %s for album %s",
                err.response.status_code,
                err.response.reason_phrase,
                self.artwork_uri,
                self.album,
            )
            LOGGER.error("Response content: %s", err.response.text)  # noqa: TRY400
        except Exception:
            LOGGER.exception(
                "Failed to download artwork from %s for album %s",
                self.artwork_uri,
                self.album,
            )
        return self.zeros()

    @backoff(httpx.HTTPStatusError, logger=LOGGER, timeout=60, max_delay=10)
    def download(self) -> GridView:
        """Download the image from the URL to store it locally for future use."""
        if not (
            self.artwork_uri
            and str(self.artwork_uri)
            and self.artist_directory
            and self.file_path
        ):
            LOGGER.error(
                "Unable to download artwork for %s. "
                "Missing required data: self.artwork_uri=%s, self.artist_directory=%s, self.file_path=%s",
                self.album,
                self.artwork_uri,
                self.artist_directory,
                self.file_path,
            )
            return self.zeros()

        self.ARTWORK_DIRECTORY.joinpath(self.artist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        LOGGER.debug("Downloading artwork from remote URL: %s", self.artwork_uri)

        res = httpx.get(self.artwork_uri, timeout=120, verify=SSL_CONTEXT)
        res.raise_for_status()
        artwork_bytes = res.content

        img_arr = np.array(
            Image.open(BytesIO(artwork_bytes))
            .resize((self.width, self.height))
            .convert("RGBA"),
        )

        np.save(force_mkdir(self.file_path, path_is_file=True), img_arr)

        LOGGER.info(
            "New image from %s saved at %s for album %r",
            self.artwork_uri,
            self.file_path,
            self.album,
        )

        return img_arr

    def refresh_content(self) -> Generator[None, None, None]:
        """Refresh the content."""
        yield

        if None in self.track_metadata.values():
            self.stop(StopType.EXPIRED)
            with suppress(AttributeError):
                del self._current_image

    def zeros(self, *, dtype: DTypeLike = np.int_) -> NDArray[Any]:
        """Return a grid of zeros."""
        return np.zeros((self.height, self.width, 4), dtype=dtype)

    @property
    def artist_directory(self) -> str | None:
        """Return the artist name, with all non-alphanumeric characters removed.

        Returns:
            str: the artist name, with all non-alphanumeric characters removed
        """
        if not self.artist:
            return None

        return self.ALPHANUM_PATTERN.sub("", self.artist).lower()

    @property
    def filename(self) -> str | None:
        """Return the album name, with all non-alphanumeric characters removed.

        Returns:
            str: the filename of the artwork image
        """
        if not self.album:
            return None

        return self.ALPHANUM_PATTERN.sub("", self.album).lower() + ".npy"

    @property
    def file_path(self) -> Path | None:
        """Return the local path to the artwork image.

        Returns:
            Path: fully-qualified path to the artwork image
        """
        if not self.artist_directory or not self.filename:
            return None

        return self.ARTWORK_DIRECTORY / self.artist_directory / self.filename

    @property
    def album(self) -> str | None:
        """Return the album name."""
        return self.track_metadata.get("album")

    @property
    def artist(self) -> str | None:
        """Return the artist name."""
        return self.track_metadata.get("artist")

    @property
    def artwork_uri(self) -> httpx.URL | None:
        """Return the URL of the artwork."""
        return self.track_metadata.get("album_artwork_url")

    @property
    def title(self) -> str | None:
        """Return the title of the track."""
        return self.track_metadata.get("title")

    def __hash__(self) -> int:
        """Return the hash of the object."""
        return hash((self.artist, self.album, self.artwork_uri))

    def __str__(self) -> str:
        """Return the string representation of the object."""
        return self.__repr__()

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        return (
            f"{self.__class__.__name__}({self.artist}, {self.album}, {self.artwork_uri})"
        )
