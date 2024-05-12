"""Class for the creation, caching, and management of artwork images."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from io import BytesIO
from pathlib import Path
from re import Pattern
from re import compile as compile_regex
from time import sleep
from typing import Annotated, ClassVar, Final, Generator, TypedDict

from content.base import StopType
from httpx import URL, HTTPStatusError, get
from models.setting import ParameterSetting  # noqa: TCH002
from PIL import Image, UnidentifiedImageError
from utils import const
from wg_utilities.functions import backoff, force_mkdir
from wg_utilities.loggers import get_streaming_logger

from .dynamic_content import DynamicContent

LOGGER = get_streaming_logger(__name__)


class TrackMeta(TypedDict):
    """Type definition for the track metadata."""

    title: str | None
    album: str | None
    artist: str | None
    artwork_uri: URL | None


_INITIAL_TRACK_META: Final[TrackMeta] = {
    "title": "",
    "album": "",
    "artist": "",
    "artwork_uri": URL(""),
}


@dataclass(kw_only=True, slots=True)
class NowPlaying(DynamicContent):
    """Class for the creation, caching, and management of artwork images."""

    ARTWORK_DIRECTORY: ClassVar[Path] = (
        (
            Path("/home/pi")  # Script is run as root so this needs to be hardcoded
            if const.IS_PI
            else Path.home()
        )
        .joinpath(
            ".cache",
            "led-matrix-controller",
            "artwork",
        )
        .resolve()
    )

    ALPHANUM_PATTERN: ClassVar[Pattern[str]] = compile_regex(r"[\W_]+")

    track_metadata: Annotated[TrackMeta, ParameterSetting()] = field(
        default_factory=lambda: _INITIAL_TRACK_META,
    )

    previous_track: TrackMeta = field(
        init=False,
        default_factory=lambda: _INITIAL_TRACK_META,
    )

    def __post_init__(self) -> None:
        """Initialize the image getter."""
        DynamicContent.__post_init__(self)

        self._image_getter = partial(self._get_artwork_pil_image)

    def _get_artwork_pil_image(self) -> Image.Image:
        """Get the Image of the artwork image from the local file/remote URL.

        Returns:
            Image: Image instance of the artwork image
        """
        if not self.file_path:
            return const.EMPTY_IMAGE

        if self.file_path.is_file():
            LOGGER.debug("Opening image from path %s for %s", self.file_path, self.album)
            with suppress(UnidentifiedImageError):
                return Image.open(self.file_path)

        LOGGER.debug("Image not found at %s for %s", self.file_path, self.album)

        try:
            return self.download()
        except HTTPStatusError as err:
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
        return const.EMPTY_IMAGE

    @backoff(HTTPStatusError, logger=LOGGER, timeout=60, max_delay=10)
    def download(self) -> Image.Image:
        """Download the image from the URL to store it locally for future use."""

        if (
            self.artwork_uri is None
            or not str(self.artwork_uri)
            or not self.artist_directory
            or not self.file_path
        ):
            return const.EMPTY_IMAGE

        self.ARTWORK_DIRECTORY.joinpath(self.artist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        LOGGER.debug("Downloading artwork from remote URL: %s", self.artwork_uri)
        res = get(self.artwork_uri, timeout=120)
        res.raise_for_status()
        artwork_bytes = res.content

        image = Image.open(BytesIO(artwork_bytes)).resize((self.width, self.height))
        force_mkdir(self.file_path, path_is_file=True)
        image.save(self.file_path, "PNG")

        LOGGER.info(
            "New image from %s saved at %s for album %s",
            self.artwork_uri,
            self.file_path,
            self.album,
        )

        return image

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

        return self.ALPHANUM_PATTERN.sub("", self.album).lower() + ".png"

    @property
    def file_path(self) -> Path | None:
        """Return the local path to the artwork image.

        Returns:
            Path: fully-qualified path to the artwork image
        """
        if not self.artist_directory or not self.filename:
            return None

        return self.ARTWORK_DIRECTORY / self.artist_directory / self.filename

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

    def refresh_content(self) -> Generator[None, None, None]:
        """Refresh the content."""
        if self.track_metadata != self.previous_track:
            self.is_sleeping = False
            self.previous_track = self.track_metadata
            yield
        else:
            self.is_sleeping = True
            sleep(const.TICK_LENGTH)

        if None in self.track_metadata.values():
            self.stop(StopType.EXPIRED)

    def teardown(self) -> Generator[None, None, None]:
        """No teardown needed."""
        yield

    @property
    def album(self) -> str | None:
        """Return the album name."""
        return self.track_metadata.get("album")

    @property
    def artist(self) -> str | None:
        """Return the artist name."""
        return self.track_metadata.get("artist")

    @property
    def artwork_uri(self) -> URL | None:
        """Return the URL of the artwork."""
        return self.track_metadata.get("artwork_uri")

    @property
    def title(self) -> str | None:
        """Return the title of the track."""
        return self.track_metadata.get("title")
