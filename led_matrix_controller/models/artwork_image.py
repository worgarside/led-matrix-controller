"""Class for the creation, caching, and management of artwork images."""

from __future__ import annotations

from io import BytesIO
from logging import DEBUG, getLogger
from os import getenv
from pathlib import Path
from re import Pattern
from re import compile as compile_regex
from threading import Thread
from typing import ClassVar, Final

from PIL.Image import Image, Resampling
from PIL.Image import open as open_image
from requests import get
from wg_utilities.functions import force_mkdir
from wg_utilities.loggers import add_stream_handler

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
add_stream_handler(LOGGER)


class ArtworkImage:
    """Class for the creation, caching, and management of artwork images."""

    ARTWORK_DIR: Final[Path] = force_mkdir(
        Path(getenv("ARTWORK_DIR", "/var/cache/led-matrix-controller/artwork")).resolve(),
    )
    ALPHANUM_PATTERN: ClassVar[Pattern[str]] = compile_regex(r"[\W_]+")

    def __init__(
        self,
        album: str,
        artist: str,
        url: str,
        *,
        pre_cache: bool = False,
        pre_cache_size: int | None = None,
    ) -> None:
        self.album = album
        self.artist = artist or "unknown"
        self.url = url

        self._image_cache: Image | None = None
        self.cache_in_progress: bool = False

        if pre_cache:
            cache_thread = Thread(
                target=self._cache_image,
                kwargs={"size": pre_cache_size},
            )
            cache_thread.start()

    def _cache_image(self, size: int) -> None:
        """Cache the image in memory for future use.

        Args:
            size (int): integer value to use as height and width of artwork, in pixels
        """
        LOGGER.debug("Pre-caching image, setting cache_in_progress to True")

        self.cache_in_progress = True

        LOGGER.debug(
            "Caching image in memory with size %s",
            size,
        )
        self._image_cache = self._get_artwork_pil_image(size)
        LOGGER.debug("Cache complete, setting cache_in_progress to False")
        self.cache_in_progress = False

    def _get_artwork_pil_image(
        self, size: int | None = None, *, ignore_cache: bool = False,
    ) -> Image:
        """Get the Image of the artwork image from the cache/local file/remote URL.

        Args:
            size (int): integer value to use as height and width of artwork, in pixels
            ignore_cache (bool): whether to ignore the cache and download/resize the
                image

        Returns:
            Image: Image instance of the artwork image
        """
        if from_cache := self._image_cache is not None and ignore_cache is False:
            LOGGER.debug("Using cached image for %s", self.album)

            pil_image = self._image_cache
        elif self.file_path.is_file():
            LOGGER.debug("Opening image from path %s for %s", self.file_path, self.album)
            with self.file_path.open("rb") as fin:
                pil_image = open_image(BytesIO(fin.read()))
        else:
            pil_image = open_image(BytesIO(self.download()))

        # If a size is specified and the image hasn't already been cached (at this size)
        if size and not from_cache:
            LOGGER.debug("Resizing image to %ix%i", size, size)
            pil_image = pil_image.resize((size, size), Resampling.LANCZOS)

        return pil_image

    def download(self) -> bytes:
        """Download the image from the URL to store it locally for future use."""

        self.ARTWORK_DIR.joinpath(self.artist_directory).mkdir(
            parents=True, exist_ok=True,
        )

        if Path(self.url).is_file():
            # Mainly used for copying the null image out of the repo into the artwork
            # directory
            LOGGER.debug("Opening local image: %s", self.url)
            artwork_bytes = Path(self.url).read_bytes()
        else:
            LOGGER.debug("Downloading artwork from remote URL: %s", self.url)
            artwork_bytes = get(self.url, timeout=120).content

        self.file_path.write_bytes(artwork_bytes)

        LOGGER.info(
            "New image from %s saved at %s for album %s",
            self.url,
            self.file_path,
            self.album,
        )

        return artwork_bytes

    def get_image(self, size: int | None = None, *, ignore_cache: bool = False) -> Image:
        """Return the image as a PIL Image object, with optional resizing.

        Args:
            size (int): integer value to use as height and width of artwork, in pixels
            ignore_cache (bool): whether to ignore the cache and download/resize the
                image again

        Returns:
            Image: PIL Image object of artwork
        """

        if self.cache_in_progress:
            LOGGER.debug("Waiting for cache to finish")
            while self.cache_in_progress:
                pass

        pil_image: Image = self._get_artwork_pil_image(size, ignore_cache=ignore_cache)

        return pil_image

    @property
    def artist_directory(self) -> str:
        """Return the artist name, with all non-alphanumeric characters removed.

        Returns:
            str: the artist name, with all non-alphanumeric characters removed
        """
        return str(ArtworkImage.ALPHANUM_PATTERN.sub("", self.artist).lower())

    @property
    def filename(self) -> str:
        """Return the album name, with all non-alphanumeric characters removed.

        Returns:
            str: the filename of the artwork image
        """
        return str(ArtworkImage.ALPHANUM_PATTERN.sub("", self.album).lower() + ".png")

    @property
    def file_path(self) -> Path:
        """Return the local path to the artwork image.

        Returns:
            Path: fully-qualified path to the artwork image
        """
        return self.ARTWORK_DIR / self.artist_directory / self.filename

    def __hash__(self) -> int:
        """Return the hash of the object."""
        return hash((self.artist, self.album, self.url))

    def __str__(self) -> str:
        """Return the string representation of the object."""
        return self.__repr__()

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        return f"ArtworkImage({self.artist}, {self.album}, {self.url})"


NULL_IMAGE = ArtworkImage(
    "null", "null", str(Path(__file__).parents[3] / "assets" / "images" / "null.png"),
)


LOGGER.debug("Artwork directory: %s", ArtworkImage.ARTWORK_DIR.as_posix())
