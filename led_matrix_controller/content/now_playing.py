"""Class for the creation, caching, and management of artwork images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from re import compile as compile_regex
from typing import Annotated, ClassVar, Generator

from httpx import get
from models.setting import ParameterSetting
from utils import const
from wg_utilities.loggers import get_streaming_logger

from .dynamic_content import DynamicContent

LOGGER = get_streaming_logger(__name__)


@dataclass(kw_only=True, slots=True)
class NowPlaying(DynamicContent):
    """Class for the creation, caching, and management of artwork images."""

    ARTWORK_DIRECTORY: ClassVar[Path] = const.ASSETS_DIRECTORY / "artwork"

    ALPHANUM_PATTERN: ClassVar[Pattern[str]] = compile_regex(r"[\W_]+")

    title: Annotated[str, ParameterSetting()] = ""
    album: Annotated[str, ParameterSetting()] = ""
    artist: Annotated[str, ParameterSetting()] = ""
    uri: Annotated[str, ParameterSetting()] = ""

    # def _get_artwork_pil_image(
    #     self,
    #     size: int | None = None,
    #     *,
    #     ignore_cache: bool = False,
    # ) -> Image:
    #     """Get the Image of the artwork image from the cache/local file/remote URL.

    #     Args:
    #         size (int): integer value to use as height and width of artwork, in pixels
    #         ignore_cache (bool): whether to ignore the cache and download/resize the
    #             image

    #     Returns:
    #         Image: Image instance of the artwork image
    #     """
    #     if from_cache := self._image_cache is not None and ignore_cache is False:
    #         LOGGER.debug("Using cached image for %s", self.album)

    #         pil_image = self._image_cache
    #     elif self.file_path.is_file():
    #         LOGGER.debug("Opening image from path %s for %s", self.file_path, self.album)
    #         with self.file_path.open("rb") as fin:
    #             pil_image = open_image(BytesIO(fin.read()))
    #     else:
    #         pil_image = open_image(BytesIO(self.download()))

    #     # If a size is specified and the image hasn't already been cached (at this size)
    #     if size and not from_cache:
    #         LOGGER.debug("Resizing image to %ix%i", size, size)
    #         pil_image = pil_image.resize((size, size), Resampling.LANCZOS)

    # return pil_image

    def download(self) -> bytes:
        """Download the image from the URL to store it locally for future use."""

        self.ARTWORK_DIRECTORY.joinpath(self.artist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        if Path(self.uri).is_file():
            # Mainly used for copying the null image out of the repo into the artwork
            # directory
            LOGGER.debug("Opening local image: %s", self.uri)
            artwork_bytes = Path(self.uri).read_bytes()
        else:
            LOGGER.debug("Downloading artwork from remote URL: %s", self.uri)
            artwork_bytes = get(self.uri, timeout=120).content

        self.file_path.write_bytes(artwork_bytes)

        LOGGER.info(
            "New image from %s saved at %s for album %s",
            self.uri,
            self.file_path,
            self.album,
        )

        return artwork_bytes

    # def get_image(self, size: int | None = None, *, ignore_cache: bool = False) -> Image:
    #     """Return the image as a PIL Image object, with optional resizing.

    #     Args:
    #         size (int): integer value to use as height and width of artwork, in pixels
    #         ignore_cache (bool): whether to ignore the cache and download/resize the
    #             image again

    #     Returns:
    #         Image: PIL Image object of artwork
    #     """

    #     if self.cache_in_progress:
    #         LOGGER.debug("Waiting for cache to finish")
    #         while self.cache_in_progress:
    #             pass

    #     pil_image: Image = self._get_artwork_pil_image(size, ignore_cache=ignore_cache)

    #     return pil_image

    @property
    def artist_directory(self) -> str:
        """Return the artist name, with all non-alphanumeric characters removed.

        Returns:
            str: the artist name, with all non-alphanumeric characters removed
        """
        return self.ALPHANUM_PATTERN.sub("", self.artist).lower()

    @property
    def filename(self) -> str:
        """Return the album name, with all non-alphanumeric characters removed.

        Returns:
            str: the filename of the artwork image
        """
        return self.ALPHANUM_PATTERN.sub("", self.album).lower() + ".png"

    @property
    def file_path(self) -> Path:
        """Return the local path to the artwork image.

        Returns:
            Path: fully-qualified path to the artwork image
        """
        return self.ARTWORK_DIRECTORY / self.artist_directory / self.filename

    def __hash__(self) -> int:
        """Return the hash of the object."""
        return hash((self.artist, self.album, self.uri))

    def __str__(self) -> str:
        """Return the string representation of the object."""
        return self.__repr__()

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        return f"ArtworkImage({self.artist}, {self.album}, {self.uri})"

    def __iter__(self) -> Generator[None, None, None]:
        yield

    def teardown(self) -> Generator[None, None, None]:
        yield


# NULL_IMAGE = ArtworkImage(
#     "null",
#     "null",
#     str(Path(__file__).parents[3] / "assets" / "images" / "null.png"),
# )


# LOGGER.debug("Artwork directory: %s", ArtworkImage.ARTWORK_DIR.as_posix())
