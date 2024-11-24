"""Class for the creation, caching, and management of artwork images."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Final, Generator, Literal, TypedDict

import numpy as np
import pyaudio
from httpx import URL
from numpy.typing import NDArray  # noqa: TCH002
from scipy.fft import rfft
from wg_utilities.loggers import get_streaming_logger

from .dynamic_content import DynamicContent

LOGGER = get_streaming_logger(__name__)


PYAUDIO = pyaudio.PyAudio()


class TrackMeta(TypedDict):
    """Type definition for the track metadata."""

    title: str | None
    album: str | None
    artist: str | None
    album_artwork_url: URL | None


_INITIAL_TRACK_META: Final[TrackMeta] = {
    "title": "",
    "album": "",
    "artist": "",
    "album_artwork_url": URL(""),
}


@dataclass(kw_only=True, slots=True)
class AudioVisualiser(DynamicContent):
    """Visualise the audio from an incoming mic/aux feed."""

    IS_OPAQUE: ClassVar[bool] = False

    channels: Literal[1, 2] = field(default=1)
    """Number of audio channels."""

    sample_rate: int = field(default=44100)
    """Number of audio samples per second."""

    chunk_size: int = field(default=441)
    """Number of audio samples per chunk.

    Set to 441 to allow for 100 FPS updates.
    """

    low_freq_focus: tuple[int, int] = field(default=(44, 44))
    """Coordinates of the lowest frequency in the grid."""

    high_freq_focus: tuple[int, int] = field(default=(0, 0))
    """Coordinates of the highest frequency in the grid."""

    freq_bin_indices: NDArray[np.int_] = field(init=False, repr=False)

    stream: pyaudio.Stream = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the audio visualiser."""
        DynamicContent.__post_init__(self)

        steps = 100

        self.colormap = np.array(
            [((0, 0, 0, 0))]
            + [
                (int(255 * (1 - i / (steps - 1))), 0, int(255 * (i / (steps - 1))), 255)
                for i in range(steps - 1)
            ],
            dtype=np.uint8,
        )

        freq_bin_count = self.chunk_size // 2 + 1

        x_coords = np.arange(self.width)
        y_coords = np.arange(self.height)

        x, y = np.meshgrid(x_coords, y_coords)

        x1, y1 = self.low_freq_focus
        x2, y2 = self.high_freq_focus

        # Compute distances from the lowest frequency focal point (X1, Y1)
        dist_low = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)

        # Compute distances from the highest frequency focal point (X2, Y2)
        dist_high = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)

        # Combine distances to create a gradient
        total_distance = dist_low + dist_high

        # Normalize distances to range [0, 1]
        normalized_distances = dist_high / total_distance

        # Map normalized distances to frequency bin indices
        freq_bin_indices = (normalized_distances * (freq_bin_count - 1)).astype(int)

        # Ensure indices are within valid range
        self.freq_bin_indices = np.clip(freq_bin_indices, 0, freq_bin_count - 1)

    def setup(self) -> None:
        """Setup the audio visualiser."""
        LOGGER.info("Opening audio stream")
        self.stream = PYAUDIO.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=0,
        )

    def refresh_content(self) -> Generator[None, None, None]:
        """Refresh the content."""
        # max_, min_ = 0, 0  # noqa: ERA001
        while self.active:
            data = np.frombuffer(
                self.stream.read(self.chunk_size, exception_on_overflow=False),
                dtype=np.int16,
            )

            fft_magnitudes = np.abs(rfft(data))

            # Normalize magnitudes to range [0, 100]
            fft_magnitudes /= np.max(fft_magnitudes)
            fft_magnitudes *= 100
            fft_magnitudes -= 1

            # Map magnitudes to the grid
            self.pixels[:, :] = fft_magnitudes[self.freq_bin_indices]

            # max_ = max(max_, self.pixels.max())  # noqa: ERA001
            # min_ = min(min_, self.pixels.min())  # noqa: ERA001

            # print(max_, min_)  # noqa: ERA001
            yield

    def teardown(self) -> None:
        """Teardown the audio visualiser."""
        self.stream.close()
