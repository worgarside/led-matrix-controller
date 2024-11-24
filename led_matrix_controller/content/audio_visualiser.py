"""Windows Media Player? ;_;"""  # noqa: D415

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import ClassVar, Generator, Literal

import numpy as np
from numpy.typing import NDArray  # noqa: TCH002
from scipy.fftpack import rfftfreq
from wg_utilities.loggers import get_streaming_logger

from .dynamic_content import DynamicContent

LOGGER = get_streaming_logger(__name__)


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

    shm: shared_memory.SharedMemory = field(init=False, repr=False)

    cutoff_frequency: int = 5000

    def __post_init__(self) -> None:  # noqa: PLR0914
        """Initialize the audio visualiser."""
        DynamicContent.__post_init__(self)

        self.shm = shared_memory.SharedMemory(name="audio1")

        steps = 100

        self.colormap = np.array(
            [((0, 0, 0, 0))]
            + [
                (int(255 * (1 - i / (steps - 1))), 0, int(255 * (i / (steps - 1))), 255)
                for i in range(steps - 1)
            ],
            dtype=np.uint8,
        )

        freqs = rfftfreq(self.chunk_size, 1 / self.sample_rate)
        cutoff_idx = np.where(freqs <= self.cutoff_frequency)[0][-1]

        freq_bin_count = cutoff_idx + 1

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

    def refresh_content(self) -> Generator[None, None, None]:
        """Refresh the content."""
        audio: NDArray[np.float64] = np.ndarray(
            shape=(self.chunk_size // 2 + 1,),
            dtype=np.float64,
            buffer=self.shm.buf,
        )

        while self.active:
            self.pixels[:, :] = audio[self.freq_bin_indices]
            yield
