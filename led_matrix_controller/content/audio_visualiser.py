"""Windows Media Player? ;_;"""  # noqa: D415

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from json import dumps
from multiprocessing import shared_memory
from typing import Annotated, ClassVar, Generator

import numpy as np
from content.setting import ParameterSetting, TransitionableParameterSetting
from numpy.typing import NDArray  # noqa: TCH002
from scipy.fftpack import rfftfreq
from utils import const
from wg_utilities.loggers import get_streaming_logger

from .dynamic_content import DynamicContent

LOGGER = get_streaming_logger(__name__)


@dataclass(kw_only=True, slots=True)
class AudioVisualiser(DynamicContent):
    """Visualise the audio from an incoming mic/aux feed."""

    IS_OPAQUE: ClassVar[bool] = False

    low_freq_x: Annotated[
        int,
        TransitionableParameterSetting(
            minimum=0,
            maximum=const.MATRIX_WIDTH - 1,
            transition_rate=0.05,
            icon="mdi:arrow-left-right",
            unit_of_measurement="",
            display_mode="slider",
            invoke_settings_callback=True,
        ),
    ] = 1
    """X coordinate of the lowest frequency in the grid."""

    low_freq_y: Annotated[
        int,
        TransitionableParameterSetting(
            minimum=0,
            maximum=const.MATRIX_HEIGHT - 1,
            transition_rate=0.05,
            icon="mdi:arrow-up-down",
            unit_of_measurement="",
            display_mode="slider",
            invoke_settings_callback=True,
        ),
    ] = 1
    """Y coordinate of the lowest frequency in the grid."""

    high_freq_x: Annotated[
        int,
        TransitionableParameterSetting(
            minimum=0,
            maximum=const.MATRIX_HEIGHT - 1,
            transition_rate=0.05,
            icon="mdi:arrow-up-down",
            unit_of_measurement="",
            display_mode="slider",
            invoke_settings_callback=True,
        ),
    ] = const.MATRIX_WIDTH - 1
    """X coordinate of the highest frequency in the grid."""

    high_freq_y: Annotated[
        int,
        TransitionableParameterSetting(
            minimum=0,
            maximum=const.MATRIX_HEIGHT - 1,
            transition_rate=0.05,
            icon="mdi:arrow-up-down",
            unit_of_measurement="",
            display_mode="slider",
            invoke_settings_callback=True,
        ),
    ] = const.MATRIX_HEIGHT - 1
    """Y coordinate of the highest frequency in the grid."""

    cutoff_frequency: Annotated[
        int,
        ParameterSetting(
            minimum=0,
            maximum=20000,
            icon="mdi:sine-wave",
            unit_of_measurement="Hz",
            display_mode="slider",
            invoke_settings_callback=True,
        ),
    ] = 5000

    colormap_length: Annotated[
        int,
        ParameterSetting(
            minimum=100,
            maximum=10000,
            icon="mdi:sine-wave",
            unit_of_measurement="colors",
            display_mode="slider",
            invoke_settings_callback=True,
        ),
    ] = 10000

    low_magnitude_hex_color: Annotated[
        str,
        ParameterSetting(
            invoke_settings_callback=True,
            icon="mdi:palette",
        ),
    ] = "#0000FF80"

    high_magnitude_hex_color: Annotated[
        str,
        ParameterSetting(
            invoke_settings_callback=True,
            icon="mdi:palette",
        ),
    ] = "#FF0000FF"

    freq_bin_indices: NDArray[np.int_] = field(init=False, repr=False)

    shm: shared_memory.SharedMemory = field(init=False, repr=False)

    sample_rate: int = field(default=44100, init=False, repr=False)
    """Number of audio samples per second."""

    chunk_size: int = field(default=441, init=False, repr=False)
    """Number of audio samples per chunk.

    Set to 441 to allow for 100 FPS updates.
    """

    def __post_init__(self) -> None:
        """Initialize the audio visualiser."""
        DynamicContent.__post_init__(self)

        self.update_colormap()
        self.update_frequency_foci()

        self.setup_shm(allow_failure=True)

    def setup(self) -> None:
        """Setup the audio visualiser."""
        if not hasattr(self, "shm"):
            self.setup_shm()

    def refresh_content(self) -> Generator[None, None, None]:
        """Refresh the content."""
        audio: NDArray[np.float64] = np.ndarray(
            shape=(self.chunk_size // 2 + 1,),
            dtype=np.float64,
            buffer=self.shm.buf,
        )

        while self.active:
            audio_ints = (audio * (self.colormap_length - 1)).astype(np.int_)
            self.pixels[:, :] = audio_ints[self.freq_bin_indices]

            yield

    def setting_update_callback(self, update_setting: str | None = None) -> None:
        """Update the colormap."""
        if update_setting in {
            "low_magnitude_hex_color",
            "high_magnitude_hex_color",
            "colormap_length",
        }:
            self.update_colormap()
        elif update_setting in {
            "low_freq_x",
            "low_freq_y",
            "high_freq_x",
            "high_freq_y",
            "cutoff_frequency",
        }:
            self.update_frequency_foci()

    def setup_shm(self, *, allow_failure: bool = False) -> None:
        """Setup the shared memory."""
        try:
            self.shm = shared_memory.SharedMemory(name=const.AUDIO_VISUALISER_SHM_NAME)
            atexit.register(self.shm.close)
        except FileNotFoundError:
            LOGGER.critical("Shared memory %r not found", const.AUDIO_VISUALISER_SHM_NAME)
            if not allow_failure:
                raise

    def update_colormap(self) -> None:
        """Update the colormap."""
        low_magnitude_color = tuple(
            int(self.low_magnitude_hex_color.lstrip("#")[i : i + 2], 16)
            for i in (0, 2, 4, 6)
        )
        high_magnitude_color = tuple(
            int(self.high_magnitude_hex_color.lstrip("#")[i : i + 2], 16)
            for i in (0, 2, 4, 6)
        )

        colors = {
            0: (0, 0, 0, 0),
            self.colormap_length // 150: (0, 0, 0, 0),
            self.colormap_length // 80: low_magnitude_color,
            self.colormap_length // 50: low_magnitude_color,
            self.colormap_length // 10: high_magnitude_color,
            self.colormap_length: high_magnitude_color,
        }

        LOGGER.debug("Colormap: %s", dumps(colors))

        gradient = []
        prev = None
        for index, color in colors.items():
            if prev is None:
                prev = (index, color)
                gradient.append(color)
                continue

            prev_index, prev_color = prev

            if (steps := index - prev_index) > 1:
                for i in range(steps):
                    r = int(prev_color[0] + (color[0] - prev_color[0]) * i / (steps - 1))
                    g = int(prev_color[1] + (color[1] - prev_color[1]) * i / (steps - 1))
                    b = int(prev_color[2] + (color[2] - prev_color[2]) * i / (steps - 1))
                    a = int(prev_color[3] + (color[3] - prev_color[3]) * i / (steps - 1))
                    gradient.append((r, g, b, a))

            prev = (index, color)

        self.colormap = np.array(gradient, dtype=np.uint8)

    def update_frequency_foci(self) -> None:
        """Update the location of the frequency foci on the visualisation."""
        freqs = rfftfreq(self.chunk_size, 1 / self.sample_rate)

        cutoff_idx = np.where(freqs <= self.cutoff_frequency)[0][-1]

        freq_bin_count = cutoff_idx + 1

        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))

        # Compute distances from the lowest/highest frequency focal points
        dist_low = np.sqrt((x - self.high_freq_x) ** 2 + (y - self.high_freq_y) ** 2)
        dist_high = np.sqrt((x - self.low_freq_x) ** 2 + (y - self.low_freq_y) ** 2)

        # Combine distances to create a gradient
        total_distance = dist_low + dist_high

        # Normalize distances to range [0, 1]
        normalized_distances = dist_high / total_distance

        # Map normalized distances to frequency bin indices
        freq_bin_indices = (normalized_distances * (freq_bin_count - 1)).astype(int)

        # Ensure indices are within valid range
        self.freq_bin_indices = np.clip(freq_bin_indices, 0, freq_bin_count - 1)