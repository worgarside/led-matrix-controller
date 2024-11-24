"""Process incoming audio and save it to shared memory."""

from __future__ import annotations

import time
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

import numpy as np
import pyaudio
from scipy.fft import rfft
from wg_utilities.loggers import get_streaming_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

CHUNK = 441
"""Number of audio samples per chunk."""


PYAUDIO = pyaudio.PyAudio()

LOGGER = get_streaming_logger(__name__)


MAX_MAGNITUDE = 0


def get_magnitudes(stream: pyaudio.Stream) -> NDArray[np.float64]:
    """Get the magnitudes of the audio stream and apply FFT."""
    global MAX_MAGNITUDE  # noqa: PLW0603

    data = np.frombuffer(
        stream.read(CHUNK, exception_on_overflow=False),
        dtype=np.int16,
    )

    fft_magnitudes: NDArray[np.float64] = np.abs(rfft(data))

    MAX_MAGNITUDE = max(MAX_MAGNITUDE, fft_magnitudes.max())

    # Get in range 0-99
    return fft_magnitudes / MAX_MAGNITUDE * 99


def process_incoming_audio(
    stream: pyaudio.Stream,
    shm: shared_memory.SharedMemory,
) -> None:
    """Take incoming audio and save it to shared memory."""
    fft_magnitudes = get_magnitudes(stream)

    dest: NDArray[np.float64] = np.ndarray(
        shape=fft_magnitudes.shape,
        dtype=fft_magnitudes.dtype,
        buffer=shm.buf,
    )
    while True:
        start_time = time.time()

        dest[:] = get_magnitudes(stream)

        # Sleep for remaining time to achieve 0.01s interval
        elapsed = time.time() - start_time
        if elapsed < 0.01:  # noqa: PLR2004
            time.sleep(0.01 - elapsed)
        else:
            LOGGER.warning("Audio processing took %fs", elapsed)


def main() -> None:
    """Main function."""
    stream = PYAUDIO.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        input=True,
        frames_per_buffer=CHUNK,
    )

    shm = shared_memory.SharedMemory(
        name="audio1",
        create=True,
        size=64 * 64 * 4 * 4,
    )

    try:
        process_incoming_audio(stream, shm)
    finally:
        stream.stop_stream()
        stream.close()
        PYAUDIO.terminate()

        shm.unlink()
        shm.close()


if __name__ == "__main__":
    main()
