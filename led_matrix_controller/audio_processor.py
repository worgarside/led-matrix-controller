"""Process incoming audio and save it to shared memory."""

from __future__ import annotations

import threading
from contextlib import suppress
from json import JSONDecodeError, loads
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pyaudio
from content import Combination
from content.audio_visualiser import AudioVisualiser
from models import Matrix
from scipy.fft import rfft
from utils import const
from wg_utilities.loggers import get_streaming_logger
from wg_utilities.utils import mqtt

if TYPE_CHECKING:
    import paho.mqtt.client
    from numpy.typing import NDArray
    from paho.mqtt.client import MQTTMessage

LOGGER = get_streaming_logger(__name__)

AV_CONTENT_ID: Final = AudioVisualiser().id
CURRENT_CONTENT_TOPIC: Final = Matrix.mqtt_topic("current-content")
_COMBO = Combination()
COMBO_CONTENT_ID: Final = _COMBO.id
COMBO_CONTENT_TOPIC: Final = _COMBO.settings["content"].mqtt_topic
del _COMBO

PYAUDIO = pyaudio.PyAudio()

CHUNK = 441
"""Number of audio samples per chunk."""

RATE = 44100
"""Sampling rate in Hz."""


class AudioProcessor:
    """Process incoming audio and save it to shared memory."""

    def __init__(self) -> None:
        self.active = False

        self.stream = PYAUDIO.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        self.max_magnitude = 1e-9

        try:
            self.shm = shared_memory.SharedMemory(
                name=const.AUDIO_VISUALISER_SHM_NAME,
                create=True,
                size=self.get_magnitudes(self.stream).nbytes,
            )
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=const.AUDIO_VISUALISER_SHM_NAME)

        self.audio_visualiser_in_combination = False
        self.current_content: str | None = None
        self.worker_thread: threading.Thread | None = None

        mqtt.CLIENT.message_callback_add(
            CURRENT_CONTENT_TOPIC,
            self._on_current_content_message,
        )
        mqtt.CLIENT.message_callback_add(
            COMBO_CONTENT_TOPIC,
            self._on_combined_content_message,
        )

    def get_magnitudes(self, stream: pyaudio.Stream) -> NDArray[np.float64]:
        """Get the magnitudes of the audio stream and apply FFT."""
        data = np.frombuffer(
            stream.read(CHUNK, exception_on_overflow=False),
            dtype=np.int16,
        )

        fft_magnitudes: NDArray[np.float64] = np.abs(rfft(data))

        self.max_magnitude = max(self.max_magnitude, fft_magnitudes.max())

        # Get in range 0-1
        return fft_magnitudes / self.max_magnitude

    def process_incoming_audio(
        self,
    ) -> None:
        """Take incoming audio and save it to shared memory."""
        fft_magnitudes = self.get_magnitudes(self.stream)

        dest: NDArray[np.float64] = np.ndarray(
            shape=fft_magnitudes.shape,
            dtype=fft_magnitudes.dtype,
            buffer=self.shm.buf,
        )

        while self.active:
            dest[:] = self.get_magnitudes(self.stream)

        LOGGER.info("Audio processing loop exiting.")

    def _on_current_content_message(
        self,
        _: paho.mqtt.client.Client,
        __: Any,
        message: MQTTMessage,
    ) -> None:
        self.current_content = message.payload.decode()
        LOGGER.info("Current content: %s", self.current_content)

        self.update_loop_status()

    def _on_combined_content_message(
        self,
        _: paho.mqtt.client.Client,
        __: Any,
        message: MQTTMessage,
    ) -> None:
        """Process incoming MQTT messages."""
        try:
            combined_content = loads(message.payload.decode())
        except JSONDecodeError:
            LOGGER.exception("Failed to decode JSON payload: %s", message.payload)
            return

        self.audio_visualiser_in_combination = AV_CONTENT_ID in combined_content

        LOGGER.info(
            "Audio visualiser is %s combination: %s",
            "in" if self.audio_visualiser_in_combination else "not in",
            combined_content,
        )

        self.update_loop_status()

    def update_loop_status(self) -> None:
        """Start or stop the audio processing loop based on the current content."""
        if self.current_content == AV_CONTENT_ID or (
            self.current_content == COMBO_CONTENT_ID
            and self.audio_visualiser_in_combination
        ):
            LOGGER.info(
                "Starting audio processing (current content: %s, AV in combination: %s)",
                self.current_content,
                self.audio_visualiser_in_combination,
            )
            self.active = True
            self.worker_thread = threading.Thread(target=self.process_incoming_audio)
            self.worker_thread.start()
        else:
            LOGGER.info("Stopping audio processing")
            self.active = False
            if self.worker_thread:
                self.worker_thread.join()

            LOGGER.info(
                "Audio processing loop stopped. Thread %s alive.",
                "is"
                if self.worker_thread and self.worker_thread.is_alive()
                else "is not",
            )


def main() -> None:
    """Main function."""
    processor = AudioProcessor()

    try:
        mqtt.CLIENT.connect(mqtt.MQTT_HOST)

        mqtt.CLIENT.subscribe(CURRENT_CONTENT_TOPIC, qos=2)
        mqtt.CLIENT.subscribe(COMBO_CONTENT_TOPIC, qos=2)

        mqtt.CLIENT.loop_forever()
    finally:
        if processor.worker_thread:
            with suppress(Exception):
                processor.worker_thread.join()

        processor.stream.stop_stream()
        processor.stream.close()
        PYAUDIO.terminate()

        processor.shm.close()
        processor.shm.unlink()


if __name__ == "__main__":
    main()
