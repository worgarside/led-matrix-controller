"""Process incoming audio and save it to shared memory."""

from __future__ import annotations

import collections
import os
import threading
import time
from contextlib import suppress
from json import JSONDecodeError, loads
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pyaudio
from content import Combination
from content.audio_visualiser import AudioVisualiser
from models import Matrix
from scipy.fft import rfft
from utils import const, get_shared_memory
from wg_utilities.functions import backoff
from wg_utilities.loggers import get_streaming_logger
from wg_utilities.utils import mqtt

if TYPE_CHECKING:
    import paho.mqtt.client
    from numpy.typing import NDArray
    from paho.mqtt.client import MQTTMessage

LOGGER = get_streaming_logger(__name__)

_AV = AudioVisualiser()
AV_CONTENT_ID: Final = _AV.id
CHUNK_SIZE_TOPIC: Final = _AV.settings["chunk_size"].mqtt_topic
SAMPLE_RATE_TOPIC: Final = _AV.settings["sample_rate"].mqtt_topic
CURRENT_CONTENT_TOPIC: Final = Matrix.mqtt_topic("current-content")
_COMBO = Combination()
COMBO_CONTENT_ID: Final = _COMBO.id
COMBO_CONTENT_TOPIC: Final = _COMBO.settings["content"].mqtt_topic
del _AV, _COMBO

PYAUDIO = pyaudio.PyAudio()

MAX_MAGNITUDE_TOPIC: Final = f"/{const.HOSTNAME}/audio-processor/max-magnitude"
MAGNITUDE_HISTORY_SIZE: Final = int(os.getenv("MAGNITUDE_HISTORY_SIZE", "200"))
MAX_MAGNITUDE_RELATIVE_STEP: Final = float(
    os.getenv("MAX_MAGNITUDE_RELATIVE_STEP", "0.01"),
)  # Max 1% change relative to current value
MIN_MAGNITUDE_ABSOLUTE_STEP: Final = 1e-6  # Min absolute change step
MAX_MAGNITUDE_UPDATE_FREQUENCY: Final = float(
    os.getenv("MAX_MAGNITUDE_UPDATE_FREQUENCY", "0.01"),
)  # seconds


class AudioProcessor:
    """Process incoming audio and save it to shared memory."""

    stream: pyaudio.Stream

    def __init__(self) -> None:
        self.active = False

        self.chunk_size = 441
        """Number of audio samples per chunk.

        /mtrxpi/audio-visualiser/parameter/chunk-size
        """

        self.sample_rate = 44100
        """Sampling rate in Hz.

        /mtrxpi/audio-visualiser/parameter/sample-rate
        """

        self._stream_config = (
            self.chunk_size,
            self.sample_rate,
        )

        self.create_stream()

        self.max_magnitude = 1e-9
        self.magnitude_history: collections.deque[float] = collections.deque(
            maxlen=MAGNITUDE_HISTORY_SIZE,
        )

        self.shm = get_shared_memory(logger=LOGGER)

        self.audio_visualiser_in_combination = False
        self.current_content: str | None = None
        self.worker_thread: threading.Thread | None = None
        self.last_magnitude_update = time.time()

        mqtt.CLIENT.message_callback_add(
            CURRENT_CONTENT_TOPIC,
            self._on_current_content_message,
        )
        mqtt.CLIENT.message_callback_add(
            COMBO_CONTENT_TOPIC,
            self._on_combined_content_message,
        )
        mqtt.CLIENT.message_callback_add(
            CHUNK_SIZE_TOPIC,
            self._on_chunk_size_message,
        )
        mqtt.CLIENT.message_callback_add(
            SAMPLE_RATE_TOPIC,
            self._on_sample_rate_message,
        )

        self.get_magnitudes()

    def create_shm_array(self) -> None:
        """Create the shared memory array."""
        fft_magnitudes = self.get_magnitudes()

        try:
            self.shm_array: NDArray[np.float64] = np.ndarray(
                shape=fft_magnitudes.shape,
                dtype=fft_magnitudes.dtype,
                buffer=self.shm.buf,
            )
        except TypeError as err:
            if "buffer is too small" in str(err):
                LOGGER.critical(
                    "Shared memory buffer too small (%s bytes required)",
                    fft_magnitudes.nbytes,
                )
            raise

    def create_stream(self) -> None:
        """Create the PyAudio stream."""
        if (
            hasattr(self, "stream")
            and self.stream.is_active()
            and self._stream_config
            == (
                self.chunk_size,
                self.sample_rate,
            )
        ):
            # i.e. the stream is already created with the correct config
            return

        if hasattr(self, "stream"):
            self.stream.close()

        self.stream = PYAUDIO.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        LOGGER.info(
            "Created stream with chunk size %s and sample rate %s",
            self.chunk_size,
            self.sample_rate,
        )

        self._stream_config = (
            self.chunk_size,
            self.sample_rate,
        )

    def get_magnitudes(self) -> NDArray[np.float64]:
        """Get the magnitudes of the audio stream and apply FFT."""
        data = np.frombuffer(
            self.stream.read(self.chunk_size, exception_on_overflow=False),
            dtype=np.int16,
        )

        fft_magnitudes: NDArray[np.float64] = np.abs(rfft(data))

        current_max_fft = fft_magnitudes.max()
        self.magnitude_history.append(current_max_fft)

        prev = self.max_magnitude  # Value from the end of the previous call

        # Calculate the target for max_magnitude
        if len(self.magnitude_history) >= MAGNITUDE_HISTORY_SIZE / 2:
            target_max_magnitude = float(np.percentile(self.magnitude_history, 95))
        else:
            # Initial phase, use a growing max while populating history
            target_max_magnitude = max(prev, current_max_fft)

        # Smooth the update to self.max_magnitude
        # Determine the maximum allowed step size for this update
        relative_step_size = prev * MAX_MAGNITUDE_RELATIVE_STEP
        actual_step_limit = max(relative_step_size, MIN_MAGNITUDE_ABSOLUTE_STEP)

        if time.time() - self.last_magnitude_update > MAX_MAGNITUDE_UPDATE_FREQUENCY:
            if target_max_magnitude > prev:
                self.max_magnitude = min(target_max_magnitude, prev + actual_step_limit)
            elif target_max_magnitude < prev:
                self.max_magnitude = max(target_max_magnitude, prev - actual_step_limit)

            if prev != self.max_magnitude:
                LOGGER.info("Max magnitude: %.3f", self.max_magnitude)

                mqtt.CLIENT.publish(
                    MAX_MAGNITUDE_TOPIC,
                    payload=round(self.max_magnitude, 3),
                    qos=2,
                    retain=True,
                )

                self.last_magnitude_update = time.time()

        # Get in range 0-1
        return fft_magnitudes / target_max_magnitude

    @backoff(OSError, logger=LOGGER, max_delay=1, max_tries=20)
    def process_incoming_audio(self) -> None:
        """Take incoming audio and save it to shared memory."""
        self.create_shm_array()

        try:
            while self.active:
                self.shm_array[:] = self.get_magnitudes()
        except OSError as err:
            if "Unanticipated host error" in str(err):
                LOGGER.warning(
                    "Error with PyAudio stream, falling back to backoff: %s",
                    err,
                )
                raise

        LOGGER.info("Audio processing loop exiting.")

    def _on_chunk_size_message(
        self,
        _: paho.mqtt.client.Client,
        __: Any,
        message: MQTTMessage,
    ) -> None:
        """Update the chunk size and refresh the stream."""
        try:
            self.chunk_size = int(message.payload)
        except ValueError:
            LOGGER.exception("Failed to decode chunk size payload: %s", message.payload)
            return

        self.create_stream()
        self.create_shm_array()

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

    def _on_current_content_message(
        self,
        _: paho.mqtt.client.Client,
        __: Any,
        message: MQTTMessage,
    ) -> None:
        self.current_content = message.payload.decode()
        LOGGER.info("Current content: %s", self.current_content)

        self.update_loop_status()

    def _on_sample_rate_message(
        self,
        _: paho.mqtt.client.Client,
        __: Any,
        message: MQTTMessage,
    ) -> None:
        """Update the sample rate and refresh the stream."""
        try:
            self.sample_rate = int(message.payload)
        except ValueError:
            LOGGER.exception("Failed to decode sample rate payload: %s", message.payload)
            return

        self.create_stream()
        self.create_shm_array()

    def update_loop_status(self) -> None:
        """Start or stop the audio processing loop based on the current content."""
        if self.current_content == AV_CONTENT_ID or (
            self.current_content == COMBO_CONTENT_ID
            and self.audio_visualiser_in_combination
        ):
            if not self.active:
                LOGGER.info(
                    "Starting audio processing (current content: %s, AV in combination: %s)",
                    self.current_content,
                    self.audio_visualiser_in_combination,
                )
                self.active = True
                self.worker_thread = threading.Thread(target=self.process_incoming_audio)
                self.worker_thread.start()
        else:
            LOGGER.info(
                "Stopping audio processing (current content: %s, AV in combination: %s)",
                self.current_content,
                self.audio_visualiser_in_combination,
            )

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
        mqtt.CLIENT.subscribe(CHUNK_SIZE_TOPIC, qos=2)
        mqtt.CLIENT.subscribe(SAMPLE_RATE_TOPIC, qos=2)

        mqtt.CLIENT.loop_forever()

        LOGGER.info("MQTT loop finished.")
    finally:
        if processor.worker_thread:
            with suppress(Exception):
                processor.worker_thread.join()
                LOGGER.info("Audio processing thread joined.")

        processor.stream.stop_stream()
        processor.stream.close()
        LOGGER.info("PyAudio stream closed.")

        PYAUDIO.terminate()
        LOGGER.info("PyAudio terminated.")

    LOGGER.info("Audio processor finished.")
    raise SystemExit


if __name__ == "__main__":
    main()
