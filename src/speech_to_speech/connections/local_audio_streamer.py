import logging
import threading
import time
from queue import Empty, Queue
from typing import Any

import numpy as np
import sounddevice as sd

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE
from speech_to_speech.pipeline.queue_types import AudioInItem, AudioOutItem

logger = logging.getLogger(__name__)

BARGE_IN_RMS_THRESHOLD = 900
BARGE_IN_CONSECUTIVE_CHUNKS = 3


class LocalAudioStreamer:
    def __init__(
        self,
        input_queue: Queue[AudioInItem],
        output_queue: Queue[AudioOutItem],
        should_listen: threading.Event,
        enabled_event: threading.Event | None = None,
        cancel_scope: CancelScope | None = None,
        interrupt_queues: list[Queue[Any]] | None = None,
        list_play_chunk_size: int = 512,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
    ) -> None:
        self.list_play_chunk_size = list_play_chunk_size

        self.stop_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.should_listen = should_listen
        self.enabled_event = enabled_event
        self.cancel_scope = cancel_scope
        self.interrupt_queues = interrupt_queues or []
        self.input_device = input_device
        self.output_device = output_device
        self._barge_in_chunks = 0

    def _enabled(self) -> bool:
        return self.enabled_event is None or self.enabled_event.is_set()

    def _drain_queue(self, queue: Queue[Any]) -> int:
        drained = 0
        while True:
            try:
                queue.get_nowait()
                drained += 1
            except Empty:
                return drained

    def _barge_in_detected(self, pcm: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float32)))))
        if rms < BARGE_IN_RMS_THRESHOLD:
            self._barge_in_chunks = 0
            return False

        self._barge_in_chunks += 1
        return self._barge_in_chunks >= BARGE_IN_CONSECUTIVE_CHUNKS

    def _interrupt_response(self, pcm: np.ndarray) -> None:
        if self.cancel_scope:
            self.cancel_scope.cancel()

        drained = self._drain_queue(self.output_queue)
        for queue in self.interrupt_queues:
            drained += self._drain_queue(queue)

        self.should_listen.set()
        self.input_queue.put(pcm.tobytes())
        self._barge_in_chunks = 0
        logger.info("Interrupted local audio response; drained %s queued item(s)", drained)

    def run(self) -> None:
        # Pre-generate a static dither buffer (±1 LSB, -96 dB) to keep the
        # audio sink active without calling numpy inside the real-time callback.
        dither = np.random.randint(-1, 2, size=(self.list_play_chunk_size, 1), dtype=np.int16)

        def callback(indata: np.ndarray, outdata: np.ndarray, frames: int, time: float, status: str) -> None:
            # During shutdown, just output silence
            if self.stop_event.is_set():
                outdata[:] = 0 * outdata
                return

            pcm = np.ascontiguousarray(indata, dtype=np.int16)

            if not self.output_queue.empty() and self._enabled() and not self.should_listen.is_set():
                if self._barge_in_detected(pcm):
                    self._interrupt_response(pcm)
                    outdata[:] = dither
                    return

            if self.output_queue.empty():
                if self._enabled() and self.should_listen.is_set():
                    self.input_queue.put(pcm.tobytes())
                outdata[:] = dither
            else:
                try:
                    audio_chunk = self.output_queue.get_nowait()
                    if isinstance(audio_chunk, np.ndarray):
                        outdata[:] = audio_chunk[:, np.newaxis]
                    elif audio_chunk == AUDIO_RESPONSE_DONE:
                        if self._enabled():
                            self.should_listen.set()
                            logger.debug("Response complete, listening re-enabled")
                        outdata[:] = 0 * outdata
                    else:
                        outdata[:] = 0 * outdata
                except Exception:
                    outdata[:] = 0 * outdata

        logger.debug("Available devices:")
        logger.debug(sd.query_devices())
        device = (self.input_device, self.output_device) if self.input_device is not None or self.output_device is not None else None
        if device is not None:
            logger.info("Using local audio devices: input=%s output=%s", self.input_device, self.output_device)
        with sd.Stream(
            samplerate=16000,
            dtype="int16",
            channels=1,
            device=device,
            callback=callback,
            blocksize=self.list_play_chunk_size,
        ):
            logger.info("Starting local audio stream")
            while not self.stop_event.is_set():
                time.sleep(0.001)
            print("Stopping recording")
