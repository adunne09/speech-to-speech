import logging
import threading
import time
from queue import Empty, Queue
from typing import Any

import numpy as np
import sounddevice as sd

from speech_to_speech.pipeline.audio_devices import AudioDeviceController
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE
from speech_to_speech.pipeline.queue_types import AudioInItem, AudioOutItem
from speech_to_speech.pipeline.speex_echo_canceller import SpeexEchoCanceller

logger = logging.getLogger(__name__)

BARGE_IN_RMS_THRESHOLD = 1800
BARGE_IN_WARMUP_RMS_THRESHOLD = 3000
BARGE_IN_CONSECUTIVE_CHUNKS = 3
DEVICE_CHECK_INTERVAL_SECONDS = 1.0
AEC_FILTER_LENGTH_MULTIPLIER = 10
BARGE_IN_DEBUG_MIN_RMS = 300
BARGE_IN_WARMUP_CHUNKS = 15


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
        audio_devices: AudioDeviceController | None = None,
        echo_canceller: SpeexEchoCanceller | None = None,
    ) -> None:
        self.list_play_chunk_size = list_play_chunk_size

        self.stop_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.should_listen = should_listen
        self.enabled_event = enabled_event
        self.cancel_scope = cancel_scope
        self.interrupt_queues = interrupt_queues or []
        self.audio_devices = audio_devices or AudioDeviceController(input_device, output_device)
        self.echo_canceller = echo_canceller or SpeexEchoCanceller(
            frame_size=list_play_chunk_size,
            filter_length=list_play_chunk_size * AEC_FILTER_LENGTH_MULTIPLIER,
        )
        self._barge_in_chunks = 0
        self._playback_chunks = 0
        self._barge_in_suppressed_chunks = 0

    def _reset_response_audio_state(self) -> None:
        self.echo_canceller.reset()
        self._barge_in_chunks = 0
        self._playback_chunks = 0
        self._barge_in_suppressed_chunks = 0

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

    def _barge_in_detected(self, pcm: np.ndarray, threshold: int = BARGE_IN_RMS_THRESHOLD) -> bool:
        rms = self._rms(pcm)
        if rms < threshold:
            self._barge_in_chunks = 0
            return False

        self._barge_in_chunks += 1
        return self._barge_in_chunks >= BARGE_IN_CONSECUTIVE_CHUNKS

    def _rms(self, pcm: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(pcm.astype(np.float32)))))

    def _interrupt_response(self, pcm: np.ndarray) -> None:
        if self.cancel_scope:
            self.cancel_scope.cancel()

        drained = self._drain_queue(self.output_queue)
        for queue in self.interrupt_queues:
            drained += self._drain_queue(queue)

        self.should_listen.set()
        self.input_queue.put(pcm.tobytes())
        self._reset_response_audio_state()
        logger.info("Interrupted local audio response; drained %s queued item(s)", drained)

    def _maybe_interrupt_response(self, pcm: np.ndarray) -> bool:
        if not self._enabled() or self.should_listen.is_set():
            self._barge_in_chunks = 0
            return False

        raw_rms = self._rms(pcm)
        clean_pcm = self.echo_canceller.process_capture(pcm)
        clean_rms = self._rms(clean_pcm)
        threshold = (
            BARGE_IN_WARMUP_RMS_THRESHOLD
            if self._playback_chunks < BARGE_IN_WARMUP_CHUNKS
            else BARGE_IN_RMS_THRESHOLD
        )
        if raw_rms >= BARGE_IN_DEBUG_MIN_RMS or clean_rms >= BARGE_IN_DEBUG_MIN_RMS:
            logger.debug(
                "barge candidate raw_rms=%.1f clean_rms=%.1f consecutive=%s threshold=%s aec_available=%s playback_chunks=%s",
                raw_rms,
                clean_rms,
                self._barge_in_chunks,
                threshold,
                self.echo_canceller.available,
                self._playback_chunks,
            )
        if not self._barge_in_detected(clean_pcm, threshold):
            if raw_rms >= threshold and clean_rms < threshold:
                self._barge_in_suppressed_chunks += 1
            return False

        logger.info(
            "barge interrupt raw_rms=%.1f clean_rms=%.1f consecutive=%s threshold=%s aec_available=%s suppressed_chunks=%s",
            raw_rms,
            clean_rms,
            self._barge_in_chunks,
            threshold,
            self.echo_canceller.available,
            self._barge_in_suppressed_chunks,
        )
        self._interrupt_response(clean_pcm)
        return True

    def _play_audio_chunk(self, audio_chunk: np.ndarray, outdata: np.ndarray) -> None:
        self.echo_canceller.feed_reference(audio_chunk)
        self._playback_chunks += 1
        outdata[:] = audio_chunk[:, np.newaxis]

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

            if self.output_queue.empty():
                if self._enabled() and self.should_listen.is_set():
                    self.input_queue.put(pcm.tobytes())
                outdata[:] = dither
            else:
                try:
                    audio_chunk = self.output_queue.get_nowait()
                    if isinstance(audio_chunk, np.ndarray):
                        if self._maybe_interrupt_response(pcm):
                            outdata[:] = dither
                            return
                        self._play_audio_chunk(audio_chunk, outdata)
                    elif audio_chunk == AUDIO_RESPONSE_DONE:
                        logger.info(
                            "local audio response done playback_chunks=%s suppressed_barge_chunks=%s aec_available=%s",
                            self._playback_chunks,
                            self._barge_in_suppressed_chunks,
                            self.echo_canceller.available,
                        )
                        self._reset_response_audio_state()
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
        logger.info("Starting local audio stream")
        try:
            while not self.stop_event.is_set():
                device = self.audio_devices.resolve_stream_device()
                version = self.audio_devices.version()
                try:
                    self._run_stream(callback, device, version)
                except sd.PortAudioError as error:
                    if device is None:
                        raise
                    logger.warning("Could not open selected local audio devices; using system defaults: %s", error)
                    self.audio_devices.mark_fallback_to_default(f"selected device failed to open: {error}")
                    self._run_stream(callback, None, version)
        finally:
            self.echo_canceller.close()
        print("Stopping recording")

    def _run_stream(self, callback: Any, device: tuple[int | None, int | None] | None, version: int) -> None:
        if device is not None:
            logger.info("Using local audio devices: input=%s output=%s", device[0], device[1])
        with sd.Stream(
            samplerate=16000,
            dtype="int16",
            channels=1,
            device=device,
            callback=callback,
            blocksize=self.list_play_chunk_size,
        ):
            next_device_check = time.monotonic() + DEVICE_CHECK_INTERVAL_SECONDS
            while not self.stop_event.is_set() and self.audio_devices.version() == version:
                if time.monotonic() >= next_device_check:
                    next_device_check = time.monotonic() + DEVICE_CHECK_INTERVAL_SECONDS
                    if self.audio_devices.resolve_stream_device() != device:
                        return
                time.sleep(0.001)
