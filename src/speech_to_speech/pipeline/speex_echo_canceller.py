from __future__ import annotations

import ctypes
import ctypes.util
import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SPEEX_ECHO_SET_SAMPLING_RATE = 24
HOMEBREW_SPEEXDSP_PATHS = [
    Path("/opt/homebrew/opt/speexdsp/lib/libspeexdsp.dylib"),
    Path("/usr/local/opt/speexdsp/lib/libspeexdsp.dylib"),
]


def speexdsp_library_candidates() -> list[str]:
    candidates = [ctypes.util.find_library("speexdsp")]
    candidates.extend(str(path) for path in HOMEBREW_SPEEXDSP_PATHS if path.exists())
    return [candidate for candidate in dict.fromkeys(candidates) if candidate is not None]


class SpeexEchoCanceller:
    def __init__(self, frame_size: int, filter_length: int, sample_rate: int = 16000) -> None:
        self.frame_size = frame_size
        self.filter_length = filter_length
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._reference_started = False

        self._lib = self._load_library()
        self._state = self._create_state() if self._lib is not None else None

        self._capture_buffer = (ctypes.c_int16 * frame_size)()
        self._playback_buffer = (ctypes.c_int16 * frame_size)()
        self._output_buffer = (ctypes.c_int16 * frame_size)()
        self._frame_bytes = frame_size * 2

    @property
    def available(self) -> bool:
        return self._state is not None

    def process_capture(self, pcm: np.ndarray) -> np.ndarray:
        if self._state is None or not self._reference_started:
            return pcm

        assert self._lib is not None
        frame = self._as_frame(pcm)
        with self._lock:
            ctypes.memmove(self._capture_buffer, frame.ctypes.data, self._frame_bytes)
            self._lib.speex_echo_capture(self._state, self._capture_buffer, self._output_buffer)
            return np.frombuffer(bytes(self._output_buffer), dtype=np.int16).reshape(pcm.shape).copy()

    def feed_reference(self, pcm: np.ndarray) -> None:
        if self._state is None:
            return

        assert self._lib is not None
        frame = self._as_frame(pcm)
        with self._lock:
            ctypes.memmove(self._playback_buffer, frame.ctypes.data, self._frame_bytes)
            self._lib.speex_echo_playback(self._state, self._playback_buffer)
            self._reference_started = True

    def reset(self) -> None:
        self._reference_started = False
        if self._state is None:
            return
        assert self._lib is not None
        with self._lock:
            self._lib.speex_echo_state_reset(self._state)

    def close(self) -> None:
        if self._state is None:
            return
        assert self._lib is not None
        with self._lock:
            self._lib.speex_echo_state_destroy(self._state)
            self._state = None

    def _as_frame(self, pcm: np.ndarray) -> np.ndarray:
        frame = np.ascontiguousarray(pcm.reshape(-1), dtype=np.int16)
        if len(frame) == self.frame_size:
            return frame
        if len(frame) > self.frame_size:
            logger.debug("AEC frame too large (%s > %s); truncating", len(frame), self.frame_size)
            return frame[: self.frame_size]
        logger.debug("AEC frame too small (%s < %s); padding", len(frame), self.frame_size)
        return np.pad(frame, (0, self.frame_size - len(frame))).astype(np.int16)

    def _load_library(self) -> ctypes.CDLL | None:
        lib = None
        errors = []
        for path in speexdsp_library_candidates():
            try:
                lib = ctypes.CDLL(path)
                break
            except OSError as exc:
                errors.append(f"{path}: {exc}")

        if lib is None:
            if errors:
                logger.warning("SpeexDSP echo cancellation unavailable; failed to load libspeexdsp: %s", "; ".join(errors))
            else:
                logger.warning(
                    "SpeexDSP echo cancellation unavailable; install libspeexdsp to enable speaker-safe barge-in"
                )
            return None

        lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.speex_echo_state_init.restype = ctypes.c_void_p
        lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
        lib.speex_echo_state_destroy.restype = None
        lib.speex_echo_state_reset.argtypes = [ctypes.c_void_p]
        lib.speex_echo_state_reset.restype = None
        lib.speex_echo_playback.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.speex_echo_playback.restype = None
        lib.speex_echo_capture.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.speex_echo_capture.restype = None
        lib.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.speex_echo_ctl.restype = ctypes.c_int
        return lib

    def _create_state(self) -> ctypes.c_void_p | None:
        assert self._lib is not None
        state = self._lib.speex_echo_state_init(self.frame_size, self.filter_length)
        if not state:
            logger.warning("SpeexDSP echo cancellation unavailable; speex_echo_state_init returned NULL")
            return None

        sample_rate = ctypes.c_int(self.sample_rate)
        self._lib.speex_echo_ctl(state, SPEEX_ECHO_SET_SAMPLING_RATE, ctypes.byref(sample_rate))
        logger.info(
            "SpeexDSP echo cancellation enabled frame_size=%s filter_length=%s sample_rate=%s",
            self.frame_size,
            self.filter_length,
            self.sample_rate,
        )
        return ctypes.c_void_p(state)
