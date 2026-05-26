from queue import Queue
from threading import Event

import numpy as np
import torch

from speech_to_speech.pipeline.events import SpeechStoppedEvent
from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.VAD.smart_turn_detector import SmartTurnResult
from speech_to_speech.VAD.vad_handler import VADHandler


class _FakeSmartTurnDetector:
    def __init__(self, results: list[SmartTurnResult]) -> None:
        self.results = iter(results)
        self.audio_seen: list[np.ndarray] = []

    def predict(self, audio: np.ndarray) -> SmartTurnResult:
        self.audio_seen.append(audio.copy())
        return next(self.results)


def _handler(detector: _FakeSmartTurnDetector) -> VADHandler:
    handler = object.__new__(VADHandler)
    handler.sample_rate = 16000
    handler.min_speech_ms = 10
    handler.max_speech_ms = float("inf")
    handler.text_output_queue = Queue()
    handler.should_listen = Event()
    handler.should_listen.set()
    handler.audio_enhancement = False
    handler.smart_turn_detector = detector
    handler._smart_turn_pending = []
    handler._speech_started_emitted = True
    handler._log_speech_ends = 0
    handler._total_samples = 3200
    return handler


def test_smart_turn_incomplete_keeps_listening_and_suppresses_final_audio() -> None:
    detector = _FakeSmartTurnDetector([SmartTurnResult(complete=False, probability=0.2)])
    handler = _handler(detector)

    outputs = list(handler._process_normal([torch.ones(1600)]))

    assert outputs == []
    assert handler.should_listen.is_set()
    assert handler._speech_started_emitted is True
    assert len(handler._smart_turn_pending) == 1
    assert handler.text_output_queue.empty()


def test_smart_turn_complete_after_incomplete_yields_combined_audio() -> None:
    detector = _FakeSmartTurnDetector(
        [
            SmartTurnResult(complete=False, probability=0.2),
            SmartTurnResult(complete=True, probability=0.8),
        ]
    )
    handler = _handler(detector)

    assert list(handler._process_normal([torch.ones(1600)])) == []
    outputs = list(handler._process_normal([torch.ones(1600) * 2]))

    assert len(outputs) == 1
    assert isinstance(outputs[0], VADAudio)
    assert outputs[0].audio.shape == (3200,)
    assert np.all(outputs[0].audio[:1600] == 1)
    assert np.all(outputs[0].audio[1600:] == 2)
    assert not handler.should_listen.is_set()
    assert handler._smart_turn_pending == []
    assert isinstance(handler.text_output_queue.get_nowait(), SpeechStoppedEvent)
    assert detector.audio_seen[1].shape == (3200,)
