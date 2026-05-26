from queue import Queue
from threading import Event

import numpy as np

from speech_to_speech.connections.local_audio_streamer import (
    BARGE_IN_CONSECUTIVE_CHUNKS,
    LocalAudioStreamer,
)
from speech_to_speech.pipeline.audio_devices import AudioDeviceController
from speech_to_speech.pipeline.cancel_scope import CancelScope


def test_barge_in_detection_requires_consecutive_loud_chunks() -> None:
    streamer = LocalAudioStreamer(
        input_queue=Queue(),
        output_queue=Queue(),
        should_listen=Event(),
    )

    quiet = np.zeros((512, 1), dtype=np.int16)
    loud = np.full((512, 1), 2000, dtype=np.int16)

    assert not streamer._barge_in_detected(loud)
    assert not streamer._barge_in_detected(quiet)

    for _ in range(BARGE_IN_CONSECUTIVE_CHUNKS - 1):
        assert not streamer._barge_in_detected(loud)

    assert streamer._barge_in_detected(loud)


def test_interrupt_response_cancels_drains_and_forwards_audio() -> None:
    input_queue = Queue()
    output_queue = Queue()
    tts_queue = Queue()
    lm_queue = Queue()
    should_listen = Event()
    cancel_scope = CancelScope()

    output_queue.put(np.ones(512, dtype=np.int16))
    output_queue.put(np.ones(512, dtype=np.int16))
    tts_queue.put("stale text")
    lm_queue.put("stale response")

    streamer = LocalAudioStreamer(
        input_queue=input_queue,
        output_queue=output_queue,
        should_listen=should_listen,
        cancel_scope=cancel_scope,
        interrupt_queues=[tts_queue, lm_queue],
    )

    pcm = np.full((512, 1), 2000, dtype=np.int16)
    streamer._interrupt_response(pcm)

    assert cancel_scope.discarding
    assert should_listen.is_set()
    assert output_queue.empty()
    assert tts_queue.empty()
    assert lm_queue.empty()
    assert input_queue.get_nowait() == pcm.tobytes()


def test_stream_exits_when_selected_device_becomes_unavailable(monkeypatch) -> None:
    controller = AudioDeviceController()
    streamer = LocalAudioStreamer(
        input_queue=Queue(),
        output_queue=Queue(),
        should_listen=Event(),
        audio_devices=controller,
    )
    calls = 0

    def resolve_stream_device():
        nonlocal calls
        calls += 1
        return (0, 1) if calls == 1 else None

    class FakeStream:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

    monkeypatch.setattr(controller, "resolve_stream_device", resolve_stream_device)
    monkeypatch.setattr("speech_to_speech.connections.local_audio_streamer.DEVICE_CHECK_INTERVAL_SECONDS", 0)
    monkeypatch.setattr("speech_to_speech.connections.local_audio_streamer.sd.Stream", FakeStream)

    streamer._run_stream(lambda *_args: None, (0, 1), controller.version())

    assert calls == 2
