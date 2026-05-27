from queue import Queue
from threading import Event

import numpy as np

from speech_to_speech.connections.local_audio_streamer import (
    BARGE_IN_CONSECUTIVE_CHUNKS,
    BARGE_IN_WARMUP_CHUNKS,
    LocalAudioStreamer,
)
from speech_to_speech.pipeline.audio_devices import AudioDeviceController
from speech_to_speech.pipeline.cancel_scope import CancelScope


class FakeEchoCanceller:
    def __init__(self, clean_pcm: np.ndarray | None = None) -> None:
        self.clean_pcm = clean_pcm
        self.references: list[np.ndarray] = []
        self.reset_called = False
        self.available = True

    def process_capture(self, pcm: np.ndarray) -> np.ndarray:
        return self.clean_pcm if self.clean_pcm is not None else pcm

    def feed_reference(self, pcm: np.ndarray) -> None:
        self.references.append(pcm.copy())

    def reset(self) -> None:
        self.reset_called = True

    def close(self) -> None:
        pass


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
    echo_canceller = FakeEchoCanceller()

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
        echo_canceller=echo_canceller,
    )
    streamer._barge_in_chunks = 2
    streamer._playback_chunks = 20
    streamer._barge_in_suppressed_chunks = 4

    pcm = np.full((512, 1), 2000, dtype=np.int16)
    streamer._interrupt_response(pcm)

    assert cancel_scope.discarding
    assert should_listen.is_set()
    assert output_queue.empty()
    assert tts_queue.empty()
    assert lm_queue.empty()
    assert input_queue.get_nowait() == pcm.tobytes()
    assert echo_canceller.reset_called
    assert streamer._barge_in_chunks == 0
    assert streamer._playback_chunks == 0
    assert streamer._barge_in_suppressed_chunks == 0


def test_maybe_interrupt_uses_echo_cancelled_audio_to_suppress_playback_echo() -> None:
    input_queue = Queue()
    output_queue = Queue()
    should_listen = Event()
    quiet_after_aec = np.zeros((512, 1), dtype=np.int16)
    streamer = LocalAudioStreamer(
        input_queue=input_queue,
        output_queue=output_queue,
        should_listen=should_listen,
        echo_canceller=FakeEchoCanceller(quiet_after_aec),
    )

    raw_echo = np.full((512, 1), 2000, dtype=np.int16)

    for _ in range(BARGE_IN_CONSECUTIVE_CHUNKS + 1):
        assert not streamer._maybe_interrupt_response(raw_echo)

    assert input_queue.empty()


def test_maybe_interrupt_cancels_with_echo_cancelled_user_audio() -> None:
    input_queue = Queue()
    output_queue = Queue()
    should_listen = Event()
    cancel_scope = CancelScope()
    cleaned_user_audio = np.full((512, 1), 2000, dtype=np.int16)
    streamer = LocalAudioStreamer(
        input_queue=input_queue,
        output_queue=output_queue,
        should_listen=should_listen,
        cancel_scope=cancel_scope,
        echo_canceller=FakeEchoCanceller(cleaned_user_audio),
    )
    streamer._playback_chunks = BARGE_IN_WARMUP_CHUNKS

    raw_mic = np.full((512, 1), 2000, dtype=np.int16)
    for _ in range(BARGE_IN_CONSECUTIVE_CHUNKS - 1):
        assert not streamer._maybe_interrupt_response(raw_mic)

    assert streamer._maybe_interrupt_response(raw_mic)
    assert cancel_scope.discarding
    assert should_listen.is_set()
    assert input_queue.get_nowait() == cleaned_user_audio.tobytes()


def test_maybe_interrupt_uses_higher_threshold_during_aec_warmup() -> None:
    streamer = LocalAudioStreamer(
        input_queue=Queue(),
        output_queue=Queue(),
        should_listen=Event(),
        echo_canceller=FakeEchoCanceller(),
    )

    early_echo = np.full((512, 1), 1500, dtype=np.int16)
    for _ in range(BARGE_IN_CONSECUTIVE_CHUNKS + 1):
        assert not streamer._maybe_interrupt_response(early_echo)


def test_maybe_interrupt_allows_strong_user_audio_during_aec_warmup() -> None:
    input_queue = Queue()
    should_listen = Event()
    streamer = LocalAudioStreamer(
        input_queue=input_queue,
        output_queue=Queue(),
        should_listen=should_listen,
        echo_canceller=FakeEchoCanceller(),
    )

    strong_user_audio = np.full((512, 1), 5000, dtype=np.int16)
    for _ in range(BARGE_IN_CONSECUTIVE_CHUNKS - 1):
        assert not streamer._maybe_interrupt_response(strong_user_audio)

    assert streamer._maybe_interrupt_response(strong_user_audio)
    assert should_listen.is_set()
    assert input_queue.get_nowait() == strong_user_audio.tobytes()


def test_maybe_interrupt_uses_normal_threshold_after_aec_warmup() -> None:
    input_queue = Queue()
    should_listen = Event()
    streamer = LocalAudioStreamer(
        input_queue=input_queue,
        output_queue=Queue(),
        should_listen=should_listen,
        echo_canceller=FakeEchoCanceller(),
    )
    streamer._playback_chunks = BARGE_IN_WARMUP_CHUNKS

    user_audio = np.full((512, 1), 2000, dtype=np.int16)
    for _ in range(BARGE_IN_CONSECUTIVE_CHUNKS - 1):
        assert not streamer._maybe_interrupt_response(user_audio)

    assert streamer._maybe_interrupt_response(user_audio)
    assert should_listen.is_set()
    assert input_queue.get_nowait() == user_audio.tobytes()


def test_maybe_interrupt_ignores_moderate_echo_after_aec_warmup() -> None:
    streamer = LocalAudioStreamer(
        input_queue=Queue(),
        output_queue=Queue(),
        should_listen=Event(),
        echo_canceller=FakeEchoCanceller(),
    )
    streamer._playback_chunks = BARGE_IN_WARMUP_CHUNKS

    moderate_echo = np.full((512, 1), 1600, dtype=np.int16)
    for _ in range(BARGE_IN_CONSECUTIVE_CHUNKS + 1):
        assert not streamer._maybe_interrupt_response(moderate_echo)


def test_play_audio_chunk_feeds_echo_reference_before_output() -> None:
    echo_canceller = FakeEchoCanceller()
    streamer = LocalAudioStreamer(
        input_queue=Queue(),
        output_queue=Queue(),
        should_listen=Event(),
        echo_canceller=echo_canceller,
    )
    audio_chunk = np.arange(512, dtype=np.int16)
    outdata = np.zeros((512, 1), dtype=np.int16)

    streamer._play_audio_chunk(audio_chunk, outdata)

    assert len(echo_canceller.references) == 1
    np.testing.assert_array_equal(echo_canceller.references[0], audio_chunk)
    np.testing.assert_array_equal(outdata, audio_chunk[:, np.newaxis])


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
