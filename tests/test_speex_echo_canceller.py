from pathlib import Path

from speech_to_speech.pipeline import speex_echo_canceller


def test_library_candidates_include_homebrew_fallback_when_find_library_misses(monkeypatch, tmp_path) -> None:
    dylib = tmp_path / "libspeexdsp.dylib"
    dylib.touch()

    monkeypatch.setattr(speex_echo_canceller.ctypes.util, "find_library", lambda _name: None)
    monkeypatch.setattr(speex_echo_canceller, "HOMEBREW_SPEEXDSP_PATHS", [dylib])

    assert speex_echo_canceller.speexdsp_library_candidates() == [str(dylib)]


def test_library_candidates_preserve_find_library_before_homebrew_fallback(monkeypatch, tmp_path) -> None:
    dylib = tmp_path / "libspeexdsp.dylib"
    dylib.touch()

    monkeypatch.setattr(speex_echo_canceller.ctypes.util, "find_library", lambda _name: "libspeexdsp.dylib")
    monkeypatch.setattr(speex_echo_canceller, "HOMEBREW_SPEEXDSP_PATHS", [dylib])

    assert speex_echo_canceller.speexdsp_library_candidates() == ["libspeexdsp.dylib", str(dylib)]


def test_library_candidates_skip_missing_homebrew_fallback(monkeypatch) -> None:
    monkeypatch.setattr(speex_echo_canceller.ctypes.util, "find_library", lambda _name: None)
    monkeypatch.setattr(speex_echo_canceller, "HOMEBREW_SPEEXDSP_PATHS", [Path("/missing/libspeexdsp.dylib")])

    assert speex_echo_canceller.speexdsp_library_candidates() == []
