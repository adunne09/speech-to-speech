from speech_to_speech.pipeline import audio_devices
from speech_to_speech.pipeline.audio_devices import AudioDeviceController

DEVICES = [
    {"name": "Mic A", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0},
    {"name": "Speaker A", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2},
]


def install_fake_devices(monkeypatch) -> None:
    monkeypatch.setattr(audio_devices.sd, "query_hostapis", lambda: [{"name": "Core Audio"}])

    def query_devices(index=None):
        if index is None:
            return DEVICES
        return DEVICES[index]

    monkeypatch.setattr(audio_devices.sd, "query_devices", query_devices)


def test_devices_groups_inputs_and_outputs(monkeypatch) -> None:
    install_fake_devices(monkeypatch)

    devices = AudioDeviceController().devices()

    assert devices["inputs"] == [{"id": "Core Audio:Mic A", "name": "Mic A", "hostapi": "Core Audio", "index": 0}]
    assert devices["outputs"] == [
        {"id": "Core Audio:Speaker A", "name": "Speaker A", "hostapi": "Core Audio", "index": 1}
    ]
    assert devices["system_default"] == {"input": None, "output": None}


def test_resolves_selected_devices_by_stable_id(monkeypatch) -> None:
    install_fake_devices(monkeypatch)

    controller = AudioDeviceController("Core Audio:Mic A", "Core Audio:Speaker A")

    assert controller.resolve_stream_device() == (0, 1)
    assert controller.settings()["fallback_reason"] is None


def test_unavailable_selection_falls_back_to_system_default(monkeypatch) -> None:
    install_fake_devices(monkeypatch)

    controller = AudioDeviceController("Missing Mic", "Core Audio:Speaker A")

    assert controller.resolve_stream_device() == (None, 1)
    assert controller.settings()["desired"] == {"input": "Missing Mic", "output": "Core Audio:Speaker A"}
    assert controller.settings()["effective"] == {"input": None, "output": "Core Audio:Speaker A"}
    assert controller.settings()["fallback_reason"] == "input device unavailable: Missing Mic"


def test_setting_devices_preserves_omitted_side(monkeypatch) -> None:
    install_fake_devices(monkeypatch)
    controller = AudioDeviceController("Core Audio:Mic A", "Core Audio:Speaker A")

    controller.set_devices(None, set_input=True, set_output=False)

    assert controller.desired() == (None, "Core Audio:Speaker A")
