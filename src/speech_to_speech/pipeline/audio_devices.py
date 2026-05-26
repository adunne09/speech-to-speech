from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Literal

import sounddevice as sd

AudioDeviceKind = Literal["input", "output"]


@dataclass(frozen=True)
class AudioDevice:
    id: str
    name: str
    hostapi: str
    index: int


class AudioDeviceController:
    def __init__(self, input_device: str | int | None = None, output_device: str | int | None = None) -> None:
        self._desired_input = input_device
        self._desired_output = output_device
        self._effective_input: int | None = None
        self._effective_output: int | None = None
        self._fallback_reason: str | None = None
        self._version = 0
        self._lock = threading.Lock()

    def set_devices(
        self,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
        *,
        set_input: bool = True,
        set_output: bool = True,
    ) -> int:
        with self._lock:
            if set_input:
                self._desired_input = input_device
            if set_output:
                self._desired_output = output_device
            self._version += 1
            return self._version

    def version(self) -> int:
        with self._lock:
            return self._version

    def desired(self) -> tuple[str | int | None, str | int | None]:
        with self._lock:
            return self._desired_input, self._desired_output

    def resolve_stream_device(self) -> tuple[int | None, int | None] | None:
        with self._lock:
            desired_input = self._desired_input
            desired_output = self._desired_output

        input_index, input_reason = self._resolve("input", desired_input)
        output_index, output_reason = self._resolve("output", desired_output)
        fallback_reason = "; ".join(reason for reason in (input_reason, output_reason) if reason) or None

        with self._lock:
            self._effective_input = input_index
            self._effective_output = output_index
            self._fallback_reason = fallback_reason

        if input_index is None and output_index is None:
            return None
        return input_index, output_index

    def settings(self) -> dict[str, Any]:
        with self._lock:
            desired_input = self._desired_input
            desired_output = self._desired_output
            effective_input = self._effective_input
            effective_output = self._effective_output
            fallback_reason = self._fallback_reason

        return {
            "desired": {
                "input": desired_input,
                "output": desired_output,
            },
            "effective": {
                "input": self._device_id(effective_input),
                "output": self._device_id(effective_output),
            },
            "fallback_reason": fallback_reason,
        }

    def mark_fallback_to_default(self, reason: str) -> None:
        with self._lock:
            self._effective_input = None
            self._effective_output = None
            self._fallback_reason = reason

    def devices(self) -> dict[str, Any]:
        devices = self._devices()
        return {
            "inputs": [device.__dict__ for device in devices if self._has_channels(device.index, "input")],
            "outputs": [device.__dict__ for device in devices if self._has_channels(device.index, "output")],
            "system_default": {"input": None, "output": None},
        }

    def _resolve(self, kind: AudioDeviceKind, desired: str | int | None) -> tuple[int | None, str | None]:
        if desired is None:
            return None, None

        devices = self._devices()
        if isinstance(desired, int):
            for device in devices:
                if device.index == desired and self._has_channels(device.index, kind):
                    return desired, None
            return None, f"{kind} device unavailable: {desired}"

        for device in devices:
            if not self._has_channels(device.index, kind):
                continue
            if desired in (device.id, device.name):
                return device.index, None

        return None, f"{kind} device unavailable: {desired}"

    def _devices(self) -> list[AudioDevice]:
        hostapis = sd.query_hostapis()
        devices = sd.query_devices()
        result = []
        for index, device in enumerate(devices):
            hostapi = hostapis[device["hostapi"]]["name"]
            name = device["name"]
            result.append(AudioDevice(id=f"{hostapi}:{name}", name=name, hostapi=hostapi, index=index))
        return result

    def _has_channels(self, index: int, kind: AudioDeviceKind) -> bool:
        device = sd.query_devices(index)
        channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
        return int(device[channel_key]) > 0

    def _device_id(self, index: int | None) -> str | None:
        if index is None:
            return None
        for device in self._devices():
            if device.index == index:
                return device.id
        return None
