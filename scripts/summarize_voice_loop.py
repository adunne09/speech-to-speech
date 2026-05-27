from __future__ import annotations

import argparse
import re
from pathlib import Path

PATTERNS = (
    "SpeexDSP echo cancellation",
    "Using local audio devices",
    "Pipeline enabled",
    "Pipeline disabled",
    "Speech started",
    "Speech ended",
    "Transcription completed",
    "USER:",
    "ASSISTANT:",
    "barge candidate",
    "barge interrupt",
    "Interrupted local audio response",
    "local audio response done",
    "Response complete",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize voice loop diagnostics from speech-to-speech.log")
    parser.add_argument("log", type=Path, help="Path to speech-to-speech.log")
    args = parser.parse_args()

    for line in args.log.read_text(errors="replace").splitlines():
        if any(pattern in line for pattern in PATTERNS):
            print(_compact(line))


def _compact(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


if __name__ == "__main__":
    main()
