from dataclasses import dataclass, field


@dataclass
class VADHandlerArguments:
    thresh: float = field(
        default=0.6,
        metadata={
            "help": "The threshold value for voice activity detection (VAD). Values typically range from 0 to 1, with higher values requiring higher confidence in speech detection."
        },
    )
    sample_rate: int = field(
        default=16000,
        metadata={
            "help": "The sample rate of the audio in Hertz. Default is 16000 Hz, which is a common setting for voice audio."
        },
    )
    min_silence_ms: int = field(
        default=300,
        metadata={
            "help": "Minimum length of silence intervals to be used for segmenting speech. Measured in milliseconds. Default is 250 ms."
        },
    )
    min_speech_ms: int = field(
        default=500,
        metadata={
            "help": "Minimum length of speech segments to be considered valid speech. Measured in milliseconds. Default is 500 ms."
        },
    )
    max_speech_ms: float = field(
        default=float("inf"),
        metadata={
            "help": "Maximum length of continuous speech before forcing a split. Default is infinite, allowing for uninterrupted speech segments."
        },
    )
    speech_pad_ms: int = field(
        default=500,
        metadata={
            "help": "Amount of audio retained before VAD triggers and prepended to detected speech segments. Once speech is detected, audio continues to be kept until VAD declares the segment done. Measured in milliseconds. Default is 500 ms."
        },
    )
    audio_enhancement: bool = field(
        default=False,
        metadata={
            "help": "improves sound quality by applying techniques like noise reduction, equalization, and echo cancellation. Default is False."
        },
    )
    enable_realtime_transcription: bool = field(
        default=False,
        metadata={"help": "Enable progressive audio release for live transcription during speech. Default is False."},
    )
    realtime_processing_pause: float = field(
        default=0.2,
        metadata={
            "help": "Interval (in seconds) for releasing progressive audio chunks during speech. Default is 0.2s."
        },
    )
    smart_turn: bool = field(
        default=False,
        metadata={
            "help": "Enable Pipecat Smart Turn endpoint detection after Silero detects candidate silence. Default is False."
        },
    )
    smart_turn_threshold: float = field(
        default=0.5,
        metadata={"help": "Completion probability threshold for Smart Turn. Default is 0.5."},
    )
    smart_turn_model_repo: str = field(
        default="pipecat-ai/smart-turn-v3",
        metadata={"help": "Hugging Face repo containing the Smart Turn ONNX model."},
    )
    smart_turn_model_filename: str = field(
        default="smart-turn-v3.2-cpu.onnx",
        metadata={"help": "ONNX model filename inside the Smart Turn Hugging Face repo."},
    )
