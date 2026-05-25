from dataclasses import dataclass, field

from speech_to_speech.arguments_classes.language_model_base_arguments import LanguageModelBaseArguments


@dataclass
class OpencodeLanguageModelHandlerArguments(LanguageModelBaseArguments):
    model_name: str = field(
        default="openai/gpt-5.5",
        metadata={"help": "The opencode provider/model to use. Default is 'openai/gpt-5.5'."},
    )
    opencode_base_url: str = field(
        default="http://localhost:4096",
        metadata={"help": "Base URL for a running opencode server. Default is http://localhost:4096."},
    )
    opencode_session_id: str | None = field(
        default=None,
        metadata={"help": "Existing opencode session ID to use. If omitted, a new session is created."},
    )
    opencode_directory: str | None = field(
        default=None,
        metadata={"help": "Project directory routed to opencode. Defaults to opencode server's current project."},
    )
    opencode_provider_id: str = field(
        default="openai",
        metadata={"help": "Provider ID used when model_name has no provider prefix. Default is 'openai'."},
    )
    opencode_request_timeout_s: float = field(
        default=120.0,
        metadata={"help": "Timeout in seconds for opencode HTTP requests. Default is 120."},
    )
    opencode_control_host: str = field(
        default="127.0.0.1",
        metadata={"help": "Host for the optional opencode session control API. Default is 127.0.0.1."},
    )
    opencode_control_port: int | None = field(
        default=None,
        metadata={"help": "Enable an HTTP API on this port to update the opencode session ID at runtime."},
    )
