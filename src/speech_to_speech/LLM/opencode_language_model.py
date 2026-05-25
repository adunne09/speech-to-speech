from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

import httpx
from nltk import sent_tokenize
from openai.types.realtime.conversation_item import RealtimeConversationItemUserMessage

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import Chat, make_assistant_message
from speech_to_speech.LLM.utils import remove_unspeechable, resolve_auto_language
from speech_to_speech.LLM.voice_prompt import build_voice_system_prompt
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import EndOfResponse, LLMResponseChunk

logger = logging.getLogger(__name__)


class OpencodeSessionState:
    def __init__(self, session_id: str | None = None) -> None:
        self._lock = threading.Lock()
        self._session_id = session_id

    def get(self) -> str | None:
        with self._lock:
            return self._session_id

    def set(self, session_id: str) -> None:
        with self._lock:
            self._session_id = session_id

    def clear(self) -> None:
        with self._lock:
            self._session_id = None


class OpencodeModelHandler(BaseHandler[LLMIn, LLMOut]):
    def setup(
        self,
        model_name: str = "openai/gpt-5.5",
        base_url: str = "http://localhost:4096",
        session_id: Optional[str] = None,
        directory: Optional[str] = None,
        provider_id: str = "openai",
        request_timeout_s: float = 120.0,
        stream_batch_sentences: int = 3,
        control_host: str = "127.0.0.1",
        control_port: int | None = None,
        enable_lang_prompt: bool = False,
        cancel_scope: CancelScope | None = None,
        **_kwargs: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_state = OpencodeSessionState(session_id)
        self.directory = directory
        self.provider_id, self.model_id = self._parse_model_name(model_name, provider_id)
        self.stream_batch_sentences = max(1, stream_batch_sentences)
        self.enable_lang_prompt = enable_lang_prompt
        self.cancel_scope = cancel_scope
        self.client = httpx.Client(timeout=httpx.Timeout(request_timeout_s, connect=min(10.0, request_timeout_s)))
        self.control_server: ThreadingHTTPServer | None = None
        self.control_thread: threading.Thread | None = None
        self._start_control_server(control_host, control_port)
        self.warmup()

    def _parse_model_name(self, model_name: str, fallback_provider_id: str) -> tuple[str, str]:
        if "/" not in model_name:
            return fallback_provider_id, model_name
        provider_id, model_id = model_name.split("/", 1)
        return provider_id, model_id

    def _request_kwargs(self) -> dict[str, Any]:
        if not self.directory:
            return {}
        return {"params": {"directory": self.directory}}

    def _ensure_session(self) -> str:
        session_id = self.session_state.get()
        if session_id:
            logger.info("Using existing opencode session: %s", session_id)
            return session_id

        response = self.client.post(
            f"{self.base_url}/session",
            json={},
            **self._request_kwargs(),
        )
        response.raise_for_status()
        session_id = response.json()["id"]
        self.session_state.set(session_id)
        logger.info("Created opencode session: %s", session_id)
        return session_id

    def set_session_id(self, session_id: str) -> None:
        response = self.client.get(f"{self.base_url}/session/{session_id}", **self._request_kwargs())
        response.raise_for_status()
        self.session_state.set(session_id)
        logger.info("Switched opencode session: %s", session_id)

    def _start_control_server(self, control_host: str, control_port: int | None) -> None:
        if control_port is None:
            return
        handler = self._make_control_handler()
        self.control_server = ThreadingHTTPServer((control_host, control_port), handler)
        self.control_thread = threading.Thread(
            target=self.control_server.serve_forever,
            name="opencode-session-control",
            daemon=True,
        )
        self.control_thread.start()
        logger.info("Opencode session control API listening on http://%s:%s", control_host, control_port)

    def _make_control_handler(self) -> type[BaseHTTPRequestHandler]:
        model_handler = self

        class ControlHandler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                logger.debug("opencode session control: " + _format, *_args)

            def _write_json(self, status: int, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                if self.path != "/opencode/session":
                    self._write_json(404, {"error": "not_found"})
                    return
                self._write_json(200, {"session_id": model_handler.session_state.get()})

            def do_PUT(self) -> None:
                if self.path != "/opencode/session":
                    self._write_json(404, {"error": "not_found"})
                    return
                try:
                    length = int(self.headers.get("content-length", "0"))
                    data = json.loads(self.rfile.read(length) or b"{}")
                    session_id = data.get("session_id")
                    if session_id is None:
                        model_handler.session_state.clear()
                        self._write_json(200, {"session_id": None})
                        return
                    if not isinstance(session_id, str) or not session_id.strip():
                        self._write_json(400, {"error": "session_id must be a non-empty string or null"})
                        return
                    model_handler.set_session_id(session_id.strip())
                except json.JSONDecodeError:
                    self._write_json(400, {"error": "invalid_json"})
                    return
                except httpx.HTTPStatusError as exc:
                    self._write_json(exc.response.status_code, {"error": "opencode rejected session_id"})
                    return
                except httpx.HTTPError as exc:
                    self._write_json(502, {"error": str(exc)})
                    return
                self._write_json(200, {"session_id": model_handler.session_state.get()})

        return ControlHandler

    def warmup(self) -> None:
        logger.info("Warming up %s", self.__class__.__name__)
        start = time.time()
        session_id = self.session_state.get()
        if session_id is None:
            logger.info(
                "%s warmed up without an opencode session! time: %.3f s",
                self.__class__.__name__,
                time.time() - start,
            )
            return
        response = self.client.get(f"{self.base_url}/session/{session_id}", **self._request_kwargs())
        response.raise_for_status()
        logger.info("%s warmed up! time: %.3f s", self.__class__.__name__, time.time() - start)

    def _latest_user_text(self, chat: Chat) -> str:
        with chat._lock:
            for item in reversed(chat.buffer):
                if isinstance(item, RealtimeConversationItemUserMessage):
                    return "\n".join(
                        part.text for part in item.content if part.type == "input_text" and part.text
                    ).strip()
        return ""

    def _assistant_text(self, payload: dict[str, Any]) -> str:
        text_parts: list[str] = []
        for part in payload.get("parts", []):
            if part.get("type") == "text" and part.get("text"):
                text_parts.append(part["text"])
        return "\n".join(text_parts).strip()

    def _prompt_opencode(self, text: str, system: str | None) -> str:
        session_id = self._ensure_session()
        body: dict[str, Any] = {
            "model": {"providerID": self.provider_id, "modelID": self.model_id},
            "parts": [{"type": "text", "text": text}],
        }
        if system:
            body["system"] = build_voice_system_prompt(system)
        response = self.client.post(
            f"{self.base_url}/session/{session_id}/message",
            json=body,
            **self._request_kwargs(),
        )
        response.raise_for_status()
        return self._assistant_text(response.json())

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        runtime_config = request.runtime_config
        response = request.response
        original_chat = runtime_config.chat
        language_code = request.language_code
        instructions = (
            response.instructions if response and response.instructions else runtime_config.session.instructions
        ) or ""
        language_code, lang_name = resolve_auto_language(language_code)
        user_text = self._latest_user_text(original_chat)
        if lang_name and self.enable_lang_prompt:
            user_text = f"{user_text}\n\nPlease reply to my message in {lang_name}."

        gen = self.cancel_scope.generation if self.cancel_scope else None
        if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
            yield EndOfResponse()
            return

        clean_text = remove_unspeechable(self._prompt_opencode(user_text, instructions))
        if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
            logger.info("opencode generation cancelled after response returned")
            yield EndOfResponse()
            return
        original_chat.add_item(make_assistant_message(clean_text))
        printable_text = clean_text
        sentence_batch: list[str] = []
        sentences = sent_tokenize(printable_text)
        for sentence in sentences:
            sentence_batch.append(sentence)
            if len(sentence_batch) >= self.stream_batch_sentences:
                yield LLMResponseChunk(
                    text=" ".join(sentence_batch),
                    language_code=language_code,
                    runtime_config=runtime_config,
                    response=response,
                )
                sentence_batch = []
        if sentence_batch or not sentences:
            yield LLMResponseChunk(
                text=" ".join(sentence_batch) if sentence_batch else clean_text,
                language_code=language_code,
                runtime_config=runtime_config,
                response=response,
            )
        original_chat.strip_images()
        yield EndOfResponse()

    def on_session_end(self) -> None:
        logger.debug("opencode language model session ended")

    def cleanup(self) -> None:
        if self.control_server is not None:
            self.control_server.shutdown()
            self.control_server.server_close()
            self.control_server = None
        self.client.close()

    @property
    def timing_log_level(self) -> int:
        return logging.INFO

    def should_log_timing(self, output: LLMOut) -> bool:
        return isinstance(output, LLMResponseChunk) and self.last_time > self.min_time_to_debug
