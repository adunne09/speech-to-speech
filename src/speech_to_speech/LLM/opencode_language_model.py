from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from collections import defaultdict
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

_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_id_lock = threading.Lock()
_last_id_timestamp = 0
_id_counter = 0


class _FlushText:
    pass


_FLUSH_TEXT = _FlushText()


def _preview(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?"))


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

    def _message_id(self) -> str:
        global _id_counter, _last_id_timestamp
        timestamp = int(time.time() * 1000)
        with _id_lock:
            if timestamp != _last_id_timestamp:
                _last_id_timestamp = timestamp
                _id_counter = 0
            _id_counter += 1
            encoded_time = (timestamp * 0x1000 + _id_counter) & 0xFFFFFFFFFFFF
        random = "".join(secrets.choice(_BASE62) for _ in range(14))
        return f"msg_{encoded_time:012x}{random}"

    def _sse_events(self, response: httpx.Response) -> Iterator[dict[str, Any]]:
        data_lines: list[str] = []

        for line in response.iter_lines():
            if line == "":
                if not data_lines:
                    continue
                raw_data = "\n".join(data_lines)
                data_lines = []
                try:
                    event = json.loads(raw_data)
                except json.JSONDecodeError:
                    logger.debug("Ignoring non-JSON opencode SSE payload: %s", raw_data)
                    continue
                if isinstance(event, dict):
                    yield event
                continue

            if line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").lstrip())

    def _post_prompt_async(self, session_id: str, message_id: str, text: str, system: str | None) -> None:
        body: dict[str, Any] = {
            "messageID": message_id,
            "model": {"providerID": self.provider_id, "modelID": self.model_id},
            "parts": [{"type": "text", "text": text}],
        }
        if system:
            body["system"] = build_voice_system_prompt(system)

        response = self.client.post(
            f"{self.base_url}/session/{session_id}/prompt_async",
            json=body,
            **self._request_kwargs(),
        )
        response.raise_for_status()
        logger.info(
            "opencode prompt_async accepted session=%s user_message=%s chars=%d",
            session_id,
            message_id,
            len(text),
        )

    def _stream_opencode_deltas(self, text: str, system: str | None) -> Iterator[str | _FlushText]:
        session_id = self._ensure_session()
        message_id = self._message_id()
        assistant_message_ids: set[str] = set()
        speakable_part_ids: set[str] = set()
        unspeakable_part_ids: set[str] = set()
        pending_deltas: dict[str, list[str]] = defaultdict(list)

        logger.info("opencode opening SSE stream session=%s user_message=%s", session_id, message_id)
        with self.client.stream(
            "GET",
            f"{self.base_url}/event",
            headers={"Accept": "text/event-stream"},
            **self._request_kwargs(),
        ) as event_response:
            event_response.raise_for_status()
            connected = False
            events = self._sse_events(event_response)
            for event in events:
                logger.debug("opencode SSE pre-prompt event type=%s", event.get("type"))
                if event.get("type") == "server.connected":
                    connected = True
                    break

            if not connected:
                raise RuntimeError("opencode event stream closed before server.connected")

            self._post_prompt_async(session_id, message_id, text, system)

            for event in events:
                event_type = event.get("type")
                properties = event.get("properties") or {}
                event_session_id = properties.get("sessionID")

                if event_type == "session.error" and event_session_id in (None, session_id):
                    error = properties.get("error") or {}
                    logger.error("opencode session.error session=%s error=%s", event_session_id, error)
                    raise RuntimeError(f"opencode prompt_async failed: {error}")

                logger.debug(
                    "opencode SSE event type=%s session=%s message=%s part=%s field=%s",
                    event_type,
                    event_session_id,
                    properties.get("messageID"),
                    properties.get("partID"),
                    properties.get("field"),
                )

                if properties.get("sessionID") != session_id:
                    logger.debug("opencode SSE ignored other session event type=%s session=%s", event_type, event_session_id)
                    continue

                if event_type == "message.updated":
                    info = properties.get("info") or {}
                    if info.get("role") == "assistant" and info.get("parentID") == message_id:
                        assistant_id = info.get("id")
                        if isinstance(assistant_id, str):
                            assistant_message_ids.add(assistant_id)
                            buffered = pending_deltas.pop(assistant_id, [])
                            logger.info(
                                "opencode assistant matched session=%s user_message=%s assistant_message=%s buffered_deltas=%d completed=%s",
                                session_id,
                                message_id,
                                assistant_id,
                                len(buffered),
                                bool((info.get("time") or {}).get("completed")),
                            )
                            for buffered_delta in buffered:
                                yield buffered_delta
                        if (info.get("time") or {}).get("completed") and info.get("finish") == "tool-calls":
                            logger.info(
                                "opencode assistant tool-call step completed session=%s user_message=%s assistant_message=%s",
                                session_id,
                                message_id,
                                assistant_id,
                            )
                            yield _FLUSH_TEXT
                        elif (info.get("time") or {}).get("completed"):
                            logger.info(
                                "opencode assistant completed session=%s user_message=%s assistant_messages=%s pending_delta_messages=%s",
                                session_id,
                                message_id,
                                sorted(assistant_message_ids),
                                sorted(pending_deltas.keys()),
                            )
                            return
                    continue

                if event_type == "message.part.updated":
                    part = properties.get("part") or {}
                    part_id = part.get("id")
                    part_message_id = part.get("messageID")
                    part_type = part.get("type")
                    if not isinstance(part_id, str) or part_message_id not in assistant_message_ids:
                        continue
                    if part_type == "text":
                        speakable_part_ids.add(part_id)
                        buffered = pending_deltas.pop(part_id, [])
                        logger.info(
                            "opencode text part matched session=%s message=%s part=%s buffered_deltas=%d",
                            session_id,
                            part_message_id,
                            part_id,
                            len(buffered),
                        )
                        for buffered_delta in buffered:
                            yield buffered_delta
                    else:
                        unspeakable_part_ids.add(part_id)
                        pending_deltas.pop(part_id, None)
                        logger.debug(
                            "opencode ignored non-text part session=%s message=%s part=%s type=%s",
                            session_id,
                            part_message_id,
                            part_id,
                            part_type,
                        )
                    continue

                if event_type == "message.part.delta":
                    if properties.get("field") != "text":
                        continue
                    delta_message_id = properties.get("messageID")
                    delta_part_id = properties.get("partID")
                    delta = properties.get("delta")
                    if not isinstance(delta_message_id, str) or not isinstance(delta_part_id, str) or not isinstance(delta, str):
                        logger.debug("opencode ignored malformed text delta properties=%s", properties)
                        continue
                    if delta_message_id not in assistant_message_ids:
                        pending_deltas[delta_part_id].append(delta)
                        logger.debug(
                            "opencode buffered unmatched text delta session=%s message=%s chars=%d preview=%r",
                            session_id,
                            delta_message_id,
                            len(delta),
                            _preview(delta),
                        )
                        continue
                    if delta_part_id in unspeakable_part_ids:
                        logger.debug(
                            "opencode ignored non-text delta session=%s message=%s part=%s chars=%d preview=%r",
                            session_id,
                            delta_message_id,
                            delta_part_id,
                            len(delta),
                            _preview(delta),
                        )
                        continue
                    if delta_part_id not in speakable_part_ids:
                        pending_deltas[delta_part_id].append(delta)
                        logger.debug(
                            "opencode buffered untyped text delta session=%s message=%s part=%s chars=%d preview=%r",
                            session_id,
                            delta_message_id,
                            delta_part_id,
                            len(delta),
                            _preview(delta),
                        )
                        continue
                    logger.debug(
                        "opencode yielding text delta session=%s message=%s part=%s chars=%d preview=%r",
                        session_id,
                        delta_message_id,
                        delta_part_id,
                        len(delta),
                        _preview(delta),
                    )
                    yield delta
                    continue

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

        clean_text = ""
        printable_text = ""
        sentence_batch: list[str] = []
        try:
            for stream_item in self._stream_opencode_deltas(user_text, instructions):
                if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                    logger.info("opencode generation cancelled while streaming")
                    yield EndOfResponse()
                    return
                if isinstance(stream_item, _FlushText):
                    if printable_text.strip():
                        sentence_batch.append(printable_text.strip())
                        printable_text = ""
                    if sentence_batch:
                        chunk_text = " ".join(sentence_batch)
                        logger.info(
                            "opencode flushing LLM chunk chars=%d sentences=%d preview=%r",
                            len(chunk_text),
                            len(sentence_batch),
                            _preview(chunk_text),
                        )
                        yield LLMResponseChunk(
                            text=chunk_text,
                            language_code=language_code,
                            runtime_config=runtime_config,
                            response=response,
                        )
                        sentence_batch = []
                    continue
                delta = stream_item
                new_text = remove_unspeechable(delta)
                clean_text += new_text
                printable_text += new_text
                sentences = sent_tokenize(printable_text)
                logger.debug(
                    "opencode chunk state delta_chars=%d clean_chars=%d printable_chars=%d sentences=%d batch=%d preview=%r",
                    len(new_text),
                    len(clean_text),
                    len(printable_text),
                    len(sentences),
                    len(sentence_batch),
                    _preview(printable_text),
                )
                complete_sentences = sentences if _ends_sentence(printable_text) else sentences[:-1]
                if complete_sentences:
                    for sentence in complete_sentences:
                        sentence_batch.append(sentence)
                        if len(sentence_batch) >= self.stream_batch_sentences:
                            chunk_text = " ".join(sentence_batch)
                            logger.info(
                                "opencode yielding LLM chunk chars=%d sentences=%d preview=%r",
                                len(chunk_text),
                                len(sentence_batch),
                                _preview(chunk_text),
                            )
                            yield LLMResponseChunk(
                                text=chunk_text,
                                language_code=language_code,
                                runtime_config=runtime_config,
                                response=response,
                            )
                            sentence_batch = []
                    printable_text = "" if len(complete_sentences) == len(sentences) else sentences[-1]
        except Exception:
            logger.exception("opencode streaming failed; ending response to re-enable listening")
            yield EndOfResponse()
            return

        if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
            logger.info("opencode generation cancelled after stream ended")
            yield EndOfResponse()
            return

        original_chat.add_item(make_assistant_message(clean_text))
        if printable_text.strip():
            sentence_batch.append(printable_text.strip())
        if sentence_batch or not clean_text:
            final_chunk_text = " ".join(sentence_batch) if sentence_batch else clean_text
            logger.info(
                "opencode yielding final LLM chunk chars=%d sentences=%d preview=%r",
                len(final_chunk_text),
                len(sentence_batch),
                _preview(final_chunk_text),
            )
            yield LLMResponseChunk(
                text=final_chunk_text,
                language_code=language_code,
                runtime_config=runtime_config,
                response=response,
            )
        logger.info("opencode response complete clean_chars=%d", len(clean_text))
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
