import json
import re
from types import SimpleNamespace

import httpx
from openai.types.realtime import RealtimeSessionCreateRequest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM import opencode_language_model as opencode_module
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.opencode_language_model import OpencodeModelHandler
from speech_to_speech.pipeline.messages import EndOfResponse, GenerateResponseRequest, LLMResponseChunk


def _sent_tokenize(text):
    matches = list(re.finditer(r"[^.!?]+[.!?]", text))
    sentences = [match.group(0).strip() for match in matches]
    if not sentences and text.strip():
        return [text.strip()]
    consumed = matches[-1].end() if matches else 0
    remainder = text[consumed:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def _sse(event):
    return [f"data: {json.dumps(event)}", ""]


def _part_updated(session_id, message_id, part_id, part_type="text"):
    return _sse(
        {
            "type": "message.part.updated",
            "properties": {
                "sessionID": session_id,
                "part": {"id": part_id, "messageID": message_id, "sessionID": session_id, "type": part_type},
            },
        }
    )


class FakeStreamResponse:
    def __init__(self, lines):
        self.lines = lines
        self.consumed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self):
        return None

    def iter_lines(self):
        if self.consumed:
            raise httpx.StreamConsumed()
        self.consumed = True
        yield from self.lines


class FakeResponse:
    def raise_for_status(self):
        return None


class FakeClient:
    def __init__(self, lines):
        self.lines = lines
        self.posts = []

    def stream(self, method, url, **kwargs):
        self.stream_call = (method, url, kwargs)
        return FakeStreamResponse(self.lines)

    def post(self, url, json, **kwargs):
        self.posts.append((url, json, kwargs))
        return FakeResponse()


def _make_handler(lines):
    handler = object.__new__(OpencodeModelHandler)
    handler.base_url = "http://opencode.test"
    handler.session_state = SimpleNamespace(get=lambda: "ses_123")
    handler.directory = None
    handler.provider_id = "openai"
    handler.model_id = "gpt-5.5"
    handler.stream_batch_sentences = 1
    handler.enable_lang_prompt = False
    handler.cancel_scope = None
    handler.client = FakeClient(lines)
    return handler


def _make_request(text="Hi"):
    cfg = RuntimeConfig(
        chat=Chat(2),
        session=RealtimeSessionCreateRequest(type="realtime", instructions="Be concise."),
    )
    cfg.chat.add_item(make_user_message(text))
    return GenerateResponseRequest(runtime_config=cfg)


def test_opencode_message_ids_use_compatible_ascending_order(monkeypatch):
    handler = _make_handler([])
    monkeypatch.setattr(opencode_module, "_last_id_timestamp", 0)
    monkeypatch.setattr(opencode_module, "_id_counter", 0)
    monkeypatch.setattr(opencode_module.time, "time", lambda: 1770000000.0)

    first = handler._message_id()
    second = handler._message_id()

    assert re.match(r"^msg_[0-9a-f]{12}[0-9A-Za-z]{14}$", first)
    assert first < second
    assert not first.startswith("msg_19")


def test_opencode_streams_text_deltas_before_completion(monkeypatch):
    monkeypatch.setattr(opencode_module, "sent_tokenize", _sent_tokenize)
    user_message_id = "msg_test_user"
    assistant_message_id = "msg_test_assistant"
    text_part_id = "prt_test_text"
    lines = []
    lines += _sse({"type": "server.connected", "properties": {}})
    lines += _sse(
        {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_123",
                "info": {"id": assistant_message_id, "role": "assistant", "parentID": user_message_id, "time": {}},
            },
        }
    )
    lines += _part_updated("ses_123", assistant_message_id, text_part_id)
    lines += _sse(
        {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses_123",
                "messageID": assistant_message_id,
                "partID": text_part_id,
                "field": "text",
                "delta": "Hello. ",
            },
        }
    )
    lines += _sse(
        {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses_123",
                "messageID": assistant_message_id,
                "partID": text_part_id,
                "field": "text",
                "delta": "How are you?",
            },
        }
    )
    lines += _sse(
        {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_123",
                "info": {
                    "id": assistant_message_id,
                    "role": "assistant",
                    "parentID": user_message_id,
                    "time": {"completed": 123},
                },
            },
        }
    )

    handler = _make_handler(lines)
    monkeypatch.setattr(handler, "_message_id", lambda: user_message_id)

    outputs = list(handler.process(_make_request()))

    assert [type(output) for output in outputs] == [LLMResponseChunk, LLMResponseChunk, EndOfResponse]
    assert outputs[0].text == "Hello."
    assert outputs[1].text == "How are you?"
    assert handler.client.posts[0][0] == "http://opencode.test/session/ses_123/prompt_async"
    assert handler.client.posts[0][1]["messageID"] == user_message_id
    assert handler.client.posts[0][1]["parts"] == [{"type": "text", "text": "Hi"}]


def test_opencode_flushes_single_complete_sentence_before_completion(monkeypatch):
    monkeypatch.setattr(opencode_module, "sent_tokenize", _sent_tokenize)
    user_message_id = "msg_test_user"
    assistant_message_id = "msg_test_assistant"
    text_part_id = "prt_test_text"
    lines = []
    lines += _sse({"type": "server.connected", "properties": {}})
    lines += _sse(
        {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_123",
                "info": {"id": assistant_message_id, "role": "assistant", "parentID": user_message_id, "time": {}},
            },
        }
    )
    lines += _part_updated("ses_123", assistant_message_id, text_part_id)
    lines += _sse(
        {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses_123",
                "messageID": assistant_message_id,
                "partID": text_part_id,
                "field": "text",
                "delta": "I'll inspect the repo first.",
            },
        }
    )
    lines += _sse({"type": "server.heartbeat", "properties": {}})
    lines += _sse(
        {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_123",
                "info": {
                    "id": assistant_message_id,
                    "role": "assistant",
                    "parentID": user_message_id,
                    "time": {"completed": 123},
                },
            },
        }
    )

    handler = _make_handler(lines)
    monkeypatch.setattr(handler, "_message_id", lambda: user_message_id)

    outputs = list(handler.process(_make_request()))

    assert [output.text for output in outputs if isinstance(output, LLMResponseChunk)] == ["I'll inspect the repo first."]


def test_opencode_ignores_deltas_for_other_messages(monkeypatch):
    monkeypatch.setattr(opencode_module, "sent_tokenize", _sent_tokenize)
    user_message_id = "msg_test_user"
    assistant_message_id = "msg_test_assistant"
    text_part_id = "prt_test_text"
    lines = []
    lines += _sse({"type": "server.connected", "properties": {}})
    lines += _sse(
        {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses_123",
                "messageID": "msg_other",
                "partID": "prt_other",
                "field": "text",
                "delta": "Ignore this.",
            },
        }
    )
    lines += _sse(
        {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_123",
                "info": {"id": assistant_message_id, "role": "assistant", "parentID": user_message_id, "time": {}},
            },
        }
    )
    lines += _part_updated("ses_123", assistant_message_id, text_part_id)
    lines += _sse(
        {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses_123",
                "messageID": assistant_message_id,
                "partID": text_part_id,
                "field": "text",
                "delta": "Use this.",
            },
        }
    )
    lines += _sse(
        {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_123",
                "info": {
                    "id": assistant_message_id,
                    "role": "assistant",
                    "parentID": user_message_id,
                    "time": {"completed": 123},
                },
            },
        }
    )

    handler = _make_handler(lines)
    monkeypatch.setattr(handler, "_message_id", lambda: user_message_id)

    outputs = list(handler.process(_make_request()))

    assert [output.text for output in outputs if isinstance(output, LLMResponseChunk)] == ["Use this."]


def test_opencode_buffers_text_deltas_until_assistant_message_is_known(monkeypatch):
    monkeypatch.setattr(opencode_module, "sent_tokenize", _sent_tokenize)
    user_message_id = "msg_test_user"
    assistant_message_id = "msg_test_assistant"
    text_part_id = "prt_test_text"
    lines = []
    lines += _sse({"type": "server.connected", "properties": {}})
    lines += _sse(
        {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses_123",
                "messageID": assistant_message_id,
                "partID": text_part_id,
                "field": "text",
                "delta": "Buffered first. ",
            },
        }
    )
    lines += _sse(
        {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_123",
                "info": {"id": assistant_message_id, "role": "assistant", "parentID": user_message_id, "time": {}},
            },
        }
    )
    lines += _part_updated("ses_123", assistant_message_id, text_part_id)
    lines += _sse(
        {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses_123",
                "messageID": assistant_message_id,
                "partID": text_part_id,
                "field": "text",
                "delta": "Then live.",
            },
        }
    )
    lines += _sse(
        {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_123",
                "info": {
                    "id": assistant_message_id,
                    "role": "assistant",
                    "parentID": user_message_id,
                    "time": {"completed": 123},
                },
            },
        }
    )

    handler = _make_handler(lines)
    monkeypatch.setattr(handler, "_message_id", lambda: user_message_id)

    outputs = list(handler.process(_make_request()))

    assert [output.text for output in outputs if isinstance(output, LLMResponseChunk)] == [
        "Buffered first.",
        "Then live.",
    ]


def test_opencode_ignores_reasoning_text_deltas(monkeypatch):
    monkeypatch.setattr(opencode_module, "sent_tokenize", _sent_tokenize)
    user_message_id = "msg_test_user"
    assistant_message_id = "msg_test_assistant"
    reasoning_part_id = "prt_reasoning"
    text_part_id = "prt_text"
    lines = []
    lines += _sse({"type": "server.connected", "properties": {}})
    lines += _sse(
        {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_123",
                "info": {"id": assistant_message_id, "role": "assistant", "parentID": user_message_id, "time": {}},
            },
        }
    )
    lines += _part_updated("ses_123", assistant_message_id, reasoning_part_id, "reasoning")
    lines += _sse(
        {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses_123",
                "messageID": assistant_message_id,
                "partID": reasoning_part_id,
                "field": "text",
                "delta": "Do not speak this reasoning.",
            },
        }
    )
    lines += _part_updated("ses_123", assistant_message_id, text_part_id)
    lines += _sse(
        {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses_123",
                "messageID": assistant_message_id,
                "partID": text_part_id,
                "field": "text",
                "delta": "Speak this.",
            },
        }
    )
    lines += _sse(
        {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_123",
                "info": {
                    "id": assistant_message_id,
                    "role": "assistant",
                    "parentID": user_message_id,
                    "time": {"completed": 123},
                },
            },
        }
    )

    handler = _make_handler(lines)
    monkeypatch.setattr(handler, "_message_id", lambda: user_message_id)

    outputs = list(handler.process(_make_request()))

    assert [output.text for output in outputs if isinstance(output, LLMResponseChunk)] == ["Speak this."]


def test_opencode_continues_after_tool_call_commentary_to_final_answer(monkeypatch):
    monkeypatch.setattr(opencode_module, "sent_tokenize", _sent_tokenize)
    user_message_id = "msg_test_user"
    commentary_message_id = "msg_commentary_assistant"
    final_message_id = "msg_final_assistant"
    commentary_part_id = "prt_commentary"
    final_part_id = "prt_final"
    lines = []
    lines += _sse({"type": "server.connected", "properties": {}})
    lines += _sse(
        {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_123",
                "info": {"id": commentary_message_id, "role": "assistant", "parentID": user_message_id, "time": {}},
            },
        }
    )
    lines += _part_updated("ses_123", commentary_message_id, commentary_part_id)
    lines += _sse(
        {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses_123",
                "messageID": commentary_message_id,
                "partID": commentary_part_id,
                "field": "text",
                "delta": "I'll inspect the repo first.",
            },
        }
    )
    lines += _sse(
        {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_123",
                "info": {
                    "id": commentary_message_id,
                    "role": "assistant",
                    "parentID": user_message_id,
                    "time": {"completed": 123},
                    "finish": "tool-calls",
                },
            },
        }
    )
    lines += _sse(
        {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_123",
                "info": {"id": final_message_id, "role": "assistant", "parentID": user_message_id, "time": {}},
            },
        }
    )
    lines += _part_updated("ses_123", final_message_id, final_part_id)
    lines += _sse(
        {
            "type": "message.part.delta",
            "properties": {
                "sessionID": "ses_123",
                "messageID": final_message_id,
                "partID": final_part_id,
                "field": "text",
                "delta": "This is the final answer.",
            },
        }
    )
    lines += _sse(
        {
            "type": "message.updated",
            "properties": {
                "sessionID": "ses_123",
                "info": {
                    "id": final_message_id,
                    "role": "assistant",
                    "parentID": user_message_id,
                    "time": {"completed": 456},
                    "finish": "stop",
                },
            },
        }
    )

    handler = _make_handler(lines)
    monkeypatch.setattr(handler, "_message_id", lambda: user_message_id)

    outputs = list(handler.process(_make_request()))

    assert [output.text for output in outputs if isinstance(output, LLMResponseChunk)] == [
        "I'll inspect the repo first.",
        "This is the final answer.",
    ]
