from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Queue
from threading import Event
from typing import Any

from speech_to_speech.pipeline.cancel_scope import CancelScope

logger = logging.getLogger(__name__)


class PipelineControlServer:
    def __init__(
        self,
        stop_event: Event,
        enabled_event: Event,
        should_listen: Event,
        host: str,
        port: int,
        queues_to_clear: list[Queue[Any]] | None = None,
        cancel_scope: CancelScope | None = None,
    ) -> None:
        self.stop_event = stop_event
        self.enabled_event = enabled_event
        self.should_listen = should_listen
        self.host = host
        self.port = port
        self.queues_to_clear = queues_to_clear or []
        self.cancel_scope = cancel_scope
        self.server: ThreadingHTTPServer | None = None

    def _set_enabled(self, enabled: bool) -> None:
        if enabled:
            self.enabled_event.set()
            self.should_listen.set()
            logger.info("Pipeline enabled")
            return

        self.enabled_event.clear()
        self.should_listen.clear()
        if self.cancel_scope is not None:
            self.cancel_scope.cancel()
        for queue in self.queues_to_clear:
            self._clear_queue(queue)
        if self.cancel_scope is not None:
            self.cancel_scope.reset()
        logger.info("Pipeline disabled")

    def _clear_queue(self, queue: Queue[Any]) -> None:
        while True:
            try:
                queue.get_nowait()
            except Empty:
                return

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        control = self

        class ControlHandler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                logger.debug("pipeline control: " + _format, *_args)

            def _write_json(self, status: int, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                if self.path != "/pipeline/enabled":
                    self._write_json(404, {"error": "not_found"})
                    return
                self._write_json(200, {"enabled": control.enabled_event.is_set()})

            def do_PUT(self) -> None:
                if self.path != "/pipeline/enabled":
                    self._write_json(404, {"error": "not_found"})
                    return
                try:
                    length = int(self.headers.get("content-length", "0"))
                    data = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    self._write_json(400, {"error": "invalid_json"})
                    return

                enabled = data.get("enabled")
                if not isinstance(enabled, bool):
                    self._write_json(400, {"error": "enabled must be a boolean"})
                    return
                control._set_enabled(enabled)
                self._write_json(200, {"enabled": control.enabled_event.is_set()})

        return ControlHandler

    def run(self) -> None:
        self.server = ThreadingHTTPServer((self.host, self.port), self._make_handler())
        self.server.timeout = 0.2
        logger.info("Pipeline control API listening on http://%s:%s", self.host, self.port)
        while not self.stop_event.is_set():
            self.server.handle_request()
        self.server.server_close()
