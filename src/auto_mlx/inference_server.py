"""Bounded localhost-only text-completions API; not a production web server."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import json
import threading
import time
import uuid

from .canonical import strict_json_loads
from .errors import AutoMLXError
from .model_bundle import integer, require
from .runners.mlx_lm_runner import validate_request

MAX_HTTP_BODY = 262144


class LocalInferenceServer(ThreadingMixIn, HTTPServer):
    daemon_threads = False
    block_on_close = True
    request_queue_size = 4

    def __init__(self, port, session, profile_resolver):
        integer(port, 0, 65535, "port")
        self.session = session
        self.profile_resolver = profile_resolver
        self._slots = threading.BoundedSemaphore(4)
        super().__init__(("127.0.0.1", port), InferenceHandler)

    def process_request(self, request, address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self._slots.release()


class InferenceHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "AutoMLXLocal/1"

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *args):
        pass  # Prompts, bodies, and URLs are not retained in access logs.

    def _json(self, status, value):
        data = json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _local_request(self):
        expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        require(self.headers.get_all("Host", []) in [[h] for h in expected], "invalid localhost Host header")
        require(self.headers.get("Origin") is None, "browser-origin requests are not supported")
        require(self.headers.get("Transfer-Encoding") is None, "transfer encoding is not supported")

    def do_GET(self):
        try:
            self._local_request()
            if self.path == "/health":
                self._json(200, {"ok": True, "bundle_id": self.server.session.bundle.bundle_id, "scope": "single-user-local"})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": self.server.session.bundle.bundle_id, "object": "model", "owned_by": "local"}]})
            else:
                self._json(404, {"error": {"message": "unknown route"}})
        except (AutoMLXError, ValueError) as exc:
            self._json(400, {"error": {"message": str(exc)}})

    def do_POST(self):
        stream = None
        headers_sent = False
        try:
            self._local_request()
            require(self.path == "/v1/completions", "only /v1/completions is supported")
            require(self.headers.get("Content-Type", "").split(";")[0].strip() == "application/json", "application/json is required")
            lengths = self.headers.get_all("Content-Length", [])
            require(len(lengths) == 1 and lengths[0].isascii() and lengths[0].isdigit(), "one numeric Content-Length is required")
            length = integer(int(lengths[0]), 1, MAX_HTTP_BODY, "Content-Length")
            raw = self.rfile.read(length)
            require(len(raw) == length, "request body was truncated")
            body = strict_json_loads(raw)
            require(type(body) is dict and not (set(body) - {"model", "prompt", "max_tokens", "stream", "temperature"}), "unsupported completion fields")
            require(body.get("model") == self.server.session.bundle.bundle_id, "model must match the loaded bundle id")
            require(type(body.get("stream", False)) is bool, "stream must be boolean")
            require(type(body.get("temperature", 0)) is int and body.get("temperature", 0) == 0, "only deterministic temperature=0 is supported")
            request = {"prompt": body.get("prompt"), "max_tokens": body.get("max_tokens", 128)}
            validate_request(request)
            profile = self.server.profile_resolver(request)
            stream = self.server.session.stream(request, prefill_step_size=profile["prefill_step_size"])
            # Advance before success headers: admission, tokenization, context,
            # model loading and first-token failures become ordinary HTTP errors.
            first = next(stream)
            identifier = "cmpl-" + uuid.uuid4().hex
            created = int(time.time())
            base = {"id": identifier, "object": "text_completion", "created": created, "model": self.server.session.bundle.bundle_id}
            def events():
                yield first
                yield from stream
            if body.get("stream", False):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                headers_sent = True
                for event in events():
                    done = event["event"] == "done"
                    result = {**base, "choices": [{"index": 0, "text": "" if done else event["text"], "finish_reason": event.get("finish_reason") if done else None}]}
                    if done:
                        result["usage"] = {"prompt_tokens": event["prompt_tokens"], "completion_tokens": event["completion_tokens"], "total_tokens": event["prompt_tokens"] + event["completion_tokens"]}
                        result["auto_mlx"] = profile
                    self.wfile.write(b"data: " + json.dumps(result).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                final = None
                for event in events():
                    if event["event"] == "done":
                        final = event
                require(final is not None, "generation ended without a final event")
                self._json(200, {**base, "choices": [{"index": 0, "text": final["text"], "finish_reason": final["finish_reason"]}], "usage": {"prompt_tokens": final["prompt_tokens"], "completion_tokens": final["completion_tokens"], "total_tokens": final["prompt_tokens"] + final["completion_tokens"]}, "auto_mlx": {**profile, "metrics": {"memory_metric": final.get("memory_metric", "unknown"), **{k: final[k] for k in ("time_to_first_token_ns", "generation_elapsed_ns", "peak_memory_bytes")}}}})
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass  # Iterator close below cancels work on disconnect/read timeout.
        except (AutoMLXError, ValueError, TypeError, StopIteration) as exc:
            if not headers_sent:
                self._json(429 if "busy" in str(exc) else 400, {"error": {"message": str(exc)}})
            else:
                try:
                    self.wfile.write(b"data: " + json.dumps({"error": {"message": str(exc)}}).encode() + b"\n\n")
                    self.wfile.flush()
                except OSError:
                    pass
        finally:
            if stream is not None:
                stream.close()
