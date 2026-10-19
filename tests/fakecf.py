"""A stand-in for Cloudflare Workers AI on a REAL socket (127.0.0.1), for end-to-end tests.

Unlike the in-memory ``httpx.MockTransport`` the unit tests use, this exercises the full stack: the
OpenAI SDK, its real HTTP client, TLS-less loopback, streaming framing and the response headers.

Behaviour is chosen by the LAST path segment of the model id, so one server serves every scenario:
``@cf/test/ok`` (default), ``stream`` is selected by the request's ``stream`` flag, and ``e401``,
``e403``, ``e429-capacity`` (code 3040 + ``Retry-After: 7``), ``e429-quota`` (code 2005), ``e500``.

It can only stand in for the OVERRIDE path (``CLOUDFLARE_BASE_URL``): the real
``api.cloudflare.com/client/v4/accounts/{id}`` URL is deliberately not overridable, since a knob that
redirected it would itself be a way to redirect the token (see ``endpoint.validate_base_url``).
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
from dataclasses import dataclass, field

MODELS = ["@cf/meta/llama-3.1-8b-instruct", "@cf/qwen/qwen3.8-27b"]


@dataclass
class Seen:
    method: str
    path: str
    authorization: str | None
    body: dict = field(default_factory=dict)


class FakeCloudflare:
    def __init__(self) -> None:
        self.seen: list[Seen] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output clean
                pass

            def _json(self, code: int, body: dict, headers: dict | None = None) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.seen.append(Seen("GET", self.path, self.headers.get("Authorization")))
                if self.path.endswith("/models"):
                    return self._json(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in MODELS]})
                self._json(404, {"errors": [{"code": 7000, "message": "No route"}], "success": False})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                outer.seen.append(Seen("POST", self.path, self.headers.get("Authorization"), body))
                scenario = str(body.get("model", "")).rsplit("/", 1)[-1]
                fail = {
                    "e401": (401, {"errors": [{"code": 10000, "message": "Authentication error"}], "success": False}, {}),
                    "e403": (403, {"errors": [{"code": 10000, "message": "Authentication error"}], "success": False}, {}),
                    "e429-capacity": (429, {"errors": [{"code": 3040, "message": "AiError: Capacity temporarily exceeded, please try again"}],
                                            "success": False}, {"Retry-After": "7"}),
                    "e429-quota": (429, {"errors": [{"code": 2005, "message": "Rate limit exceeded"}], "success": False}, {}),
                    "e500": (500, {"errors": [{"code": 3043, "message": "Internal server error"}], "success": False}, {}),
                }.get(scenario)
                if fail:
                    return self._json(fail[0], fail[1], fail[2])
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for piece in ("he", "llo"):
                        chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                                 "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                self._json(200, {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                                 "choices": [{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant", "content": "ok"}}]})

        self._server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/client/v4/accounts/e2e/ai/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def posts(self) -> list[Seen]:
        return [s for s in self.seen if s.method == "POST"]
