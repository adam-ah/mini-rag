#!/usr/bin/env python3
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import backend
from settings import AISettings


class OpenAIStubHandler(BaseHTTPRequestHandler):
    def _send_json(self, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        if self.headers.get("Authorization") == "Bearer test-key":
            return True
        self.send_error(401)
        return False

    def do_GET(self):
        if self.path != "/v1/models" or not self._authorized():
            if self.path != "/v1/models":
                self.send_error(404)
            return
        self._send_json({"data": [{"id": "test-model"}]})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if not self._authorized():
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        if request.get("stream"):
            body = (
                b'data: {"choices":[{"delta":{"content":"Tyre "}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"pressure"}}]}\n\n'
                b'data: [DONE]\n\n'
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_json({"choices": [{"message": {"content": "Tyre pressure"}}]})

    def log_message(self, _format, *args):
        pass


class OpenAIBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), OpenAIStubHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.settings = AISettings(
            backend="openai",
            base_url=f"http://127.0.0.1:{cls.server.server_port}/v1",
            model="test-model",
            api_key="test-key",
            timeout_seconds=2,
        )

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_connection_uses_configured_key(self):
        self.assertEqual(backend.test_connection(self.settings), (True, "Connection successful"))

    def test_answer_parses_openai_response(self):
        self.assertEqual(backend.answer("question", "context", 1, self.settings), "Tyre pressure")

    def test_stream_parses_sse_tokens(self):
        self.assertEqual(
            "".join(backend.stream("question", "context", 1, self.settings)),
            "Tyre pressure",
        )


if __name__ == "__main__":
    unittest.main()
