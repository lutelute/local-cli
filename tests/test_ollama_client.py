"""Tests for local_cli.ollama_client module."""

import io
import json
import socket
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from local_cli.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaRequestError,
    OllamaStreamError,
    _DEFAULT_TIMEOUT,
    _STREAM_TIMEOUT,
)


class TestOllamaClientInit(unittest.TestCase):
    """Tests for OllamaClient construction."""

    def test_default_base_url(self) -> None:
        """Default base URL is http://localhost:11434."""
        client = OllamaClient()
        self.assertEqual(client.base_url, "http://localhost:11434")

    def test_custom_localhost_url(self) -> None:
        """Accept custom localhost URLs."""
        client = OllamaClient("http://127.0.0.1:11435")
        self.assertEqual(client.base_url, "http://127.0.0.1:11435")

    def test_trailing_slash_stripped(self) -> None:
        """Trailing slash is stripped from the base URL."""
        client = OllamaClient("http://localhost:11434/")
        self.assertEqual(client.base_url, "http://localhost:11434")

    def test_remote_host_rejected(self) -> None:
        """Reject non-localhost URLs."""
        with self.assertRaises(ValueError) as ctx:
            OllamaClient("http://evil.com:11434")
        self.assertIn("localhost", str(ctx.exception))

    def test_remote_ip_rejected(self) -> None:
        """Reject remote IP addresses."""
        with self.assertRaises(ValueError):
            OllamaClient("http://192.168.1.100:11434")


class TestOllamaClientRequest(unittest.TestCase):
    """Tests for OllamaClient._request() via public methods."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_get_version_request_format(self, mock_urlopen: MagicMock) -> None:
        """get_version sends correct GET request to /api/version."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"version": "0.5.1"}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.get_version()

        self.assertEqual(result, {"version": "0.5.1"})
        # Verify the request object passed to urlopen.
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        self.assertEqual(req.full_url, "http://localhost:11434/api/version")
        self.assertEqual(req.method, "GET")
        self.assertIsNone(req.data)
        # Verify timeout.
        self.assertEqual(call_args[1].get("timeout", call_args[0][1] if len(call_args[0]) > 1 else None), _DEFAULT_TIMEOUT)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_post_request_with_json_body(self, mock_urlopen: MagicMock) -> None:
        """POST requests include Content-Type header and JSON body."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        messages = [{"role": "user", "content": "Hi"}]
        self.client.chat("qwen3:8b", messages)

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        self.assertEqual(req.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(req.method, "POST")
        self.assertEqual(req.get_header("Content-type"), "application/json")

        # Verify JSON body.
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen3:8b")
        self.assertEqual(body["messages"], messages)
        self.assertFalse(body["stream"])

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_with_tools_includes_tools_in_payload(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat() includes tools in the request payload when provided."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": ""},
            "done": True,
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        tools = [{"type": "function", "function": {"name": "read_file"}}]
        self.client.chat("qwen3:8b", [{"role": "user", "content": "Read"}], tools=tools)

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["tools"], tools)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_without_tools_excludes_tools_key(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat() omits tools key when no tools are provided."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.client.chat("qwen3:8b", [{"role": "user", "content": "Hi"}])

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertNotIn("tools", body)


class TestOllamaClientVersionParsing(unittest.TestCase):
    """Tests for version parsing via get_version()."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_version_parsed_correctly(self, mock_urlopen: MagicMock) -> None:
        """Version string is returned as a dict."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"version": "0.5.1"}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.get_version()
        self.assertEqual(result["version"], "0.5.1")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_version_with_extra_fields(self, mock_urlopen: MagicMock) -> None:
        """Version response with extra fields is handled."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"version": "0.6.0", "build": "abc123"}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.get_version()
        self.assertEqual(result["version"], "0.6.0")
        self.assertEqual(result["build"], "abc123")


class TestOllamaClientListModels(unittest.TestCase):
    """Tests for model list parsing via list_models()."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_list_models_returns_models_list(self, mock_urlopen: MagicMock) -> None:
        """list_models extracts the 'models' key from the response."""
        models_data = {
            "models": [
                {"name": "qwen3:8b", "size": 4_000_000_000},
                {"name": "all-minilm", "size": 50_000_000},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(models_data).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.list_models()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "qwen3:8b")
        self.assertEqual(result[1]["name"], "all-minilm")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_list_models_empty(self, mock_urlopen: MagicMock) -> None:
        """list_models returns empty list when no models are available."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"models": []}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.list_models()
        self.assertEqual(result, [])

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_list_models_missing_key(self, mock_urlopen: MagicMock) -> None:
        """list_models returns empty list when 'models' key is absent."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.list_models()
        self.assertEqual(result, [])


class TestOllamaClientStreaming(unittest.TestCase):
    """Tests for streaming NDJSON parsing via chat_stream()."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    def _make_stream_response(self, lines: list[bytes]) -> MagicMock:
        """Create a mock response that iterates over NDJSON lines."""
        mock_resp = MagicMock()
        mock_resp.__iter__ = MagicMock(return_value=iter(lines))
        mock_resp.close = MagicMock()
        return mock_resp

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_stream_multi_line_response(self, mock_urlopen: MagicMock) -> None:
        """Parse multiple NDJSON lines from a streaming response."""
        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}).encode() + b"\n",
            json.dumps({"message": {"content": " world"}, "done": False}).encode() + b"\n",
            json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        chunks = list(self.client.chat_stream("qwen3:8b", [{"role": "user", "content": "Hi"}]))
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["message"]["content"], "Hello")
        self.assertEqual(chunks[1]["message"]["content"], " world")
        self.assertTrue(chunks[2]["done"])

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_stream_skips_empty_lines(self, mock_urlopen: MagicMock) -> None:
        """Empty lines in the NDJSON stream are skipped."""
        lines = [
            json.dumps({"message": {"content": "a"}, "done": False}).encode() + b"\n",
            b"\n",
            b"  \n",
            json.dumps({"message": {"content": "b"}, "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        chunks = list(self.client.chat_stream("qwen3:8b", [{"role": "user", "content": "Hi"}]))
        self.assertEqual(len(chunks), 2)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_stream_skips_invalid_json_lines(self, mock_urlopen: MagicMock) -> None:
        """Invalid JSON lines are silently skipped."""
        lines = [
            json.dumps({"message": {"content": "a"}, "done": False}).encode() + b"\n",
            b"this is not json\n",
            json.dumps({"message": {"content": "b"}, "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        chunks = list(self.client.chat_stream("qwen3:8b", [{"role": "user", "content": "Hi"}]))
        self.assertEqual(len(chunks), 2)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_stream_request_format(self, mock_urlopen: MagicMock) -> None:
        """chat_stream sends correct POST request with stream=true."""
        lines = [
            json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        list(self.client.chat_stream("qwen3:8b", [{"role": "user", "content": "Hi"}]))

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(req.method, "POST")
        self.assertEqual(req.get_header("Content-type"), "application/json")

        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen3:8b")
        self.assertTrue(body["stream"])

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_stream_with_tools_in_payload(self, mock_urlopen: MagicMock) -> None:
        """chat_stream includes tools in the payload when provided."""
        lines = [
            json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        tools = [{"type": "function", "function": {"name": "run_shell"}}]
        list(self.client.chat_stream(
            "qwen3:8b",
            [{"role": "user", "content": "List files"}],
            tools=tools,
        ))

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["tools"], tools)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_stream_without_tools_excludes_key(self, mock_urlopen: MagicMock) -> None:
        """chat_stream omits tools key when not provided."""
        lines = [
            json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        list(self.client.chat_stream("qwen3:8b", [{"role": "user", "content": "Hi"}]))

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertNotIn("tools", body)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_stream_response_closed_after_iteration(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Response object is closed after stream is consumed."""
        lines = [
            json.dumps({"message": {"content": "done"}, "done": True}).encode() + b"\n",
        ]
        mock_resp = self._make_stream_response(lines)
        mock_urlopen.return_value = mock_resp

        list(self.client.chat_stream("qwen3:8b", [{"role": "user", "content": "Hi"}]))
        mock_resp.close.assert_called_once()


class TestOllamaClientToolCallExtraction(unittest.TestCase):
    """Tests for tool call extraction from streaming responses."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    def _make_stream_response(self, lines: list[bytes]) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.__iter__ = MagicMock(return_value=iter(lines))
        mock_resp.close = MagicMock()
        return mock_resp

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_tool_call_in_final_chunk(self, mock_urlopen: MagicMock) -> None:
        """Tool calls appear in the final done=true chunk."""
        tool_calls = [
            {
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "/tmp/test.py"},
                },
            }
        ]
        lines = [
            json.dumps({
                "message": {"role": "assistant", "content": ""},
                "done": False,
            }).encode() + b"\n",
            json.dumps({
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                },
                "done": True,
            }).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        chunks = list(self.client.chat_stream(
            "qwen3:8b",
            [{"role": "user", "content": "Read file"}],
            tools=[{"type": "function", "function": {"name": "read_file"}}],
        ))

        # The final chunk should contain tool_calls.
        final = chunks[-1]
        self.assertTrue(final["done"])
        self.assertIn("tool_calls", final["message"])
        self.assertEqual(final["message"]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(
            final["message"]["tool_calls"][0]["function"]["arguments"]["path"],
            "/tmp/test.py",
        )

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_multiple_tool_calls(self, mock_urlopen: MagicMock) -> None:
        """Multiple tool calls in a single final chunk."""
        tool_calls = [
            {"function": {"name": "read_file", "arguments": {"path": "a.py"}}},
            {"function": {"name": "run_shell", "arguments": {"command": "ls"}}},
        ]
        lines = [
            json.dumps({
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                },
                "done": True,
            }).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        chunks = list(self.client.chat_stream(
            "qwen3:8b",
            [{"role": "user", "content": "Read and list"}],
        ))

        final = chunks[-1]
        self.assertEqual(len(final["message"]["tool_calls"]), 2)
        self.assertEqual(final["message"]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(final["message"]["tool_calls"][1]["function"]["name"], "run_shell")


class TestOllamaClientErrorHandling(unittest.TestCase):
    """Tests for error handling: connection refused, timeout, mid-stream, invalid JSON."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    # --- Connection errors for _request ---

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_connection_refused(self, mock_urlopen: MagicMock) -> None:
        """URLError (connection refused) raises OllamaConnectionError."""
        mock_urlopen.side_effect = urllib.error.URLError(
            ConnectionRefusedError("Connection refused")
        )

        with self.assertRaises(OllamaConnectionError) as ctx:
            self.client.get_version()
        self.assertIn("Failed to connect", str(ctx.exception))

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_timeout_error(self, mock_urlopen: MagicMock) -> None:
        """socket.timeout raises OllamaConnectionError."""
        mock_urlopen.side_effect = socket.timeout("timed out")

        with self.assertRaises(OllamaConnectionError) as ctx:
            self.client.get_version()
        self.assertIn("timed out", str(ctx.exception))

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_url_error_generic(self, mock_urlopen: MagicMock) -> None:
        """Generic URLError raises OllamaConnectionError."""
        mock_urlopen.side_effect = urllib.error.URLError("Name resolution failed")

        with self.assertRaises(OllamaConnectionError):
            self.client.list_models()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_invalid_json_response(self, mock_urlopen: MagicMock) -> None:
        """Invalid JSON in non-streaming response raises OllamaRequestError."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not valid json {"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with self.assertRaises(OllamaRequestError) as ctx:
            self.client.get_version()
        self.assertIn("Invalid JSON", str(ctx.exception))

    # --- Connection errors for _stream_request ---

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_stream_connection_refused(self, mock_urlopen: MagicMock) -> None:
        """URLError during stream connection raises OllamaConnectionError."""
        mock_urlopen.side_effect = urllib.error.URLError(
            ConnectionRefusedError("Connection refused")
        )

        with self.assertRaises(OllamaConnectionError):
            list(self.client.chat_stream(
                "qwen3:8b",
                [{"role": "user", "content": "Hi"}],
            ))

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_stream_timeout_on_connect(self, mock_urlopen: MagicMock) -> None:
        """socket.timeout during stream connection raises OllamaConnectionError."""
        mock_urlopen.side_effect = socket.timeout("timed out")

        with self.assertRaises(OllamaConnectionError):
            list(self.client.chat_stream(
                "qwen3:8b",
                [{"role": "user", "content": "Hi"}],
            ))

    # --- Mid-stream errors ---

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_mid_stream_error_raises_stream_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        """An error object mid-stream raises OllamaStreamError."""
        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}).encode() + b"\n",
            json.dumps({"error": "model not found"}).encode() + b"\n",
            json.dumps({"message": {"content": "end"}, "done": True}).encode() + b"\n",
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__ = MagicMock(return_value=iter(lines))
        mock_resp.close = MagicMock()
        mock_urlopen.return_value = mock_resp

        chunks = []
        with self.assertRaises(OllamaStreamError) as ctx:
            for chunk in self.client.chat_stream(
                "qwen3:8b",
                [{"role": "user", "content": "Hi"}],
            ):
                chunks.append(chunk)

        # First chunk should have been yielded before the error.
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["message"]["content"], "Hello")
        self.assertIn("model not found", str(ctx.exception))
        # Response should be closed even on error.
        mock_resp.close.assert_called_once()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_stream_error_at_start(self, mock_urlopen: MagicMock) -> None:
        """Error object as the first chunk raises OllamaStreamError."""
        lines = [
            json.dumps({"error": "server overloaded"}).encode() + b"\n",
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__ = MagicMock(return_value=iter(lines))
        mock_resp.close = MagicMock()
        mock_urlopen.return_value = mock_resp

        with self.assertRaises(OllamaStreamError) as ctx:
            list(self.client.chat_stream(
                "qwen3:8b",
                [{"role": "user", "content": "Hi"}],
            ))
        self.assertIn("server overloaded", str(ctx.exception))

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_stream_invalid_json_lines_skipped(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Invalid JSON lines in the stream are silently skipped."""
        lines = [
            b"not json at all\n",
            b"{broken json\n",
            json.dumps({"message": {"content": "ok"}, "done": True}).encode() + b"\n",
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__ = MagicMock(return_value=iter(lines))
        mock_resp.close = MagicMock()
        mock_urlopen.return_value = mock_resp

        chunks = list(self.client.chat_stream(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
        ))
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0]["done"])


class TestOllamaClientEmbed(unittest.TestCase):
    """Tests for the embed() method."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_embed_single_string(self, mock_urlopen: MagicMock) -> None:
        """embed() with a single string returns embeddings list."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "embeddings": [[0.1, 0.2, 0.3]],
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.embed("all-minilm", "hello world")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [0.1, 0.2, 0.3])

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "all-minilm")
        self.assertEqual(body["input"], "hello world")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_embed_multiple_strings(self, mock_urlopen: MagicMock) -> None:
        """embed() with a list of strings returns multiple embeddings."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.embed("all-minilm", ["hello", "world"])
        self.assertEqual(len(result), 2)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_embed_missing_embeddings_key(self, mock_urlopen: MagicMock) -> None:
        """embed() returns empty list when 'embeddings' key is missing."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.embed("all-minilm", "test")
        self.assertEqual(result, [])


class TestOllamaClientPullModel(unittest.TestCase):
    """Tests for the pull_model() streaming method."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_pull_model_streams_progress(self, mock_urlopen: MagicMock) -> None:
        """pull_model yields progress update chunks."""
        lines = [
            json.dumps({"status": "pulling manifest"}).encode() + b"\n",
            json.dumps({"status": "downloading", "completed": 100, "total": 1000}).encode() + b"\n",
            json.dumps({"status": "downloading", "completed": 1000, "total": 1000}).encode() + b"\n",
            json.dumps({"status": "success"}).encode() + b"\n",
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__ = MagicMock(return_value=iter(lines))
        mock_resp.close = MagicMock()
        mock_urlopen.return_value = mock_resp

        chunks = list(self.client.pull_model("qwen3:8b"))
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0]["status"], "pulling manifest")
        self.assertEqual(chunks[-1]["status"], "success")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_pull_model_request_format(self, mock_urlopen: MagicMock) -> None:
        """pull_model sends correct POST to /api/pull with stream=true."""
        lines = [
            json.dumps({"status": "success"}).encode() + b"\n",
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__ = MagicMock(return_value=iter(lines))
        mock_resp.close = MagicMock()
        mock_urlopen.return_value = mock_resp

        list(self.client.pull_model("qwen3:8b"))

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "http://localhost:11434/api/pull")
        self.assertEqual(req.method, "POST")

        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen3:8b")
        self.assertTrue(body["stream"])


class TestOllamaClientTimeouts(unittest.TestCase):
    """Tests for timeout parameter usage."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_default_timeout_for_simple_requests(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Simple requests (get_version, list_models) use _DEFAULT_TIMEOUT."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"version": "0.5.1"}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.client.get_version()

        call_kwargs = mock_urlopen.call_args
        self.assertEqual(call_kwargs[1]["timeout"], _DEFAULT_TIMEOUT)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_stream_timeout_for_chat(self, mock_urlopen: MagicMock) -> None:
        """Chat requests use _STREAM_TIMEOUT."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "message": {"content": "Hi"}, "done": True,
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.client.chat("qwen3:8b", [{"role": "user", "content": "Hi"}])

        call_kwargs = mock_urlopen.call_args
        self.assertEqual(call_kwargs[1]["timeout"], _STREAM_TIMEOUT)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_stream_timeout_for_chat_stream(self, mock_urlopen: MagicMock) -> None:
        """chat_stream uses _STREAM_TIMEOUT."""
        lines = [
            json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n",
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__ = MagicMock(return_value=iter(lines))
        mock_resp.close = MagicMock()
        mock_urlopen.return_value = mock_resp

        list(self.client.chat_stream("qwen3:8b", [{"role": "user", "content": "Hi"}]))

        call_kwargs = mock_urlopen.call_args
        self.assertEqual(call_kwargs[1]["timeout"], _STREAM_TIMEOUT)


class TestOllamaClientExceptionChaining(unittest.TestCase):
    """Tests that exceptions are properly chained with __cause__."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_connection_error_chains_original(self, mock_urlopen: MagicMock) -> None:
        """OllamaConnectionError chains the original URLError."""
        original = urllib.error.URLError(ConnectionRefusedError("refused"))
        mock_urlopen.side_effect = original

        with self.assertRaises(OllamaConnectionError) as ctx:
            self.client.get_version()
        self.assertIs(ctx.exception.__cause__, original)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_timeout_error_chains_original(self, mock_urlopen: MagicMock) -> None:
        """OllamaConnectionError chains the original socket.timeout."""
        original = socket.timeout("timed out")
        mock_urlopen.side_effect = original

        with self.assertRaises(OllamaConnectionError) as ctx:
            self.client.get_version()
        self.assertIs(ctx.exception.__cause__, original)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_json_error_chains_original(self, mock_urlopen: MagicMock) -> None:
        """OllamaRequestError chains the original JSONDecodeError."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"invalid"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with self.assertRaises(OllamaRequestError) as ctx:
            self.client.get_version()
        self.assertIsInstance(ctx.exception.__cause__, json.JSONDecodeError)


if __name__ == "__main__":
    unittest.main()
