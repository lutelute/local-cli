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
from local_cli.providers.base import (
    ProviderConnectionError,
    ProviderRequestError,
    ProviderStreamError,
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


class TestOllamaExceptionInheritance(unittest.TestCase):
    """Tests that Ollama exceptions inherit from provider base exceptions."""

    def test_connection_error_is_provider_connection_error(self) -> None:
        """OllamaConnectionError is a subclass of ProviderConnectionError."""
        self.assertTrue(issubclass(OllamaConnectionError, ProviderConnectionError))
        exc = OllamaConnectionError("test")
        self.assertIsInstance(exc, ProviderConnectionError)

    def test_request_error_is_provider_request_error(self) -> None:
        """OllamaRequestError is a subclass of ProviderRequestError."""
        self.assertTrue(issubclass(OllamaRequestError, ProviderRequestError))
        exc = OllamaRequestError("test")
        self.assertIsInstance(exc, ProviderRequestError)

    def test_stream_error_is_provider_stream_error(self) -> None:
        """OllamaStreamError is a subclass of ProviderStreamError."""
        self.assertTrue(issubclass(OllamaStreamError, ProviderStreamError))
        exc = OllamaStreamError("test")
        self.assertIsInstance(exc, ProviderStreamError)

    def test_connection_error_still_catches_as_ollama_type(self) -> None:
        """OllamaConnectionError is still caught by except OllamaConnectionError."""
        with self.assertRaises(OllamaConnectionError):
            raise OllamaConnectionError("test")

    def test_stream_error_caught_by_provider_except(self) -> None:
        """OllamaStreamError is caught by except ProviderStreamError."""
        with self.assertRaises(ProviderStreamError):
            raise OllamaStreamError("test")

    def test_connection_error_caught_by_provider_except(self) -> None:
        """OllamaConnectionError is caught by except ProviderConnectionError."""
        with self.assertRaises(ProviderConnectionError):
            raise OllamaConnectionError("test")


class TestOllamaClientShowModel(unittest.TestCase):
    """Tests for the show_model() method."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_show_model_returns_details(self, mock_urlopen: MagicMock) -> None:
        """show_model returns model details dict."""
        model_info = {
            "modelfile": "FROM qwen3:8b\nSYSTEM You are helpful.",
            "parameters": "temperature 0.7",
            "template": "{{ .System }}\n{{ .Prompt }}",
            "details": {
                "family": "qwen3",
                "parameter_size": "8B",
                "quantization_level": "Q4_K_M",
            },
            "capabilities": ["completion", "tools"],
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(model_info).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.show_model("qwen3:8b")

        self.assertEqual(result["details"]["family"], "qwen3")
        self.assertEqual(result["capabilities"], ["completion", "tools"])

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_show_model_request_format(self, mock_urlopen: MagicMock) -> None:
        """show_model sends POST to /api/show with model in body."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"details": {}}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.client.show_model("qwen3:8b")

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "http://localhost:11434/api/show")
        self.assertEqual(req.method, "POST")
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen3:8b")

    def test_show_model_validates_model_name(self) -> None:
        """show_model rejects invalid model names."""
        with self.assertRaises(ValueError):
            self.client.show_model("../../../etc/passwd")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_show_model_without_capabilities(self, mock_urlopen: MagicMock) -> None:
        """show_model works even when capabilities is absent."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "details": {"family": "custom"},
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.show_model("custom-model")
        self.assertNotIn("capabilities", result)
        self.assertEqual(result["details"]["family"], "custom")


class TestOllamaClientListRunningModels(unittest.TestCase):
    """Tests for the list_running_models() method."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_list_running_models_returns_models(self, mock_urlopen: MagicMock) -> None:
        """list_running_models returns list of running model dicts."""
        running_data = {
            "models": [
                {
                    "name": "qwen3:8b",
                    "size": 5_200_000_000,
                    "size_vram": 5_200_000_000,
                },
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(running_data).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.list_running_models()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "qwen3:8b")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_list_running_models_empty(self, mock_urlopen: MagicMock) -> None:
        """list_running_models returns empty list when no models loaded."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"models": []}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.list_running_models()
        self.assertEqual(result, [])

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_list_running_models_missing_key(self, mock_urlopen: MagicMock) -> None:
        """list_running_models returns empty list when 'models' key is absent."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.list_running_models()
        self.assertEqual(result, [])

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_list_running_models_request_format(self, mock_urlopen: MagicMock) -> None:
        """list_running_models sends GET to /api/ps."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"models": []}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.client.list_running_models()

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "http://localhost:11434/api/ps")
        self.assertEqual(req.method, "GET")
        self.assertIsNone(req.data)


class TestOllamaClientDeleteModel(unittest.TestCase):
    """Tests for the delete_model() method."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_delete_model_request_format(self, mock_urlopen: MagicMock) -> None:
        """delete_model sends DELETE to /api/delete with model in JSON body."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b""
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.client.delete_model("phi4-mini")

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "http://localhost:11434/api/delete")
        self.assertEqual(req.method, "DELETE")
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "phi4-mini")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_delete_model_returns_none(self, mock_urlopen: MagicMock) -> None:
        """delete_model returns None on success (no JSON body)."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b""
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.delete_model("phi4-mini")
        self.assertIsNone(result)

    def test_delete_model_validates_model_name(self) -> None:
        """delete_model rejects invalid model names."""
        with self.assertRaises(ValueError):
            self.client.delete_model("../../../etc/passwd")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_delete_model_connection_error(self, mock_urlopen: MagicMock) -> None:
        """delete_model raises OllamaConnectionError on connection failure."""
        mock_urlopen.side_effect = urllib.error.URLError(
            ConnectionRefusedError("Connection refused")
        )

        with self.assertRaises(OllamaConnectionError):
            self.client.delete_model("phi4-mini")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_delete_model_timeout(self, mock_urlopen: MagicMock) -> None:
        """delete_model raises OllamaConnectionError on timeout."""
        mock_urlopen.side_effect = socket.timeout("timed out")

        with self.assertRaises(OllamaConnectionError):
            self.client.delete_model("phi4-mini")


class TestOllamaClientCopyModel(unittest.TestCase):
    """Tests for the copy_model() method."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_copy_model_request_format(self, mock_urlopen: MagicMock) -> None:
        """copy_model sends POST to /api/copy with source and destination."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b""
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.client.copy_model("qwen3:8b", "my-qwen3:8b")

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "http://localhost:11434/api/copy")
        self.assertEqual(req.method, "POST")
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["source"], "qwen3:8b")
        self.assertEqual(body["destination"], "my-qwen3:8b")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_copy_model_returns_none(self, mock_urlopen: MagicMock) -> None:
        """copy_model returns None on success (no JSON body)."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b""
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.copy_model("qwen3:8b", "my-qwen3:8b")
        self.assertIsNone(result)

    def test_copy_model_validates_source_name(self) -> None:
        """copy_model rejects invalid source model names."""
        with self.assertRaises(ValueError):
            self.client.copy_model("../../../etc/passwd", "valid-name")

    def test_copy_model_validates_destination_name(self) -> None:
        """copy_model rejects invalid destination model names."""
        with self.assertRaises(ValueError):
            self.client.copy_model("qwen3:8b", "../../../etc/passwd")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_copy_model_connection_error(self, mock_urlopen: MagicMock) -> None:
        """copy_model raises OllamaConnectionError on connection failure."""
        mock_urlopen.side_effect = urllib.error.URLError(
            ConnectionRefusedError("Connection refused")
        )

        with self.assertRaises(OllamaConnectionError):
            self.client.copy_model("qwen3:8b", "my-qwen3:8b")


class TestOllamaClientCreateModel(unittest.TestCase):
    """Tests for the create_model() streaming method."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    def _make_stream_response(self, lines: list[bytes]) -> MagicMock:
        """Create a mock response that iterates over NDJSON lines."""
        mock_resp = MagicMock()
        mock_resp.__iter__ = MagicMock(return_value=iter(lines))
        mock_resp.close = MagicMock()
        return mock_resp

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_create_model_with_from_model(self, mock_urlopen: MagicMock) -> None:
        """create_model with from_model sends 'from' key in payload."""
        lines = [
            json.dumps({"status": "using existing layer"}).encode() + b"\n",
            json.dumps({"status": "success"}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        chunks = list(self.client.create_model("my-model", from_model="qwen3:8b"))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[-1]["status"], "success")

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "http://localhost:11434/api/create")
        self.assertEqual(req.method, "POST")

        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "my-model")
        self.assertEqual(body["from"], "qwen3:8b")
        self.assertTrue(body["stream"])
        self.assertNotIn("modelfile", body)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_create_model_with_modelfile(self, mock_urlopen: MagicMock) -> None:
        """create_model with modelfile sends modelfile content in payload."""
        lines = [
            json.dumps({"status": "success"}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        modelfile_content = "FROM qwen3:8b\nSYSTEM You are a coding assistant."
        chunks = list(self.client.create_model("my-coder", modelfile=modelfile_content))

        self.assertEqual(len(chunks), 1)

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "my-coder")
        self.assertEqual(body["modelfile"], modelfile_content)
        self.assertNotIn("from", body)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_create_model_with_both_from_and_modelfile(
        self, mock_urlopen: MagicMock
    ) -> None:
        """create_model with both from_model and modelfile sends both."""
        lines = [
            json.dumps({"status": "success"}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        list(self.client.create_model(
            "my-model",
            from_model="qwen3:8b",
            modelfile="SYSTEM You are helpful.",
        ))

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["from"], "qwen3:8b")
        self.assertEqual(body["modelfile"], "SYSTEM You are helpful.")

    def test_create_model_requires_from_or_modelfile(self) -> None:
        """create_model raises ValueError when neither from_model nor modelfile given."""
        with self.assertRaises(ValueError) as ctx:
            list(self.client.create_model("my-model"))
        self.assertIn("from_model", str(ctx.exception))
        self.assertIn("modelfile", str(ctx.exception))

    def test_create_model_validates_name(self) -> None:
        """create_model rejects invalid model names."""
        with self.assertRaises(ValueError):
            list(self.client.create_model(
                "../../../etc/passwd",
                from_model="qwen3:8b",
            ))

    def test_create_model_validates_from_model(self) -> None:
        """create_model rejects invalid from_model names."""
        with self.assertRaises(ValueError):
            list(self.client.create_model(
                "my-model",
                from_model="../../../etc/passwd",
            ))

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_create_model_streams_progress(self, mock_urlopen: MagicMock) -> None:
        """create_model yields progress update chunks."""
        lines = [
            json.dumps({"status": "reading model metadata"}).encode() + b"\n",
            json.dumps({"status": "creating system layer"}).encode() + b"\n",
            json.dumps({"status": "writing manifest"}).encode() + b"\n",
            json.dumps({"status": "success"}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        chunks = list(self.client.create_model("my-model", from_model="qwen3:8b"))
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0]["status"], "reading model metadata")
        self.assertEqual(chunks[-1]["status"], "success")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_create_model_mid_stream_error(self, mock_urlopen: MagicMock) -> None:
        """create_model raises OllamaStreamError on mid-stream error."""
        lines = [
            json.dumps({"status": "reading model metadata"}).encode() + b"\n",
            json.dumps({"error": "model not found"}).encode() + b"\n",
        ]
        mock_resp = self._make_stream_response(lines)
        mock_urlopen.return_value = mock_resp

        with self.assertRaises(OllamaStreamError) as ctx:
            list(self.client.create_model("my-model", from_model="qwen3:8b"))
        self.assertIn("model not found", str(ctx.exception))
        mock_resp.close.assert_called_once()


class TestOllamaClientThinkParam(unittest.TestCase):
    """Tests for the think parameter in chat() and chat_stream()."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    def _make_stream_response(self, lines: list[bytes]) -> MagicMock:
        """Create a mock response that iterates over NDJSON lines."""
        mock_resp = MagicMock()
        mock_resp.__iter__ = MagicMock(return_value=iter(lines))
        mock_resp.close = MagicMock()
        return mock_resp

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_think_true_included_in_payload(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat() with think=True includes think in the request payload."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.client.chat(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
            think=True,
        )

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertTrue(body["think"])

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_think_false_included_in_payload(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat() with think=False includes think=False in the payload."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.client.chat(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
            think=False,
        )

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertFalse(body["think"])

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_think_none_excluded_from_payload(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat() with think=None (default) omits think from the payload."""
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
        self.assertNotIn("think", body)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_stream_think_true_included_in_payload(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat_stream() with think=True includes think in the payload."""
        lines = [
            json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        list(self.client.chat_stream(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
            think=True,
        ))

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertTrue(body["think"])

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_stream_think_false_included_in_payload(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat_stream() with think=False includes think=False in the payload."""
        lines = [
            json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        list(self.client.chat_stream(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
            think=False,
        ))

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertFalse(body["think"])

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_stream_think_none_excluded_from_payload(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat_stream() with think=None (default) omits think from the payload."""
        lines = [
            json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        list(self.client.chat_stream(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
        ))

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertNotIn("think", body)


class TestOllamaClientKeepAliveParam(unittest.TestCase):
    """Tests for the keep_alive parameter in chat() and chat_stream()."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    def _make_stream_response(self, lines: list[bytes]) -> MagicMock:
        """Create a mock response that iterates over NDJSON lines."""
        mock_resp = MagicMock()
        mock_resp.__iter__ = MagicMock(return_value=iter(lines))
        mock_resp.close = MagicMock()
        return mock_resp

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_keep_alive_string_included(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat() with keep_alive='5m' includes keep_alive in the payload."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.client.chat(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
            keep_alive="5m",
        )

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["keep_alive"], "5m")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_keep_alive_int_included(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat() with keep_alive=300 (seconds) includes keep_alive in the payload."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.client.chat(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
            keep_alive=300,
        )

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["keep_alive"], 300)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_keep_alive_none_excluded(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat() with keep_alive=None (default) omits keep_alive from the payload."""
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
        self.assertNotIn("keep_alive", body)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_stream_keep_alive_string_included(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat_stream() with keep_alive='10m' includes keep_alive in the payload."""
        lines = [
            json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        list(self.client.chat_stream(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
            keep_alive="10m",
        ))

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["keep_alive"], "10m")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_stream_keep_alive_int_included(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat_stream() with keep_alive=600 includes keep_alive in the payload."""
        lines = [
            json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        list(self.client.chat_stream(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
            keep_alive=600,
        ))

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["keep_alive"], 600)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_stream_keep_alive_none_excluded(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat_stream() with keep_alive=None (default) omits keep_alive."""
        lines = [
            json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        list(self.client.chat_stream(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
        ))

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertNotIn("keep_alive", body)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_think_and_keep_alive_combined(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat() with both think and keep_alive includes both in the payload."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.client.chat(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
            think=True,
            keep_alive="5m",
        )

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertTrue(body["think"])
        self.assertEqual(body["keep_alive"], "5m")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_chat_stream_think_and_keep_alive_combined(
        self, mock_urlopen: MagicMock
    ) -> None:
        """chat_stream() with both think and keep_alive includes both."""
        lines = [
            json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        list(self.client.chat_stream(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
            think=True,
            keep_alive="10m",
        ))

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertTrue(body["think"])
        self.assertEqual(body["keep_alive"], "10m")


class TestOllamaClientGenerateStream(unittest.TestCase):
    """Tests for the generate_stream() method."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    def _make_stream_response(self, lines: list[bytes]) -> MagicMock:
        """Create a mock response that iterates over NDJSON lines."""
        mock_resp = MagicMock()
        mock_resp.__iter__ = MagicMock(return_value=iter(lines))
        mock_resp.close = MagicMock()
        return mock_resp

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_generate_stream_request_format(
        self, mock_urlopen: MagicMock
    ) -> None:
        """generate_stream sends POST to /api/generate with stream=true."""
        lines = [
            json.dumps({"response": "Hello", "done": False}).encode() + b"\n",
            json.dumps({"response": "", "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        list(self.client.generate_stream("qwen3:8b", "Say hello"))

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "http://localhost:11434/api/generate")
        self.assertEqual(req.method, "POST")
        self.assertEqual(req.get_header("Content-type"), "application/json")

        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen3:8b")
        self.assertEqual(body["prompt"], "Say hello")
        self.assertTrue(body["stream"])

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_generate_stream_yields_chunks(
        self, mock_urlopen: MagicMock
    ) -> None:
        """generate_stream yields parsed chunks from the NDJSON stream."""
        lines = [
            json.dumps({"response": "Hello", "done": False}).encode() + b"\n",
            json.dumps({"response": " world", "done": False}).encode() + b"\n",
            json.dumps({"response": "", "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        chunks = list(self.client.generate_stream("qwen3:8b", "Say hello"))
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["response"], "Hello")
        self.assertEqual(chunks[1]["response"], " world")
        self.assertTrue(chunks[2]["done"])

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_generate_stream_with_kwargs(
        self, mock_urlopen: MagicMock
    ) -> None:
        """generate_stream forwards extra kwargs to the payload."""
        lines = [
            json.dumps({"response": "", "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        list(self.client.generate_stream(
            "qwen3:8b",
            "Think deeply",
            think=True,
            keep_alive="5m",
        ))

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertTrue(body["think"])
        self.assertEqual(body["keep_alive"], "5m")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_generate_stream_without_kwargs_excludes_extra_keys(
        self, mock_urlopen: MagicMock
    ) -> None:
        """generate_stream without kwargs only includes model, prompt, stream."""
        lines = [
            json.dumps({"response": "", "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        list(self.client.generate_stream("qwen3:8b", "Hello"))

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(set(body.keys()), {"model", "prompt", "stream"})

    def test_generate_stream_validates_model_name(self) -> None:
        """generate_stream rejects invalid model names."""
        with self.assertRaises(ValueError):
            list(self.client.generate_stream("../../../etc/passwd", "test"))

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_generate_stream_connection_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        """generate_stream raises OllamaConnectionError on connection failure."""
        mock_urlopen.side_effect = urllib.error.URLError(
            ConnectionRefusedError("Connection refused")
        )

        with self.assertRaises(OllamaConnectionError):
            list(self.client.generate_stream("qwen3:8b", "test"))

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_generate_stream_timeout(
        self, mock_urlopen: MagicMock
    ) -> None:
        """generate_stream raises OllamaConnectionError on timeout."""
        mock_urlopen.side_effect = socket.timeout("timed out")

        with self.assertRaises(OllamaConnectionError):
            list(self.client.generate_stream("qwen3:8b", "test"))

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_generate_stream_mid_stream_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        """generate_stream raises OllamaStreamError on mid-stream error."""
        lines = [
            json.dumps({"response": "Hello", "done": False}).encode() + b"\n",
            json.dumps({"error": "out of memory"}).encode() + b"\n",
        ]
        mock_resp = self._make_stream_response(lines)
        mock_urlopen.return_value = mock_resp

        chunks = []
        with self.assertRaises(OllamaStreamError) as ctx:
            for chunk in self.client.generate_stream("qwen3:8b", "test"):
                chunks.append(chunk)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["response"], "Hello")
        self.assertIn("out of memory", str(ctx.exception))
        mock_resp.close.assert_called_once()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_generate_stream_uses_stream_timeout(
        self, mock_urlopen: MagicMock
    ) -> None:
        """generate_stream uses _STREAM_TIMEOUT for the request."""
        lines = [
            json.dumps({"response": "", "done": True}).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        list(self.client.generate_stream("qwen3:8b", "test"))

        call_kwargs = mock_urlopen.call_args
        self.assertEqual(call_kwargs[1]["timeout"], _STREAM_TIMEOUT)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_generate_stream_response_closed_after_iteration(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Response object is closed after generate_stream is consumed."""
        lines = [
            json.dumps({"response": "done", "done": True}).encode() + b"\n",
        ]
        mock_resp = self._make_stream_response(lines)
        mock_urlopen.return_value = mock_resp

        list(self.client.generate_stream("qwen3:8b", "test"))
        mock_resp.close.assert_called_once()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_generate_stream_http_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        """generate_stream raises OllamaRequestError on HTTP error."""
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "http://localhost:11434/api/generate",
            404,
            "Not Found",
            {},
            None,
        )

        with self.assertRaises(OllamaRequestError):
            list(self.client.generate_stream("qwen3:8b", "test"))


class TestOllamaClientThinkingFieldHandling(unittest.TestCase):
    """Tests for handling the 'thinking' field in streaming response chunks."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    def _make_stream_response(self, lines: list[bytes]) -> MagicMock:
        """Create a mock response that iterates over NDJSON lines."""
        mock_resp = MagicMock()
        mock_resp.__iter__ = MagicMock(return_value=iter(lines))
        mock_resp.close = MagicMock()
        return mock_resp

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_thinking_field_in_stream_chunks(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Streaming chunks with 'thinking' field are yielded correctly."""
        lines = [
            json.dumps({
                "message": {"role": "assistant", "content": "", "thinking": "Let me think..."},
                "done": False,
            }).encode() + b"\n",
            json.dumps({
                "message": {"role": "assistant", "content": "", "thinking": "I need to consider..."},
                "done": False,
            }).encode() + b"\n",
            json.dumps({
                "message": {"role": "assistant", "content": "Here is my answer."},
                "done": False,
            }).encode() + b"\n",
            json.dumps({
                "message": {"role": "assistant", "content": ""},
                "done": True,
            }).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        chunks = list(self.client.chat_stream(
            "qwen3:8b",
            [{"role": "user", "content": "Think about this"}],
            think=True,
        ))

        self.assertEqual(len(chunks), 4)
        # Thinking chunks have the thinking field.
        self.assertEqual(chunks[0]["message"]["thinking"], "Let me think...")
        self.assertEqual(chunks[1]["message"]["thinking"], "I need to consider...")
        # Content chunks have normal content.
        self.assertEqual(chunks[2]["message"]["content"], "Here is my answer.")
        # Final done chunk.
        self.assertTrue(chunks[3]["done"])

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_thinking_field_absent_when_think_not_set(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Chunks without thinking field work normally (think not enabled)."""
        lines = [
            json.dumps({
                "message": {"role": "assistant", "content": "Hello"},
                "done": False,
            }).encode() + b"\n",
            json.dumps({
                "message": {"role": "assistant", "content": ""},
                "done": True,
            }).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        chunks = list(self.client.chat_stream(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
        ))

        self.assertEqual(len(chunks), 2)
        self.assertNotIn("thinking", chunks[0]["message"])
        self.assertEqual(chunks[0]["message"]["content"], "Hello")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_thinking_field_mixed_with_content(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Chunks can have both thinking and content fields simultaneously."""
        lines = [
            json.dumps({
                "message": {
                    "role": "assistant",
                    "content": "partial",
                    "thinking": "reasoning step",
                },
                "done": False,
            }).encode() + b"\n",
            json.dumps({
                "message": {"role": "assistant", "content": " answer"},
                "done": True,
            }).encode() + b"\n",
        ]
        mock_urlopen.return_value = self._make_stream_response(lines)

        chunks = list(self.client.chat_stream(
            "qwen3:8b",
            [{"role": "user", "content": "Think"}],
            think=True,
        ))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["message"]["thinking"], "reasoning step")
        self.assertEqual(chunks[0]["message"]["content"], "partial")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_non_streaming_chat_with_thinking_response(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Non-streaming chat() returns thinking field when present."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "thinking": "Let me calculate step by step...",
            },
            "done": True,
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = self.client.chat(
            "qwen3:8b",
            [{"role": "user", "content": "What is 6*7?"}],
            think=True,
        )

        self.assertEqual(result["message"]["content"], "The answer is 42.")
        self.assertEqual(
            result["message"]["thinking"],
            "Let me calculate step by step...",
        )


class TestOllamaClientRequestNoContent(unittest.TestCase):
    """Tests for the _request_no_content() helper."""

    def setUp(self) -> None:
        self.client = OllamaClient()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_request_no_content_does_not_parse_json(
        self, mock_urlopen: MagicMock
    ) -> None:
        """_request_no_content consumes body without attempting JSON parse."""
        mock_resp = MagicMock()
        # Even though this is not valid JSON, it should not raise.
        mock_resp.read.return_value = b"OK"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        # Should not raise any exception.
        self.client._request_no_content("POST", "/api/copy", data={"source": "a", "destination": "b"})
        mock_resp.read.assert_called_once()

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_request_no_content_empty_body(self, mock_urlopen: MagicMock) -> None:
        """_request_no_content handles empty response body."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b""
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.client._request_no_content("DELETE", "/api/delete", data={"model": "test"})

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_request_no_content_connection_error(self, mock_urlopen: MagicMock) -> None:
        """_request_no_content raises OllamaConnectionError on connection failure."""
        mock_urlopen.side_effect = urllib.error.URLError(
            ConnectionRefusedError("Connection refused")
        )

        with self.assertRaises(OllamaConnectionError):
            self.client._request_no_content("DELETE", "/api/delete", data={"model": "x"})

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_request_no_content_timeout(self, mock_urlopen: MagicMock) -> None:
        """_request_no_content raises OllamaConnectionError on timeout."""
        mock_urlopen.side_effect = socket.timeout("timed out")

        with self.assertRaises(OllamaConnectionError):
            self.client._request_no_content("DELETE", "/api/delete", data={"model": "x"})

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_request_no_content_sends_json_body(
        self, mock_urlopen: MagicMock
    ) -> None:
        """_request_no_content includes Content-Type and JSON body."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b""
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.client._request_no_content(
            "DELETE", "/api/delete", data={"model": "phi4-mini"}
        )

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("Content-type"), "application/json")
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "phi4-mini")

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_request_no_content_without_data(self, mock_urlopen: MagicMock) -> None:
        """_request_no_content works without a JSON body."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b""
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        self.client._request_no_content("DELETE", "/api/delete")

        req = mock_urlopen.call_args[0][0]
        self.assertIsNone(req.data)


class TestOllamaClientParameterPassthrough(unittest.TestCase):
    """Tests for chat/chat_stream parameter passthrough to payloads."""

    def setUp(self) -> None:
        self.client = OllamaClient()
        self.messages = [{"role": "user", "content": "Hi"}]

    @patch.object(OllamaClient, "_stream_request")
    def test_chat_stream_options_in_payload(
        self, mock_stream: MagicMock
    ) -> None:
        """chat_stream passes options dict into the payload."""
        mock_stream.return_value = iter([])
        opts = {"num_ctx": 16384, "temperature": 0.5}

        list(self.client.chat_stream("qwen3:8b", self.messages, options=opts))

        call_args = mock_stream.call_args
        payload = call_args[0][1]  # second positional arg is data dict
        self.assertEqual(payload["options"], opts)

    @patch.object(OllamaClient, "_request")
    def test_chat_options_in_payload(self, mock_req: MagicMock) -> None:
        """chat passes options dict into the payload."""
        mock_req.return_value = {
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
        }
        opts = {"num_ctx": 4096, "top_p": 0.9}

        self.client.chat("qwen3:8b", self.messages, options=opts)

        call_kwargs = mock_req.call_args
        payload = call_kwargs[1]["data"]  # keyword arg 'data'
        self.assertEqual(payload["options"], opts)

    @patch.object(OllamaClient, "_request")
    def test_default_num_ctx(self, mock_req: MagicMock) -> None:
        """When no options are passed, payload has options.num_ctx=8192."""
        mock_req.return_value = {
            "message": {"role": "assistant", "content": "Hi"},
            "done": True,
        }

        self.client.chat("qwen3:8b", self.messages)

        payload = mock_req.call_args[1]["data"]
        self.assertIn("options", payload)
        self.assertEqual(payload["options"]["num_ctx"], 8192)

    @patch.object(OllamaClient, "_stream_request")
    def test_default_num_ctx_stream(self, mock_stream: MagicMock) -> None:
        """When no options are passed to chat_stream, default num_ctx=8192."""
        mock_stream.return_value = iter([])

        list(self.client.chat_stream("qwen3:8b", self.messages))

        payload = mock_stream.call_args[0][1]
        self.assertIn("options", payload)
        self.assertEqual(payload["options"]["num_ctx"], 8192)

    @patch.object(OllamaClient, "_request")
    def test_think_top_level(self, mock_req: MagicMock) -> None:
        """think=True is sent as a top-level key, not inside options."""
        mock_req.return_value = {
            "message": {"role": "assistant", "content": ""},
            "done": True,
        }

        self.client.chat("qwen3:8b", self.messages, think=True)

        payload = mock_req.call_args[1]["data"]
        self.assertTrue(payload["think"])
        # Must not be nested inside options.
        self.assertNotIn("think", payload.get("options", {}))

    @patch.object(OllamaClient, "_request")
    def test_format_top_level(self, mock_req: MagicMock) -> None:
        """format is sent as a top-level key in the payload."""
        mock_req.return_value = {
            "message": {"role": "assistant", "content": "{}"},
            "done": True,
        }

        self.client.chat("qwen3:8b", self.messages, format="json")

        payload = mock_req.call_args[1]["data"]
        self.assertEqual(payload["format"], "json")
        self.assertNotIn("format", payload.get("options", {}))

    @patch.object(OllamaClient, "_request")
    def test_format_json_schema(self, mock_req: MagicMock) -> None:
        """format accepts a JSON schema dict as well as a string."""
        mock_req.return_value = {
            "message": {"role": "assistant", "content": "{}"},
            "done": True,
        }
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

        self.client.chat("qwen3:8b", self.messages, format=schema)

        payload = mock_req.call_args[1]["data"]
        self.assertEqual(payload["format"], schema)

    @patch.object(OllamaClient, "_request")
    def test_keep_alive_top_level(self, mock_req: MagicMock) -> None:
        """keep_alive is sent as a top-level key in the payload."""
        mock_req.return_value = {
            "message": {"role": "assistant", "content": "Hi"},
            "done": True,
        }

        self.client.chat("qwen3:8b", self.messages, keep_alive="5m")

        payload = mock_req.call_args[1]["data"]
        self.assertEqual(payload["keep_alive"], "5m")
        self.assertNotIn("keep_alive", payload.get("options", {}))

    @patch.object(OllamaClient, "_request")
    def test_keep_alive_integer(self, mock_req: MagicMock) -> None:
        """keep_alive accepts integer seconds."""
        mock_req.return_value = {
            "message": {"role": "assistant", "content": "Hi"},
            "done": True,
        }

        self.client.chat("qwen3:8b", self.messages, keep_alive=300)

        payload = mock_req.call_args[1]["data"]
        self.assertEqual(payload["keep_alive"], 300)

    @patch.object(OllamaClient, "_request")
    def test_options_merge(self, mock_req: MagicMock) -> None:
        """User-provided options dict replaces defaults entirely."""
        mock_req.return_value = {
            "message": {"role": "assistant", "content": "Hi"},
            "done": True,
        }
        user_opts = {"num_ctx": 32768, "temperature": 0.3, "top_k": 40}

        self.client.chat("qwen3:8b", self.messages, options=user_opts)

        payload = mock_req.call_args[1]["data"]
        self.assertEqual(payload["options"], user_opts)
        # User-supplied options override the default num_ctx.
        self.assertEqual(payload["options"]["num_ctx"], 32768)

    @patch("local_cli.ollama_client.urllib.request.urlopen")
    def test_backward_compat_no_options(self, mock_urlopen: MagicMock) -> None:
        """Calling chat without new params still works (backward compat)."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
        }).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        # Old-style call: only model, messages, and tools.
        result = self.client.chat(
            "qwen3:8b", self.messages, tools=None,
        )

        self.assertEqual(result["message"]["content"], "Hello")
        # Payload should still be well-formed.
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen3:8b")
        self.assertFalse(body["stream"])
        self.assertNotIn("tools", body)
        self.assertNotIn("think", body)
        self.assertNotIn("format", body)
        self.assertNotIn("keep_alive", body)
        # Default options should still be present.
        self.assertIn("options", body)
        self.assertEqual(body["options"]["num_ctx"], 8192)

    @patch.object(OllamaClient, "_request")
    def test_options_none_omitted(self, mock_req: MagicMock) -> None:
        """None values for think, format, keep_alive are not sent in payload."""
        mock_req.return_value = {
            "message": {"role": "assistant", "content": "Hi"},
            "done": True,
        }

        self.client.chat(
            "qwen3:8b",
            self.messages,
            think=None,
            format=None,
            keep_alive=None,
        )

        payload = mock_req.call_args[1]["data"]
        self.assertNotIn("think", payload)
        self.assertNotIn("format", payload)
        self.assertNotIn("keep_alive", payload)

    @patch.object(OllamaClient, "_stream_request")
    def test_chat_stream_think_top_level(self, mock_stream: MagicMock) -> None:
        """chat_stream also sends think as a top-level key."""
        mock_stream.return_value = iter([])

        list(self.client.chat_stream("qwen3:8b", self.messages, think=True))

        payload = mock_stream.call_args[0][1]
        self.assertTrue(payload["think"])
        self.assertNotIn("think", payload.get("options", {}))

    @patch.object(OllamaClient, "_stream_request")
    def test_chat_stream_format_top_level(self, mock_stream: MagicMock) -> None:
        """chat_stream sends format as a top-level key."""
        mock_stream.return_value = iter([])

        list(self.client.chat_stream("qwen3:8b", self.messages, format="json"))

        payload = mock_stream.call_args[0][1]
        self.assertEqual(payload["format"], "json")

    @patch.object(OllamaClient, "_stream_request")
    def test_chat_stream_keep_alive_top_level(self, mock_stream: MagicMock) -> None:
        """chat_stream sends keep_alive as a top-level key."""
        mock_stream.return_value = iter([])

        list(self.client.chat_stream("qwen3:8b", self.messages, keep_alive="10m"))

        payload = mock_stream.call_args[0][1]
        self.assertEqual(payload["keep_alive"], "10m")

    @patch.object(OllamaClient, "_stream_request")
    def test_chat_stream_options_none_omitted(self, mock_stream: MagicMock) -> None:
        """chat_stream omits None values for think, format, keep_alive."""
        mock_stream.return_value = iter([])

        list(self.client.chat_stream(
            "qwen3:8b",
            self.messages,
            think=None,
            format=None,
            keep_alive=None,
        ))

        payload = mock_stream.call_args[0][1]
        self.assertNotIn("think", payload)
        self.assertNotIn("format", payload)
        self.assertNotIn("keep_alive", payload)


if __name__ == "__main__":
    unittest.main()
