"""Tests for local_cli.providers.claude_provider module.

Verifies that :class:`ClaudeProvider` correctly implements the
:class:`LLMProvider` interface, handles authentication, builds correct
request bodies, parses Claude API responses, handles SSE streaming with
tool use, and retries rate-limited requests.

All tests mock ``urllib.request.urlopen`` to avoid requiring a real
Anthropic API key or network access.
"""

import io
import json
import unittest
import urllib.error
from typing import Any, Generator
from unittest.mock import MagicMock, patch

from local_cli.providers.base import (
    LLMProvider,
    ProviderConnectionError,
    ProviderRequestError,
    ProviderStreamError,
)
from local_cli.providers.claude_provider import (
    ClaudeConnectionError,
    ClaudeProvider,
    ClaudeRateLimitError,
    ClaudeRequestError,
    ClaudeStreamError,
    _API_VERSION,
    _BASE_URL,
    _DEFAULT_MAX_TOKENS,
    _MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTool:
    """Minimal stand-in for a Tool instance used by format_tools tests."""

    def __init__(self, name: str, description: str, parameters: dict) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return self._parameters

    def to_claude_tool(self) -> dict:
        return {
            "name": self._name,
            "description": self._description,
            "input_schema": self._parameters,
        }


def _make_response(body: dict[str, Any], status: int = 200) -> MagicMock:
    """Create a mock HTTP response with a JSON body."""
    raw = json.dumps(body).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw
    mock_resp.status = status
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_sse_response(events: list[str]) -> MagicMock:
    """Create a mock HTTP response that yields SSE lines.

    Args:
        events: A list of raw SSE lines (including ``event:``, ``data:``,
            and blank-line separators).
    """
    lines = [line.encode("utf-8") + b"\n" for line in events]
    mock_resp = MagicMock()
    mock_resp.__iter__ = MagicMock(return_value=iter(lines))
    mock_resp.close = MagicMock()
    return mock_resp


def _claude_chat_response(
    content: str = "Hello!",
    tool_use: list[dict[str, Any]] | None = None,
    stop_reason: str = "end_turn",
) -> dict[str, Any]:
    """Build a Claude Messages API non-streaming response."""
    blocks: list[dict[str, Any]] = []
    if content:
        blocks.append({"type": "text", "text": content})
    if tool_use:
        blocks.extend(tool_use)

    return {
        "id": "msg_test_123",
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": "claude-sonnet-4-5",
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 10, "output_tokens": 25},
    }


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestClaudeProviderInit(unittest.TestCase):
    """Tests for ClaudeProvider construction and authentication."""

    def test_explicit_api_key(self) -> None:
        """Provider can be created with an explicit API key."""
        provider = ClaudeProvider(api_key="sk-ant-test-key")
        self.assertIsInstance(provider, ClaudeProvider)

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-env-key"})
    def test_env_var_api_key(self) -> None:
        """Provider reads API key from ANTHROPIC_API_KEY env var."""
        provider = ClaudeProvider()
        self.assertIsInstance(provider, ClaudeProvider)

    @patch.dict("os.environ", {}, clear=True)
    def test_no_api_key_raises(self) -> None:
        """Provider raises ValueError when no API key is available."""
        # Ensure no key in env
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with self.assertRaises(ValueError) as ctx:
            ClaudeProvider()
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    def test_explicit_key_takes_precedence(self) -> None:
        """Explicit api_key overrides environment variable."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            provider = ClaudeProvider(api_key="explicit-key")
            # The explicit key should be used (we verify via headers).
            headers = provider._headers()
            self.assertEqual(headers["x-api-key"], "explicit-key")

    def test_custom_max_tokens(self) -> None:
        """Provider accepts custom default_max_tokens."""
        provider = ClaudeProvider(api_key="test-key", default_max_tokens=8192)
        self.assertEqual(provider._default_max_tokens, 8192)

    def test_custom_base_url(self) -> None:
        """Provider accepts a custom base_url."""
        provider = ClaudeProvider(
            api_key="test-key",
            base_url="https://custom.api.example.com/",
        )
        self.assertEqual(provider._base_url, "https://custom.api.example.com")

    def test_implements_llm_provider(self) -> None:
        """ClaudeProvider is an instance of LLMProvider."""
        provider = ClaudeProvider(api_key="test-key")
        self.assertIsInstance(provider, LLMProvider)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestClaudeProviderProperties(unittest.TestCase):
    """Tests for ClaudeProvider properties."""

    def setUp(self) -> None:
        self.provider = ClaudeProvider(api_key="test-key")

    def test_name_is_claude(self) -> None:
        """Provider name is 'claude'."""
        self.assertEqual(self.provider.name, "claude")

    def test_name_is_string(self) -> None:
        """Provider name is a string."""
        self.assertIsInstance(self.provider.name, str)


# ---------------------------------------------------------------------------
# Headers tests
# ---------------------------------------------------------------------------


class TestClaudeProviderHeaders(unittest.TestCase):
    """Tests for ClaudeProvider._headers()."""

    def setUp(self) -> None:
        self.provider = ClaudeProvider(api_key="sk-ant-test-123")

    def test_has_api_key_header(self) -> None:
        """Headers include x-api-key."""
        headers = self.provider._headers()
        self.assertEqual(headers["x-api-key"], "sk-ant-test-123")

    def test_has_version_header(self) -> None:
        """Headers include anthropic-version."""
        headers = self.provider._headers()
        self.assertEqual(headers["anthropic-version"], _API_VERSION)

    def test_has_content_type(self) -> None:
        """Headers include content-type: application/json."""
        headers = self.provider._headers()
        self.assertEqual(headers["content-type"], "application/json")

    def test_exactly_three_headers(self) -> None:
        """Headers dict has exactly 3 entries."""
        headers = self.provider._headers()
        self.assertEqual(len(headers), 3)


# ---------------------------------------------------------------------------
# Request body building tests
# ---------------------------------------------------------------------------


class TestClaudeProviderBuildRequestBody(unittest.TestCase):
    """Tests for ClaudeProvider._build_request_body()."""

    def setUp(self) -> None:
        self.provider = ClaudeProvider(api_key="test-key")

    def test_basic_body(self) -> None:
        """Request body contains model, max_tokens, and messages."""
        body = self.provider._build_request_body(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertEqual(body["model"], "claude-sonnet-4-5")
        self.assertEqual(body["max_tokens"], _DEFAULT_MAX_TOKENS)
        self.assertIn("messages", body)

    def test_custom_max_tokens(self) -> None:
        """max_tokens parameter overrides the default."""
        body = self.provider._build_request_body(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=2048,
        )
        self.assertEqual(body["max_tokens"], 2048)

    def test_system_message_extracted(self) -> None:
        """System messages are extracted to a separate 'system' field."""
        body = self.provider._build_request_body(
            model="claude-sonnet-4-5",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        )
        self.assertEqual(body["system"], "You are helpful.")
        # System message should not appear in messages list.
        roles = [m["role"] for m in body["messages"]]
        self.assertNotIn("system", roles)

    def test_no_system_message(self) -> None:
        """When there are no system messages, 'system' key is absent."""
        body = self.provider._build_request_body(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertNotIn("system", body)

    def test_tools_included(self) -> None:
        """Tools are included when provided."""
        tools = [{"name": "bash", "description": "Run commands", "input_schema": {}}]
        body = self.provider._build_request_body(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "list files"}],
            tools=tools,
        )
        self.assertEqual(body["tools"], tools)

    def test_no_tools(self) -> None:
        """When tools is None, 'tools' key is absent."""
        body = self.provider._build_request_body(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            tools=None,
        )
        self.assertNotIn("tools", body)

    def test_stream_flag(self) -> None:
        """stream=True adds 'stream' key to body."""
        body = self.provider._build_request_body(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        self.assertTrue(body["stream"])

    def test_no_stream_flag(self) -> None:
        """stream=False does not add 'stream' key to body."""
        body = self.provider._build_request_body(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )
        self.assertNotIn("stream", body)

    def test_messages_converted_to_claude_format(self) -> None:
        """Messages are converted to Claude's content-block format."""
        body = self.provider._build_request_body(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
        )
        # Claude format: content is a list of blocks.
        msg = body["messages"][0]
        self.assertIsInstance(msg["content"], list)
        self.assertEqual(msg["content"][0]["type"], "text")
        self.assertEqual(msg["content"][0]["text"], "Hello")

    def test_tool_result_messages_converted(self) -> None:
        """Tool result messages are converted to Claude format."""
        body = self.provider._build_request_body(
            model="claude-sonnet-4-5",
            messages=[
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {"name": "bash", "arguments": {"command": "ls"}},
                            "id": "toolu_123",
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_name": "bash",
                    "tool_call_id": "toolu_123",
                    "content": "file1.py\nfile2.py",
                },
            ],
        )
        # Tool result should become a user message with tool_result block.
        messages = body["messages"]
        tool_result_msg = messages[-1]
        self.assertEqual(tool_result_msg["role"], "user")
        self.assertEqual(tool_result_msg["content"][0]["type"], "tool_result")
        self.assertEqual(
            tool_result_msg["content"][0]["tool_use_id"], "toolu_123"
        )


# ---------------------------------------------------------------------------
# chat() tests
# ---------------------------------------------------------------------------


class TestClaudeProviderChat(unittest.TestCase):
    """Tests for ClaudeProvider.chat() non-streaming completion."""

    def setUp(self) -> None:
        self.provider = ClaudeProvider(api_key="test-key")

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_basic_chat(self, mock_urlopen: MagicMock) -> None:
        """chat() sends request and returns normalized response."""
        mock_urlopen.return_value = _make_response(
            _claude_chat_response("Hello!")
        )

        result = self.provider.chat(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
        )

        self.assertIn("message", result)
        self.assertEqual(result["message"]["role"], "assistant")
        self.assertEqual(result["message"]["content"], "Hello!")

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_chat_with_tool_use(self, mock_urlopen: MagicMock) -> None:
        """chat() handles tool_use blocks in the response."""
        response = _claude_chat_response(
            content="Let me run that.",
            tool_use=[
                {
                    "type": "tool_use",
                    "id": "toolu_abc123",
                    "name": "bash",
                    "input": {"command": "ls -la"},
                }
            ],
            stop_reason="tool_use",
        )
        mock_urlopen.return_value = _make_response(response)

        result = self.provider.chat(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "list files"}],
        )

        self.assertIn("tool_calls", result["message"])
        tool_call = result["message"]["tool_calls"][0]
        self.assertEqual(tool_call["function"]["name"], "bash")
        self.assertEqual(tool_call["function"]["arguments"]["command"], "ls -la")
        self.assertEqual(tool_call["id"], "toolu_abc123")

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_chat_passes_tools(self, mock_urlopen: MagicMock) -> None:
        """chat() includes tools in the request body."""
        mock_urlopen.return_value = _make_response(
            _claude_chat_response("OK")
        )

        tools = [{"name": "bash", "description": "Run commands", "input_schema": {}}]
        self.provider.chat(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
        )

        # Verify the request body includes tools.
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertIn("tools", body)
        self.assertEqual(body["tools"], tools)

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_chat_max_tokens(self, mock_urlopen: MagicMock) -> None:
        """chat() forwards max_tokens to the request body."""
        mock_urlopen.return_value = _make_response(
            _claude_chat_response("OK")
        )

        self.provider.chat(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=2048,
        )

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["max_tokens"], 2048)

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_chat_default_max_tokens(self, mock_urlopen: MagicMock) -> None:
        """chat() uses default max_tokens when not specified."""
        mock_urlopen.return_value = _make_response(
            _claude_chat_response("OK")
        )

        self.provider.chat(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["max_tokens"], _DEFAULT_MAX_TOKENS)

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_chat_sends_to_correct_url(self, mock_urlopen: MagicMock) -> None:
        """chat() sends request to /v1/messages."""
        mock_urlopen.return_value = _make_response(
            _claude_chat_response("OK")
        )

        self.provider.chat(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        self.assertTrue(request.full_url.endswith("/v1/messages"))

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_chat_sets_correct_headers(self, mock_urlopen: MagicMock) -> None:
        """chat() sets correct authentication and content headers."""
        mock_urlopen.return_value = _make_response(
            _claude_chat_response("OK")
        )

        self.provider.chat(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
        )

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        self.assertEqual(request.get_header("X-api-key"), "test-key")
        self.assertEqual(request.get_header("Anthropic-version"), _API_VERSION)
        self.assertEqual(
            request.get_header("Content-type"), "application/json"
        )


# ---------------------------------------------------------------------------
# chat() error handling tests
# ---------------------------------------------------------------------------


class TestClaudeProviderChatErrors(unittest.TestCase):
    """Tests for ClaudeProvider.chat() error handling."""

    def setUp(self) -> None:
        self.provider = ClaudeProvider(api_key="test-key")

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_connection_error(self, mock_urlopen: MagicMock) -> None:
        """chat() raises ClaudeConnectionError on URLError."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with self.assertRaises(ClaudeConnectionError):
            self.provider.chat(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
            )

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_connection_error_is_provider_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        """ClaudeConnectionError is a ProviderConnectionError."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with self.assertRaises(ProviderConnectionError):
            self.provider.chat(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
            )

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_http_error(self, mock_urlopen: MagicMock) -> None:
        """chat() raises ClaudeRequestError on HTTP errors."""
        error = urllib.error.HTTPError(
            url="https://api.anthropic.com/v1/messages",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=io.BytesIO(b'{"error": "invalid request"}'),
        )
        mock_urlopen.side_effect = error

        with self.assertRaises(ClaudeRequestError):
            self.provider.chat(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
            )

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_http_error_is_provider_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        """ClaudeRequestError is a ProviderRequestError."""
        error = urllib.error.HTTPError(
            url="https://api.anthropic.com/v1/messages",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=io.BytesIO(b"server error"),
        )
        mock_urlopen.side_effect = error

        with self.assertRaises(ProviderRequestError):
            self.provider.chat(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
            )

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_timeout_error(self, mock_urlopen: MagicMock) -> None:
        """chat() raises ClaudeConnectionError on socket timeout."""
        import socket
        mock_urlopen.side_effect = socket.timeout("timed out")

        with self.assertRaises(ClaudeConnectionError):
            self.provider.chat(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
            )


# ---------------------------------------------------------------------------
# Rate limit retry tests
# ---------------------------------------------------------------------------


class TestClaudeProviderRateLimit(unittest.TestCase):
    """Tests for ClaudeProvider rate limit retry logic."""

    def setUp(self) -> None:
        self.provider = ClaudeProvider(api_key="test-key")

    @patch("local_cli.providers.claude_provider.time.sleep")
    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_retries_on_429(
        self, mock_urlopen: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """chat() retries on HTTP 429 and succeeds on a subsequent attempt."""
        rate_limit_error = urllib.error.HTTPError(
            url="https://api.anthropic.com/v1/messages",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=io.BytesIO(b"rate limited"),
        )

        # First call: 429, second call: success.
        mock_urlopen.side_effect = [
            rate_limit_error,
            _make_response(_claude_chat_response("After retry")),
        ]

        result = self.provider.chat(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
        )

        self.assertEqual(result["message"]["content"], "After retry")
        self.assertEqual(mock_urlopen.call_count, 2)
        mock_sleep.assert_called_once()  # One backoff sleep.

    @patch("local_cli.providers.claude_provider.time.sleep")
    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_exponential_backoff(
        self, mock_urlopen: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Retry uses exponential backoff delays (1, 2, 4 seconds)."""
        rate_limit_error = urllib.error.HTTPError(
            url="https://api.anthropic.com/v1/messages",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=io.BytesIO(b"rate limited"),
        )

        # All retries fail with 429.
        mock_urlopen.side_effect = [rate_limit_error] * _MAX_RETRIES

        with self.assertRaises(ClaudeRateLimitError):
            self.provider.chat(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
            )

        # Verify backoff delays: 1.0, 2.0, 4.0
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        self.assertEqual(delays, [1.0, 2.0, 4.0])

    @patch("local_cli.providers.claude_provider.time.sleep")
    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_rate_limit_exhausted(
        self, mock_urlopen: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """chat() raises ClaudeRateLimitError when all retries exhausted."""
        rate_limit_error = urllib.error.HTTPError(
            url="https://api.anthropic.com/v1/messages",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=io.BytesIO(b"rate limited"),
        )

        mock_urlopen.side_effect = [rate_limit_error] * _MAX_RETRIES

        with self.assertRaises(ClaudeRateLimitError):
            self.provider.chat(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
            )

        self.assertEqual(mock_urlopen.call_count, _MAX_RETRIES)

    @patch("local_cli.providers.claude_provider.time.sleep")
    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_rate_limit_is_request_error(
        self, mock_urlopen: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """ClaudeRateLimitError can be caught as ProviderRequestError."""
        rate_limit_error = urllib.error.HTTPError(
            url="https://api.anthropic.com/v1/messages",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=io.BytesIO(b"rate limited"),
        )

        mock_urlopen.side_effect = [rate_limit_error] * _MAX_RETRIES

        with self.assertRaises(ProviderRequestError):
            self.provider.chat(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
            )


# ---------------------------------------------------------------------------
# chat_stream() tests
# ---------------------------------------------------------------------------


class TestClaudeProviderChatStream(unittest.TestCase):
    """Tests for ClaudeProvider.chat_stream() SSE streaming."""

    def setUp(self) -> None:
        self.provider = ClaudeProvider(api_key="test-key")

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_basic_stream(self, mock_urlopen: MagicMock) -> None:
        """chat_stream() yields text delta chunks and a done chunk."""
        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
        mock_urlopen.return_value = _make_sse_response(sse_lines)

        chunks = list(self.provider.chat_stream(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
        ))

        # Should have text delta chunks and a final done chunk.
        text_chunks = [c for c in chunks if c["message"].get("content")]
        self.assertTrue(len(text_chunks) >= 1)

        # Last chunk should be done.
        done_chunks = [c for c in chunks if c.get("done")]
        self.assertTrue(len(done_chunks) >= 1)

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_returns_generator(self, mock_urlopen: MagicMock) -> None:
        """chat_stream() returns a generator."""
        sse_lines = [
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
        mock_urlopen.return_value = _make_sse_response(sse_lines)

        stream = self.provider.chat_stream(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
        )
        self.assertIsInstance(stream, Generator)

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_stream_with_tool_use(self, mock_urlopen: MagicMock) -> None:
        """chat_stream() accumulates tool use input_json_delta and emits tool_calls."""
        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_abc","name":"bash"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"com"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"mand\\": \\"ls\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":10}}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
        mock_urlopen.return_value = _make_sse_response(sse_lines)

        chunks = list(self.provider.chat_stream(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "list files"}],
        ))

        # Find the chunk with tool_calls (should be in a done chunk).
        tool_chunks = [
            c for c in chunks
            if c["message"].get("tool_calls")
        ]
        self.assertTrue(len(tool_chunks) >= 1)

        tool_call = tool_chunks[0]["message"]["tool_calls"][0]
        self.assertEqual(tool_call["function"]["name"], "bash")
        self.assertEqual(tool_call["function"]["arguments"]["command"], "ls")
        self.assertEqual(tool_call["id"], "toolu_abc")

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_stream_text_and_tool_use(self, mock_urlopen: MagicMock) -> None:
        """chat_stream() handles mixed text and tool_use content blocks."""
        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"role":"assistant"}}',
            "",
            # Text block
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Running command"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            # Tool use block
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_xyz","name":"bash"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\": \\"pwd\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":1}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":15}}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
        mock_urlopen.return_value = _make_sse_response(sse_lines)

        chunks = list(self.provider.chat_stream(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "where am I"}],
        ))

        # Should have a text delta chunk.
        text_chunks = [
            c for c in chunks if c["message"].get("content")
        ]
        self.assertTrue(len(text_chunks) >= 1)

        # Should have tool calls in a done chunk.
        tool_chunks = [
            c for c in chunks if c["message"].get("tool_calls")
        ]
        self.assertTrue(len(tool_chunks) >= 1)
        self.assertEqual(
            tool_chunks[0]["message"]["tool_calls"][0]["id"], "toolu_xyz"
        )

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_stream_sends_stream_flag(self, mock_urlopen: MagicMock) -> None:
        """chat_stream() sends stream: true in request body."""
        sse_lines = [
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
        mock_urlopen.return_value = _make_sse_response(sse_lines)

        list(self.provider.chat_stream(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
        ))

        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertTrue(body["stream"])

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_stream_closes_response(self, mock_urlopen: MagicMock) -> None:
        """chat_stream() closes the HTTP response when done."""
        sse_lines = [
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
        mock_resp = _make_sse_response(sse_lines)
        mock_urlopen.return_value = mock_resp

        list(self.provider.chat_stream(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
        ))

        mock_resp.close.assert_called_once()


# ---------------------------------------------------------------------------
# chat_stream() error handling tests
# ---------------------------------------------------------------------------


class TestClaudeProviderChatStreamErrors(unittest.TestCase):
    """Tests for ClaudeProvider.chat_stream() error handling."""

    def setUp(self) -> None:
        self.provider = ClaudeProvider(api_key="test-key")

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_stream_connection_error(self, mock_urlopen: MagicMock) -> None:
        """chat_stream() raises ClaudeConnectionError on URLError."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with self.assertRaises(ClaudeConnectionError):
            list(self.provider.chat_stream(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
            ))

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_stream_error_event(self, mock_urlopen: MagicMock) -> None:
        """chat_stream() raises ClaudeStreamError on error SSE events."""
        sse_lines = [
            "event: error",
            'data: {"type":"error","error":{"type":"overloaded_error","message":"Server overloaded"}}',
            "",
        ]
        mock_urlopen.return_value = _make_sse_response(sse_lines)

        with self.assertRaises(ClaudeStreamError):
            list(self.provider.chat_stream(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
            ))

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_stream_error_is_provider_error(
        self, mock_urlopen: MagicMock
    ) -> None:
        """ClaudeStreamError can be caught as ProviderStreamError."""
        sse_lines = [
            "event: error",
            'data: {"type":"error","error":{"message":"Server error"}}',
            "",
        ]
        mock_urlopen.return_value = _make_sse_response(sse_lines)

        with self.assertRaises(ProviderStreamError):
            list(self.provider.chat_stream(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
            ))

    @patch("local_cli.providers.claude_provider.time.sleep")
    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_stream_rate_limit_retry(
        self, mock_urlopen: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """chat_stream() retries on HTTP 429 during connection."""
        rate_limit_error = urllib.error.HTTPError(
            url="https://api.anthropic.com/v1/messages",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=io.BytesIO(b"rate limited"),
        )

        sse_lines = [
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]

        mock_urlopen.side_effect = [
            rate_limit_error,
            _make_sse_response(sse_lines),
        ]

        chunks = list(self.provider.chat_stream(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
        ))

        self.assertEqual(mock_urlopen.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_stream_http_error(self, mock_urlopen: MagicMock) -> None:
        """chat_stream() raises ClaudeRequestError on non-429 HTTP errors."""
        error = urllib.error.HTTPError(
            url="https://api.anthropic.com/v1/messages",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=io.BytesIO(b"bad request"),
        )
        mock_urlopen.side_effect = error

        with self.assertRaises(ClaudeRequestError):
            list(self.provider.chat_stream(
                model="claude-sonnet-4-5",
                messages=[{"role": "user", "content": "Hi"}],
            ))


# ---------------------------------------------------------------------------
# list_models() tests
# ---------------------------------------------------------------------------


class TestClaudeProviderListModels(unittest.TestCase):
    """Tests for ClaudeProvider.list_models()."""

    def setUp(self) -> None:
        self.provider = ClaudeProvider(api_key="test-key")

    def test_returns_list(self) -> None:
        """list_models() returns a list."""
        result = self.provider.list_models()
        self.assertIsInstance(result, list)

    def test_returns_non_empty(self) -> None:
        """list_models() returns a non-empty list."""
        result = self.provider.list_models()
        self.assertTrue(len(result) > 0)

    def test_items_have_name(self) -> None:
        """Each model dict has a 'name' key."""
        for model in self.provider.list_models():
            self.assertIn("name", model)

    def test_includes_known_models(self) -> None:
        """list_models() includes well-known Claude model names."""
        names = [m["name"] for m in self.provider.list_models()]
        self.assertIn("claude-sonnet-4-5", names)

    def test_returns_copy(self) -> None:
        """list_models() returns a fresh list each time (not a reference)."""
        result1 = self.provider.list_models()
        result2 = self.provider.list_models()
        self.assertIsNot(result1, result2)


# ---------------------------------------------------------------------------
# get_model_info() tests
# ---------------------------------------------------------------------------


class TestClaudeProviderGetModelInfo(unittest.TestCase):
    """Tests for ClaudeProvider.get_model_info()."""

    def setUp(self) -> None:
        self.provider = ClaudeProvider(api_key="test-key")

    def test_returns_dict(self) -> None:
        """get_model_info() returns a dict."""
        result = self.provider.get_model_info("claude-sonnet-4-5")
        self.assertIsInstance(result, dict)

    def test_result_has_name(self) -> None:
        """Result includes 'name' key matching the argument."""
        result = self.provider.get_model_info("claude-sonnet-4-5")
        self.assertEqual(result["name"], "claude-sonnet-4-5")

    def test_result_has_provider(self) -> None:
        """Result includes 'provider' key."""
        result = self.provider.get_model_info("claude-sonnet-4-5")
        self.assertEqual(result["provider"], "claude")

    def test_result_has_capabilities(self) -> None:
        """Result includes 'capabilities' list."""
        result = self.provider.get_model_info("claude-sonnet-4-5")
        self.assertIn("capabilities", result)
        self.assertIsInstance(result["capabilities"], list)

    def test_any_model_name(self) -> None:
        """get_model_info() works with any model name string."""
        result = self.provider.get_model_info("custom-model-name")
        self.assertEqual(result["name"], "custom-model-name")


# ---------------------------------------------------------------------------
# format_tools() tests
# ---------------------------------------------------------------------------


class TestClaudeProviderFormatTools(unittest.TestCase):
    """Tests for ClaudeProvider.format_tools()."""

    def setUp(self) -> None:
        self.provider = ClaudeProvider(api_key="test-key")

    def test_returns_list(self) -> None:
        """format_tools() returns a list."""
        result = self.provider.format_tools([])
        self.assertIsInstance(result, list)

    def test_empty_tools(self) -> None:
        """format_tools() with empty list returns empty list."""
        result = self.provider.format_tools([])
        self.assertEqual(result, [])

    def test_converts_single_tool(self) -> None:
        """format_tools() converts a single Tool to Claude format."""
        tool = _FakeTool(
            name="bash",
            description="Execute a shell command.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        )
        result = self.provider.format_tools([tool])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "bash")
        self.assertEqual(
            result[0]["description"],
            "Execute a shell command.",
        )
        self.assertIn("input_schema", result[0])
        self.assertIn("properties", result[0]["input_schema"])

    def test_converts_multiple_tools(self) -> None:
        """format_tools() converts multiple Tools."""
        tools = [
            _FakeTool(
                name="bash",
                description="Run shell commands.",
                parameters={"type": "object", "properties": {}},
            ),
            _FakeTool(
                name="read",
                description="Read a file.",
                parameters={"type": "object", "properties": {}},
            ),
        ]
        result = self.provider.format_tools(tools)
        self.assertEqual(len(result), 2)
        names = [t["name"] for t in result]
        self.assertEqual(names, ["bash", "read"])

    def test_claude_format_no_type_wrapper(self) -> None:
        """Claude format does not use {"type": "function"} wrapper."""
        tool = _FakeTool(
            name="edit",
            description="Edit a file.",
            parameters={"type": "object", "properties": {}},
        )
        result = self.provider.format_tools([tool])
        self.assertNotIn("type", result[0])
        self.assertNotIn("function", result[0])

    def test_format_matches_to_claude_tool(self) -> None:
        """format_tools() output matches Tool.to_claude_tool()."""
        tool = _FakeTool(
            name="write",
            description="Write a file.",
            parameters={"type": "object", "properties": {}},
        )
        result = self.provider.format_tools([tool])
        expected = tool.to_claude_tool()
        self.assertEqual(result[0], expected)


# ---------------------------------------------------------------------------
# Interface conformance tests
# ---------------------------------------------------------------------------


class TestClaudeProviderConformance(unittest.TestCase):
    """Verify ClaudeProvider satisfies the LLMProvider interface."""

    def test_is_subclass_of_llm_provider(self) -> None:
        """ClaudeProvider is a subclass of LLMProvider."""
        self.assertTrue(issubclass(ClaudeProvider, LLMProvider))

    def test_is_instance_of_llm_provider(self) -> None:
        """ClaudeProvider instances pass isinstance check."""
        provider = ClaudeProvider(api_key="test-key")
        self.assertIsInstance(provider, LLMProvider)

    def test_has_all_abstract_methods(self) -> None:
        """ClaudeProvider implements all abstract methods from LLMProvider."""
        provider = ClaudeProvider(api_key="test-key")
        self.assertIsNotNone(provider)

    def test_name_property_type(self) -> None:
        """name property returns a non-empty string."""
        provider = ClaudeProvider(api_key="test-key")
        self.assertIsInstance(provider.name, str)
        self.assertTrue(len(provider.name) > 0)


# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------


class TestClaudeProviderExceptionHierarchy(unittest.TestCase):
    """Verify that Claude-specific exceptions inherit from provider base."""

    def test_connection_error_is_provider_connection_error(self) -> None:
        """ClaudeConnectionError is a ProviderConnectionError."""
        self.assertTrue(
            issubclass(ClaudeConnectionError, ProviderConnectionError)
        )

    def test_request_error_is_provider_request_error(self) -> None:
        """ClaudeRequestError is a ProviderRequestError."""
        self.assertTrue(
            issubclass(ClaudeRequestError, ProviderRequestError)
        )

    def test_stream_error_is_provider_stream_error(self) -> None:
        """ClaudeStreamError is a ProviderStreamError."""
        self.assertTrue(
            issubclass(ClaudeStreamError, ProviderStreamError)
        )

    def test_rate_limit_error_is_request_error(self) -> None:
        """ClaudeRateLimitError is a ClaudeRequestError."""
        self.assertTrue(
            issubclass(ClaudeRateLimitError, ClaudeRequestError)
        )

    def test_rate_limit_error_is_provider_request_error(self) -> None:
        """ClaudeRateLimitError is a ProviderRequestError."""
        self.assertTrue(
            issubclass(ClaudeRateLimitError, ProviderRequestError)
        )

    def test_catch_provider_connection_error(self) -> None:
        """ClaudeConnectionError can be caught as ProviderConnectionError."""
        with self.assertRaises(ProviderConnectionError):
            raise ClaudeConnectionError("test")

    def test_catch_provider_request_error(self) -> None:
        """ClaudeRequestError can be caught as ProviderRequestError."""
        with self.assertRaises(ProviderRequestError):
            raise ClaudeRequestError("test")

    def test_catch_provider_stream_error(self) -> None:
        """ClaudeStreamError can be caught as ProviderStreamError."""
        with self.assertRaises(ProviderStreamError):
            raise ClaudeStreamError("test")


# ---------------------------------------------------------------------------
# Full round-trip streaming test
# ---------------------------------------------------------------------------


class TestClaudeProviderStreamRoundTrip(unittest.TestCase):
    """End-to-end test simulating a full Claude API streaming response."""

    def setUp(self) -> None:
        self.provider = ClaudeProvider(api_key="test-key")

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_full_text_stream(self, mock_urlopen: MagicMock) -> None:
        """Simulate a full text-only streaming response."""
        sse_lines = [
            ": ping",
            "",
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5","stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" answer"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" is 42."}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":6}}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
        mock_urlopen.return_value = _make_sse_response(sse_lines)

        chunks = list(self.provider.chat_stream(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "What is the meaning of life?"}],
        ))

        # Collect text from all chunks.
        full_text = "".join(
            c["message"].get("content", "") for c in chunks
        )
        self.assertIn("The", full_text)
        self.assertIn("answer", full_text)
        self.assertIn("42", full_text)

        # Final chunk should be done.
        self.assertTrue(chunks[-1]["done"])

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_multiple_tool_calls(self, mock_urlopen: MagicMock) -> None:
        """Simulate a streaming response with multiple tool calls."""
        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"role":"assistant"}}',
            "",
            # First tool
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_001","name":"bash"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\": \\"ls\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            # Second tool
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_002","name":"read"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\": \\"/tmp/test.txt\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":1}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
        mock_urlopen.return_value = _make_sse_response(sse_lines)

        chunks = list(self.provider.chat_stream(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "list and read"}],
        ))

        # Find chunk with tool_calls.
        tool_chunks = [
            c for c in chunks if c["message"].get("tool_calls")
        ]
        self.assertTrue(len(tool_chunks) >= 1)

        tool_calls = tool_chunks[0]["message"]["tool_calls"]
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["function"]["name"], "bash")
        self.assertEqual(tool_calls[0]["id"], "toolu_001")
        self.assertEqual(tool_calls[1]["function"]["name"], "read")
        self.assertEqual(tool_calls[1]["id"], "toolu_002")

    @patch("local_cli.providers.claude_provider.urllib.request.urlopen")
    def test_empty_tool_input(self, mock_urlopen: MagicMock) -> None:
        """Handles tool use with no input_json_delta (empty arguments)."""
        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_empty","name":"status"}}',
            "",
            # No input_json_delta events -- empty arguments.
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
        mock_urlopen.return_value = _make_sse_response(sse_lines)

        chunks = list(self.provider.chat_stream(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "check status"}],
        ))

        tool_chunks = [
            c for c in chunks if c["message"].get("tool_calls")
        ]
        self.assertTrue(len(tool_chunks) >= 1)
        tool_call = tool_chunks[0]["message"]["tool_calls"][0]
        self.assertEqual(tool_call["function"]["arguments"], {})


if __name__ == "__main__":
    unittest.main()
