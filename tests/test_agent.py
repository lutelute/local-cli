"""Tests for local_cli.agent module."""

import sys
import unittest
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

from local_cli.agent import (
    _COMPACT_ASSISTANT_MAX,
    _COMPACT_KEEP_RECENT,
    _COMPACT_MESSAGE_THRESHOLD,
    _COMPACT_TOKEN_THRESHOLD,
    _COMPACT_TOOL_RESULT_MAX,
    _MAX_DISPLAY_RESULT,
    _compact_message,
    _estimate_tokens,
    _execute_tool,
    _needs_compaction,
    _truncate,
    agent_loop,
    collect_streaming_response,
    compact_messages,
)
from local_cli.ollama_client import OllamaStreamError
from local_cli.tools.base import Tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyTool(Tool):
    """Minimal concrete tool for testing."""

    def __init__(
        self,
        name: str = "dummy",
        result: str = "ok",
        *,
        side_effect: Exception | None = None,
    ) -> None:
        self._name = name
        self._result = result
        self._side_effect = side_effect

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A dummy tool for testing."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "arg": {"type": "string", "description": "An argument."},
            },
            "required": [],
        }

    def execute(self, **kwargs: object) -> str:
        if self._side_effect is not None:
            raise self._side_effect
        return self._result


def _make_chunks(
    content_parts: list[str],
    tool_calls: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a list of NDJSON-style chunks for streaming tests.

    All chunks except the last have ``done: False``.  Tool calls (if any)
    appear in the final ``done: True`` chunk.
    """
    chunks: list[dict[str, Any]] = []
    for i, text in enumerate(content_parts):
        is_last = i == len(content_parts) - 1
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if is_last and tool_calls:
            msg["tool_calls"] = tool_calls
        chunks.append({"message": msg, "done": is_last})
    return chunks


# ---------------------------------------------------------------------------
# collect_streaming_response
# ---------------------------------------------------------------------------


class TestCollectStreamingResponse(unittest.TestCase):
    """Tests for collect_streaming_response()."""

    def setUp(self) -> None:
        # Suppress stdout writes during streaming tests.
        self._orig_stdout = sys.stdout
        sys.stdout = StringIO()

    def tearDown(self) -> None:
        sys.stdout = self._orig_stdout

    def test_accumulates_content(self) -> None:
        """Content deltas across chunks are concatenated."""
        chunks = _make_chunks(["Hello", " ", "world"])
        result = collect_streaming_response(iter(chunks))

        self.assertEqual(result["message"]["content"], "Hello world")
        self.assertEqual(result["message"]["role"], "assistant")

    def test_empty_content(self) -> None:
        """A single empty-content chunk produces empty content string."""
        chunks = _make_chunks([""])
        result = collect_streaming_response(iter(chunks))
        self.assertEqual(result["message"]["content"], "")

    def test_tool_calls_accumulated(self) -> None:
        """Tool calls from the final chunk are collected."""
        tc = [{"function": {"name": "read", "arguments": {"path": "a.py"}}}]
        chunks = _make_chunks([""], tool_calls=tc)
        result = collect_streaming_response(iter(chunks))

        self.assertIn("tool_calls", result["message"])
        self.assertEqual(len(result["message"]["tool_calls"]), 1)
        self.assertEqual(
            result["message"]["tool_calls"][0]["function"]["name"], "read"
        )

    def test_tool_calls_across_multiple_chunks(self) -> None:
        """Tool calls appearing in multiple chunks are accumulated."""
        tc1 = [{"function": {"name": "read", "arguments": {}}}]
        tc2 = [{"function": {"name": "write", "arguments": {}}}]
        chunks = [
            {"message": {"role": "assistant", "content": "", "tool_calls": tc1}, "done": False},
            {"message": {"role": "assistant", "content": "", "tool_calls": tc2}, "done": True},
        ]
        result = collect_streaming_response(iter(chunks))

        self.assertEqual(len(result["message"]["tool_calls"]), 2)
        names = [tc["function"]["name"] for tc in result["message"]["tool_calls"]]
        self.assertIn("read", names)
        self.assertIn("write", names)

    def test_no_tool_calls_omits_key(self) -> None:
        """When no tool calls are present, 'tool_calls' is not in the message."""
        chunks = _make_chunks(["Hello"])
        result = collect_streaming_response(iter(chunks))
        self.assertNotIn("tool_calls", result["message"])

    def test_final_chunk_fields_preserved(self) -> None:
        """Top-level fields from the final chunk (model, done) are preserved."""
        chunks = [
            {"message": {"role": "assistant", "content": "Hi"}, "done": True, "model": "qwen3:8b"},
        ]
        result = collect_streaming_response(iter(chunks))
        self.assertTrue(result["done"])
        self.assertEqual(result["model"], "qwen3:8b")

    def test_content_printed_to_stdout(self) -> None:
        """Content tokens are printed to stdout as they arrive."""
        chunks = _make_chunks(["Hello", " world"])
        collect_streaming_response(iter(chunks))

        output = sys.stdout.getvalue()
        self.assertIn("Hello", output)
        self.assertIn(" world", output)

    def test_stream_error_reraised(self) -> None:
        """OllamaStreamError from the stream is re-raised."""

        def error_stream():
            yield {"message": {"role": "assistant", "content": "partial"}, "done": False}
            raise OllamaStreamError("model not found")

        with self.assertRaises(OllamaStreamError) as ctx:
            collect_streaming_response(error_stream())

        self.assertIn("model not found", str(ctx.exception))

    def test_empty_stream(self) -> None:
        """An empty stream produces empty content and empty last_chunk."""
        result = collect_streaming_response(iter([]))
        self.assertEqual(result["message"]["content"], "")
        self.assertNotIn("tool_calls", result["message"])

    def test_thinking_not_in_content(self) -> None:
        """Thinking content is accumulated separately from message content."""
        chunks = [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                    "thinking": "Let me think...",
                },
                "done": False,
            },
            {
                "message": {
                    "role": "assistant",
                    "content": " world",
                    "thinking": " more thoughts",
                },
                "done": True,
            },
        ]
        result = collect_streaming_response(iter(chunks))

        # Content should NOT include thinking tokens.
        self.assertEqual(result["message"]["content"], "Hello world")
        self.assertNotIn("thinking", result["message"])

        # Thinking should be a separate top-level key in the result.
        self.assertIn("thinking", result)
        self.assertEqual(result["thinking"], "Let me think... more thoughts")


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


class TestTruncate(unittest.TestCase):
    """Tests for _truncate()."""

    def test_short_string_unchanged(self) -> None:
        """Strings within the limit are returned unchanged."""
        text = "short"
        self.assertEqual(_truncate(text, 100), text)

    def test_exact_length_unchanged(self) -> None:
        """Strings at exactly the limit are not truncated."""
        text = "a" * _MAX_DISPLAY_RESULT
        self.assertEqual(_truncate(text, _MAX_DISPLAY_RESULT), text)

    def test_long_string_truncated(self) -> None:
        """Strings exceeding the limit are truncated with '...'."""
        text = "a" * 300
        result = _truncate(text, 100)
        self.assertEqual(len(result), 103)  # 100 + len("...")
        self.assertTrue(result.endswith("..."))

    def test_default_max_len(self) -> None:
        """Uses _MAX_DISPLAY_RESULT as default."""
        text = "a" * (_MAX_DISPLAY_RESULT + 50)
        result = _truncate(text)
        self.assertEqual(len(result), _MAX_DISPLAY_RESULT + 3)


# ---------------------------------------------------------------------------
# _execute_tool
# ---------------------------------------------------------------------------


class TestExecuteTool(unittest.TestCase):
    """Tests for _execute_tool()."""

    def test_successful_execution(self) -> None:
        """Tool returns its result string."""
        tool = _DummyTool(result="file contents here")
        result = _execute_tool(tool, {})
        self.assertEqual(result, "file contents here")

    def test_tool_with_arguments(self) -> None:
        """Arguments are passed through to the tool."""

        class ArgTool(_DummyTool):
            def execute(self, **kwargs: object) -> str:
                return f"got: {kwargs.get('arg', '')}"

        tool = ArgTool()
        result = _execute_tool(tool, {"arg": "hello"})
        self.assertEqual(result, "got: hello")

    def test_exception_returns_error_string(self) -> None:
        """Tool exceptions are caught and returned as error strings."""
        tool = _DummyTool(side_effect=FileNotFoundError("no such file"))
        result = _execute_tool(tool, {})
        self.assertIn("Error:", result)
        self.assertIn("FileNotFoundError", result)
        self.assertIn("no such file", result)

    def test_runtime_error_returns_error_string(self) -> None:
        """RuntimeError from tool is converted to error string."""
        tool = _DummyTool(side_effect=RuntimeError("something broke"))
        result = _execute_tool(tool, {})
        self.assertIn("RuntimeError", result)
        self.assertIn("something broke", result)

    @patch("sys.stderr", new_callable=StringIO)
    def test_debug_prints_arguments(self, mock_stderr: StringIO) -> None:
        """Debug mode prints tool arguments to stderr."""
        tool = _DummyTool()
        _execute_tool(tool, {"arg": "test_value"}, debug=True)
        output = mock_stderr.getvalue()
        self.assertIn("dummy", output)
        self.assertIn("test_value", output)


# ---------------------------------------------------------------------------
# Context compaction helpers
# ---------------------------------------------------------------------------


class TestEstimateTokens(unittest.TestCase):
    """Tests for _estimate_tokens()."""

    def test_empty_messages(self) -> None:
        """Empty message list returns 0."""
        self.assertEqual(_estimate_tokens([]), 0)

    def test_single_message(self) -> None:
        """Estimate based on content length."""
        messages = [{"role": "user", "content": "Hello world"}]
        tokens = _estimate_tokens(messages)
        self.assertGreater(tokens, 0)

    def test_tool_call_arguments_counted(self) -> None:
        """Tool call arguments are included in the estimate."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read", "arguments": {"path": "a" * 100}}},
                ],
            },
        ]
        tokens = _estimate_tokens(messages)
        self.assertGreater(tokens, 0)


class TestCompactMessage(unittest.TestCase):
    """Tests for _compact_message()."""

    def test_system_message_never_compacted(self) -> None:
        """System messages are always returned unchanged."""
        msg = {"role": "system", "content": "x" * 10000}
        result = _compact_message(msg)
        self.assertIs(result, msg)

    def test_user_message_never_compacted(self) -> None:
        """User messages are returned unchanged."""
        msg = {"role": "user", "content": "x" * 10000}
        result = _compact_message(msg)
        self.assertIs(result, msg)

    def test_short_tool_message_unchanged(self) -> None:
        """Short tool messages are returned as-is."""
        msg = {"role": "tool", "tool_name": "read", "content": "short"}
        result = _compact_message(msg)
        self.assertIs(result, msg)

    def test_long_tool_message_truncated(self) -> None:
        """Long tool results are truncated."""
        long_content = "x" * (_COMPACT_TOOL_RESULT_MAX + 500)
        msg = {"role": "tool", "tool_name": "read", "content": long_content}
        result = _compact_message(msg)
        self.assertIsNot(result, msg)
        self.assertIn("[truncated for context]", result["content"])
        self.assertLessEqual(
            len(result["content"]),
            _COMPACT_TOOL_RESULT_MAX + 50,  # truncated text + suffix
        )

    def test_long_assistant_message_truncated(self) -> None:
        """Long assistant messages are truncated."""
        long_content = "x" * (_COMPACT_ASSISTANT_MAX + 500)
        msg = {"role": "assistant", "content": long_content}
        result = _compact_message(msg)
        self.assertIsNot(result, msg)
        self.assertIn("[truncated for context]", result["content"])

    def test_assistant_tool_calls_stripped(self) -> None:
        """Tool calls are stripped from compacted assistant messages."""
        long_content = "x" * (_COMPACT_ASSISTANT_MAX + 500)
        msg = {
            "role": "assistant",
            "content": long_content,
            "tool_calls": [
                {"function": {"name": "read", "arguments": {}}},
                {"function": {"name": "write", "arguments": {}}},
            ],
        }
        result = _compact_message(msg)
        self.assertNotIn("tool_calls", result)
        self.assertIn("2 tool call(s) omitted", result["content"])


class TestNeedsCompaction(unittest.TestCase):
    """Tests for _needs_compaction()."""

    def test_below_thresholds(self) -> None:
        """Small message list does not need compaction."""
        messages = [{"role": "user", "content": "hi"}] * 5
        self.assertFalse(_needs_compaction(messages))

    def test_message_count_threshold(self) -> None:
        """Exceeding message count threshold triggers compaction."""
        messages = [{"role": "user", "content": "hi"}] * (
            _COMPACT_MESSAGE_THRESHOLD + 1
        )
        self.assertTrue(_needs_compaction(messages))

    def test_token_threshold(self) -> None:
        """Exceeding token count threshold triggers compaction."""
        # Create messages with enough content to exceed the token threshold.
        big_content = "x" * (_COMPACT_TOKEN_THRESHOLD * 5)
        messages = [{"role": "user", "content": big_content}]
        self.assertTrue(_needs_compaction(messages))


class TestCompactMessages(unittest.TestCase):
    """Tests for compact_messages()."""

    def test_small_list_unchanged(self) -> None:
        """Lists smaller than _COMPACT_KEEP_RECENT are not modified."""
        messages = [{"role": "user", "content": "hi"}] * 3
        original = [dict(m) for m in messages]
        compact_messages(messages)
        self.assertEqual(messages, original)

    def test_system_messages_preserved(self) -> None:
        """System messages at the start are never compacted."""
        system_msg = {"role": "system", "content": "x" * 10000}
        messages = [system_msg]
        # Add enough messages to trigger compaction.
        for i in range(_COMPACT_KEEP_RECENT + 20):
            messages.append(
                {"role": "tool", "tool_name": "t", "content": "x" * 1000}
            )
        compact_messages(messages)
        # System message content unchanged.
        self.assertEqual(messages[0]["content"], "x" * 10000)

    def test_recent_messages_preserved(self) -> None:
        """The most recent messages are not compacted."""
        messages = [{"role": "system", "content": "sys"}]
        # Old messages (will be compacted).
        for _ in range(20):
            messages.append(
                {"role": "tool", "tool_name": "t", "content": "x" * 1000}
            )
        # Recent messages.
        for _ in range(_COMPACT_KEEP_RECENT):
            messages.append(
                {"role": "user", "content": "recent " + "y" * 1000}
            )

        compact_messages(messages)

        # Recent messages should still contain "recent".
        for msg in messages[-_COMPACT_KEEP_RECENT:]:
            self.assertIn("recent", msg["content"])


# ---------------------------------------------------------------------------
# agent_loop
# ---------------------------------------------------------------------------


class TestAgentLoopNoToolCalls(unittest.TestCase):
    """Tests for agent_loop when the LLM responds without tool calls."""

    def setUp(self) -> None:
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def tearDown(self) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

    def test_simple_response_appended_to_messages(self) -> None:
        """LLM response without tool calls appends assistant message and exits."""
        client = MagicMock()
        chunks = _make_chunks(["Hello!"])
        client.chat_stream.return_value = iter(chunks)

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
        ]
        agent_loop(client, "qwen3:8b", [], messages)

        # Assistant message should have been appended.
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "Hello!")

    def test_no_tool_calls_means_single_iteration(self) -> None:
        """Without tool calls, chat_stream is called exactly once."""
        client = MagicMock()
        chunks = _make_chunks(["Done."])
        client.chat_stream.return_value = iter(chunks)

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Say done"},
        ]
        agent_loop(client, "test-model", [], messages)

        client.chat_stream.assert_called_once()


class TestAgentLoopWithToolCalls(unittest.TestCase):
    """Tests for agent_loop when the LLM requests tool calls."""

    def setUp(self) -> None:
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def tearDown(self) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

    def test_tool_call_then_final_response(self) -> None:
        """LLM calls a tool, gets result, then responds without tools."""
        tool = _DummyTool(name="read", result="file content here")

        # First call: LLM requests a tool call.
        tc = [{"function": {"name": "read", "arguments": {"arg": "test"}}}]
        first_chunks = _make_chunks([""], tool_calls=tc)

        # Second call: LLM responds without tool calls.
        second_chunks = _make_chunks(["I read the file."])

        client = MagicMock()
        client.chat_stream.side_effect = [
            iter(first_chunks),
            iter(second_chunks),
        ]

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Read the file"},
        ]
        agent_loop(client, "qwen3:8b", [tool], messages)

        # Expected messages: user, assistant(tool_call), tool(result), assistant(final)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertIn("tool_calls", messages[1])
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_name"], "read")
        self.assertEqual(messages[2]["content"], "file content here")
        self.assertEqual(messages[3]["role"], "assistant")
        self.assertEqual(messages[3]["content"], "I read the file.")

    def test_multiple_tool_calls_in_single_response(self) -> None:
        """LLM requests multiple tool calls in a single response."""
        tool_a = _DummyTool(name="read", result="content_a")
        tool_b = _DummyTool(name="write", result="written")

        tc = [
            {"function": {"name": "read", "arguments": {}}},
            {"function": {"name": "write", "arguments": {}}},
        ]
        first_chunks = _make_chunks([""], tool_calls=tc)
        second_chunks = _make_chunks(["All done."])

        client = MagicMock()
        client.chat_stream.side_effect = [
            iter(first_chunks),
            iter(second_chunks),
        ]

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Do both"},
        ]
        agent_loop(client, "qwen3:8b", [tool_a, tool_b], messages)

        # user + assistant(tc) + tool(read) + tool(write) + assistant(final)
        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_name"], "read")
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["tool_name"], "write")

    def test_chained_tool_calls(self) -> None:
        """LLM requests tools in two consecutive rounds."""
        tool = _DummyTool(name="bash", result="output")

        # Round 1: one tool call.
        tc1 = [{"function": {"name": "bash", "arguments": {}}}]
        chunks1 = _make_chunks([""], tool_calls=tc1)

        # Round 2: another tool call.
        tc2 = [{"function": {"name": "bash", "arguments": {}}}]
        chunks2 = _make_chunks([""], tool_calls=tc2)

        # Round 3: final text response.
        chunks3 = _make_chunks(["Done."])

        client = MagicMock()
        client.chat_stream.side_effect = [
            iter(chunks1),
            iter(chunks2),
            iter(chunks3),
        ]

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Run commands"},
        ]
        agent_loop(client, "qwen3:8b", [tool], messages)

        # user + assistant + tool + assistant + tool + assistant
        self.assertEqual(len(messages), 6)
        self.assertEqual(client.chat_stream.call_count, 3)


class TestAgentLoopErrorHandling(unittest.TestCase):
    """Tests for agent_loop error handling."""

    def setUp(self) -> None:
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def tearDown(self) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

    def test_unknown_tool_returns_error_string(self) -> None:
        """Calling an unknown tool appends an error tool message."""
        tc = [{"function": {"name": "nonexistent", "arguments": {}}}]
        first_chunks = _make_chunks([""], tool_calls=tc)
        second_chunks = _make_chunks(["I see the error."])

        client = MagicMock()
        client.chat_stream.side_effect = [
            iter(first_chunks),
            iter(second_chunks),
        ]

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Do something"},
        ]
        agent_loop(client, "qwen3:8b", [], messages)

        # The tool message should contain an error.
        tool_msg = messages[2]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_name"], "nonexistent")
        self.assertIn("Error", tool_msg["content"])
        self.assertIn("unknown tool", tool_msg["content"])

    def test_tool_exception_returns_error_string(self) -> None:
        """Tool raising an exception produces an error tool message."""
        tool = _DummyTool(
            name="failing",
            side_effect=ValueError("bad input"),
        )

        tc = [{"function": {"name": "failing", "arguments": {}}}]
        first_chunks = _make_chunks([""], tool_calls=tc)
        second_chunks = _make_chunks(["Noted the error."])

        client = MagicMock()
        client.chat_stream.side_effect = [
            iter(first_chunks),
            iter(second_chunks),
        ]

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Try this"},
        ]
        agent_loop(client, "qwen3:8b", [tool], messages)

        tool_msg = messages[2]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertIn("Error", tool_msg["content"])
        self.assertIn("ValueError", tool_msg["content"])
        self.assertIn("bad input", tool_msg["content"])

    def test_stream_error_breaks_loop(self) -> None:
        """OllamaStreamError during streaming breaks the loop gracefully."""

        def error_stream():
            yield {"message": {"role": "assistant", "content": ""}, "done": False}
            raise OllamaStreamError("server error")

        client = MagicMock()
        client.chat_stream.return_value = error_stream()

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
        ]
        # Should not raise -- the error is caught and the loop exits.
        agent_loop(client, "qwen3:8b", [], messages)

        # Only the original user message should remain (no assistant appended
        # because the error was caught before building the response).
        stderr_output = sys.stderr.getvalue()
        self.assertIn("Error from Ollama", stderr_output)

    def test_tool_defs_passed_to_chat_stream(self) -> None:
        """Tool definitions are converted and passed to chat_stream."""
        tool = _DummyTool(name="mytool")

        chunks = _make_chunks(["Hello"])
        client = MagicMock()
        client.chat_stream.return_value = iter(chunks)

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
        ]
        agent_loop(client, "qwen3:8b", [tool], messages)

        call_args = client.chat_stream.call_args
        tool_defs = call_args[1].get("tools") or call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("tools")
        self.assertIsNotNone(tool_defs)
        self.assertEqual(len(tool_defs), 1)
        self.assertEqual(tool_defs[0]["type"], "function")
        self.assertEqual(tool_defs[0]["function"]["name"], "mytool")

    def test_tool_call_with_missing_function_key(self) -> None:
        """Tool call with missing 'function' key is handled gracefully."""
        tc = [{"not_function": {}}]  # malformed tool call
        first_chunks = _make_chunks([""], tool_calls=tc)
        second_chunks = _make_chunks(["Ok."])

        client = MagicMock()
        client.chat_stream.side_effect = [
            iter(first_chunks),
            iter(second_chunks),
        ]

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Do it"},
        ]
        # Should not crash.
        agent_loop(client, "qwen3:8b", [], messages)

        # A tool message with error for the unknown/empty tool name should be added.
        tool_msg = messages[2]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertIn("Error", tool_msg["content"])


# ---------------------------------------------------------------------------
# Agent loop: thinking mode, options, and dynamic compaction
# ---------------------------------------------------------------------------


class TestAgentLoopThinkingAndOptions(unittest.TestCase):
    """Tests for thinking mode handling, options passthrough, and dynamic compaction."""

    def setUp(self) -> None:
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def tearDown(self) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

    def test_thinking_not_in_history(self) -> None:
        """After agent_loop, messages list has no thinking key."""
        client = MagicMock()
        # Build chunks with thinking content in the message.
        chunks = [
            {
                "message": {
                    "role": "assistant",
                    "content": "Answer",
                    "thinking": "internal reasoning",
                },
                "done": True,
            },
        ]
        client.chat_stream.return_value = iter(chunks)

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Question"},
        ]
        agent_loop(client, "qwen3:8b", [], messages)

        # No message in the history should contain a "thinking" key.
        for msg in messages:
            self.assertNotIn("thinking", msg)

        # The assistant content should still be recorded.
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "Answer")

    def test_thinking_debug_output(self) -> None:
        """Thinking content is shown in stderr when debug=True."""
        client = MagicMock()
        chunks = [
            {
                "message": {
                    "role": "assistant",
                    "content": "Answer",
                    "thinking": "my deep reasoning",
                },
                "done": True,
            },
        ]
        client.chat_stream.return_value = iter(chunks)

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Question"},
        ]
        agent_loop(client, "qwen3:8b", [], messages, debug=True)

        stderr_output = sys.stderr.getvalue()
        self.assertIn("Thinking", stderr_output)
        self.assertIn("my deep reasoning", stderr_output)

    def test_agent_loop_with_options(self) -> None:
        """Options dict is passed through to client.chat_stream."""
        client = MagicMock()
        chunks = _make_chunks(["Done."])
        client.chat_stream.return_value = iter(chunks)

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
        ]
        opts = {"temperature": 0.5, "num_ctx": 8192}
        agent_loop(client, "qwen3:8b", [], messages, options=opts)

        # Verify options were forwarded as a keyword argument.
        call_kwargs = client.chat_stream.call_args[1]
        self.assertIn("options", call_kwargs)
        self.assertEqual(call_kwargs["options"], opts)

    def test_dynamic_compaction_threshold(self) -> None:
        """Compaction threshold adjusts with num_ctx via _needs_compaction."""
        # Create messages with ~5000 estimated tokens (20000 chars / 4).
        messages = [{"role": "user", "content": "x" * 20000}]

        # Default threshold is 24_000 tokens — 5000 < 24000 → no compaction.
        self.assertFalse(_needs_compaction(messages))

        # Simulating num_ctx=4096 → threshold would be int(4096 * 0.75) = 3072.
        # 5000 > 3072 → compaction needed.
        self.assertTrue(_needs_compaction(messages, token_threshold=3072))

        # Verify the threshold computation inside agent_loop by checking that
        # compact_messages is called when num_ctx is small enough.
        client = MagicMock()
        chunks = _make_chunks(["Ok."])
        client.chat_stream.return_value = iter(chunks)

        # Enough messages to make compaction meaningful.
        big_messages: list[dict[str, Any]] = [
            {"role": "system", "content": "sys"},
        ]
        for _ in range(20):
            big_messages.append(
                {"role": "tool", "tool_name": "t", "content": "y" * 2000}
            )
        big_messages.append({"role": "user", "content": "go"})

        opts = {"num_ctx": 4096}
        with patch("local_cli.agent.compact_messages") as mock_compact:
            agent_loop(client, "qwen3:8b", [], big_messages, options=opts)

        # compact_messages should have been triggered due to the low threshold.
        mock_compact.assert_called()


if __name__ == "__main__":
    unittest.main()
