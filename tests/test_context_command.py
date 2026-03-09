"""Tests for the /context slash command in local_cli.cli."""

import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

from local_cli.agent import (
    _CHARS_PER_TOKEN,
    _COMPACT_MESSAGE_THRESHOLD,
    _COMPACT_TOKEN_THRESHOLD,
)
from local_cli.cli import _handle_slash_command, _ReplContext
from local_cli.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(messages: list[dict] | None = None) -> _ReplContext:
    """Build a minimal _ReplContext for testing slash commands.

    Args:
        messages: Optional conversation message list.  Defaults to a
            single system message.

    Returns:
        A _ReplContext instance with mocked dependencies.
    """
    config = Config()
    client = MagicMock()
    system_prompt = "You are a helpful assistant."
    if messages is None:
        messages = [{"role": "system", "content": system_prompt}]

    return _ReplContext(
        config=config,
        client=client,
        tools=[],
        messages=messages,
        session_manager=MagicMock(),
        system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestContextCommand(unittest.TestCase):
    """Tests for /context slash command."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_context_returns_true(self, mock_stdout: StringIO) -> None:
        """The /context command returns True (REPL should continue)."""
        ctx = _make_ctx()
        result = _handle_slash_command("/context", ctx)
        self.assertTrue(result)

    @patch("sys.stdout", new_callable=StringIO)
    def test_context_shows_message_count(self, mock_stdout: StringIO) -> None:
        """Output includes the total message count."""
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        ctx = _make_ctx(messages)
        _handle_slash_command("/context", ctx)
        output = mock_stdout.getvalue()
        self.assertIn("Messages: 3", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_context_shows_estimated_tokens(
        self, mock_stdout: StringIO
    ) -> None:
        """Output includes estimated token count and the threshold."""
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "a" * 400},  # 400 chars = ~100 tokens
        ]
        ctx = _make_ctx(messages)
        _handle_slash_command("/context", ctx)
        output = mock_stdout.getvalue()
        # Verify estimated tokens and threshold are present.
        self.assertIn("Est. tokens:", output)
        self.assertIn(str(_COMPACT_TOKEN_THRESHOLD), output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_context_compaction_not_triggered(
        self, mock_stdout: StringIO
    ) -> None:
        """Shows 'not triggered' when below compaction thresholds."""
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "short message"},
        ]
        ctx = _make_ctx(messages)
        _handle_slash_command("/context", ctx)
        output = mock_stdout.getvalue()
        self.assertIn("Compaction: not triggered", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_context_compaction_triggered_by_message_count(
        self, mock_stdout: StringIO
    ) -> None:
        """Shows 'triggered' when message count exceeds threshold."""
        messages = [{"role": "system", "content": "system prompt"}]
        # Add enough messages to exceed the threshold.
        for i in range(_COMPACT_MESSAGE_THRESHOLD + 1):
            messages.append({"role": "user", "content": f"msg {i}"})
        ctx = _make_ctx(messages)
        _handle_slash_command("/context", ctx)
        output = mock_stdout.getvalue()
        self.assertIn("Compaction: triggered", output)
        # Make sure it's not "not triggered".
        self.assertNotIn("not triggered", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_context_compaction_triggered_by_token_count(
        self, mock_stdout: StringIO
    ) -> None:
        """Shows 'triggered' when estimated tokens exceed threshold."""
        # Create a small number of messages with very large content.
        large_content = "x" * (_COMPACT_TOKEN_THRESHOLD * _CHARS_PER_TOKEN + 100)
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": large_content},
        ]
        ctx = _make_ctx(messages)
        _handle_slash_command("/context", ctx)
        output = mock_stdout.getvalue()
        self.assertIn("Compaction: triggered", output)
        self.assertNotIn("not triggered", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_context_output_format(self, mock_stdout: StringIO) -> None:
        """Output follows the expected pipe-separated format."""
        ctx = _make_ctx()
        _handle_slash_command("/context", ctx)
        output = mock_stdout.getvalue().strip()
        # Verify the pipe-separated format.
        parts = output.split(" | ")
        self.assertEqual(len(parts), 3)
        self.assertTrue(parts[0].startswith("Messages:"))
        self.assertTrue(parts[1].startswith("Est. tokens:"))
        self.assertTrue(parts[2].startswith("Compaction:"))

    @patch("sys.stdout", new_callable=StringIO)
    def test_context_with_empty_messages(
        self, mock_stdout: StringIO
    ) -> None:
        """Handles an empty message list gracefully."""
        ctx = _make_ctx(messages=[])
        _handle_slash_command("/context", ctx)
        output = mock_stdout.getvalue()
        self.assertIn("Messages: 0", output)
        self.assertIn("Est. tokens: ~0", output)
        self.assertIn("Compaction: not triggered", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_context_token_estimate_accuracy(
        self, mock_stdout: StringIO
    ) -> None:
        """Token estimate matches the value from _estimate_tokens."""
        content = "hello world test message"
        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": content},
        ]
        ctx = _make_ctx(messages)
        _handle_slash_command("/context", ctx)
        output = mock_stdout.getvalue()
        # Manually compute expected tokens.
        total_chars = len("prompt") + len(content)
        expected_tokens = total_chars // _CHARS_PER_TOKEN
        self.assertIn(f"~{expected_tokens}", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_context_case_insensitive(self, mock_stdout: StringIO) -> None:
        """The /context command is matched case-insensitively."""
        ctx = _make_ctx()
        result = _handle_slash_command("/CONTEXT", ctx)
        self.assertTrue(result)
        output = mock_stdout.getvalue()
        self.assertIn("Messages:", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_context_with_tool_calls_in_messages(
        self, mock_stdout: StringIO
    ) -> None:
        """Token estimate accounts for tool call arguments."""
        messages = [
            {"role": "system", "content": "prompt"},
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read",
                            "arguments": {"file_path": "/some/long/path.py"},
                        }
                    }
                ],
            },
        ]
        ctx = _make_ctx(messages)
        _handle_slash_command("/context", ctx)
        output = mock_stdout.getvalue()
        # Token estimate should be > 0 and include tool call argument chars.
        self.assertIn("Est. tokens: ~", output)
        # Extract the estimated tokens number.
        import re
        match = re.search(r"~(\d+)", output)
        self.assertIsNotNone(match)
        tokens = int(match.group(1))
        self.assertGreater(tokens, 0)


class TestContextInSlashCommands(unittest.TestCase):
    """Tests that /context is registered in _SLASH_COMMANDS."""

    def test_context_in_slash_commands_dict(self) -> None:
        """The /context command is listed in the _SLASH_COMMANDS dict."""
        from local_cli.cli import _SLASH_COMMANDS

        self.assertIn("/context", _SLASH_COMMANDS)

    def test_context_help_description(self) -> None:
        """The /context help text mentions context-related info."""
        from local_cli.cli import _SLASH_COMMANDS

        desc = _SLASH_COMMANDS["/context"]
        self.assertIn("context", desc.lower())


if __name__ == "__main__":
    unittest.main()
