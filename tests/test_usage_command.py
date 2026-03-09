"""Tests for the /usage slash command in local_cli.cli."""

import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

from local_cli.cli import _handle_slash_command, _ReplContext, _SLASH_COMMANDS
from local_cli.config import Config
from local_cli.token_tracker import TokenTracker, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    token_tracker: TokenTracker | None = None,
    messages: list[dict] | None = None,
) -> _ReplContext:
    """Build a minimal _ReplContext for testing the /usage command.

    Args:
        token_tracker: Optional token tracker instance.  Defaults to None.
        messages: Optional conversation message list.

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
        token_tracker=token_tracker,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUsageCommand(unittest.TestCase):
    """Tests for /usage slash command."""

    @patch("sys.stdout", new_callable=StringIO)
    def test_usage_returns_true(self, mock_stdout: StringIO) -> None:
        """The /usage command returns True (REPL should continue)."""
        tracker = TokenTracker()
        ctx = _make_ctx(token_tracker=tracker)
        result = _handle_slash_command("/usage", ctx)
        self.assertTrue(result)

    @patch("sys.stdout", new_callable=StringIO)
    def test_usage_no_tracker(self, mock_stdout: StringIO) -> None:
        """Shows 'not available' when token_tracker is None."""
        ctx = _make_ctx(token_tracker=None)
        result = _handle_slash_command("/usage", ctx)
        self.assertTrue(result)
        output = mock_stdout.getvalue()
        self.assertIn("Token tracking not available", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_usage_empty_tracker(self, mock_stdout: StringIO) -> None:
        """Shows 'No token usage recorded' when tracker has no records."""
        tracker = TokenTracker()
        ctx = _make_ctx(token_tracker=tracker)
        _handle_slash_command("/usage", ctx)
        output = mock_stdout.getvalue()
        self.assertIn("No token usage recorded", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_usage_single_ollama_message(self, mock_stdout: StringIO) -> None:
        """Displays a table row for a single Ollama message exchange."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50, provider="ollama"))
        ctx = _make_ctx(token_tracker=tracker)
        _handle_slash_command("/usage", ctx)
        output = mock_stdout.getvalue()
        # Should have header, separator, data row, separator, totals.
        self.assertIn("Input", output)
        self.assertIn("Output", output)
        self.assertIn("Total", output)
        self.assertIn("100", output)
        self.assertIn("50", output)
        self.assertIn("150", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_usage_ollama_shows_na_cost(self, mock_stdout: StringIO) -> None:
        """Ollama usage shows 'N/A' for cost column."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=200, output_tokens=100, provider="ollama"))
        ctx = _make_ctx(token_tracker=tracker)
        _handle_slash_command("/usage", ctx)
        output = mock_stdout.getvalue()
        self.assertIn("N/A", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_usage_claude_shows_cost(self, mock_stdout: StringIO) -> None:
        """Claude usage shows estimated cost in the cost column."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=1000, output_tokens=500, provider="claude"))
        ctx = _make_ctx(token_tracker=tracker)
        _handle_slash_command("/usage", ctx)
        output = mock_stdout.getvalue()
        # Should contain a dollar-sign cost value, not N/A.
        self.assertIn("$", output)
        # Should NOT have N/A in the totals row for Claude-only usage.
        lines = output.strip().split("\n")
        totals_line = lines[-1]
        self.assertNotIn("N/A", totals_line)

    @patch("sys.stdout", new_callable=StringIO)
    def test_usage_multiple_messages(self, mock_stdout: StringIO) -> None:
        """Displays multiple rows for multiple message exchanges."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50, provider="ollama"))
        tracker.record(TokenUsage(input_tokens=200, output_tokens=80, provider="ollama"))
        tracker.record(TokenUsage(input_tokens=150, output_tokens=60, provider="ollama"))
        ctx = _make_ctx(token_tracker=tracker)
        _handle_slash_command("/usage", ctx)
        output = mock_stdout.getvalue()
        # Check that session totals are correct.
        # Total input: 450, output: 190, total: 640.
        self.assertIn("TOTAL", output)
        self.assertIn("450", output)
        self.assertIn("190", output)
        self.assertIn("640", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_usage_table_has_header(self, mock_stdout: StringIO) -> None:
        """Output includes table header with column names."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=10, output_tokens=5, provider="ollama"))
        ctx = _make_ctx(token_tracker=tracker)
        _handle_slash_command("/usage", ctx)
        output = mock_stdout.getvalue()
        self.assertIn("Provider", output)
        self.assertIn("Input", output)
        self.assertIn("Output", output)
        self.assertIn("Total", output)
        self.assertIn("Cost", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_usage_case_insensitive(self, mock_stdout: StringIO) -> None:
        """The /usage command is matched case-insensitively."""
        tracker = TokenTracker()
        ctx = _make_ctx(token_tracker=tracker)
        result = _handle_slash_command("/USAGE", ctx)
        self.assertTrue(result)
        output = mock_stdout.getvalue()
        # Should show the empty tracker message, not "Unknown command".
        self.assertNotIn("Unknown command", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_usage_mixed_providers(self, mock_stdout: StringIO) -> None:
        """Handles mixed Ollama and Claude records correctly."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=100, output_tokens=50, provider="ollama"))
        tracker.record(TokenUsage(input_tokens=200, output_tokens=100, provider="claude"))
        ctx = _make_ctx(token_tracker=tracker)
        _handle_slash_command("/usage", ctx)
        output = mock_stdout.getvalue()
        # Both providers should appear.
        self.assertIn("ollama", output)
        self.assertIn("claude", output)
        # Ollama row has N/A, Claude row has $ cost.
        self.assertIn("N/A", output)
        self.assertIn("$", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_usage_table_has_separator_lines(
        self, mock_stdout: StringIO,
    ) -> None:
        """Output includes separator lines (dashes) in the table."""
        tracker = TokenTracker()
        tracker.record(TokenUsage(input_tokens=10, output_tokens=5, provider="ollama"))
        ctx = _make_ctx(token_tracker=tracker)
        _handle_slash_command("/usage", ctx)
        output = mock_stdout.getvalue()
        # Table should have at least two separator lines (after header, before totals).
        dash_lines = [line for line in output.split("\n") if line.startswith("---")]
        self.assertGreaterEqual(len(dash_lines), 2)


class TestUsageInSlashCommands(unittest.TestCase):
    """Tests that /usage is registered in _SLASH_COMMANDS."""

    def test_usage_in_slash_commands_dict(self) -> None:
        """The /usage command is listed in the _SLASH_COMMANDS dict."""
        self.assertIn("/usage", _SLASH_COMMANDS)

    def test_usage_help_description(self) -> None:
        """The /usage help text mentions token-related info."""
        desc = _SLASH_COMMANDS["/usage"]
        self.assertIn("token", desc.lower())


if __name__ == "__main__":
    unittest.main()
