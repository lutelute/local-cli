"""Tests for local_cli.ideation module."""

import sys
import unittest
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

from local_cli.ideation import (
    IdeationEngine,
    IdeationError,
    _DEFAULT_SYSTEM_PROMPT,
    collect_generate_response,
)
from local_cli.providers.base import (
    ProviderRequestError,
    ProviderStreamError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_chunks(
    content_parts: list[str],
) -> list[dict[str, Any]]:
    """Build a list of NDJSON-style chunks for chat streaming tests.

    All chunks except the last have ``done: False``.
    """
    chunks: list[dict[str, Any]] = []
    for i, text in enumerate(content_parts):
        is_last = i == len(content_parts) - 1
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        chunks.append({"message": msg, "done": is_last})
    return chunks


def _make_generate_chunks(
    response_parts: list[str],
) -> list[dict[str, Any]]:
    """Build a list of NDJSON-style chunks for generate streaming tests.

    Generate API uses ``"response"`` key instead of ``"message": {"content": ...}``.
    """
    chunks: list[dict[str, Any]] = []
    for i, text in enumerate(response_parts):
        is_last = i == len(response_parts) - 1
        chunks.append({"response": text, "done": is_last})
    return chunks


# ---------------------------------------------------------------------------
# collect_generate_response
# ---------------------------------------------------------------------------


class TestCollectGenerateResponse(unittest.TestCase):
    """Tests for collect_generate_response()."""

    def setUp(self) -> None:
        self._orig_stdout = sys.stdout
        sys.stdout = StringIO()

    def tearDown(self) -> None:
        sys.stdout = self._orig_stdout

    def test_accumulates_response_parts(self) -> None:
        """Response deltas across chunks are concatenated."""
        chunks = _make_generate_chunks(["Hello", " ", "world"])
        result = collect_generate_response(iter(chunks))
        self.assertEqual(result, "Hello world")

    def test_empty_stream_returns_empty_string(self) -> None:
        """An empty stream produces an empty string."""
        result = collect_generate_response(iter([]))
        self.assertEqual(result, "")

    def test_single_chunk(self) -> None:
        """A single chunk with response text is collected."""
        chunks = _make_generate_chunks(["Complete response."])
        result = collect_generate_response(iter(chunks))
        self.assertEqual(result, "Complete response.")

    def test_content_printed_to_stdout(self) -> None:
        """Response tokens are printed to stdout as they arrive."""
        chunks = _make_generate_chunks(["Hello", " world"])
        collect_generate_response(iter(chunks))
        output = sys.stdout.getvalue()
        self.assertIn("Hello", output)
        self.assertIn(" world", output)

    def test_trailing_newline_after_content(self) -> None:
        """A trailing newline is printed after streamed content."""
        chunks = _make_generate_chunks(["Hello"])
        collect_generate_response(iter(chunks))
        output = sys.stdout.getvalue()
        self.assertTrue(output.endswith("\n"))

    def test_no_trailing_newline_for_empty_content(self) -> None:
        """No trailing newline when no content was streamed."""
        chunks = _make_generate_chunks([""])
        collect_generate_response(iter(chunks))
        output = sys.stdout.getvalue()
        self.assertEqual(output, "")

    def test_spinner_stopped_on_first_content(self) -> None:
        """Spinner is stopped when the first content token arrives."""
        spinner = MagicMock()
        chunks = _make_generate_chunks(["Hello", " world"])
        collect_generate_response(iter(chunks), spinner=spinner)
        spinner.stop.assert_called_once()

    def test_spinner_stopped_on_empty_stream(self) -> None:
        """Spinner is stopped even if stream is empty."""
        spinner = MagicMock()
        collect_generate_response(iter([]), spinner=spinner)
        spinner.stop.assert_called_once()

    def test_spinner_stopped_on_stream_error(self) -> None:
        """Spinner is stopped when a ProviderStreamError occurs."""

        def error_stream():
            yield {"response": "partial", "done": False}
            raise ProviderStreamError("stream failed")

        spinner = MagicMock()
        with self.assertRaises(ProviderStreamError):
            collect_generate_response(error_stream(), spinner=spinner)
        spinner.stop.assert_called_once()

    def test_stream_error_reraised(self) -> None:
        """ProviderStreamError from the stream is re-raised."""

        def error_stream():
            yield {"response": "partial", "done": False}
            raise ProviderStreamError("server error")

        with self.assertRaises(ProviderStreamError) as ctx:
            collect_generate_response(error_stream())
        self.assertIn("server error", str(ctx.exception))

    def test_keyboard_interrupt_returns_partial(self) -> None:
        """KeyboardInterrupt returns the content accumulated so far."""

        def interrupted_stream():
            yield {"response": "partial", "done": False}
            raise KeyboardInterrupt

        result = collect_generate_response(interrupted_stream())
        self.assertEqual(result, "partial")

    def test_keyboard_interrupt_stops_spinner(self) -> None:
        """Spinner is stopped on KeyboardInterrupt."""

        def interrupted_stream():
            if False:
                yield  # pragma: no cover — makes this a generator
            raise KeyboardInterrupt

        spinner = MagicMock()
        collect_generate_response(interrupted_stream(), spinner=spinner)
        # spinner.stop() may be called more than once (handler + cleanup).
        spinner.stop.assert_called()

    def test_chunks_without_response_key_ignored(self) -> None:
        """Chunks missing the 'response' key produce no content."""
        chunks = [{"done": False}, {"done": True}]
        result = collect_generate_response(iter(chunks))
        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# IdeationEngine — session management
# ---------------------------------------------------------------------------


class TestIdeationEngineInit(unittest.TestCase):
    """Tests for IdeationEngine construction."""

    def test_initial_state(self) -> None:
        """Engine starts with empty message history."""
        client = MagicMock()
        engine = IdeationEngine(client)
        self.assertEqual(engine.get_history(), [])
        self.assertFalse(engine.has_session)


class TestIdeationEngineStartSession(unittest.TestCase):
    """Tests for IdeationEngine.start_session()."""

    def test_default_system_prompt(self) -> None:
        """start_session with no args uses the default system prompt."""
        client = MagicMock()
        engine = IdeationEngine(client)
        engine.start_session()

        history = engine.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["role"], "system")
        self.assertEqual(history[0]["content"], _DEFAULT_SYSTEM_PROMPT)

    def test_custom_system_prompt(self) -> None:
        """start_session with a custom prompt uses it."""
        client = MagicMock()
        engine = IdeationEngine(client)
        engine.start_session("You are a creative thinker.")

        history = engine.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["content"], "You are a creative thinker.")

    def test_replaces_existing_system_prompt(self) -> None:
        """start_session on existing session replaces the system prompt."""
        client = MagicMock()
        engine = IdeationEngine(client)
        engine.start_session("First prompt")
        engine.start_session("Second prompt")

        history = engine.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["content"], "Second prompt")

    def test_preserves_conversation_history(self) -> None:
        """start_session replaces system prompt but keeps other messages."""
        client = MagicMock()
        engine = IdeationEngine(client)
        engine.start_session("Original")

        # Manually add conversation messages.
        engine.get_history().append({"role": "user", "content": "Hello"})
        engine.get_history().append({"role": "assistant", "content": "Hi there"})

        engine.start_session("Updated")

        history = engine.get_history()
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["content"], "Updated")
        self.assertEqual(history[1]["role"], "user")
        self.assertEqual(history[2]["role"], "assistant")

    def test_has_session_true_after_start(self) -> None:
        """has_session returns True after start_session."""
        client = MagicMock()
        engine = IdeationEngine(client)
        engine.start_session()
        self.assertTrue(engine.has_session)

    def test_none_system_prompt_uses_default(self) -> None:
        """Passing None explicitly uses default system prompt."""
        client = MagicMock()
        engine = IdeationEngine(client)
        engine.start_session(None)

        history = engine.get_history()
        self.assertEqual(history[0]["content"], _DEFAULT_SYSTEM_PROMPT)


class TestIdeationEngineClearHistory(unittest.TestCase):
    """Tests for IdeationEngine.clear_history()."""

    def test_clears_all_messages(self) -> None:
        """clear_history removes all messages including system prompt."""
        client = MagicMock()
        engine = IdeationEngine(client)
        engine.start_session()
        engine.get_history().append({"role": "user", "content": "Hello"})

        engine.clear_history()

        self.assertEqual(engine.get_history(), [])
        self.assertFalse(engine.has_session)

    def test_clear_empty_history_is_noop(self) -> None:
        """Clearing an already empty history does not raise."""
        client = MagicMock()
        engine = IdeationEngine(client)
        engine.clear_history()
        self.assertEqual(engine.get_history(), [])


class TestIdeationEngineGetHistory(unittest.TestCase):
    """Tests for IdeationEngine.get_history()."""

    def test_returns_internal_list(self) -> None:
        """get_history returns the internal message list (not a copy)."""
        client = MagicMock()
        engine = IdeationEngine(client)
        engine.start_session()

        history = engine.get_history()
        history.append({"role": "user", "content": "test"})

        self.assertEqual(len(engine.get_history()), 2)


class TestIdeationEngineHasSession(unittest.TestCase):
    """Tests for IdeationEngine.has_session property."""

    def test_false_on_empty_history(self) -> None:
        """has_session is False when history is empty."""
        client = MagicMock()
        engine = IdeationEngine(client)
        self.assertFalse(engine.has_session)

    def test_false_when_first_message_not_system(self) -> None:
        """has_session is False if first message is not system role."""
        client = MagicMock()
        engine = IdeationEngine(client)
        engine.get_history().append({"role": "user", "content": "Hello"})
        self.assertFalse(engine.has_session)

    def test_true_when_system_prompt_present(self) -> None:
        """has_session is True after start_session."""
        client = MagicMock()
        engine = IdeationEngine(client)
        engine.start_session()
        self.assertTrue(engine.has_session)

    def test_false_after_clear_history(self) -> None:
        """has_session is False after clearing history."""
        client = MagicMock()
        engine = IdeationEngine(client)
        engine.start_session()
        engine.clear_history()
        self.assertFalse(engine.has_session)


# ---------------------------------------------------------------------------
# IdeationEngine — chat_turn (multi-turn, no tools)
# ---------------------------------------------------------------------------


class TestIdeationEngineChatTurn(unittest.TestCase):
    """Tests for IdeationEngine.chat_turn()."""

    def setUp(self) -> None:
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def tearDown(self) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

    @patch("local_cli.ideation.Spinner")
    def test_basic_chat_turn(self, mock_spinner_cls: MagicMock) -> None:
        """chat_turn sends user message and returns assistant response."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        chunks = _make_chat_chunks(["Great idea!"])
        client.chat_stream.return_value = iter(chunks)

        engine = IdeationEngine(client)
        engine.start_session()
        result = engine.chat_turn("Brainstorm features", "qwen3:8b")

        self.assertEqual(result, "Great idea!")

    @patch("local_cli.ideation.Spinner")
    def test_messages_appended_to_history(self, mock_spinner_cls: MagicMock) -> None:
        """Both user and assistant messages are appended to history."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        chunks = _make_chat_chunks(["Response"])
        client.chat_stream.return_value = iter(chunks)

        engine = IdeationEngine(client)
        engine.start_session()
        engine.chat_turn("Hello", "qwen3:8b")

        history = engine.get_history()
        self.assertEqual(len(history), 3)  # system + user + assistant
        self.assertEqual(history[0]["role"], "system")
        self.assertEqual(history[1]["role"], "user")
        self.assertEqual(history[1]["content"], "Hello")
        self.assertEqual(history[2]["role"], "assistant")
        self.assertEqual(history[2]["content"], "Response")

    @patch("local_cli.ideation.Spinner")
    def test_chat_stream_called_without_tools(self, mock_spinner_cls: MagicMock) -> None:
        """chat_turn calls chat_stream without the tools parameter."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        chunks = _make_chat_chunks(["Ok"])
        client.chat_stream.return_value = iter(chunks)

        engine = IdeationEngine(client)
        engine.start_session()
        engine.chat_turn("Idea", "qwen3:8b")

        call_kwargs = client.chat_stream.call_args
        # tools should not be passed.
        self.assertNotIn("tools", call_kwargs.kwargs)

    @patch("local_cli.ideation.Spinner")
    def test_think_parameter_passed_to_chat_stream(self, mock_spinner_cls: MagicMock) -> None:
        """chat_turn passes think parameter to chat_stream."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        chunks = _make_chat_chunks(["Thinking..."])
        client.chat_stream.return_value = iter(chunks)

        engine = IdeationEngine(client)
        engine.start_session()
        engine.chat_turn("Deep thought", "qwen3:8b", think=True)

        call_kwargs = client.chat_stream.call_args
        self.assertEqual(call_kwargs.kwargs.get("think"), True)

    @patch("local_cli.ideation.Spinner")
    def test_think_none_omits_parameter(self, mock_spinner_cls: MagicMock) -> None:
        """chat_turn with think=None omits think from chat_stream call."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        chunks = _make_chat_chunks(["No think"])
        client.chat_stream.return_value = iter(chunks)

        engine = IdeationEngine(client)
        engine.start_session()
        engine.chat_turn("Quick thought", "qwen3:8b", think=None)

        call_kwargs = client.chat_stream.call_args
        self.assertIsNone(call_kwargs.kwargs.get("think"))

    @patch("local_cli.ideation.Spinner")
    def test_auto_starts_session_if_not_active(self, mock_spinner_cls: MagicMock) -> None:
        """chat_turn auto-starts a session if one is not active."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        chunks = _make_chat_chunks(["Auto started"])
        client.chat_stream.return_value = iter(chunks)

        engine = IdeationEngine(client)
        # Don't call start_session -- chat_turn should auto-start.
        result = engine.chat_turn("Hello", "qwen3:8b")

        self.assertEqual(result, "Auto started")
        self.assertTrue(engine.has_session)

    @patch("local_cli.ideation.Spinner")
    def test_multi_turn_preserves_history(self, mock_spinner_cls: MagicMock) -> None:
        """Multiple chat_turn calls preserve conversation context."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()

        chunks1 = _make_chat_chunks(["First answer"])
        chunks2 = _make_chat_chunks(["Second answer"])
        client.chat_stream.side_effect = [iter(chunks1), iter(chunks2)]

        engine = IdeationEngine(client)
        engine.start_session()

        engine.chat_turn("First question", "qwen3:8b")
        engine.chat_turn("Follow up", "qwen3:8b")

        history = engine.get_history()
        # system + user1 + assistant1 + user2 + assistant2
        self.assertEqual(len(history), 5)
        self.assertEqual(history[1]["content"], "First question")
        self.assertEqual(history[2]["content"], "First answer")
        self.assertEqual(history[3]["content"], "Follow up")
        self.assertEqual(history[4]["content"], "Second answer")

    @patch("local_cli.ideation.Spinner")
    def test_messages_sent_to_chat_stream(self, mock_spinner_cls: MagicMock) -> None:
        """chat_turn passes the accumulated message history to chat_stream."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()

        # Capture message list lengths at the time of each call.
        captured_lengths: list[int] = []

        def _capture_chat_stream(model: str, messages: list, **kwargs: Any) -> Any:
            captured_lengths.append(len(messages))
            idx = len(captured_lengths) - 1
            chunks = _make_chat_chunks([f"R{idx + 1}"])
            return iter(chunks)

        client.chat_stream.side_effect = _capture_chat_stream

        engine = IdeationEngine(client)
        engine.start_session("Be creative")
        engine.chat_turn("Q1", "qwen3:8b")
        engine.chat_turn("Q2", "qwen3:8b")

        # First call: system + Q1 = 2 messages.
        self.assertEqual(captured_lengths[0], 2)
        # Second call: system + Q1 + R1 + Q2 = 4 messages.
        self.assertEqual(captured_lengths[1], 4)

    @patch("local_cli.ideation.Spinner")
    def test_spinner_started_and_stopped(self, mock_spinner_cls: MagicMock) -> None:
        """A spinner is started and stopped during chat_turn."""
        mock_spinner = MagicMock()
        mock_spinner_cls.return_value = mock_spinner
        client = MagicMock()
        chunks = _make_chat_chunks(["Done"])
        client.chat_stream.return_value = iter(chunks)

        engine = IdeationEngine(client)
        engine.start_session()
        engine.chat_turn("Go", "qwen3:8b")

        mock_spinner.start.assert_called_once()


# ---------------------------------------------------------------------------
# IdeationEngine — chat_turn think fallback
# ---------------------------------------------------------------------------


class TestIdeationEngineChatTurnThinkFallback(unittest.TestCase):
    """Tests for think parameter fallback on ProviderRequestError."""

    def setUp(self) -> None:
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def tearDown(self) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

    @patch("local_cli.ideation.Spinner")
    def test_think_fallback_on_provider_error(self, mock_spinner_cls: MagicMock) -> None:
        """If think=True causes ProviderRequestError, retries without think."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()

        # First call with think=True fails, second without think succeeds.
        chunks = _make_chat_chunks(["Fallback response"])
        client.chat_stream.side_effect = [
            ProviderRequestError("think not supported"),
            iter(chunks),
        ]

        engine = IdeationEngine(client)
        engine.start_session()
        result = engine.chat_turn("Test", "qwen3:8b", think=True)

        self.assertEqual(result, "Fallback response")
        self.assertEqual(client.chat_stream.call_count, 2)
        # Warning message should be printed to stderr.
        stderr_output = sys.stderr.getvalue()
        self.assertIn("thinking mode", stderr_output)

    @patch("local_cli.ideation.Spinner")
    def test_think_none_no_fallback(self, mock_spinner_cls: MagicMock) -> None:
        """With think=None, ProviderRequestError is re-raised without fallback."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        client.chat_stream.side_effect = ProviderRequestError("api error")

        engine = IdeationEngine(client)
        engine.start_session()

        with self.assertRaises(ProviderRequestError):
            engine.chat_turn("Test", "qwen3:8b", think=None)

    @patch("local_cli.ideation.Spinner")
    def test_fallback_also_fails_raises_ideation_error(self, mock_spinner_cls: MagicMock) -> None:
        """If both think and non-think attempts fail, raises IdeationError."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        client.chat_stream.side_effect = [
            ProviderRequestError("think not supported"),
            ProviderRequestError("still broken"),
        ]

        engine = IdeationEngine(client)
        engine.start_session()

        with self.assertRaises(IdeationError) as ctx:
            engine.chat_turn("Test", "qwen3:8b", think=True)

        self.assertIn("failed", str(ctx.exception))

    @patch("local_cli.ideation.Spinner")
    def test_user_message_removed_on_failure(self, mock_spinner_cls: MagicMock) -> None:
        """User message is removed from history when chat_turn fails."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        client.chat_stream.side_effect = [
            ProviderRequestError("think not supported"),
            ProviderRequestError("still broken"),
        ]

        engine = IdeationEngine(client)
        engine.start_session()

        with self.assertRaises(IdeationError):
            engine.chat_turn("Should be removed", "qwen3:8b", think=True)

        # Only system message should remain.
        history = engine.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["role"], "system")

    @patch("local_cli.ideation.Spinner")
    def test_stream_error_raises_ideation_error(self, mock_spinner_cls: MagicMock) -> None:
        """ProviderStreamError is wrapped in IdeationError."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()

        def error_stream():
            yield {"message": {"role": "assistant", "content": ""}, "done": False}
            raise ProviderStreamError("connection lost")

        client.chat_stream.return_value = error_stream()

        engine = IdeationEngine(client)
        engine.start_session()

        with self.assertRaises(IdeationError) as ctx:
            engine.chat_turn("Test", "qwen3:8b")

        self.assertIn("stream error", str(ctx.exception))

    @patch("local_cli.ideation.Spinner")
    def test_stream_error_removes_user_message(self, mock_spinner_cls: MagicMock) -> None:
        """User message is removed on ProviderStreamError."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()

        def error_stream():
            yield {"message": {"role": "assistant", "content": ""}, "done": False}
            raise ProviderStreamError("broken")

        client.chat_stream.return_value = error_stream()

        engine = IdeationEngine(client)
        engine.start_session()

        with self.assertRaises(IdeationError):
            engine.chat_turn("Remove me", "qwen3:8b")

        history = engine.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["role"], "system")


# ---------------------------------------------------------------------------
# IdeationEngine — chat_turn keyboard interrupt
# ---------------------------------------------------------------------------


class TestIdeationEngineChatTurnInterrupt(unittest.TestCase):
    """Tests for KeyboardInterrupt handling in chat_turn."""

    def setUp(self) -> None:
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def tearDown(self) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

    @patch("local_cli.ideation.Spinner")
    def test_keyboard_interrupt_returns_empty_string(self, mock_spinner_cls: MagicMock) -> None:
        """KeyboardInterrupt during chat returns empty string."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()

        def interrupted_stream():
            if False:
                yield  # pragma: no cover — makes this a generator
            raise KeyboardInterrupt

        client.chat_stream.return_value = interrupted_stream()

        engine = IdeationEngine(client)
        engine.start_session()
        result = engine.chat_turn("Test", "qwen3:8b")

        self.assertEqual(result, "")

    @patch("local_cli.ideation.Spinner")
    def test_keyboard_interrupt_appends_messages(self, mock_spinner_cls: MagicMock) -> None:
        """User message and empty assistant response are in history after interrupt."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()

        def interrupted_stream():
            if False:
                yield  # pragma: no cover — makes this a generator
            raise KeyboardInterrupt

        client.chat_stream.return_value = interrupted_stream()

        engine = IdeationEngine(client)
        engine.start_session()
        engine.chat_turn("Interrupted", "qwen3:8b")

        history = engine.get_history()
        # system + user + assistant (empty)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[1]["content"], "Interrupted")
        self.assertEqual(history[2]["content"], "")


# ---------------------------------------------------------------------------
# IdeationEngine — single_shot (generate mode)
# ---------------------------------------------------------------------------


class TestIdeationEngineSingleShot(unittest.TestCase):
    """Tests for IdeationEngine.single_shot()."""

    def setUp(self) -> None:
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def tearDown(self) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

    @patch("local_cli.ideation.Spinner")
    def test_basic_single_shot(self, mock_spinner_cls: MagicMock) -> None:
        """single_shot returns generated text."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        chunks = _make_generate_chunks(["Generated text"])
        client.generate_stream.return_value = iter(chunks)

        engine = IdeationEngine(client)
        result = engine.single_shot("A prompt", "qwen3:8b")

        self.assertEqual(result, "Generated text")

    @patch("local_cli.ideation.Spinner")
    def test_single_shot_calls_generate_stream(self, mock_spinner_cls: MagicMock) -> None:
        """single_shot uses generate_stream, not chat_stream."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        chunks = _make_generate_chunks(["Ok"])
        client.generate_stream.return_value = iter(chunks)

        engine = IdeationEngine(client)
        engine.single_shot("Prompt", "qwen3:8b")

        client.generate_stream.assert_called_once()
        client.chat_stream.assert_not_called()

    @patch("local_cli.ideation.Spinner")
    def test_single_shot_passes_prompt_and_model(self, mock_spinner_cls: MagicMock) -> None:
        """single_shot passes prompt and model to generate_stream."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        chunks = _make_generate_chunks(["Ok"])
        client.generate_stream.return_value = iter(chunks)

        engine = IdeationEngine(client)
        engine.single_shot("My prompt", "qwen3:8b")

        call_args = client.generate_stream.call_args
        self.assertEqual(call_args[0][0], "qwen3:8b")
        self.assertEqual(call_args[0][1], "My prompt")

    @patch("local_cli.ideation.Spinner")
    def test_single_shot_does_not_modify_history(self, mock_spinner_cls: MagicMock) -> None:
        """single_shot does not affect ideation message history."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        chunks = _make_generate_chunks(["Result"])
        client.generate_stream.return_value = iter(chunks)

        engine = IdeationEngine(client)
        engine.start_session()
        original_len = len(engine.get_history())

        engine.single_shot("One off", "qwen3:8b")

        self.assertEqual(len(engine.get_history()), original_len)

    @patch("local_cli.ideation.Spinner")
    def test_single_shot_with_think_param(self, mock_spinner_cls: MagicMock) -> None:
        """single_shot passes think parameter to generate_stream."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        chunks = _make_generate_chunks(["Thought"])
        client.generate_stream.return_value = iter(chunks)

        engine = IdeationEngine(client)
        engine.single_shot("Deep", "qwen3:8b", think=True)

        call_kwargs = client.generate_stream.call_args
        self.assertEqual(call_kwargs.kwargs.get("think"), True)

    @patch("local_cli.ideation.Spinner")
    def test_single_shot_think_none_omits_param(self, mock_spinner_cls: MagicMock) -> None:
        """single_shot with think=None does not include think in kwargs."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        chunks = _make_generate_chunks(["Simple"])
        client.generate_stream.return_value = iter(chunks)

        engine = IdeationEngine(client)
        engine.single_shot("Quick", "qwen3:8b", think=None)

        call_kwargs = client.generate_stream.call_args
        self.assertNotIn("think", call_kwargs.kwargs)

    @patch("local_cli.ideation.Spinner")
    def test_single_shot_spinner_started(self, mock_spinner_cls: MagicMock) -> None:
        """A spinner is started during single_shot."""
        mock_spinner = MagicMock()
        mock_spinner_cls.return_value = mock_spinner
        client = MagicMock()
        chunks = _make_generate_chunks(["Ok"])
        client.generate_stream.return_value = iter(chunks)

        engine = IdeationEngine(client)
        engine.single_shot("Test", "qwen3:8b")

        mock_spinner.start.assert_called_once()


# ---------------------------------------------------------------------------
# IdeationEngine — single_shot think fallback
# ---------------------------------------------------------------------------


class TestIdeationEngineSingleShotThinkFallback(unittest.TestCase):
    """Tests for think parameter fallback in single_shot."""

    def setUp(self) -> None:
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def tearDown(self) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

    @patch("local_cli.ideation.Spinner")
    def test_think_fallback_on_provider_error(self, mock_spinner_cls: MagicMock) -> None:
        """If think=True causes error, retries without think."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()

        chunks = _make_generate_chunks(["Fallback result"])
        client.generate_stream.side_effect = [
            ProviderRequestError("think not supported"),
            iter(chunks),
        ]

        engine = IdeationEngine(client)
        result = engine.single_shot("Test", "qwen3:8b", think=True)

        self.assertEqual(result, "Fallback result")
        self.assertEqual(client.generate_stream.call_count, 2)
        stderr_output = sys.stderr.getvalue()
        self.assertIn("thinking mode", stderr_output)

    @patch("local_cli.ideation.Spinner")
    def test_think_none_no_fallback(self, mock_spinner_cls: MagicMock) -> None:
        """With think=None, ProviderRequestError is re-raised."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        client.generate_stream.side_effect = ProviderRequestError("broken")

        engine = IdeationEngine(client)

        with self.assertRaises(ProviderRequestError):
            engine.single_shot("Test", "qwen3:8b", think=None)

    @patch("local_cli.ideation.Spinner")
    def test_fallback_failure_raises_ideation_error(self, mock_spinner_cls: MagicMock) -> None:
        """If both attempts fail, raises IdeationError."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        client.generate_stream.side_effect = [
            ProviderRequestError("no think"),
            ProviderRequestError("still broken"),
        ]

        engine = IdeationEngine(client)

        with self.assertRaises(IdeationError) as ctx:
            engine.single_shot("Test", "qwen3:8b", think=True)

        self.assertIn("failed", str(ctx.exception))

    @patch("local_cli.ideation.Spinner")
    def test_stream_error_raises_ideation_error(self, mock_spinner_cls: MagicMock) -> None:
        """ProviderStreamError is wrapped in IdeationError."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()

        def error_stream():
            yield {"response": "partial", "done": False}
            raise ProviderStreamError("stream lost")

        client.generate_stream.return_value = error_stream()

        engine = IdeationEngine(client)

        with self.assertRaises(IdeationError) as ctx:
            engine.single_shot("Test", "qwen3:8b")

        self.assertIn("stream error", str(ctx.exception))


# ---------------------------------------------------------------------------
# IdeationEngine — single_shot keyboard interrupt
# ---------------------------------------------------------------------------


class TestIdeationEngineSingleShotInterrupt(unittest.TestCase):
    """Tests for KeyboardInterrupt handling in single_shot."""

    def setUp(self) -> None:
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def tearDown(self) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

    @patch("local_cli.ideation.Spinner")
    def test_keyboard_interrupt_returns_empty_string(self, mock_spinner_cls: MagicMock) -> None:
        """KeyboardInterrupt during single_shot returns empty string."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()

        def interrupted_stream():
            if False:
                yield  # pragma: no cover — makes this a generator
            raise KeyboardInterrupt

        client.generate_stream.return_value = interrupted_stream()

        engine = IdeationEngine(client)
        result = engine.single_shot("Test", "qwen3:8b")

        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# IdeationEngine — history isolation
# ---------------------------------------------------------------------------


class TestIdeationEngineHistoryIsolation(unittest.TestCase):
    """Tests verifying ideation history is separate from main conversation."""

    def setUp(self) -> None:
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def tearDown(self) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

    @patch("local_cli.ideation.Spinner")
    def test_history_separate_from_external_list(self, mock_spinner_cls: MagicMock) -> None:
        """Ideation history is a separate list from any external conversation."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()
        chunks = _make_chat_chunks(["Idea"])
        client.chat_stream.return_value = iter(chunks)

        # Simulate a main conversation.
        main_messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Main conversation"},
        ]

        engine = IdeationEngine(client)
        engine.start_session()
        engine.chat_turn("Brainstorm", "qwen3:8b")

        # Main conversation should not be affected.
        self.assertEqual(len(main_messages), 1)
        self.assertEqual(main_messages[0]["content"], "Main conversation")

        # Ideation history should have its own messages.
        ideation_history = engine.get_history()
        self.assertEqual(len(ideation_history), 3)  # system + user + assistant

    @patch("local_cli.ideation.Spinner")
    def test_history_preserved_across_exit_reentry(self, mock_spinner_cls: MagicMock) -> None:
        """Ideation history is preserved if engine is not cleared."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()

        chunks1 = _make_chat_chunks(["First response"])
        chunks2 = _make_chat_chunks(["After reentry"])
        client.chat_stream.side_effect = [iter(chunks1), iter(chunks2)]

        engine = IdeationEngine(client)
        engine.start_session()
        engine.chat_turn("Before exit", "qwen3:8b")

        # "Exit" ideation (conceptually) -- just stop calling chat_turn.
        first_len = len(engine.get_history())

        # "Re-enter" ideation -- call chat_turn again.
        engine.chat_turn("After reentry", "qwen3:8b")

        # History should have grown by 2 (user + assistant).
        self.assertEqual(len(engine.get_history()), first_len + 2)

    @patch("local_cli.ideation.Spinner")
    def test_clear_then_restart(self, mock_spinner_cls: MagicMock) -> None:
        """After clear_history and start_session, history starts fresh."""
        mock_spinner_cls.return_value = MagicMock()
        client = MagicMock()

        chunks1 = _make_chat_chunks(["R1"])
        chunks2 = _make_chat_chunks(["R2"])
        client.chat_stream.side_effect = [iter(chunks1), iter(chunks2)]

        engine = IdeationEngine(client)
        engine.start_session()
        engine.chat_turn("Old", "qwen3:8b")

        engine.clear_history()
        engine.start_session("Fresh start")
        engine.chat_turn("New", "qwen3:8b")

        history = engine.get_history()
        self.assertEqual(len(history), 3)  # system + user + assistant
        self.assertEqual(history[0]["content"], "Fresh start")
        self.assertEqual(history[1]["content"], "New")

    def test_two_engines_independent_histories(self) -> None:
        """Two IdeationEngine instances have independent histories."""
        client = MagicMock()
        engine_a = IdeationEngine(client)
        engine_b = IdeationEngine(client)

        engine_a.start_session("Prompt A")
        engine_b.start_session("Prompt B")

        self.assertEqual(engine_a.get_history()[0]["content"], "Prompt A")
        self.assertEqual(engine_b.get_history()[0]["content"], "Prompt B")

        engine_a.get_history().append({"role": "user", "content": "A only"})
        self.assertEqual(len(engine_a.get_history()), 2)
        self.assertEqual(len(engine_b.get_history()), 1)


# ---------------------------------------------------------------------------
# IdeationError exception hierarchy
# ---------------------------------------------------------------------------


class TestIdeationErrorHierarchy(unittest.TestCase):
    """Tests for IdeationError exception."""

    def test_is_exception(self) -> None:
        """IdeationError is an Exception subclass."""
        self.assertTrue(issubclass(IdeationError, Exception))

    def test_message_preserved(self) -> None:
        """Error message is preserved."""
        err = IdeationError("something failed")
        self.assertEqual(str(err), "something failed")

    def test_can_be_caught_as_exception(self) -> None:
        """IdeationError can be caught as a generic Exception."""
        with self.assertRaises(Exception):
            raise IdeationError("test")


# ---------------------------------------------------------------------------
# Default system prompt constant
# ---------------------------------------------------------------------------


class TestDefaultSystemPrompt(unittest.TestCase):
    """Tests for _DEFAULT_SYSTEM_PROMPT constant."""

    def test_is_string(self) -> None:
        """Default system prompt is a non-empty string."""
        self.assertIsInstance(_DEFAULT_SYSTEM_PROMPT, str)
        self.assertTrue(len(_DEFAULT_SYSTEM_PROMPT) > 0)

    def test_mentions_brainstorming(self) -> None:
        """Default prompt mentions its purpose."""
        self.assertIn("brainstorming", _DEFAULT_SYSTEM_PROMPT.lower())

    def test_mentions_no_tools(self) -> None:
        """Default prompt indicates no tool access."""
        self.assertIn("tool", _DEFAULT_SYSTEM_PROMPT.lower())


if __name__ == "__main__":
    unittest.main()
