"""Tests for local_cli.model_selector module."""

import curses
import unittest
from unittest.mock import MagicMock, patch

from local_cli.cli import _SLASH_COMMANDS, _ReplContext, _handle_slash_command, build_parser
from local_cli.config import Config
from local_cli.model_selector import (
    _build_model_display_data,
    _curses_main,
    _format_size,
    _select_model_simple,
    select_model_interactive,
)
from local_cli.ollama_client import OllamaClient, OllamaConnectionError


class TestFormatSize(unittest.TestCase):
    """Tests for _format_size() byte formatting helper."""

    def test_gigabytes(self) -> None:
        """Sizes >= 1 billion bytes are formatted as GB."""
        self.assertEqual(_format_size(5_200_000_000), "5.2 GB")

    def test_gigabytes_exact(self) -> None:
        """Exactly 1 GB boundary is formatted as GB."""
        self.assertEqual(_format_size(1_000_000_000), "1.0 GB")

    def test_megabytes(self) -> None:
        """Sizes >= 1 million bytes but < 1 billion are formatted as MB."""
        self.assertEqual(_format_size(450_000_000), "450.0 MB")

    def test_megabytes_exact(self) -> None:
        """Exactly 1 MB boundary is formatted as MB."""
        self.assertEqual(_format_size(1_000_000), "1.0 MB")

    def test_kilobytes(self) -> None:
        """Sizes < 1 million bytes are formatted as KB."""
        self.assertEqual(_format_size(500_000), "500.0 KB")

    def test_zero_bytes(self) -> None:
        """Zero bytes is formatted as KB."""
        self.assertEqual(_format_size(0), "0.0 KB")

    def test_small_bytes(self) -> None:
        """Very small byte counts are formatted as KB with decimals."""
        self.assertEqual(_format_size(1500), "1.5 KB")

    def test_large_gigabytes(self) -> None:
        """Multi-digit GB values are formatted correctly."""
        self.assertEqual(_format_size(12_500_000_000), "12.5 GB")


class TestBuildModelDisplayData(unittest.TestCase):
    """Tests for _build_model_display_data() metadata extraction."""

    def test_full_metadata(self) -> None:
        """All fields are extracted from a complete model info dict."""
        model = {
            "name": "qwen3:8b",
            "size": 5_200_000_000,
            "details": {
                "parameter_size": "8B",
                "quantization_level": "Q4_K_M",
                "family": "qwen3",
            },
        }
        result = _build_model_display_data(model)

        self.assertEqual(result["name"], "qwen3:8b")
        self.assertEqual(result["size"], "5.2 GB")
        self.assertEqual(result["parameter_size"], "8B")
        self.assertEqual(result["quantization_level"], "Q4_K_M")
        self.assertEqual(result["family"], "qwen3")

    def test_missing_details(self) -> None:
        """Missing 'details' key defaults detail fields to N/A."""
        model = {
            "name": "custom-model",
            "size": 2_000_000_000,
        }
        result = _build_model_display_data(model)

        self.assertEqual(result["name"], "custom-model")
        self.assertEqual(result["size"], "2.0 GB")
        self.assertEqual(result["parameter_size"], "N/A")
        self.assertEqual(result["quantization_level"], "N/A")
        self.assertEqual(result["family"], "N/A")

    def test_missing_fields_in_details(self) -> None:
        """Missing fields within the 'details' dict default to N/A."""
        model = {
            "name": "partial-model",
            "size": 800_000_000,
            "details": {
                "family": "llama",
            },
        }
        result = _build_model_display_data(model)

        self.assertEqual(result["name"], "partial-model")
        self.assertEqual(result["size"], "800.0 MB")
        self.assertEqual(result["parameter_size"], "N/A")
        self.assertEqual(result["quantization_level"], "N/A")
        self.assertEqual(result["family"], "llama")

    def test_empty_dict(self) -> None:
        """Empty model dict defaults all fields to N/A."""
        result = _build_model_display_data({})

        self.assertEqual(result["name"], "N/A")
        self.assertEqual(result["size"], "N/A")
        self.assertEqual(result["parameter_size"], "N/A")
        self.assertEqual(result["quantization_level"], "N/A")
        self.assertEqual(result["family"], "N/A")

    def test_none_details(self) -> None:
        """Explicit None for 'details' is handled gracefully."""
        model = {
            "name": "test-model",
            "size": 100_000_000,
            "details": None,
        }
        result = _build_model_display_data(model)

        self.assertEqual(result["name"], "test-model")
        self.assertEqual(result["size"], "100.0 MB")
        self.assertEqual(result["parameter_size"], "N/A")
        self.assertEqual(result["quantization_level"], "N/A")
        self.assertEqual(result["family"], "N/A")

    def test_zero_size(self) -> None:
        """Zero size is displayed as N/A."""
        model = {"name": "zero-size", "size": 0}
        result = _build_model_display_data(model)

        self.assertEqual(result["size"], "N/A")

    def test_negative_size(self) -> None:
        """Negative size is displayed as N/A."""
        model = {"name": "neg-size", "size": -100}
        result = _build_model_display_data(model)

        self.assertEqual(result["size"], "N/A")


class TestSelectModelSimple(unittest.TestCase):
    """Tests for _select_model_simple() numbered-list fallback selector."""

    def _make_models(self) -> list[dict]:
        """Create a sample list of model info dicts."""
        return [
            {
                "name": "qwen3:8b",
                "size": 5_200_000_000,
                "details": {
                    "parameter_size": "8B",
                    "quantization_level": "Q4_K_M",
                    "family": "qwen3",
                },
            },
            {
                "name": "gemma3:4b",
                "size": 3_300_000_000,
                "details": {
                    "parameter_size": "4B",
                    "quantization_level": "Q4_0",
                    "family": "gemma3",
                },
            },
            {
                "name": "llama3.2:latest",
                "size": 2_000_000_000,
                "details": {
                    "parameter_size": "3B",
                    "quantization_level": "Q4_K_M",
                    "family": "llama",
                },
            },
        ]

    @patch("builtins.input", return_value="1")
    def test_valid_selection_first(self, mock_input: MagicMock) -> None:
        """Selecting index 1 returns the first model name."""
        result = _select_model_simple(self._make_models())

        self.assertEqual(result, "qwen3:8b")
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="3")
    def test_valid_selection_last(self, mock_input: MagicMock) -> None:
        """Selecting the last index returns the last model name."""
        result = _select_model_simple(self._make_models())

        self.assertEqual(result, "llama3.2:latest")
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="2")
    def test_valid_selection_middle(self, mock_input: MagicMock) -> None:
        """Selecting a middle index returns the correct model name."""
        result = _select_model_simple(self._make_models())

        self.assertEqual(result, "gemma3:4b")

    @patch("builtins.input", return_value="q")
    def test_cancel_with_q(self, mock_input: MagicMock) -> None:
        """Typing 'q' cancels selection and returns None."""
        result = _select_model_simple(self._make_models())

        self.assertIsNone(result)

    @patch("builtins.input", return_value="Q")
    def test_cancel_with_uppercase_q(self, mock_input: MagicMock) -> None:
        """Typing 'Q' (uppercase) also cancels selection."""
        result = _select_model_simple(self._make_models())

        self.assertIsNone(result)

    @patch("builtins.input", return_value="")
    def test_cancel_with_empty(self, mock_input: MagicMock) -> None:
        """Empty input cancels selection and returns None."""
        result = _select_model_simple(self._make_models())

        self.assertIsNone(result)

    @patch("builtins.input", side_effect=["abc", "1"])
    def test_invalid_non_numeric_input(self, mock_input: MagicMock) -> None:
        """Non-numeric input shows error and re-prompts."""
        result = _select_model_simple(self._make_models())

        self.assertEqual(result, "qwen3:8b")
        self.assertEqual(mock_input.call_count, 2)

    @patch("builtins.input", side_effect=["0", "1"])
    def test_out_of_range_zero(self, mock_input: MagicMock) -> None:
        """Index 0 (below range) shows error and re-prompts."""
        result = _select_model_simple(self._make_models())

        self.assertEqual(result, "qwen3:8b")
        self.assertEqual(mock_input.call_count, 2)

    @patch("builtins.input", side_effect=["99", "2"])
    def test_out_of_range_high(self, mock_input: MagicMock) -> None:
        """Index above range shows error and re-prompts."""
        result = _select_model_simple(self._make_models())

        self.assertEqual(result, "gemma3:4b")
        self.assertEqual(mock_input.call_count, 2)

    @patch("builtins.input", side_effect=["-1", "1"])
    def test_out_of_range_negative(self, mock_input: MagicMock) -> None:
        """Negative index shows error and re-prompts."""
        result = _select_model_simple(self._make_models())

        self.assertEqual(result, "qwen3:8b")
        self.assertEqual(mock_input.call_count, 2)

    @patch("builtins.input", return_value="1")
    def test_single_model(self, mock_input: MagicMock) -> None:
        """Single model in the list can be selected."""
        models = [
            {
                "name": "only-model:latest",
                "size": 1_000_000_000,
                "details": {
                    "parameter_size": "1B",
                    "quantization_level": "Q4_0",
                    "family": "test",
                },
            },
        ]
        result = _select_model_simple(models)

        self.assertEqual(result, "only-model:latest")

    @patch("builtins.input", return_value="1")
    @patch("sys.stdout")
    def test_current_model_marking(
        self, mock_stdout: MagicMock, mock_input: MagicMock
    ) -> None:
        """The current model is marked with '*' in the output."""
        models = self._make_models()
        _select_model_simple(models, current_model="gemma3:4b")

        # Collect all write() calls into a single output string.
        output = "".join(
            call.args[0] for call in mock_stdout.write.call_args_list
        )
        # The current model line should contain '*' marker.
        for line in output.split("\n"):
            if "gemma3:4b" in line:
                self.assertIn("*", line)
            elif "qwen3:8b" in line:
                self.assertNotIn("*", line)

    def test_empty_model_list(self) -> None:
        """Empty model list returns None immediately."""
        result = _select_model_simple([])

        self.assertIsNone(result)

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_keyboard_interrupt(self, mock_input: MagicMock) -> None:
        """KeyboardInterrupt during input returns None."""
        result = _select_model_simple(self._make_models())

        self.assertIsNone(result)

    @patch("builtins.input", side_effect=EOFError)
    def test_eof_error(self, mock_input: MagicMock) -> None:
        """EOFError during input returns None."""
        result = _select_model_simple(self._make_models())

        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Shared test data helper
# ---------------------------------------------------------------------------


def _make_sample_models() -> list[dict]:
    """Create a reusable sample list of model info dicts."""
    return [
        {
            "name": "qwen3:8b",
            "size": 5_200_000_000,
            "details": {
                "parameter_size": "8B",
                "quantization_level": "Q4_K_M",
                "family": "qwen3",
            },
        },
        {
            "name": "gemma3:4b",
            "size": 3_300_000_000,
            "details": {
                "parameter_size": "4B",
                "quantization_level": "Q4_0",
                "family": "gemma3",
            },
        },
        {
            "name": "llama3.2:latest",
            "size": 2_000_000_000,
            "details": {
                "parameter_size": "3B",
                "quantization_level": "Q4_K_M",
                "family": "llama",
            },
        },
    ]


# ---------------------------------------------------------------------------
# Tests for select_model_interactive() routing
# ---------------------------------------------------------------------------


class TestSelectModelInteractiveRouting(unittest.TestCase):
    """Tests for routing logic in select_model_interactive()."""

    def setUp(self) -> None:
        self.client = MagicMock(spec=OllamaClient)
        self.client.list_models.return_value = _make_sample_models()

    @patch("local_cli.model_selector._select_model_curses", return_value="qwen3:8b")
    @patch("local_cli.model_selector.sys.stdin")
    @patch("local_cli.model_selector._CURSES_AVAILABLE", True)
    def test_tty_routes_to_curses(
        self, mock_stdin: MagicMock, mock_curses_sel: MagicMock
    ) -> None:
        """TTY terminal with curses available routes to _select_model_curses."""
        mock_stdin.isatty.return_value = True

        result = select_model_interactive(self.client, "qwen3:8b")

        self.assertEqual(result, "qwen3:8b")
        mock_curses_sel.assert_called_once_with(
            _make_sample_models(), "qwen3:8b"
        )

    @patch("local_cli.model_selector._select_model_simple", return_value="gemma3:4b")
    @patch("local_cli.model_selector.sys.stdin")
    @patch("local_cli.model_selector._CURSES_AVAILABLE", True)
    def test_non_tty_routes_to_simple(
        self, mock_stdin: MagicMock, mock_simple_sel: MagicMock
    ) -> None:
        """Non-TTY terminal routes to _select_model_simple even if curses available."""
        mock_stdin.isatty.return_value = False

        result = select_model_interactive(self.client, "qwen3:8b")

        self.assertEqual(result, "gemma3:4b")
        mock_simple_sel.assert_called_once_with(
            _make_sample_models(), "qwen3:8b"
        )

    @patch("local_cli.model_selector._select_model_simple", return_value="gemma3:4b")
    @patch("local_cli.model_selector.sys.stdin")
    @patch("local_cli.model_selector._CURSES_AVAILABLE", False)
    def test_curses_unavailable_routes_to_simple(
        self, mock_stdin: MagicMock, mock_simple_sel: MagicMock
    ) -> None:
        """When curses is not available, routes to _select_model_simple."""
        mock_stdin.isatty.return_value = True

        result = select_model_interactive(self.client, "qwen3:8b")

        self.assertEqual(result, "gemma3:4b")
        mock_simple_sel.assert_called_once()

    @patch("local_cli.model_selector._select_model_simple", return_value="llama3.2:latest")
    @patch(
        "local_cli.model_selector._select_model_curses",
        side_effect=curses.error("Terminal too small"),
    )
    @patch("local_cli.model_selector.sys.stdin")
    @patch("local_cli.model_selector._CURSES_AVAILABLE", True)
    def test_curses_error_falls_back_to_simple(
        self,
        mock_stdin: MagicMock,
        mock_curses_sel: MagicMock,
        mock_simple_sel: MagicMock,
    ) -> None:
        """curses.error during TUI falls back to _select_model_simple."""
        mock_stdin.isatty.return_value = True

        result = select_model_interactive(self.client, "qwen3:8b")

        self.assertEqual(result, "llama3.2:latest")
        mock_curses_sel.assert_called_once()
        mock_simple_sel.assert_called_once()

    @patch("local_cli.model_selector._select_model_curses", return_value=None)
    @patch("local_cli.model_selector.sys.stdin")
    @patch("local_cli.model_selector._CURSES_AVAILABLE", True)
    def test_cancel_returns_none(
        self, mock_stdin: MagicMock, mock_curses_sel: MagicMock
    ) -> None:
        """Cancelled selection returns None."""
        mock_stdin.isatty.return_value = True

        result = select_model_interactive(self.client, "qwen3:8b")

        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Tests for empty model list handling
# ---------------------------------------------------------------------------


class TestSelectModelNoModels(unittest.TestCase):
    """Tests for select_model_interactive() with no models available."""

    def setUp(self) -> None:
        self.client = MagicMock(spec=OllamaClient)
        self.client.list_models.return_value = []

    @patch("local_cli.model_selector.sys.stderr")
    def test_empty_list_returns_none(self, mock_stderr: MagicMock) -> None:
        """Empty model list returns None."""
        result = select_model_interactive(self.client, "qwen3:8b")

        self.assertIsNone(result)

    @patch("local_cli.model_selector.sys.stderr")
    def test_empty_list_shows_message(self, mock_stderr: MagicMock) -> None:
        """Empty model list writes a helpful message to stderr."""
        select_model_interactive(self.client, "qwen3:8b")

        output = "".join(
            call_arg.args[0] for call_arg in mock_stderr.write.call_args_list
        )
        self.assertIn("No models found", output)
        self.assertIn("ollama pull", output)


# ---------------------------------------------------------------------------
# Tests for OllamaConnectionError handling
# ---------------------------------------------------------------------------


class TestSelectModelConnectionError(unittest.TestCase):
    """Tests for select_model_interactive() when Ollama is unreachable."""

    def setUp(self) -> None:
        self.client = MagicMock(spec=OllamaClient)
        self.client.list_models.side_effect = OllamaConnectionError(
            "Connection refused"
        )

    @patch("local_cli.model_selector.sys.stderr")
    def test_connection_error_returns_none(
        self, mock_stderr: MagicMock
    ) -> None:
        """OllamaConnectionError returns None instead of raising."""
        result = select_model_interactive(self.client, "qwen3:8b")

        self.assertIsNone(result)

    @patch("local_cli.model_selector.sys.stderr")
    def test_connection_error_shows_message(
        self, mock_stderr: MagicMock
    ) -> None:
        """OllamaConnectionError writes a helpful error message to stderr."""
        select_model_interactive(self.client, "qwen3:8b")

        output = "".join(
            call_arg.args[0] for call_arg in mock_stderr.write.call_args_list
        )
        self.assertIn("Could not connect to Ollama", output)


# ---------------------------------------------------------------------------
# Tests for curses TUI (mocked)
# ---------------------------------------------------------------------------


class TestSelectModelCursesBasic(unittest.TestCase):
    """Tests for _curses_main() curses TUI with mocked stdscr.

    ``_draw_curses_ui`` is patched out because ``curses.ACS_VLINE`` and
    other curses constants are only initialised after ``curses.initscr()``,
    which is not available in headless test environments.  This allows us
    to test the key-handling and navigation logic in isolation.
    """

    def _make_stdscr(self, max_y: int = 24, max_x: int = 80) -> MagicMock:
        """Create a mock curses stdscr with configurable dimensions."""
        stdscr = MagicMock()
        stdscr.getmaxyx.return_value = (max_y, max_x)
        return stdscr

    @patch("local_cli.model_selector._draw_curses_ui")
    def test_enter_selects_first_model(self, mock_draw: MagicMock) -> None:
        """Pressing Enter immediately selects the first model."""
        stdscr = self._make_stdscr()
        stdscr.getch.return_value = ord("\n")

        result = _curses_main(stdscr, _make_sample_models(), "")

        self.assertEqual(result, "qwen3:8b")

    @patch("local_cli.model_selector._draw_curses_ui")
    def test_enter_selects_current_model(self, mock_draw: MagicMock) -> None:
        """Pressing Enter with current_model set selects that model."""
        stdscr = self._make_stdscr()
        stdscr.getch.return_value = ord("\n")

        result = _curses_main(stdscr, _make_sample_models(), "gemma3:4b")

        # The cursor starts on the current model.
        self.assertEqual(result, "gemma3:4b")

    @patch("local_cli.model_selector._draw_curses_ui")
    def test_escape_cancels(self, mock_draw: MagicMock) -> None:
        """Pressing Escape returns None (cancel)."""
        stdscr = self._make_stdscr()
        stdscr.getch.return_value = 27  # Escape

        result = _curses_main(stdscr, _make_sample_models(), "")

        self.assertIsNone(result)

    @patch("local_cli.model_selector._draw_curses_ui")
    def test_q_cancels(self, mock_draw: MagicMock) -> None:
        """Pressing 'q' returns None (cancel)."""
        stdscr = self._make_stdscr()
        stdscr.getch.return_value = ord("q")

        result = _curses_main(stdscr, _make_sample_models(), "")

        self.assertIsNone(result)

    @patch("local_cli.model_selector._draw_curses_ui")
    def test_uppercase_q_cancels(self, mock_draw: MagicMock) -> None:
        """Pressing 'Q' returns None (cancel)."""
        stdscr = self._make_stdscr()
        stdscr.getch.return_value = ord("Q")

        result = _curses_main(stdscr, _make_sample_models(), "")

        self.assertIsNone(result)

    @patch("local_cli.model_selector._draw_curses_ui")
    def test_down_arrow_then_enter(self, mock_draw: MagicMock) -> None:
        """Down arrow moves selection down, Enter selects it."""
        stdscr = self._make_stdscr()
        stdscr.getch.side_effect = [curses.KEY_DOWN, ord("\n")]

        result = _curses_main(stdscr, _make_sample_models(), "")

        self.assertEqual(result, "gemma3:4b")

    @patch("local_cli.model_selector._draw_curses_ui")
    def test_multiple_down_arrows(self, mock_draw: MagicMock) -> None:
        """Multiple Down arrows navigate to later models."""
        stdscr = self._make_stdscr()
        stdscr.getch.side_effect = [
            curses.KEY_DOWN,
            curses.KEY_DOWN,
            ord("\n"),
        ]

        result = _curses_main(stdscr, _make_sample_models(), "")

        self.assertEqual(result, "llama3.2:latest")

    @patch("local_cli.model_selector._draw_curses_ui")
    def test_up_arrow_at_top_stays(self, mock_draw: MagicMock) -> None:
        """Up arrow at top of list does not move past the first item."""
        stdscr = self._make_stdscr()
        stdscr.getch.side_effect = [curses.KEY_UP, ord("\n")]

        result = _curses_main(stdscr, _make_sample_models(), "")

        self.assertEqual(result, "qwen3:8b")

    @patch("local_cli.model_selector._draw_curses_ui")
    def test_down_arrow_at_bottom_stays(self, mock_draw: MagicMock) -> None:
        """Down arrow at bottom of list does not move past the last item."""
        stdscr = self._make_stdscr()
        stdscr.getch.side_effect = [
            curses.KEY_DOWN,
            curses.KEY_DOWN,
            curses.KEY_DOWN,  # Past the end — should stay on last.
            ord("\n"),
        ]

        result = _curses_main(stdscr, _make_sample_models(), "")

        self.assertEqual(result, "llama3.2:latest")

    @patch("local_cli.model_selector._draw_curses_ui")
    def test_up_down_navigation(self, mock_draw: MagicMock) -> None:
        """Up and Down arrows navigate correctly in combination."""
        stdscr = self._make_stdscr()
        stdscr.getch.side_effect = [
            curses.KEY_DOWN,   # -> index 1 (gemma3:4b)
            curses.KEY_DOWN,   # -> index 2 (llama3.2:latest)
            curses.KEY_UP,     # -> index 1 (gemma3:4b)
            ord("\n"),
        ]

        result = _curses_main(stdscr, _make_sample_models(), "")

        self.assertEqual(result, "gemma3:4b")

    @patch("local_cli.model_selector._draw_curses_ui")
    def test_carriage_return_selects(self, mock_draw: MagicMock) -> None:
        """Carriage return (\\r) also confirms selection."""
        stdscr = self._make_stdscr()
        stdscr.getch.return_value = ord("\r")

        result = _curses_main(stdscr, _make_sample_models(), "")

        self.assertEqual(result, "qwen3:8b")

    def test_terminal_too_small_raises(self) -> None:
        """Terminal smaller than minimum dimensions raises curses.error."""
        stdscr = self._make_stdscr(max_y=5, max_x=40)

        with self.assertRaises(curses.error):
            _curses_main(stdscr, _make_sample_models(), "")

    def test_terminal_too_narrow_raises(self) -> None:
        """Terminal too narrow raises curses.error."""
        stdscr = self._make_stdscr(max_y=24, max_x=50)

        with self.assertRaises(curses.error):
            _curses_main(stdscr, _make_sample_models(), "")

    def test_terminal_too_short_raises(self) -> None:
        """Terminal too short raises curses.error."""
        stdscr = self._make_stdscr(max_y=8, max_x=80)

        with self.assertRaises(curses.error):
            _curses_main(stdscr, _make_sample_models(), "")

    @patch("local_cli.model_selector._draw_curses_ui")
    def test_keyboard_interrupt_returns_none(
        self, mock_draw: MagicMock
    ) -> None:
        """KeyboardInterrupt during getch returns None."""
        stdscr = self._make_stdscr()
        stdscr.getch.side_effect = KeyboardInterrupt

        result = _curses_main(stdscr, _make_sample_models(), "")

        self.assertIsNone(result)

    @patch("local_cli.model_selector._draw_curses_ui")
    def test_single_model_enter_selects(self, mock_draw: MagicMock) -> None:
        """Single model in list can be selected with Enter."""
        stdscr = self._make_stdscr()
        stdscr.getch.return_value = ord("\n")
        models = [_make_sample_models()[0]]

        result = _curses_main(stdscr, models, "")

        self.assertEqual(result, "qwen3:8b")

    @patch("local_cli.model_selector._draw_curses_ui")
    def test_current_model_starts_highlighted(
        self, mock_draw: MagicMock
    ) -> None:
        """Current model is initially highlighted when it exists in the list."""
        stdscr = self._make_stdscr()
        stdscr.getch.return_value = ord("\n")

        # Current model is the third one.
        result = _curses_main(
            stdscr, _make_sample_models(), "llama3.2:latest"
        )

        self.assertEqual(result, "llama3.2:latest")

    @patch("local_cli.model_selector._draw_curses_ui")
    def test_draw_ui_called_each_iteration(
        self, mock_draw: MagicMock
    ) -> None:
        """_draw_curses_ui is called on each key press iteration."""
        stdscr = self._make_stdscr()
        stdscr.getch.side_effect = [curses.KEY_DOWN, ord("\n")]

        _curses_main(stdscr, _make_sample_models(), "")

        # Called once for initial draw, once after KEY_DOWN.
        self.assertEqual(mock_draw.call_count, 2)


# ---------------------------------------------------------------------------
# Tests for /models slash command
# ---------------------------------------------------------------------------


class TestSlashModelsCommand(unittest.TestCase):
    """Tests for /models slash command in _handle_slash_command()."""

    def _make_ctx(self) -> _ReplContext:
        """Create a minimal _ReplContext for testing."""
        config = MagicMock(spec=Config)
        config.model = "qwen3:8b"
        client = MagicMock(spec=OllamaClient)
        tools: list = []
        messages = [{"role": "system", "content": "test"}]
        session_manager = MagicMock()
        return _ReplContext(
            config=config,
            client=client,
            tools=tools,
            messages=messages,
            session_manager=session_manager,
            system_prompt="test",
        )

    def test_models_in_slash_commands_dict(self) -> None:
        """/models is registered in _SLASH_COMMANDS."""
        self.assertIn("/models", _SLASH_COMMANDS)

    @patch(
        "local_cli.model_selector.select_model_interactive",
        return_value="gemma3:4b",
    )
    def test_models_command_updates_config(
        self, mock_selector: MagicMock
    ) -> None:
        """/models command updates ctx.config.model on selection."""
        ctx = self._make_ctx()

        result = _handle_slash_command("/models", ctx)

        self.assertTrue(result)
        self.assertEqual(ctx.config.model, "gemma3:4b")
        mock_selector.assert_called_once_with(ctx.client, "qwen3:8b")

    @patch(
        "local_cli.model_selector.select_model_interactive",
        return_value=None,
    )
    def test_models_command_cancel_keeps_model(
        self, mock_selector: MagicMock
    ) -> None:
        """/models command keeps current model when user cancels."""
        ctx = self._make_ctx()

        result = _handle_slash_command("/models", ctx)

        self.assertTrue(result)
        self.assertEqual(ctx.config.model, "qwen3:8b")

    @patch(
        "local_cli.model_selector.select_model_interactive",
        side_effect=Exception("unexpected error"),
    )
    def test_models_command_handles_exception(
        self, mock_selector: MagicMock
    ) -> None:
        """/models command handles unexpected exceptions gracefully."""
        ctx = self._make_ctx()

        result = _handle_slash_command("/models", ctx)

        # Should continue REPL, not crash.
        self.assertTrue(result)
        # Model should be unchanged.
        self.assertEqual(ctx.config.model, "qwen3:8b")

    def test_models_command_continues_repl(self) -> None:
        """/models command returns True to continue the REPL."""
        ctx = self._make_ctx()

        with patch(
            "local_cli.model_selector.select_model_interactive",
            return_value=None,
        ):
            result = _handle_slash_command("/models", ctx)

        self.assertTrue(result)


# ---------------------------------------------------------------------------
# Tests for --select-model CLI flag
# ---------------------------------------------------------------------------


class TestSelectModelFlag(unittest.TestCase):
    """Tests for --select-model argparse flag in build_parser()."""

    def test_flag_parsed_true(self) -> None:
        """--select-model flag sets select_model to True."""
        parser = build_parser()
        args = parser.parse_args(["--select-model"])

        self.assertTrue(args.select_model)

    def test_flag_default_none(self) -> None:
        """Without --select-model, select_model defaults to None."""
        parser = build_parser()
        args = parser.parse_args([])

        self.assertIsNone(args.select_model)

    def test_flag_coexists_with_model(self) -> None:
        """--select-model can be used together with --model."""
        parser = build_parser()
        args = parser.parse_args(["--select-model", "--model", "gemma3:4b"])

        self.assertTrue(args.select_model)
        self.assertEqual(args.model, "gemma3:4b")

    def test_flag_in_help_output(self) -> None:
        """--select-model appears in the parser's help output."""
        parser = build_parser()
        help_text = parser.format_help()

        self.assertIn("--select-model", help_text)

    def test_flag_with_other_flags(self) -> None:
        """--select-model works alongside --debug and --rag flags."""
        parser = build_parser()
        args = parser.parse_args(["--select-model", "--debug", "--rag"])

        self.assertTrue(args.select_model)
        self.assertTrue(args.debug)
        self.assertTrue(args.rag)


if __name__ == "__main__":
    unittest.main()
