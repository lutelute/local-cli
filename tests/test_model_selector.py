"""Tests for local_cli.model_selector module."""

import unittest
from unittest.mock import MagicMock, patch

from local_cli.model_selector import (
    _build_model_display_data,
    _format_size,
    _select_model_simple,
)


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


if __name__ == "__main__":
    unittest.main()
