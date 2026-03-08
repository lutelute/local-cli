"""Tests for local_cli.health_check module."""

import unittest
from collections import namedtuple
from unittest.mock import MagicMock, patch

from local_cli.health_check import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_WARNING,
    CheckResult,
    _MIN_FREE_DISK_BYTES,
    _STATUS_COLORS,
    _STATUS_SYMBOLS,
    check_disk_space,
    check_model_availability,
    check_ollama_connectivity,
    format_check_result,
    format_health_check,
    run_health_check,
)
from local_cli.ollama_client import OllamaClient, OllamaConnectionError


# ---------------------------------------------------------------------------
# CheckResult tests
# ---------------------------------------------------------------------------


class TestCheckResult(unittest.TestCase):
    """Tests for the CheckResult data class."""

    def test_create_ok_result(self) -> None:
        """CheckResult stores name, status, message, and details."""
        result = CheckResult(
            name="Ollama",
            status=STATUS_OK,
            message="Connected (v0.5.1)",
            details={"version": "0.5.1"},
        )
        self.assertEqual(result.name, "Ollama")
        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(result.message, "Connected (v0.5.1)")
        self.assertEqual(result.details, {"version": "0.5.1"})

    def test_create_result_without_details(self) -> None:
        """CheckResult defaults details to None."""
        result = CheckResult(name="Test", status=STATUS_OK, message="OK")
        self.assertIsNone(result.details)

    def test_repr(self) -> None:
        """CheckResult __repr__ includes name, status, and message."""
        result = CheckResult(name="Test", status=STATUS_OK, message="OK")
        text = repr(result)
        self.assertIn("Test", text)
        self.assertIn("ok", text)
        self.assertIn("OK", text)

    def test_status_constants(self) -> None:
        """Status constants have expected string values."""
        self.assertEqual(STATUS_OK, "ok")
        self.assertEqual(STATUS_WARNING, "warning")
        self.assertEqual(STATUS_ERROR, "error")


# ---------------------------------------------------------------------------
# Ollama connectivity check tests
# ---------------------------------------------------------------------------


class TestCheckOllamaConnectivity(unittest.TestCase):
    """Tests for check_ollama_connectivity()."""

    def test_ollama_reachable(self) -> None:
        """Returns ok status when Ollama responds with version info."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.get_version.return_value = {"version": "0.5.1"}

        result = check_ollama_connectivity(mock_client)

        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual(result.name, "Ollama")
        self.assertIn("0.5.1", result.message)
        self.assertEqual(result.details, {"version": "0.5.1"})
        mock_client.get_version.assert_called_once()

    def test_ollama_unreachable(self) -> None:
        """Returns error status when Ollama connection fails."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.get_version.side_effect = OllamaConnectionError(
            "Connection refused"
        )

        result = check_ollama_connectivity(mock_client)

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertEqual(result.name, "Ollama")
        self.assertIn("Cannot connect", result.message)
        self.assertIn("error", result.details)

    def test_ollama_unknown_version(self) -> None:
        """Returns ok with 'unknown' when version key is missing."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.get_version.return_value = {}

        result = check_ollama_connectivity(mock_client)

        self.assertEqual(result.status, STATUS_OK)
        self.assertIn("unknown", result.message)
        self.assertEqual(result.details, {"version": "unknown"})

    def test_ollama_unexpected_error(self) -> None:
        """Returns error status on unexpected exception."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.get_version.side_effect = RuntimeError("unexpected")

        result = check_ollama_connectivity(mock_client)

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertIn("Unexpected error", result.message)


# ---------------------------------------------------------------------------
# Model availability check tests
# ---------------------------------------------------------------------------


class TestCheckModelAvailability(unittest.TestCase):
    """Tests for check_model_availability()."""

    def test_model_found_exact_match(self) -> None:
        """Returns ok when the exact model name is in the list."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.list_models.return_value = [
            {"name": "qwen3:8b"},
            {"name": "llama3.2:latest"},
        ]

        result = check_model_availability(mock_client, "qwen3:8b")

        self.assertEqual(result.status, STATUS_OK)
        self.assertIn("qwen3:8b", result.message)
        self.assertIn("available", result.message.lower())

    def test_model_found_base_name_match(self) -> None:
        """Returns ok when the model matches the base name without tag."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.list_models.return_value = [
            {"name": "qwen3:8b"},
        ]

        # "qwen3" should match "qwen3:8b" (base name match)
        result = check_model_availability(mock_client, "qwen3")

        self.assertEqual(result.status, STATUS_OK)
        self.assertIn("qwen3", result.message)

    def test_model_not_found(self) -> None:
        """Returns warning when the model is not in the list."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.list_models.return_value = [
            {"name": "llama3.2:latest"},
        ]

        result = check_model_availability(mock_client, "qwen3:8b")

        self.assertEqual(result.status, STATUS_WARNING)
        self.assertIn("not found", result.message)
        self.assertIn("llama3.2:latest", result.message)

    def test_no_models_available(self) -> None:
        """Returns warning when no models are on the server."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.list_models.return_value = []

        result = check_model_availability(mock_client, "qwen3:8b")

        self.assertEqual(result.status, STATUS_WARNING)
        self.assertIn("not found", result.message)
        self.assertIn("none", result.message)

    def test_connection_error(self) -> None:
        """Returns error when Ollama connection fails during model check."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.list_models.side_effect = OllamaConnectionError(
            "Connection refused"
        )

        result = check_model_availability(mock_client, "qwen3:8b")

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertIn("Cannot check models", result.message)

    def test_unexpected_error(self) -> None:
        """Returns error on unexpected exception during model check."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.list_models.side_effect = RuntimeError("boom")

        result = check_model_availability(mock_client, "qwen3:8b")

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertIn("Unexpected error", result.message)

    def test_model_details_include_available_models(self) -> None:
        """Details dict includes available model names."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.list_models.return_value = [
            {"name": "model-a"},
            {"name": "model-b"},
        ]

        result = check_model_availability(mock_client, "model-a")

        self.assertEqual(result.details["model"], "model-a")
        self.assertEqual(
            result.details["available_models"], ["model-a", "model-b"]
        )

    def test_missing_name_key_in_model(self) -> None:
        """Handles models dicts that are missing the 'name' key."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.list_models.return_value = [
            {"size": 1234},  # no "name" key
        ]

        result = check_model_availability(mock_client, "qwen3:8b")

        self.assertEqual(result.status, STATUS_WARNING)
        self.assertIn("not found", result.message)


# ---------------------------------------------------------------------------
# Disk space check tests
# ---------------------------------------------------------------------------

# Named tuple matching shutil.disk_usage return type.
_DiskUsage = namedtuple("_DiskUsage", ["total", "used", "free"])


class TestCheckDiskSpace(unittest.TestCase):
    """Tests for check_disk_space()."""

    @patch("local_cli.health_check.shutil.disk_usage")
    def test_sufficient_disk_space(self, mock_usage: MagicMock) -> None:
        """Returns ok when free space exceeds the threshold."""
        # 10 GB free
        mock_usage.return_value = _DiskUsage(
            total=100 * 1024 ** 3,
            used=90 * 1024 ** 3,
            free=10 * 1024 ** 3,
        )

        result = check_disk_space("/")

        self.assertEqual(result.status, STATUS_OK)
        self.assertIn("10.0 GB free", result.message)
        self.assertEqual(result.details["free_bytes"], 10 * 1024 ** 3)

    @patch("local_cli.health_check.shutil.disk_usage")
    def test_low_disk_space(self, mock_usage: MagicMock) -> None:
        """Returns warning when free space is below the threshold."""
        # 500 MB free (below 1 GB threshold)
        free_bytes = 500 * 1024 ** 2
        mock_usage.return_value = _DiskUsage(
            total=100 * 1024 ** 3,
            used=100 * 1024 ** 3 - free_bytes,
            free=free_bytes,
        )

        result = check_disk_space("/")

        self.assertEqual(result.status, STATUS_WARNING)
        self.assertIn("Low disk space", result.message)

    @patch("local_cli.health_check.shutil.disk_usage")
    def test_exact_threshold(self, mock_usage: MagicMock) -> None:
        """Returns ok when free space is exactly at the threshold."""
        mock_usage.return_value = _DiskUsage(
            total=100 * 1024 ** 3,
            used=99 * 1024 ** 3,
            free=_MIN_FREE_DISK_BYTES,
        )

        result = check_disk_space("/")

        self.assertEqual(result.status, STATUS_OK)

    @patch("local_cli.health_check.shutil.disk_usage")
    def test_just_below_threshold(self, mock_usage: MagicMock) -> None:
        """Returns warning when free space is 1 byte below the threshold."""
        mock_usage.return_value = _DiskUsage(
            total=100 * 1024 ** 3,
            used=99 * 1024 ** 3 + 1,
            free=_MIN_FREE_DISK_BYTES - 1,
        )

        result = check_disk_space("/")

        self.assertEqual(result.status, STATUS_WARNING)

    @patch("local_cli.health_check.shutil.disk_usage")
    def test_disk_check_os_error(self, mock_usage: MagicMock) -> None:
        """Returns error when disk_usage raises OSError."""
        mock_usage.side_effect = OSError("Permission denied")

        result = check_disk_space("/")

        self.assertEqual(result.status, STATUS_ERROR)
        self.assertIn("Cannot check disk space", result.message)

    @patch("local_cli.health_check.shutil.disk_usage")
    def test_custom_path(self, mock_usage: MagicMock) -> None:
        """check_disk_space passes the custom path to shutil.disk_usage."""
        mock_usage.return_value = _DiskUsage(
            total=50 * 1024 ** 3,
            used=20 * 1024 ** 3,
            free=30 * 1024 ** 3,
        )

        result = check_disk_space("/home/user")

        mock_usage.assert_called_once_with("/home/user")
        self.assertEqual(result.status, STATUS_OK)

    @patch("local_cli.health_check.shutil.disk_usage")
    def test_custom_min_free_bytes(self, mock_usage: MagicMock) -> None:
        """check_disk_space uses a custom min_free_bytes threshold."""
        # 2 GB free, but threshold is 5 GB
        mock_usage.return_value = _DiskUsage(
            total=100 * 1024 ** 3,
            used=98 * 1024 ** 3,
            free=2 * 1024 ** 3,
        )

        result = check_disk_space("/", min_free_bytes=5 * 1024 ** 3)

        self.assertEqual(result.status, STATUS_WARNING)
        self.assertIn("Low disk space", result.message)

    @patch("local_cli.health_check.shutil.disk_usage")
    def test_disk_details_structure(self, mock_usage: MagicMock) -> None:
        """Details dict includes total, used, and free bytes."""
        mock_usage.return_value = _DiskUsage(
            total=100_000,
            used=60_000,
            free=40_000,
        )

        result = check_disk_space("/", min_free_bytes=0)

        self.assertEqual(result.details["total_bytes"], 100_000)
        self.assertEqual(result.details["used_bytes"], 60_000)
        self.assertEqual(result.details["free_bytes"], 40_000)


# ---------------------------------------------------------------------------
# Aggregated health check tests
# ---------------------------------------------------------------------------


class TestRunHealthCheck(unittest.TestCase):
    """Tests for run_health_check() — the aggregated check runner."""

    @patch("local_cli.health_check.shutil.disk_usage")
    def test_all_checks_pass(self, mock_usage: MagicMock) -> None:
        """Returns three ok results when everything is healthy."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.get_version.return_value = {"version": "0.5.1"}
        mock_client.list_models.return_value = [{"name": "qwen3:8b"}]
        mock_usage.return_value = _DiskUsage(
            total=100 * 1024 ** 3,
            used=50 * 1024 ** 3,
            free=50 * 1024 ** 3,
        )

        results = run_health_check(mock_client, "qwen3:8b")

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].status, STATUS_OK)  # Ollama
        self.assertEqual(results[1].status, STATUS_OK)  # Model
        self.assertEqual(results[2].status, STATUS_OK)  # Disk

    @patch("local_cli.health_check.shutil.disk_usage")
    def test_ollama_down_skips_model_check(
        self, mock_usage: MagicMock
    ) -> None:
        """Skips model check when Ollama is unreachable."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.get_version.side_effect = OllamaConnectionError("refused")
        mock_usage.return_value = _DiskUsage(
            total=100 * 1024 ** 3,
            used=50 * 1024 ** 3,
            free=50 * 1024 ** 3,
        )

        results = run_health_check(mock_client, "qwen3:8b")

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].status, STATUS_ERROR)    # Ollama
        self.assertEqual(results[1].status, STATUS_WARNING)  # Model (skipped)
        self.assertIn("Skipped", results[1].message)
        self.assertEqual(results[2].status, STATUS_OK)       # Disk
        # Model check was NOT called because Ollama was down.
        mock_client.list_models.assert_not_called()

    @patch("local_cli.health_check.shutil.disk_usage")
    def test_model_not_found_warning(self, mock_usage: MagicMock) -> None:
        """Returns warning for model when it's not available."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.get_version.return_value = {"version": "0.5.1"}
        mock_client.list_models.return_value = [{"name": "llama3.2:latest"}]
        mock_usage.return_value = _DiskUsage(
            total=100 * 1024 ** 3,
            used=50 * 1024 ** 3,
            free=50 * 1024 ** 3,
        )

        results = run_health_check(mock_client, "qwen3:8b")

        self.assertEqual(results[0].status, STATUS_OK)       # Ollama
        self.assertEqual(results[1].status, STATUS_WARNING)  # Model
        self.assertEqual(results[2].status, STATUS_OK)       # Disk

    @patch("local_cli.health_check.shutil.disk_usage")
    def test_low_disk_space_warning(self, mock_usage: MagicMock) -> None:
        """Returns warning for disk when space is low."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.get_version.return_value = {"version": "0.5.1"}
        mock_client.list_models.return_value = [{"name": "qwen3:8b"}]
        # 100 MB free (below 1 GB)
        mock_usage.return_value = _DiskUsage(
            total=100 * 1024 ** 3,
            used=100 * 1024 ** 3 - 100 * 1024 ** 2,
            free=100 * 1024 ** 2,
        )

        results = run_health_check(mock_client, "qwen3:8b")

        self.assertEqual(results[0].status, STATUS_OK)       # Ollama
        self.assertEqual(results[1].status, STATUS_OK)       # Model
        self.assertEqual(results[2].status, STATUS_WARNING)  # Disk

    @patch("local_cli.health_check.shutil.disk_usage")
    def test_custom_disk_path(self, mock_usage: MagicMock) -> None:
        """run_health_check passes disk_path to check_disk_space."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.get_version.return_value = {"version": "0.5.1"}
        mock_client.list_models.return_value = [{"name": "qwen3:8b"}]
        mock_usage.return_value = _DiskUsage(
            total=100 * 1024 ** 3,
            used=50 * 1024 ** 3,
            free=50 * 1024 ** 3,
        )

        run_health_check(mock_client, "qwen3:8b", disk_path="/tmp")

        mock_usage.assert_called_once_with("/tmp")

    @patch("local_cli.health_check.shutil.disk_usage")
    def test_all_checks_fail(self, mock_usage: MagicMock) -> None:
        """All checks return non-ok when everything is broken."""
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.get_version.side_effect = OllamaConnectionError("refused")
        mock_usage.side_effect = OSError("disk error")

        results = run_health_check(mock_client, "qwen3:8b")

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].status, STATUS_ERROR)    # Ollama
        self.assertEqual(results[1].status, STATUS_WARNING)  # Model skipped
        self.assertEqual(results[2].status, STATUS_ERROR)    # Disk


# ---------------------------------------------------------------------------
# Output formatting tests
# ---------------------------------------------------------------------------


class TestFormatCheckResult(unittest.TestCase):
    """Tests for format_check_result()."""

    def test_format_ok_with_color(self) -> None:
        """Formats an ok result with green color and checkmark."""
        result = CheckResult(
            name="Ollama", status=STATUS_OK, message="Connected (v0.5.1)"
        )
        formatted = format_check_result(result, color=True)

        self.assertIn("✓", formatted)
        self.assertIn("Ollama", formatted)
        self.assertIn("Connected (v0.5.1)", formatted)
        self.assertIn("\033[32m", formatted)  # Green ANSI code

    def test_format_warning_with_color(self) -> None:
        """Formats a warning result with yellow color and warning symbol."""
        result = CheckResult(
            name="Model", status=STATUS_WARNING, message="Not found"
        )
        formatted = format_check_result(result, color=True)

        self.assertIn("⚠", formatted)
        self.assertIn("\033[33m", formatted)  # Yellow ANSI code

    def test_format_error_with_color(self) -> None:
        """Formats an error result with red color and X symbol."""
        result = CheckResult(
            name="Ollama", status=STATUS_ERROR, message="Connection failed"
        )
        formatted = format_check_result(result, color=True)

        self.assertIn("✗", formatted)
        self.assertIn("\033[31m", formatted)  # Red ANSI code

    def test_format_without_color(self) -> None:
        """Formats a result without ANSI color codes."""
        result = CheckResult(
            name="Ollama", status=STATUS_OK, message="Connected"
        )
        formatted = format_check_result(result, color=False)

        self.assertIn("✓", formatted)
        self.assertIn("Ollama", formatted)
        self.assertNotIn("\033[", formatted)  # No ANSI codes

    def test_format_includes_indentation(self) -> None:
        """Formatted output starts with indentation."""
        result = CheckResult(name="Test", status=STATUS_OK, message="OK")
        formatted = format_check_result(result, color=False)
        self.assertTrue(formatted.startswith("  "))


class TestFormatHealthCheck(unittest.TestCase):
    """Tests for format_health_check()."""

    def test_format_multiple_results(self) -> None:
        """Formats multiple results as a multi-line string."""
        results = [
            CheckResult(name="Ollama", status=STATUS_OK, message="OK"),
            CheckResult(name="Model", status=STATUS_WARNING, message="Warn"),
            CheckResult(name="Disk", status=STATUS_ERROR, message="Error"),
        ]

        formatted = format_health_check(results, color=False)
        lines = formatted.split("\n")

        self.assertEqual(len(lines), 3)
        self.assertIn("Ollama", lines[0])
        self.assertIn("Model", lines[1])
        self.assertIn("Disk", lines[2])

    def test_format_empty_results(self) -> None:
        """Formats an empty result list as an empty string."""
        formatted = format_health_check([], color=False)
        self.assertEqual(formatted, "")

    def test_format_single_result(self) -> None:
        """Formats a single result without trailing newline."""
        results = [
            CheckResult(name="Ollama", status=STATUS_OK, message="OK"),
        ]
        formatted = format_health_check(results, color=False)
        self.assertNotIn("\n", formatted)
        self.assertIn("Ollama", formatted)

    def test_format_with_color_includes_ansi(self) -> None:
        """Color mode includes ANSI codes in output."""
        results = [
            CheckResult(name="Ollama", status=STATUS_OK, message="OK"),
        ]
        formatted = format_health_check(results, color=True)
        self.assertIn("\033[32m", formatted)


# ---------------------------------------------------------------------------
# Status constants and symbols tests
# ---------------------------------------------------------------------------


class TestStatusMappings(unittest.TestCase):
    """Tests for status-related constant dictionaries."""

    def test_all_statuses_have_colors(self) -> None:
        """Every status code has a corresponding ANSI color."""
        for status in (STATUS_OK, STATUS_WARNING, STATUS_ERROR):
            self.assertIn(status, _STATUS_COLORS)

    def test_all_statuses_have_symbols(self) -> None:
        """Every status code has a corresponding Unicode symbol."""
        for status in (STATUS_OK, STATUS_WARNING, STATUS_ERROR):
            self.assertIn(status, _STATUS_SYMBOLS)

    def test_symbols_are_single_char(self) -> None:
        """Status symbols are single characters."""
        for symbol in _STATUS_SYMBOLS.values():
            self.assertEqual(len(symbol), 1)


if __name__ == "__main__":
    unittest.main()
