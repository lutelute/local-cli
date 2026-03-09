"""Tests for local_cli.clipboard module."""

import platform
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from local_cli.clipboard import (
    ClipboardError,
    ClipboardUnavailableError,
    _MAX_COPY_LENGTH,
    _detect_clipboard_command,
    _sanitize_text,
    copy_to_clipboard,
    get_clipboard_command,
)


class TestDetectClipboardCommand(unittest.TestCase):
    """Tests for _detect_clipboard_command()."""

    @patch("local_cli.clipboard.platform.system", return_value="Darwin")
    @patch("local_cli.clipboard.shutil.which", return_value="/usr/bin/pbcopy")
    def test_macos_returns_pbcopy(
        self, mock_which: MagicMock, mock_system: MagicMock
    ) -> None:
        """On macOS, returns pbcopy command."""
        result = _detect_clipboard_command()
        self.assertEqual(result, ["pbcopy"])
        mock_which.assert_called_with("pbcopy")

    @patch("local_cli.clipboard.platform.system", return_value="Linux")
    @patch(
        "local_cli.clipboard.shutil.which",
        side_effect=lambda x: "/usr/bin/xclip" if x == "xclip" else None,
    )
    def test_linux_returns_xclip(
        self, mock_which: MagicMock, mock_system: MagicMock
    ) -> None:
        """On Linux with xclip installed, returns xclip command."""
        result = _detect_clipboard_command()
        self.assertEqual(result, ["xclip", "-selection", "clipboard"])

    @patch("local_cli.clipboard.platform.system", return_value="Linux")
    @patch(
        "local_cli.clipboard.shutil.which",
        side_effect=lambda x: "/usr/bin/xsel" if x == "xsel" else None,
    )
    def test_linux_falls_back_to_xsel(
        self, mock_which: MagicMock, mock_system: MagicMock
    ) -> None:
        """On Linux without xclip, falls back to xsel."""
        result = _detect_clipboard_command()
        self.assertEqual(result, ["xsel", "--clipboard", "--input"])

    @patch("local_cli.clipboard.platform.system", return_value="Linux")
    @patch("local_cli.clipboard.shutil.which", return_value=None)
    def test_linux_no_clipboard_returns_none(
        self, mock_which: MagicMock, mock_system: MagicMock
    ) -> None:
        """On Linux with neither xclip nor xsel, returns None."""
        result = _detect_clipboard_command()
        self.assertIsNone(result)

    @patch("local_cli.clipboard.platform.system", return_value="Windows")
    @patch("local_cli.clipboard.shutil.which", return_value="C:\\Windows\\clip.exe")
    def test_windows_returns_clip(
        self, mock_which: MagicMock, mock_system: MagicMock
    ) -> None:
        """On Windows, returns clip command."""
        result = _detect_clipboard_command()
        self.assertEqual(result, ["clip"])

    @patch("local_cli.clipboard.platform.system", return_value="FreeBSD")
    def test_unknown_platform_returns_none(
        self, mock_system: MagicMock
    ) -> None:
        """On unsupported platforms, returns None."""
        result = _detect_clipboard_command()
        self.assertIsNone(result)

    @patch("local_cli.clipboard.platform.system", return_value="Darwin")
    @patch("local_cli.clipboard.shutil.which", return_value=None)
    def test_macos_pbcopy_missing_returns_none(
        self, mock_which: MagicMock, mock_system: MagicMock
    ) -> None:
        """On macOS without pbcopy (unlikely), returns None."""
        result = _detect_clipboard_command()
        self.assertIsNone(result)


class TestGetClipboardCommand(unittest.TestCase):
    """Tests for get_clipboard_command()."""

    @patch("local_cli.clipboard.platform.system", return_value="Darwin")
    @patch("local_cli.clipboard.shutil.which", return_value="/usr/bin/pbcopy")
    def test_returns_detected_command(
        self, mock_which: MagicMock, mock_system: MagicMock
    ) -> None:
        """get_clipboard_command() returns the detected command."""
        result = get_clipboard_command()
        self.assertEqual(result, ["pbcopy"])

    @patch("local_cli.clipboard.platform.system", return_value="FreeBSD")
    def test_returns_none_when_unavailable(
        self, mock_system: MagicMock
    ) -> None:
        """get_clipboard_command() returns None on unsupported platform."""
        result = get_clipboard_command()
        self.assertIsNone(result)


class TestSanitizeText(unittest.TestCase):
    """Tests for _sanitize_text()."""

    def test_strips_null_bytes(self) -> None:
        """Null bytes are removed from text."""
        self.assertEqual(_sanitize_text("hello\x00world"), "helloworld")

    def test_preserves_normal_text(self) -> None:
        """Normal text is returned unchanged."""
        text = "Hello, world! 日本語テスト"
        self.assertEqual(_sanitize_text(text), text)

    def test_preserves_newlines(self) -> None:
        """Newlines and other whitespace are preserved."""
        text = "line1\nline2\ttab"
        self.assertEqual(_sanitize_text(text), text)

    def test_empty_string(self) -> None:
        """Empty string is returned unchanged."""
        self.assertEqual(_sanitize_text(""), "")

    def test_multiple_null_bytes(self) -> None:
        """Multiple null bytes are all removed."""
        self.assertEqual(
            _sanitize_text("\x00a\x00b\x00c\x00"), "abc"
        )


class TestCopyToClipboard(unittest.TestCase):
    """Tests for copy_to_clipboard()."""

    @patch("local_cli.clipboard._detect_clipboard_command", return_value=["pbcopy"])
    @patch("local_cli.clipboard.subprocess.run")
    def test_successful_copy(
        self, mock_run: MagicMock, mock_detect: MagicMock
    ) -> None:
        """Successfully copies text to clipboard."""
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        result = copy_to_clipboard("Hello, clipboard!")
        self.assertTrue(result)
        mock_run.assert_called_once_with(
            ["pbcopy"],
            input=b"Hello, clipboard!",
            capture_output=True,
            timeout=5,
        )

    @patch("local_cli.clipboard._detect_clipboard_command", return_value=["pbcopy"])
    @patch("local_cli.clipboard.subprocess.run")
    def test_copies_utf8_text(
        self, mock_run: MagicMock, mock_detect: MagicMock
    ) -> None:
        """UTF-8 text is correctly encoded before sending to clipboard."""
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        result = copy_to_clipboard("日本語テスト")
        self.assertTrue(result)
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        self.assertEqual(
            call_kwargs.kwargs["input"],
            "日本語テスト".encode("utf-8"),
        )

    @patch("local_cli.clipboard._detect_clipboard_command", return_value=None)
    def test_raises_unavailable_when_no_command(
        self, mock_detect: MagicMock
    ) -> None:
        """Raises ClipboardUnavailableError when no command is found."""
        with self.assertRaises(ClipboardUnavailableError) as ctx:
            copy_to_clipboard("test")
        self.assertIn("Clipboard not available", str(ctx.exception))

    @patch(
        "local_cli.clipboard._detect_clipboard_command",
        return_value=["xclip", "-selection", "clipboard"],
    )
    @patch("local_cli.clipboard.subprocess.run")
    def test_linux_xclip_full_command(
        self, mock_run: MagicMock, mock_detect: MagicMock
    ) -> None:
        """On Linux, passes full xclip command with arguments."""
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        copy_to_clipboard("test")
        mock_run.assert_called_once_with(
            ["xclip", "-selection", "clipboard"],
            input=b"test",
            capture_output=True,
            timeout=5,
        )

    @patch("local_cli.clipboard._detect_clipboard_command", return_value=["pbcopy"])
    @patch(
        "local_cli.clipboard.subprocess.run",
        side_effect=FileNotFoundError("No such file"),
    )
    def test_raises_unavailable_on_file_not_found(
        self, mock_run: MagicMock, mock_detect: MagicMock
    ) -> None:
        """Raises ClipboardUnavailableError when command binary not found."""
        with self.assertRaises(ClipboardUnavailableError) as ctx:
            copy_to_clipboard("test")
        self.assertIn("not found", str(ctx.exception))

    @patch("local_cli.clipboard._detect_clipboard_command", return_value=["pbcopy"])
    @patch(
        "local_cli.clipboard.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="pbcopy", timeout=5),
    )
    def test_raises_error_on_timeout(
        self, mock_run: MagicMock, mock_detect: MagicMock
    ) -> None:
        """Raises ClipboardError when clipboard command times out."""
        with self.assertRaises(ClipboardError) as ctx:
            copy_to_clipboard("test")
        self.assertIn("timed out", str(ctx.exception))

    @patch("local_cli.clipboard._detect_clipboard_command", return_value=["pbcopy"])
    @patch(
        "local_cli.clipboard.subprocess.run",
        side_effect=OSError("Permission denied"),
    )
    def test_raises_error_on_os_error(
        self, mock_run: MagicMock, mock_detect: MagicMock
    ) -> None:
        """Raises ClipboardError on generic OS errors."""
        with self.assertRaises(ClipboardError) as ctx:
            copy_to_clipboard("test")
        self.assertIn("failed", str(ctx.exception))

    @patch("local_cli.clipboard._detect_clipboard_command", return_value=["pbcopy"])
    @patch("local_cli.clipboard.subprocess.run")
    def test_raises_error_on_nonzero_exit(
        self, mock_run: MagicMock, mock_detect: MagicMock
    ) -> None:
        """Raises ClipboardError when clipboard command exits with error."""
        mock_run.return_value = MagicMock(
            returncode=1, stderr=b"some error message"
        )
        with self.assertRaises(ClipboardError) as ctx:
            copy_to_clipboard("test")
        self.assertIn("exited with code 1", str(ctx.exception))
        self.assertIn("some error message", str(ctx.exception))

    @patch("local_cli.clipboard._detect_clipboard_command", return_value=["pbcopy"])
    @patch("local_cli.clipboard.subprocess.run")
    def test_nonzero_exit_no_stderr(
        self, mock_run: MagicMock, mock_detect: MagicMock
    ) -> None:
        """Error message handles empty stderr gracefully."""
        mock_run.return_value = MagicMock(returncode=1, stderr=b"")
        with self.assertRaises(ClipboardError) as ctx:
            copy_to_clipboard("test")
        msg = str(ctx.exception)
        self.assertIn("exited with code 1", msg)
        # No trailing colon when stderr is empty.
        self.assertNotIn(": \n", msg)

    def test_rejects_text_exceeding_max_length(self) -> None:
        """Raises ValueError when text exceeds maximum allowed length."""
        huge_text = "x" * (_MAX_COPY_LENGTH + 1)
        with self.assertRaises(ValueError) as ctx:
            copy_to_clipboard(huge_text)
        self.assertIn("too large", str(ctx.exception))

    def test_accepts_text_at_max_length(self) -> None:
        """Text exactly at the maximum length is accepted."""
        with patch(
            "local_cli.clipboard._detect_clipboard_command",
            return_value=["pbcopy"],
        ), patch("local_cli.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr=b"")
            text = "x" * _MAX_COPY_LENGTH
            result = copy_to_clipboard(text)
            self.assertTrue(result)

    def test_rejects_non_string_input(self) -> None:
        """Raises TypeError when input is not a string."""
        with self.assertRaises(TypeError):
            copy_to_clipboard(12345)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            copy_to_clipboard(None)  # type: ignore[arg-type]

    @patch("local_cli.clipboard._detect_clipboard_command", return_value=["pbcopy"])
    @patch("local_cli.clipboard.subprocess.run")
    def test_sanitizes_null_bytes(
        self, mock_run: MagicMock, mock_detect: MagicMock
    ) -> None:
        """Null bytes are stripped before sending to clipboard."""
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        copy_to_clipboard("hello\x00world")
        call_kwargs = mock_run.call_args
        self.assertEqual(call_kwargs.kwargs["input"], b"helloworld")

    @patch("local_cli.clipboard._detect_clipboard_command", return_value=["pbcopy"])
    @patch("local_cli.clipboard.subprocess.run")
    def test_empty_string_is_accepted(
        self, mock_run: MagicMock, mock_detect: MagicMock
    ) -> None:
        """Empty string can be copied to clipboard."""
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        result = copy_to_clipboard("")
        self.assertTrue(result)

    @patch("local_cli.clipboard._detect_clipboard_command", return_value=["pbcopy"])
    @patch("local_cli.clipboard.subprocess.run")
    def test_multiline_text(
        self, mock_run: MagicMock, mock_detect: MagicMock
    ) -> None:
        """Multi-line text is correctly passed to clipboard."""
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        text = "line 1\nline 2\nline 3"
        copy_to_clipboard(text)
        call_kwargs = mock_run.call_args
        self.assertEqual(call_kwargs.kwargs["input"], text.encode("utf-8"))


class TestClipboardExceptionHierarchy(unittest.TestCase):
    """Tests for the exception class hierarchy."""

    def test_clipboard_error_is_exception(self) -> None:
        """ClipboardError inherits from Exception."""
        self.assertTrue(issubclass(ClipboardError, Exception))

    def test_unavailable_is_clipboard_error(self) -> None:
        """ClipboardUnavailableError is a subclass of ClipboardError."""
        self.assertTrue(
            issubclass(ClipboardUnavailableError, ClipboardError)
        )

    def test_can_catch_unavailable_as_clipboard_error(self) -> None:
        """ClipboardUnavailableError can be caught as ClipboardError."""
        with self.assertRaises(ClipboardError):
            raise ClipboardUnavailableError("no clipboard")


class TestClipboardReturnsCopy(unittest.TestCase):
    """Tests that _detect_clipboard_command returns copies, not references."""

    @patch("local_cli.clipboard.platform.system", return_value="Darwin")
    @patch("local_cli.clipboard.shutil.which", return_value="/usr/bin/pbcopy")
    def test_returns_new_list_each_call(
        self, mock_which: MagicMock, mock_system: MagicMock
    ) -> None:
        """Each call returns a fresh list to prevent mutation issues."""
        result1 = _detect_clipboard_command()
        result2 = _detect_clipboard_command()
        self.assertEqual(result1, result2)
        # Verify they are different objects.
        self.assertIsNot(result1, result2)


if __name__ == "__main__":
    unittest.main()
