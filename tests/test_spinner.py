"""Tests for local_cli.spinner."""

import io
import sys
import time
import unittest

from local_cli.spinner import Spinner


class TestSpinner(unittest.TestCase):
    """Tests for the Spinner class."""

    def test_context_manager_starts_and_stops(self) -> None:
        """Spinner starts and stops cleanly as a context manager."""
        with Spinner("Test"):
            # Thread should be alive inside the context.
            pass
        # After exiting, the thread should be stopped.

    def test_start_stop(self) -> None:
        """Spinner can be started and stopped explicitly."""
        s = Spinner("Working")
        s.start()
        self.assertIsNotNone(s._thread)
        self.assertTrue(s._thread.is_alive())
        s.stop()
        self.assertIsNone(s._thread)

    def test_stop_without_start(self) -> None:
        """Stopping a spinner that was never started does not raise."""
        s = Spinner("Idle")
        s.stop()  # Should not raise.

    def test_double_stop(self) -> None:
        """Stopping twice does not raise."""
        s = Spinner("Twice")
        s.start()
        s.stop()
        s.stop()  # Should not raise.

    def test_writes_to_stderr(self) -> None:
        """Spinner writes animation frames to stderr."""
        captured = io.StringIO()
        original_stderr = sys.stderr
        sys.stderr = captured
        try:
            s = Spinner("Loading")
            s.start()
            time.sleep(0.15)  # Let a few frames render.
            s.stop()
        finally:
            sys.stderr = original_stderr

        output = captured.getvalue()
        self.assertIn("Loading", output)

    def test_clears_line_on_stop(self) -> None:
        """Spinner clears its line when stopped."""
        captured = io.StringIO()
        original_stderr = sys.stderr
        sys.stderr = captured
        try:
            s = Spinner("Clear")
            s.start()
            time.sleep(0.1)
            s.stop()
        finally:
            sys.stderr = original_stderr

        output = captured.getvalue()
        # The last write should be a clearing line (spaces + \r).
        self.assertTrue(output.rstrip(" ").endswith("\r"))


class TestSpinnerStyle(unittest.TestCase):
    """Tests for the process-wide spinner style (mascot mode)."""

    def tearDown(self) -> None:
        from local_cli.spinner import set_spinner_style
        set_spinner_style("dots")

    def test_default_style_is_dots(self) -> None:
        from local_cli.spinner import get_spinner_style
        self.assertEqual(get_spinner_style(), "dots")

    def test_set_and_get_style(self) -> None:
        from local_cli.spinner import get_spinner_style, set_spinner_style
        set_spinner_style("mascot")
        self.assertEqual(get_spinner_style(), "mascot")

    def test_unknown_style_rejected(self) -> None:
        from local_cli.spinner import set_spinner_style
        with self.assertRaises(ValueError):
            set_spinner_style("disco")

    def test_mascot_frames_render_and_clear(self) -> None:
        """Mascot mode writes cat frames and clears the whole line."""
        from io import StringIO
        from local_cli.spinner import Spinner, set_spinner_style

        set_spinner_style("mascot")
        captured = StringIO()
        original_stderr = sys.stderr
        sys.stderr = captured
        try:
            s = Spinner("Thinking")
            s.start()
            time.sleep(0.1)
            s.stop()
        finally:
            sys.stderr = original_stderr

        output = captured.getvalue()
        self.assertIn("ω", output)  # a cat face was drawn
        self.assertIn("Thinking...", output)
        # Ends with a clearing write that covers the widest frame line.
        self.assertTrue(output.rstrip(" ").endswith("\r"))

    def test_style_is_captured_per_instance(self) -> None:
        """A spinner keeps the style active when it was constructed."""
        from local_cli.spinner import (
            _MASCOT_FRAMES,
            Spinner,
            set_spinner_style,
        )
        set_spinner_style("mascot")
        s = Spinner("x")
        set_spinner_style("dots")
        self.assertEqual(s._frames, _MASCOT_FRAMES)


class TestPixelMascot(unittest.TestCase):
    """Tests for the multi-row pixel-art mascot."""

    def tearDown(self) -> None:
        from local_cli.spinner import set_spinner_style
        set_spinner_style("dots")

    def test_render_produces_colored_rows(self) -> None:
        from local_cli.spinner import (
            _PIXEL_IDLE,
            _PIXEL_MESSAGE_ROW,
            _render_pixel_frame,
        )
        lines = _render_pixel_frame(_PIXEL_IDLE, "Thinking")
        self.assertEqual(len(lines), len(_PIXEL_IDLE))
        self.assertTrue(any("\033[38;5;214m" in line for line in lines))
        self.assertIn("Thinking...", lines[_PIXEL_MESSAGE_ROW])
        # Every color code is closed by a reset (no bleed into output).
        for line in lines:
            self.assertEqual(line.count("\033[38;5;"), line.count("\033[0m"))

    def test_all_frames_same_dimensions(self) -> None:
        """Idle / blink / ear maps share width and height (clean redraw)."""
        from local_cli.spinner import _PIXEL_BLINK, _PIXEL_EAR, _PIXEL_IDLE
        for frame in (_PIXEL_BLINK, _PIXEL_EAR):
            self.assertEqual(len(frame), len(_PIXEL_IDLE))
            self.assertEqual(
                {len(r) for r in frame}, {len(r) for r in _PIXEL_IDLE},
            )

    def test_blink_frame_hides_the_eyes(self) -> None:
        """The eye color disappears entirely when Loca blinks."""
        from local_cli.spinner import (
            _PIXEL_BLINK,
            _PIXEL_IDLE,
            _render_pixel_frame,
        )
        open_eyes = _render_pixel_frame(_PIXEL_IDLE, "")
        closed = _render_pixel_frame(_PIXEL_BLINK, "")
        self.assertIn("\033[38;5;236m", "".join(open_eyes))
        self.assertNotIn("\033[38;5;236m", "".join(closed))

    def test_non_tty_falls_back_to_one_line_mascot(self) -> None:
        """Pixel mode degrades to the line mascot when not a terminal."""
        from io import StringIO
        from local_cli.spinner import Spinner, set_spinner_style

        set_spinner_style("pixel")
        captured = StringIO()  # StringIO.isatty() is False
        original_stderr = sys.stderr
        sys.stderr = captured
        try:
            s = Spinner("Thinking")
            self.assertEqual(s._style, "mascot")
            s.start()
            time.sleep(0.05)
            s.stop()
        finally:
            sys.stderr = original_stderr

        output = captured.getvalue()
        self.assertIn("ω", output)          # line mascot drew
        self.assertNotIn("\033[5F", output)  # no cursor control leaked

    def test_pixel_style_selected_on_tty(self) -> None:
        """On a TTY-like stderr the pixel style stays active."""
        from unittest.mock import patch
        from local_cli.spinner import Spinner, set_spinner_style

        set_spinner_style("pixel")
        with patch.object(sys, "stderr") as fake_err:
            fake_err.isatty.return_value = True
            s = Spinner("x")
        self.assertEqual(s._style, "pixel")


if __name__ == "__main__":
    unittest.main()
