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


if __name__ == "__main__":
    unittest.main()
