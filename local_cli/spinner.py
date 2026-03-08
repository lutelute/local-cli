"""Terminal spinner for visual feedback during long-running operations.

Provides a threaded spinner that animates in the terminal while the main
thread is blocked (e.g. waiting for LLM response or executing a tool).
Uses only stdlib -- no external dependencies.
"""

import sys
import threading
import time

# Braille-dot spinner frames (smooth 10-frame cycle).
_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Interval between frame updates (seconds).
_INTERVAL = 0.08


class Spinner:
    """A terminal spinner that runs in a background thread.

    Usage::

        with Spinner("Thinking"):
            # ... blocking work ...
            pass

    The spinner automatically stops and clears itself when the context
    manager exits (even on exception).
    """

    def __init__(self, message: str = "Thinking") -> None:
        self._message = message
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _animate(self) -> None:
        idx = 0
        while not self._stop_event.is_set():
            frame = _FRAMES[idx % len(_FRAMES)]
            line = f"\r  {frame} {self._message}..."
            sys.stderr.write(line)
            sys.stderr.flush()
            idx += 1
            self._stop_event.wait(_INTERVAL)
        # Clear the spinner line.
        sys.stderr.write("\r" + " " * (len(self._message) + 10) + "\r")
        sys.stderr.flush()

    def start(self) -> None:
        """Start the spinner animation."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the spinner and clear the line."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
