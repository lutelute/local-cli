"""Terminal spinner for visual feedback during long-running operations.

Provides a threaded spinner that animates in the terminal while the main
thread is blocked (e.g. waiting for LLM response or executing a tool).
Uses only stdlib -- no external dependencies.

Two styles are available process-wide via :func:`set_spinner_style`:

- ``"dots"`` (default): the classic braille-dot cycle.
- ``"mascot"``: Loca the local cat (``(=･ω･=)``) blinks and fidgets
  while you wait.  Enabled by ``--mascot`` / ``LOCAL_CLI_MASCOT=1``.
"""

import sys
import threading

# Braille-dot spinner frames (smooth 10-frame cycle).
_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Interval between frame updates (seconds).
_INTERVAL = 0.08

# Mascot frames: mostly resting, an occasional blink and a wink.  The
# animation reads as "alive" rather than "busy", so the interval is
# slower than the dots.
_MASCOT_FRAMES = (
    "(=･ω･=)",
    "(=･ω･=)",
    "(=･ω･=)",
    "(=-ω-=)",
    "(=･ω･=)",
    "(=･ω･=)",
    "(=･ω-=)",
    "(=･ω･=)",
)

# Mascot frame interval (seconds) — a lazy cat blinks slowly.
_MASCOT_INTERVAL = 0.24

# Styles recognised by set_spinner_style.
_STYLES = ("dots", "mascot")

# Process-wide style; new Spinner instances read this at start().
_active_style = "dots"


def set_spinner_style(style: str) -> None:
    """Set the process-wide spinner style.

    Args:
        style: ``"dots"`` (default braille cycle) or ``"mascot"``
            (Loca the local cat).

    Raises:
        ValueError: On an unknown style name.
    """
    global _active_style
    if style not in _STYLES:
        raise ValueError(f"unknown spinner style: {style!r}")
    _active_style = style


def get_spinner_style() -> str:
    """Return the active process-wide spinner style."""
    return _active_style


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
        if _active_style == "mascot":
            self._frames: tuple[str, ...] | str = _MASCOT_FRAMES
            self._interval = _MASCOT_INTERVAL
        else:
            self._frames = _FRAMES
            self._interval = _INTERVAL

    def _animate(self) -> None:
        idx = 0
        max_len = 0
        while not self._stop_event.is_set():
            frame = self._frames[idx % len(self._frames)]
            line = f"  {frame} {self._message}..."
            # Track the longest line written so the clear wipes all of
            # it (mascot frames are wider than one dot character).
            max_len = max(max_len, len(line))
            sys.stderr.write("\r" + line)
            sys.stderr.flush()
            idx += 1
            self._stop_event.wait(self._interval)
        # Clear the spinner line.  Wide glyphs occupy two terminal
        # cells, so clear generously.
        sys.stderr.write("\r" + " " * (max_len * 2) + "\r")
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
