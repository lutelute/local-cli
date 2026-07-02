"""Terminal spinner for visual feedback during long-running operations.

Provides a threaded spinner that animates in the terminal while the main
thread is blocked (e.g. waiting for LLM response or executing a tool).
Uses only stdlib -- no external dependencies.

Three styles are available process-wide via :func:`set_spinner_style`:

- ``"dots"`` (default): the classic braille-dot cycle.
- ``"mascot"``: Loca the local cat (``(=･ω･=)``) blinks and fidgets
  on a single line.  Enabled by ``--mascot`` / ``LOCAL_CLI_MASCOT=cat``.
- ``"pixel"``: a five-row pixel-art Loca, drawn with colored block
  characters and animated with ANSI cursor movement (blinks, ear
  twitches).  Enabled by ``--mascot pixel`` / ``LOCAL_CLI_MASCOT=pixel``.
  Falls back to the single-line mascot when stderr is not a TTY (pipes,
  CI) so cursor-control codes never pollute logs.
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

# ---------------------------------------------------------------------------
# Pixel-art mascot
# ---------------------------------------------------------------------------

# Pixel maps: one character per pixel, drawn at two terminal columns per
# pixel ("██") with 256-color ANSI codes.  Loca is a round-faced orange
# kitten: o = outline/mouth, y = orange fur, w = cream muzzle,
# p = ear-inner pink, c = cheek blush, k = eye, h = eye highlight,
# n = nose, . = transparent.  Each map is left/right symmetric.
_PIXEL_IDLE = (
    "..ooo.....ooo..",
    ".oypyo...oypyo.",
    "oyppyyyyyyyppyo",
    "oyyyyyyyyyyyyyo",
    "oywwwwwwwwwwwyo",
    "oywhkkwwwkkhwyo",
    "oywkkwwwwwkkwyo",
    "oycwwwwnwwwwcyo",
    "oywwwwmwmwwwwyo",
    ".oywwwwwwwwwyo.",
    "..oyywwywwyyo..",
    "...ooooooooo...",
)
# Blink: the eyes close to a short brown line.
_PIXEL_BLINK = (
    "..ooo.....ooo..",
    ".oypyo...oypyo.",
    "oyppyyyyyyyppyo",
    "oyyyyyyyyyyyyyo",
    "oywwwwwwwwwwwyo",
    "oywwwwwwwwwwwyo",
    "oywoowwwwwoowyo",
    "oycwwwwnwwwwcyo",
    "oywwwwmwmwwwwyo",
    ".oywwwwwwwwwyo.",
    "..oyywwywwyyo..",
    "...ooooooooo...",
)
# Ear twitch: the ear tips lean inward a touch.
_PIXEL_EAR = (
    "...ooo...ooo...",
    "..oypyo.oypyo..",
    "oyppyyyyyyyppyo",
    "oyyyyyyyyyyyyyo",
    "oywwwwwwwwwwwyo",
    "oywhkkwwwkkhwyo",
    "oywkkwwwwwkkwyo",
    "oycwwwwnwwwwcyo",
    "oywwwwmwmwwwwyo",
    ".oywwwwwwwwwyo.",
    "..oyywwywwyyo..",
    "...ooooooooo...",
)

# Animation sequence: mostly idle, an occasional blink and ear twitch.
_PIXEL_FRAMES = (
    _PIXEL_IDLE, _PIXEL_IDLE, _PIXEL_IDLE, _PIXEL_BLINK,
    _PIXEL_IDLE, _PIXEL_IDLE, _PIXEL_EAR, _PIXEL_IDLE,
)

# Pixel frame interval (seconds).
_PIXEL_INTERVAL = 0.28

# 256-color codes per pixel kind.  A closed eye is drawn with the
# outline color (see _PIXEL_BLINK) so the eyes visibly disappear.
_PIXEL_COLORS = {
    "o": "\033[38;5;94m",    # outline / mouth (soft brown)
    "y": "\033[38;5;214m",   # orange fur
    "w": "\033[38;5;230m",   # cream muzzle
    "p": "\033[38;5;218m",   # ear inner pink
    "c": "\033[38;5;217m",   # cheek blush
    "k": "\033[38;5;236m",   # eye
    "h": "\033[38;5;231m",   # eye highlight (white)
    "n": "\033[38;5;211m",   # nose
}
_ANSI_RESET = "\033[0m"

# Row (0-based) whose right side carries the "Thinking..." message —
# the eye row, so the label sits beside Loca's face.
_PIXEL_MESSAGE_ROW = 5


def _render_pixel_frame(pixels: tuple[str, ...], message: str) -> list[str]:
    """Render a pixel map into colored terminal lines.

    Each pixel becomes two columns ("██") so the art is roughly square;
    the spinner message is attached to the right of the face row.

    Args:
        pixels: Rows of pixel characters (see the ``_PIXEL_*`` maps).
        message: The spinner message (e.g. ``"Thinking"``).

    Returns:
        One string per row, including ANSI color codes.
    """
    lines: list[str] = []
    for row_index, row in enumerate(pixels):
        parts: list[str] = ["  "]
        for pixel in row:
            color = _PIXEL_COLORS.get(pixel)
            if color is None:
                parts.append("  ")
            else:
                parts.append(f"{color}██{_ANSI_RESET}")
        if row_index == _PIXEL_MESSAGE_ROW and message:
            parts.append(f"  {message}...")
        lines.append("".join(parts))
    return lines


# Styles recognised by set_spinner_style.
_STYLES = ("dots", "mascot", "pixel")

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

        style = _active_style
        # Pixel mode needs cursor-control escapes; fall back to the
        # one-line mascot when stderr is not a terminal (pipes, CI).
        if style == "pixel":
            try:
                if not sys.stderr.isatty():
                    style = "mascot"
            except Exception:
                style = "mascot"
        self._style = style

        if style == "mascot":
            self._frames: tuple = _MASCOT_FRAMES
            self._interval = _MASCOT_INTERVAL
        elif style == "pixel":
            self._frames = _PIXEL_FRAMES
            self._interval = _PIXEL_INTERVAL
        else:
            self._frames = _FRAMES
            self._interval = _INTERVAL

    def _animate(self) -> None:
        if self._style == "pixel":
            self._animate_pixel()
            return
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

    def _animate_pixel(self) -> None:
        """Animate the multi-row pixel mascot with cursor movement.

        Each redraw moves the cursor back to the first art row
        (``ESC[nF``), rewrites every row after erasing it (``ESC[K``),
        and on stop erases the whole block and returns the cursor, so
        subsequent output starts exactly where the spinner began.
        """
        height = len(_PIXEL_IDLE)
        idx = 0
        drawn = False
        while not self._stop_event.is_set():
            frame = self._frames[idx % len(self._frames)]
            lines = _render_pixel_frame(frame, self._message)
            if drawn:
                sys.stderr.write(f"\033[{height}F")
            for line in lines:
                sys.stderr.write("\033[K" + line + "\n")
            sys.stderr.flush()
            drawn = True
            idx += 1
            self._stop_event.wait(self._interval)
        if drawn:
            sys.stderr.write(f"\033[{height}F")
            for _ in range(height):
                sys.stderr.write("\033[K\n")
            sys.stderr.write(f"\033[{height}F")
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
