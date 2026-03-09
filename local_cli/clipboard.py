"""Cross-platform clipboard utility for local-cli.

Provides clipboard write access using platform-native command-line tools:
- macOS: ``pbcopy``
- Linux: ``xclip`` or ``xsel``
- Windows: ``clip``

All clipboard interaction is done through :func:`subprocess.run` to avoid
external Python dependencies.  When no clipboard utility is available the
functions degrade gracefully instead of raising.
"""

import platform
import shutil
import subprocess

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

# Mapping from :func:`platform.system` return values to the ordered list of
# clipboard command candidates.  The first available command is used.
_CLIPBOARD_COMMANDS: dict[str, list[list[str]]] = {
    "Darwin": [["pbcopy"]],
    "Linux": [
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
    ],
    "Windows": [["clip"]],
}

# Maximum text length accepted by :func:`copy_to_clipboard`.  Prevents
# accidentally piping very large strings to a subprocess.
_MAX_COPY_LENGTH: int = 1_000_000  # 1 MB of text


def _detect_clipboard_command() -> list[str] | None:
    """Return the first available clipboard command for the current platform.

    Checks :func:`shutil.which` to verify the executable exists on *PATH*.

    Returns:
        A list of command-line arguments suitable for :func:`subprocess.run`,
        or ``None`` if no clipboard utility is found.
    """
    system = platform.system()
    candidates = _CLIPBOARD_COMMANDS.get(system, [])
    for cmd in candidates:
        # Only check the executable name (first element).
        if shutil.which(cmd[0]) is not None:
            return list(cmd)  # return a copy to avoid mutation
    return None


def get_clipboard_command() -> list[str] | None:
    """Public wrapper around clipboard command detection.

    Returns:
        The clipboard command as a list of strings, or ``None`` if
        no clipboard utility is available on this platform.
    """
    return _detect_clipboard_command()


# ---------------------------------------------------------------------------
# Clipboard operations
# ---------------------------------------------------------------------------

class ClipboardError(Exception):
    """Raised when a clipboard operation fails."""


class ClipboardUnavailableError(ClipboardError):
    """Raised when no clipboard utility is found on the system."""


def _sanitize_text(text: str) -> str:
    """Sanitize text before sending to the clipboard subprocess.

    Strips null bytes that could cause issues with clipboard utilities.

    Args:
        text: The raw text to sanitize.

    Returns:
        The sanitized text string.
    """
    # Remove null bytes — they can confuse clipboard utilities and are
    # never meaningful in user-visible text.
    return text.replace("\x00", "")


def copy_to_clipboard(text: str) -> bool:
    """Copy *text* to the system clipboard.

    Uses the first available platform-native clipboard command
    (``pbcopy``, ``xclip``, ``xsel``, or ``clip``).

    The text is passed via **stdin** to the subprocess — it is never
    interpolated into a shell command string — so shell injection is
    not possible.

    Args:
        text: The text to place on the clipboard.

    Returns:
        ``True`` if the text was successfully copied.

    Raises:
        ClipboardUnavailableError: If no clipboard utility is found.
        ClipboardError: If the clipboard command fails.
        ValueError: If *text* exceeds :data:`_MAX_COPY_LENGTH`.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    if len(text) > _MAX_COPY_LENGTH:
        raise ValueError(
            f"Text too large for clipboard: {len(text)} chars "
            f"(max {_MAX_COPY_LENGTH})"
        )

    cmd = _detect_clipboard_command()
    if cmd is None:
        raise ClipboardUnavailableError(
            "Clipboard not available. Install pbcopy (macOS), "
            "xclip or xsel (Linux), or use Windows."
        )

    sanitized = _sanitize_text(text)

    try:
        result = subprocess.run(
            cmd,
            input=sanitized.encode("utf-8"),
            capture_output=True,
            timeout=5,
        )
    except FileNotFoundError:
        raise ClipboardUnavailableError(
            f"Clipboard command not found: {cmd[0]}"
        )
    except subprocess.TimeoutExpired:
        raise ClipboardError("Clipboard command timed out")
    except OSError as exc:
        raise ClipboardError(f"Clipboard command failed: {exc}")

    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
        raise ClipboardError(
            f"Clipboard command exited with code {result.returncode}"
            + (f": {stderr_text}" if stderr_text else "")
        )

    return True
