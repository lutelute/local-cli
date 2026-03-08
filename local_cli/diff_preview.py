"""Multi-file diff preview for local-cli using difflib.

Generates unified diffs of file changes with optional ANSI color
formatting, binary file detection, and truncation support.  Uses only
:mod:`difflib` from the standard library — no external dependencies.
"""

import difflib

# ---------------------------------------------------------------------------
# ANSI colour codes for terminal output
# ---------------------------------------------------------------------------

_COLOR_RED: str = "\033[91m"
_COLOR_GREEN: str = "\033[92m"
_COLOR_CYAN: str = "\033[96m"
_COLOR_BOLD: str = "\033[1m"
_COLOR_RESET: str = "\033[0m"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of output lines before truncation.
_MAX_DIFF_LINES: int = 500

# Number of bytes inspected when checking for binary content.
_BINARY_CHECK_SIZE: int = 8192


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DiffPreviewError(Exception):
    """Raised when diff generation fails."""


# ---------------------------------------------------------------------------
# Binary detection
# ---------------------------------------------------------------------------

def is_binary_content(data: bytes) -> bool:
    """Determine whether *data* looks like binary content.

    A file is considered binary if its first :data:`_BINARY_CHECK_SIZE`
    bytes contain a null (``\\x00``) byte.

    Args:
        data: Raw bytes to inspect.

    Returns:
        ``True`` if *data* appears to be binary, ``False`` otherwise.
    """
    if not isinstance(data, bytes):
        raise TypeError(f"Expected bytes, got {type(data).__name__}")
    return b"\x00" in data[:_BINARY_CHECK_SIZE]


# ---------------------------------------------------------------------------
# Single-file diff generation
# ---------------------------------------------------------------------------

def generate_diff(
    from_lines: list[str],
    to_lines: list[str],
    from_file: str = "a",
    to_file: str = "b",
    context_lines: int = 3,
) -> str:
    """Generate a unified diff between two sequences of lines.

    Uses :func:`difflib.unified_diff` from the standard library.

    Args:
        from_lines: Original file lines (each without trailing newline).
        to_lines:   Modified file lines (each without trailing newline).
        from_file:  Label for the original file header.
        to_file:    Label for the modified file header.
        context_lines: Number of unchanged context lines around each change.

    Returns:
        A unified diff string, or an empty string if the files are identical.
    """
    diff_iter = difflib.unified_diff(
        from_lines,
        to_lines,
        fromfile=from_file,
        tofile=to_file,
        lineterm="",
        n=context_lines,
    )
    diff_lines = list(diff_iter)
    if not diff_lines:
        return ""
    return "\n".join(diff_lines) + "\n"


# ---------------------------------------------------------------------------
# Colorisation
# ---------------------------------------------------------------------------

def colorize_diff(diff_text: str) -> str:
    """Add ANSI colour codes to unified diff output.

    Colour mapping:

    * ``+`` lines (additions, excluding ``+++``) → green
    * ``-`` lines (deletions, excluding ``---``) → red
    * ``@@`` hunk headers → cyan
    * ``---`` / ``+++`` file headers → bold

    Args:
        diff_text: Raw unified diff text.

    Returns:
        The diff text with ANSI escape sequences inserted, or the
        original text unchanged if it is empty.
    """
    if not diff_text:
        return diff_text

    colored_lines: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            colored_lines.append(f"{_COLOR_BOLD}{line}{_COLOR_RESET}")
        elif line.startswith("+"):
            colored_lines.append(f"{_COLOR_GREEN}{line}{_COLOR_RESET}")
        elif line.startswith("-"):
            colored_lines.append(f"{_COLOR_RED}{line}{_COLOR_RESET}")
        elif line.startswith("@@"):
            colored_lines.append(f"{_COLOR_CYAN}{line}{_COLOR_RESET}")
        else:
            colored_lines.append(line)

    return "\n".join(colored_lines) + "\n"


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

def truncate_diff(
    diff_text: str,
    max_lines: int = _MAX_DIFF_LINES,
) -> str:
    """Truncate *diff_text* if it exceeds *max_lines*.

    When truncation occurs, a summary line is appended indicating how
    many lines were omitted.

    Args:
        diff_text: The diff text to potentially truncate.
        max_lines: Maximum number of lines to keep.

    Returns:
        The original text if it fits within *max_lines*, otherwise
        the first *max_lines* lines followed by a truncation notice.
    """
    if not diff_text:
        return diff_text

    lines = diff_text.splitlines()
    if len(lines) <= max_lines:
        return diff_text

    remaining = len(lines) - max_lines
    truncated = lines[:max_lines]
    truncated.append(f"... truncated ({remaining} more lines)")
    return "\n".join(truncated) + "\n"


# ---------------------------------------------------------------------------
# Output formatting (colour + truncation)
# ---------------------------------------------------------------------------

def format_diff_output(
    diff_text: str,
    color: bool = True,
    max_lines: int = _MAX_DIFF_LINES,
) -> str:
    """Format diff output with optional colour and truncation.

    Truncation is applied **before** colourisation so that only visible
    lines receive ANSI escape sequences.

    Args:
        diff_text: Raw unified diff text.
        color:     Whether to add ANSI colour codes.
        max_lines: Maximum lines before truncation (``0`` disables).

    Returns:
        The formatted diff string, or the empty string if *diff_text*
        is empty.
    """
    if not diff_text:
        return diff_text

    result = diff_text

    # Truncate first to avoid colouring lines that won't be shown.
    if max_lines > 0:
        result = truncate_diff(result, max_lines)

    if color:
        result = colorize_diff(result)

    return result


# ---------------------------------------------------------------------------
# Multi-file diff generation
# ---------------------------------------------------------------------------

def generate_multi_file_diff(
    file_changes: list[tuple[str, list[str], list[str]]],
    color: bool = True,
    max_lines: int = _MAX_DIFF_LINES,
) -> str:
    """Generate a combined unified diff for multiple files.

    Each entry in *file_changes* is a ``(file_path, from_lines, to_lines)``
    tuple.  Use an empty list for *from_lines* when the file is new, or
    an empty list for *to_lines* when the file has been deleted.

    Binary files (detected by :func:`is_binary_content`) should be
    excluded by the caller; if included, they can be represented with
    a ``[binary file: <path>]`` placeholder by using
    :func:`binary_file_placeholder`.

    Args:
        file_changes: List of ``(file_path, from_lines, to_lines)`` tuples.
        color:        Whether to add ANSI colour codes.
        max_lines:    Maximum total output lines before truncation.

    Returns:
        A combined diff string for all files, or an empty string if
        there are no changes.
    """
    diff_parts: list[str] = []

    for file_path, from_lines, to_lines in file_changes:
        diff = generate_diff(
            from_lines,
            to_lines,
            from_file=f"a/{file_path}",
            to_file=f"b/{file_path}",
        )
        if diff:
            diff_parts.append(diff)

    if not diff_parts:
        return ""

    combined = "".join(diff_parts)
    return format_diff_output(combined, color=color, max_lines=max_lines)


def binary_file_placeholder(file_path: str) -> str:
    """Return a placeholder string for a binary file in diff output.

    Args:
        file_path: Path to the binary file.

    Returns:
        A human-readable placeholder string.
    """
    return f"[binary file: {file_path}]\n"
