"""Edit tool for replacing text in files.

Performs exact string matching and replacement within files.  Reads the
file, locates an exact match of the ``old_text`` parameter, replaces it
with ``new_text``, and writes the result back.  Returns a diff-like
output showing the changes made.

When ``old_text`` is not found, the error includes the most similar
block from the file (located with :mod:`difflib`) so the model can copy
the exact text — small local models most often fail edits by guessing
whitespace or indentation slightly wrong, and without the hint they
retry the same wrong ``old_text`` forever.
"""

import difflib
from pathlib import Path

from local_cli.tools._fileio import atomic_write_text, not_found_error
from local_cli.tools.base import Tool

# Skip the similarity search on big inputs — it is O(lines * window).
_HINT_MAX_CONTENT_LINES = 5_000
_HINT_MAX_OLD_TEXT_CHARS = 3_000

# Minimum similarity ratio for a hint to be worth showing.
_HINT_MIN_RATIO = 0.55


def _closest_block(content: str, old_text: str) -> tuple[int, str] | None:
    """Find the block of *content* most similar to *old_text*.

    Slides a window of ``len(old_text.splitlines())`` lines over the
    file and scores each window with :class:`difflib.SequenceMatcher`
    (``old_text`` is kept as ``seq2`` so difflib's internal cache is
    reused across windows).

    Args:
        content: The full file content.
        old_text: The text the model tried (and failed) to match.

    Returns:
        ``(start_line, block_text)`` with a 1-based start line, or
        ``None`` when the file is empty, the inputs are too large, or
        nothing scores above ``_HINT_MIN_RATIO``.
    """
    if len(old_text) > _HINT_MAX_OLD_TEXT_CHARS:
        return None
    content_lines = content.splitlines()
    if not content_lines or len(content_lines) > _HINT_MAX_CONTENT_LINES:
        return None

    window_size = max(1, len(old_text.splitlines()))
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(old_text)

    best_ratio = 0.0
    best_start = -1
    for i in range(max(1, len(content_lines) - window_size + 1)):
        window = "\n".join(content_lines[i : i + window_size])
        matcher.set_seq1(window)
        # Cheap upper bounds first; only compute the real ratio when the
        # window could beat the current best.
        if matcher.real_quick_ratio() <= best_ratio:
            continue
        if matcher.quick_ratio() <= best_ratio:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    if best_start < 0 or best_ratio < _HINT_MIN_RATIO:
        return None
    block = "\n".join(content_lines[best_start : best_start + window_size])
    return best_start + 1, block


def _not_found_error(file_path: str, content: str, old_text: str) -> str:
    """Build the old_text-not-found error, with a closest-match hint."""
    error = f"Error: old_text not found in {file_path}."
    closest = _closest_block(content, old_text)
    if closest is None:
        return error
    start_line, block = closest
    return (
        f"{error} The most similar text in the file starts at line "
        f"{start_line}:\n"
        "---\n"
        f"{block}\n"
        "---\n"
        "Retry with old_text copied EXACTLY from the file above (watch "
        "whitespace and indentation)."
    )


def _make_diff_output(
    file_path: str,
    old_text: str,
    new_text: str,
    occurrences: int,
) -> str:
    """Build a diff-like summary of the replacement.

    Args:
        file_path: Path to the edited file.
        old_text: The original text that was replaced.
        new_text: The replacement text.
        occurrences: Number of replacements made.

    Returns:
        A human-readable diff-like string.
    """
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()

    parts: list[str] = [f"--- {file_path}"]
    parts.append(f"+++ {file_path}")
    parts.append(f"@@ replaced {occurrences} occurrence(s) @@")
    for line in old_lines:
        parts.append(f"-{line}")
    for line in new_lines:
        parts.append(f"+{line}")

    return "\n".join(parts)


class EditTool(Tool):
    """Replace exact text matches in a file."""

    @property
    def name(self) -> str:
        return "edit"

    @property
    def description(self) -> str:
        return (
            "Replace an exact text match in a file. Reads the file, "
            "finds the exact occurrence of old_text, replaces it with "
            "new_text, and writes the result back. Returns a diff-like "
            "output showing the changes. Use replace_all to control "
            "whether all occurrences or just the first are replaced."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to edit.",
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find in the file.",
                },
                "new_text": {
                    "type": "string",
                    "description": "The text to replace old_text with.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": (
                        "If true, replace all occurrences. "
                        "If false (default), replace only the first occurrence."
                    ),
                },
            },
            "required": ["file_path", "old_text", "new_text"],
        }

    def execute(self, **kwargs: object) -> str:
        """Replace exact text in a file and return a diff-like summary.

        Args:
            **kwargs: Must include ``file_path`` (str), ``old_text``
                (str), and ``new_text`` (str).  May include
                ``replace_all`` (bool, default False).

        Returns:
            A diff-like string showing the changes, or an error message
            if the operation failed.
        """
        file_path = kwargs.get("file_path")
        if not isinstance(file_path, str) or not file_path.strip():
            return "Error: 'file_path' parameter is required and must be a non-empty string."

        old_text = kwargs.get("old_text")
        if not isinstance(old_text, str):
            return "Error: 'old_text' parameter is required and must be a string."
        if not old_text:
            return "Error: 'old_text' must not be empty."

        new_text = kwargs.get("new_text")
        if not isinstance(new_text, str):
            return "Error: 'new_text' parameter is required and must be a string."

        # A no-op edit still reports "1 occurrence replaced", which small
        # models read as success (observed live: old_text="add",
        # new_text="add", followed by "fixed it").  Reject it instead.
        if old_text == new_text:
            return (
                "Error: old_text and new_text are identical, so this edit "
                "changes nothing. Put the exact current text in old_text "
                "and the corrected text in new_text."
            )

        replace_all = kwargs.get("replace_all", False)
        if not isinstance(replace_all, bool):
            replace_all = False

        path = Path(file_path)

        if not path.exists():
            return not_found_error(file_path)

        if not path.is_file():
            return f"Error: not a regular file: {file_path}"

        # Read the current content.
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = path.read_text(encoding="latin-1")
            except OSError as exc:
                return f"Error: could not read file: {exc}"
        except PermissionError:
            return f"Error: permission denied: {file_path}"
        except OSError as exc:
            return f"Error: could not read file: {exc}"

        # Check that old_text exists in the file.
        count = content.count(old_text)
        if count == 0:
            # Fallback: the model usually emits LF line endings, but the
            # file on disk may use CRLF (or bare CR).  Normalize both sides
            # to LF and retry; on a match, operate on the normalized content
            # so the rewritten file is internally consistent.
            norm_content = content.replace("\r\n", "\n").replace("\r", "\n")
            norm_old = old_text.replace("\r\n", "\n").replace("\r", "\n")
            if norm_old and norm_content.count(norm_old) > 0:
                content = norm_content
                old_text = norm_old
                count = content.count(old_text)
            else:
                return _not_found_error(file_path, content, old_text)

        # Perform the replacement.
        if replace_all:
            new_content = content.replace(old_text, new_text)
            occurrences = count
        else:
            new_content = content.replace(old_text, new_text, 1)
            occurrences = 1

        # Write the modified content back atomically so a failure mid-write
        # cannot leave the file truncated or corrupted.
        try:
            atomic_write_text(path, new_content)
        except PermissionError:
            return f"Error: permission denied: {file_path}"
        except OSError as exc:
            return f"Error: could not write file: {exc}"

        return _make_diff_output(file_path, old_text, new_text, occurrences)
