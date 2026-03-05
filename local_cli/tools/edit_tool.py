"""Edit tool for replacing text in files.

Performs exact string matching and replacement within files.  Reads the
file, locates an exact match of the ``old_text`` parameter, replaces it
with ``new_text``, and writes the result back.  Returns a diff-like
output showing the changes made.
"""

from pathlib import Path

from local_cli.tools.base import Tool


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

        replace_all = kwargs.get("replace_all", False)
        if not isinstance(replace_all, bool):
            replace_all = False

        path = Path(file_path)

        if not path.exists():
            return f"Error: file not found: {file_path}"

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
            return f"Error: old_text not found in {file_path}"

        # Perform the replacement.
        if replace_all:
            new_content = content.replace(old_text, new_text)
            occurrences = count
        else:
            new_content = content.replace(old_text, new_text, 1)
            occurrences = 1

        # Write the modified content back.
        try:
            path.write_text(new_content, encoding="utf-8")
        except PermissionError:
            return f"Error: permission denied: {file_path}"
        except OSError as exc:
            return f"Error: could not write file: {exc}"

        return _make_diff_output(file_path, old_text, new_text, occurrences)
