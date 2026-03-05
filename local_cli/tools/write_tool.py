"""Write tool for creating and overwriting files.

Creates or overwrites files using ``pathlib.Path``, automatically creating
parent directories as needed (equivalent to ``mkdir -p``).  Validates
paths to prevent directory traversal attacks.
"""

from pathlib import Path

from local_cli.tools.base import Tool


def _is_path_safe(file_path: str) -> bool:
    """Check that a file path does not contain directory traversal.

    Resolves the path and verifies it does not escape the current
    working directory via ``..`` components.

    Args:
        file_path: The file path string to validate.

    Returns:
        True if the path is safe, False otherwise.
    """
    try:
        resolved = Path(file_path).resolve()
        cwd = Path.cwd().resolve()
        # Allow absolute paths, but reject paths with '..' that escape cwd
        # when the path is relative.
        if not Path(file_path).is_absolute():
            return str(resolved).startswith(str(cwd))
        return True
    except (OSError, ValueError):
        return False


class WriteTool(Tool):
    """Create or overwrite a file with the given content."""

    @property
    def name(self) -> str:
        return "write"

    @property
    def description(self) -> str:
        return (
            "Create or overwrite a file with the provided content. "
            "Parent directories are created automatically if they do "
            "not exist. Use this for writing new files or completely "
            "replacing existing file contents."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["file_path", "content"],
        }

    def execute(self, **kwargs: object) -> str:
        """Create or overwrite a file with the given content.

        Parent directories are created automatically.  The path is
        validated to prevent directory traversal.

        Args:
            **kwargs: Must include ``file_path`` (str) and
                ``content`` (str).

        Returns:
            A confirmation message, or an error message if the file
            could not be written.
        """
        file_path = kwargs.get("file_path")
        if not isinstance(file_path, str) or not file_path.strip():
            return "Error: 'file_path' parameter is required and must be a non-empty string."

        content = kwargs.get("content")
        if not isinstance(content, str):
            return "Error: 'content' parameter is required and must be a string."

        # Security check: reject directory traversal.
        if not _is_path_safe(file_path):
            return f"Error: path rejected (directory traversal not allowed): {file_path}"

        path = Path(file_path)

        # Reject writing to a directory.
        if path.exists() and path.is_dir():
            return f"Error: path is a directory, not a file: {file_path}"

        # Create parent directories if they don't exist.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return f"Error: permission denied creating directories for: {file_path}"
        except OSError as exc:
            return f"Error: could not create parent directories: {exc}"

        # Write the file.
        try:
            path.write_text(content, encoding="utf-8")
        except PermissionError:
            return f"Error: permission denied: {file_path}"
        except OSError as exc:
            return f"Error: could not write file: {exc}"

        num_lines = len(content.splitlines())
        num_bytes = len(content.encode("utf-8"))
        return f"Successfully wrote {num_bytes} bytes ({num_lines} lines) to {file_path}"
