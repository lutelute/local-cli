"""Read tool for reading file contents.

Reads files using ``pathlib.Path``, detects binary files via null-byte
check, and returns content with line numbers.  Supports ``offset`` and
``limit`` parameters for reading sections of large files.
"""

from pathlib import Path

from local_cli.tools.base import Tool


class ReadTool(Tool):
    """Read the contents of a file and return it with line numbers."""

    @property
    def cacheable(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        return (
            "Read a file and return its contents with line numbers. "
            "Detects binary files automatically. Use offset and limit "
            "to read specific sections of large files."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to read.",
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "Line number to start reading from (1-based). "
                        "Defaults to 1."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Maximum number of lines to read. "
                        "Defaults to reading the entire file."
                    ),
                },
            },
            "required": ["file_path"],
        }

    def execute(self, **kwargs: object) -> str:
        """Read a file and return its contents with line numbers.

        Args:
            **kwargs: Must include ``file_path`` (str).  May include
                ``offset`` (int, default 1) and ``limit`` (int).

        Returns:
            The file contents prefixed with line numbers, or an error
            message if the file cannot be read.
        """
        file_path = kwargs.get("file_path")
        if not isinstance(file_path, str) or not file_path.strip():
            return "Error: 'file_path' parameter is required and must be a non-empty string."

        offset = kwargs.get("offset", 1)
        if not isinstance(offset, (int, float)):
            offset = 1
        offset = max(1, int(offset))

        limit = kwargs.get("limit")
        if limit is not None:
            if not isinstance(limit, (int, float)):
                limit = None
            else:
                limit = max(1, int(limit))

        path = Path(file_path)

        if not path.exists():
            return f"Error: file not found: {file_path}"

        if not path.is_file():
            return f"Error: not a regular file: {file_path}"

        # Detect binary files by checking for null bytes in the first 8KB.
        # Read only the first 8KB rather than the whole file, so a huge file
        # is not loaded into memory just for this check -- important on the
        # memory-constrained machines local-cli targets.
        try:
            with path.open("rb") as fh:
                raw = fh.read(8192)
        except PermissionError:
            return f"Error: permission denied: {file_path}"
        except OSError as exc:
            return f"Error: could not read file: {exc}"

        if b"\x00" in raw:
            return f"Error: {file_path} appears to be a binary file."

        # Read text content.
        decoded_as_latin1 = False
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback: latin-1 accepts all byte values.  Flag it so the
            # model knows the decoding may be imperfect (possible mojibake).
            try:
                content = path.read_text(encoding="latin-1")
                decoded_as_latin1 = True
            except OSError as exc:
                return f"Error: could not read file: {exc}"
        except PermissionError:
            return f"Error: permission denied: {file_path}"
        except OSError as exc:
            return f"Error: could not read file: {exc}"

        lines = content.splitlines()
        total_lines = len(lines)

        # Apply offset (1-based).
        start_idx = offset - 1
        selected = lines[start_idx:]

        # Apply limit.
        if limit is not None:
            selected = selected[:limit]

        if not selected and total_lines > 0:
            return f"Error: offset {offset} is beyond end of file ({total_lines} lines)."

        # Format with line numbers.
        numbered_lines: list[str] = []
        for i, line in enumerate(selected, start=offset):
            numbered_lines.append(f"{i:6d}\t{line}")

        body = "\n".join(numbered_lines)
        if decoded_as_latin1:
            body = (
                f"[note: {file_path} is not valid UTF-8; decoded as latin-1, "
                "some characters may be wrong]\n" + body
            )
        return body
