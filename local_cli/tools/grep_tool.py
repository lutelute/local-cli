"""Grep tool for searching file contents.

Searches file contents using the ``re`` module for regex pattern
matching.  Returns matching lines with file paths and line numbers.
Supports case-insensitive search and glob-based file filtering.
"""

import re
from pathlib import Path

from local_cli.tools.base import Tool


class GrepTool(Tool):
    """Search file contents using regular expressions."""

    @property
    def cacheable(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents using a regular expression pattern. "
            "Returns matching lines with file paths and line numbers. "
            "Supports case-insensitive search and glob-based file filtering."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "The regular expression pattern to search for."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": (
                        "The file or directory to search in. "
                        "Defaults to the current working directory."
                    ),
                },
                "include": {
                    "type": "string",
                    "description": (
                        "Glob pattern to filter which files to search "
                        "(e.g. '*.py', '*.ts'). Defaults to all files."
                    ),
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": (
                        "Whether to perform case-insensitive matching. "
                        "Defaults to false."
                    ),
                },
            },
            "required": ["pattern"],
        }

    def execute(self, **kwargs: object) -> str:
        """Search file contents for a regex pattern.

        Args:
            **kwargs: Must include ``pattern`` (str).  May include
                ``path`` (str, defaults to ``"."``), ``include``
                (str glob filter), and ``case_insensitive`` (bool).

        Returns:
            Matching lines formatted as ``file:line_number:content``,
            or an error message if the search fails.
        """
        pattern = kwargs.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            return "Error: 'pattern' parameter is required and must be a non-empty string."

        search_path = kwargs.get("path", ".")
        if not isinstance(search_path, str) or not search_path.strip():
            search_path = "."

        include = kwargs.get("include")
        if include is not None and (not isinstance(include, str) or not include.strip()):
            include = None

        case_insensitive = kwargs.get("case_insensitive", False)
        if not isinstance(case_insensitive, bool):
            case_insensitive = False

        # Compile the regex pattern.
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return f"Error: invalid regex pattern: {exc}"

        base = Path(search_path)

        if not base.exists():
            return f"Error: path not found: {search_path}"

        # Collect files to search.
        files: list[Path] = []
        if base.is_file():
            files.append(base)
        elif base.is_dir():
            if include:
                files = sorted(base.rglob(include))
            else:
                files = sorted(
                    p for p in base.rglob("*") if p.is_file()
                )
        else:
            return f"Error: not a file or directory: {search_path}"

        results: list[str] = []
        max_results = 500

        for file_path in files:
            if not file_path.is_file():
                continue

            # Skip binary files by checking for null bytes.
            try:
                raw = file_path.read_bytes()[:8192]
            except (PermissionError, OSError):
                continue

            if b"\x00" in raw:
                continue

            # Read text content.
            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError, OSError):
                continue

            for line_num, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    results.append(f"{file_path}:{line_num}:{line}")
                    if len(results) >= max_results:
                        results.append(
                            f"... truncated ({max_results} matches shown)"
                        )
                        return "\n".join(results)

        if not results:
            return f"No matches found for pattern: {pattern}"

        return "\n".join(results)
