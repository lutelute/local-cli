"""Glob tool for file pattern matching.

Uses ``pathlib.Path.glob()`` to find files matching a glob pattern.
Returns matching file paths sorted by modification time (most recent
first).
"""

from pathlib import Path

from local_cli.tools.base import Tool


class GlobTool(Tool):
    """Find files matching a glob pattern."""

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return (
            "Find files matching a glob pattern. Returns matching file "
            "paths sorted by modification time (most recent first). "
            "Use patterns like '*.py', '**/*.ts', or 'src/**/*.js'."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "The glob pattern to match files against "
                        "(e.g. '*.py', '**/*.ts')."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": (
                        "The directory to search in. "
                        "Defaults to the current working directory."
                    ),
                },
            },
            "required": ["pattern"],
        }

    def execute(self, **kwargs: object) -> str:
        """Find files matching a glob pattern.

        Args:
            **kwargs: Must include ``pattern`` (str).  May include
                ``path`` (str, defaults to current working directory).

        Returns:
            Matching file paths sorted by modification time, one per
            line, or an error message if the search fails.
        """
        pattern = kwargs.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            return "Error: 'pattern' parameter is required and must be a non-empty string."

        search_path = kwargs.get("path", ".")
        if not isinstance(search_path, str) or not search_path.strip():
            search_path = "."

        base = Path(search_path)

        if not base.exists():
            return f"Error: directory not found: {search_path}"

        if not base.is_dir():
            return f"Error: not a directory: {search_path}"

        try:
            matches = list(base.glob(pattern))
        except ValueError as exc:
            return f"Error: invalid glob pattern: {exc}"
        except OSError as exc:
            return f"Error: could not search directory: {exc}"

        if not matches:
            return f"No files matched pattern: {pattern}"

        # Sort by modification time, most recent first.
        try:
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            # Fall back to alphabetical if stat fails on any file.
            matches.sort()

        return "\n".join(str(m) for m in matches)
