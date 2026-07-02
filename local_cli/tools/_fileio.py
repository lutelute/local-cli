"""Shared low-level file helpers for tools.

The ``edit`` and ``write`` tools both need to persist file content without
risking a half-written file if something fails mid-write.  This module is
the single place that implements that, so both tools behave identically.
It also hosts the not-found path suggestion shared by ``read`` and
``edit`` error messages.
"""

import os
import tempfile
from pathlib import Path


def find_similar_path(requested: str) -> str | None:
    """Find a file under the cwd sharing the requested path's file name.

    Small models frequently hallucinate an absolute prefix
    (``/app/app.py`` for a file that is really ``./app.py``) and then
    burn several loop iterations rediscovering the path.  When a
    requested file does not exist but a file with the same name does —
    in the working directory or up to two levels below it — returning
    that path lets the model recover in one step.

    The search is name-exact and shallow (``name``, ``*/name``,
    ``*/*/name``), so it stays fast in large repositories, and a hint
    is only returned when the match is unambiguous.

    Args:
        requested: The path the caller asked for (which did not exist).

    Returns:
        The unique matching path, relative to the cwd, or ``None``.
    """
    name = Path(requested).name
    if not name:
        return None
    cwd = Path.cwd()

    direct = cwd / name
    if direct.is_file():
        return name

    matches: list[Path] = []
    try:
        for pattern in (f"*/{name}", f"*/*/{name}"):
            for candidate in cwd.glob(pattern):
                if candidate.is_file():
                    matches.append(candidate)
                    if len(matches) > 1:
                        return None  # ambiguous — no hint
    except OSError:
        return None

    if len(matches) == 1:
        try:
            return str(matches[0].relative_to(cwd))
        except ValueError:
            return None
    return None


def not_found_error(file_path: str) -> str:
    """Build a file-not-found error, suggesting the real path when known."""
    error = f"Error: file not found: {file_path}"
    hint = find_similar_path(file_path)
    if hint is not None and hint != file_path:
        return (
            f"{error}. A file with that name exists at '{hint}' "
            "(relative to the working directory) — use that path."
        )
    return error


def atomic_write_text(path: Path, content: str) -> None:
    """Write *content* to *path* atomically (temp file + rename).

    The content is written to a sibling temp file which is then
    ``os.replace``-d into place, so a failure mid-write leaves any existing
    file intact rather than truncated.

    Permission handling:

    * When *path* already exists, its permission bits are preserved.
    * When *path* is new, the default ``0666 & ~umask`` is applied (the same
      mode a normal ``open(...,'w')`` would produce) rather than the
      restrictive ``0600`` that ``mkstemp`` creates.

    Args:
        path: Destination file path.  Its parent directory must already
            exist.
        content: Text to write (UTF-8).
    """
    try:
        orig_mode: int | None = path.stat().st_mode
    except OSError:
        orig_mode = None

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".tmp-", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        if orig_mode is not None:
            target_mode: int | None = orig_mode
        else:
            # New file: mirror what open()'s default would give (0666 minus
            # the process umask) instead of mkstemp's locked-down 0600.
            current_umask = os.umask(0)
            os.umask(current_umask)
            target_mode = 0o666 & ~current_umask
        try:
            os.chmod(tmp_name, target_mode)
        except OSError:
            pass
        os.replace(tmp_name, str(path))
    except BaseException:
        # Clean up the temp file on any failure (including the rename).
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
