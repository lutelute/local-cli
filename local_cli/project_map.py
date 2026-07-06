"""Project map injection — deterministic orientation at session start.

Small models burn their first iterations (and often their best
attention) exploring: ls, glob, wrong paths, retries.  A compact,
sorted file list injected once at session start gives the model exact
paths to read instead of guesses — the same reason a human opens the
file tree first.

The map is a snapshot: cheap (git ls-files when available, a pruned
walk otherwise), hard-capped in entries and characters so it can never
crowd a small context window, and labelled as a snapshot so the model
still verifies before editing.  Disable with LOCAL_CLI_PROJECT_MAP=0.
"""

import os
import subprocess
from pathlib import Path
from typing import Any

_MAX_ENTRIES = 120
_MAX_CHARS = 2_000

_EXCLUDE_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", "target", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".idea", ".vscode", ".cache",
})

_DISABLE_VALUES = frozenset({"0", "false", "off", "no"})

_WALK_MAX_DEPTH = 4


def project_map_enabled() -> bool:
    """Whether LOCAL_CLI_PROJECT_MAP allows injection."""
    value = os.environ.get("LOCAL_CLI_PROJECT_MAP", "").strip().lower()
    return value not in _DISABLE_VALUES


def _git_files(cwd: str) -> list[str] | None:
    """Tracked + untracked-but-not-ignored files, or None outside git."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _walk_files(cwd: str) -> list[str]:
    """Fallback listing: pruned walk, bounded depth, no hidden dirs."""
    root = Path(cwd)
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(cwd):
        rel_dir = Path(dirpath).relative_to(root)
        depth = 0 if str(rel_dir) == "." else len(rel_dir.parts)
        if depth >= _WALK_MAX_DEPTH:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in _EXCLUDE_DIRS and not d.startswith(".")
        )
        for name in filenames:
            if name.startswith("."):
                continue
            rel = name if str(rel_dir) == "." else f"{rel_dir}/{name}"
            found.append(rel)
            if len(found) > _MAX_ENTRIES * 3:
                return found  # plenty for the capped map
    return found


def build_project_map(cwd: str | None = None) -> str | None:
    """A sorted, capped file listing for *cwd* (None when empty)."""
    directory = cwd or os.getcwd()
    files = _git_files(directory)
    if files is None:
        files = _walk_files(directory)
    files = sorted(set(files))
    if not files:
        return None
    total = len(files)
    shown = files[:_MAX_ENTRIES]
    text = "\n".join(shown)
    if total > len(shown):
        text += f"\n... and {total - len(shown)} more files"
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "\n... [map truncated]"
    return text


def project_map_message(cwd: str | None = None) -> dict[str, Any] | None:
    """The injectable system message, or None (disabled/empty dir)."""
    if not project_map_enabled():
        return None
    listing = build_project_map(cwd)
    if listing is None:
        return None
    return {
        "role": "system",
        "content": (
            "--- PROJECT MAP (snapshot at session start) ---\n"
            f"{listing}\n"
            "--- END PROJECT MAP ---\n\n"
            "These are the project's file paths. Read the relevant ones "
            "directly instead of exploring; the map is a snapshot, so "
            "verify with read before editing."
        ),
    }
