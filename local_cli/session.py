"""Session persistence for local-cli.

Saves and loads conversation sessions in JSONL format (one JSON message
object per line) to the configured state directory.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


class SessionManager:
    """Manages conversation session persistence.

    Sessions are stored as JSONL files in a ``sessions`` subdirectory of the
    configured state directory.  Each line in a session file is a single JSON
    object representing one message in the conversation.

    Args:
        state_dir: Base directory for application state (e.g.
            ``~/.local/state/local-cli``).  A ``sessions`` subdirectory will
            be created automatically.
    """

    # Subdirectory within state_dir where session files are stored.
    _SESSIONS_SUBDIR = "sessions"

    # File extension for session files.
    _SESSION_EXT = ".jsonl"

    def __init__(self, state_dir: str) -> None:
        self._state_dir = Path(state_dir).expanduser()
        self._sessions_dir = self._state_dir / self._SESSIONS_SUBDIR
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_session_id(self) -> str:
        """Create a timestamped unique session identifier.

        Format: ``YYYYMMDD-HHMMSS-<short_uuid>`` (UTC).

        Returns:
            A unique session identifier string.
        """
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        return f"{timestamp}-{short_id}"

    def save_session(
        self,
        messages: list[dict],
        session_id: str | None = None,
    ) -> str:
        """Save a list of messages as a JSONL session file.

        Each message dict is written as a single JSON line.

        Args:
            messages: List of message dictionaries to persist.
            session_id: Optional session identifier.  If ``None``, a new
                identifier is generated via :meth:`generate_session_id`.

        Returns:
            The session identifier used for saving.
        """
        if session_id is None:
            session_id = self.generate_session_id()

        file_path = self._session_path(session_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as fh:
            for msg in messages:
                line = json.dumps(msg, ensure_ascii=False)
                fh.write(line + "\n")

        return session_id

    def load_session(self, session_id: str) -> list[dict]:
        """Load a session from its JSONL file.

        Corrupt or incomplete lines are silently skipped so that partially
        written sessions can still be recovered.

        Args:
            session_id: The session identifier to load.

        Returns:
            A list of message dictionaries.  Returns an empty list if the
            session file does not exist or is entirely unreadable.
        """
        file_path = self._session_path(session_id)

        if not file_path.exists():
            return []

        messages: list[dict] = []
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            messages.append(obj)
                    except (json.JSONDecodeError, ValueError):
                        # Skip corrupt / incomplete lines gracefully.
                        continue
        except OSError:
            return []

        return messages

    def list_sessions(self) -> list[str]:
        """Return available session identifiers sorted by date (newest first).

        Session IDs are derived from filenames by stripping the ``.jsonl``
        extension.  Sorting is lexicographic on the filename, which—given the
        ``YYYYMMDD-HHMMSS-*`` naming convention—corresponds to chronological
        order.

        Returns:
            A list of session identifier strings, newest first.
        """
        if not self._sessions_dir.is_dir():
            return []

        session_ids: list[str] = []
        try:
            for entry in self._sessions_dir.iterdir():
                if entry.is_file() and entry.suffix == self._SESSION_EXT:
                    session_ids.append(entry.stem)
        except OSError:
            return []

        # Sort descending (newest first) — timestamps sort lexicographically.
        session_ids.sort(reverse=True)
        return session_ids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session_path(self, session_id: str) -> Path:
        """Return the filesystem path for a given session identifier."""
        return self._sessions_dir / f"{session_id}{self._SESSION_EXT}"
