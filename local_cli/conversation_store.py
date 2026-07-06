"""Per-project last-conversation autosave — quit no longer loses work.

The flight recorder (session_log.py) records *events* for diagnosis;
this store keeps the *faithful message history* needed to resume.
After every turn the frontend saves the conversation to
``<state_dir>/projects/<cwd-slug>/last-conversation.jsonl``; the next
session in the same folder can restore it (CLI ``/resume``, desktop
"restore last conversation").

System messages are not saved: the system prompt and project
instructions are version-dependent and re-built fresh on resume, and
mid-conversation skill injections re-fire on the next matching input.

Same constitution as the flight recorder: fail-open (a broken save
never breaks the session), atomic (write temp + rename, so a crash
mid-save cannot corrupt the previous good copy), tolerant load
(corrupt lines are skipped).
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_cli.session_log import project_slug

_FILENAME = "last-conversation.jsonl"

# A resumable conversation of thousands of messages would blow a small
# model's context anyway; keep the newest tail.
_MAX_MESSAGES = 400


class ConversationStore:
    """Save/load the last conversation for one project directory."""

    def __init__(
        self,
        state_dir: str,
        cwd: str | None = None,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        try:
            self._cwd = str(Path(cwd or os.getcwd()).resolve())
        except OSError:
            self._cwd = str(cwd or "?")
        self._dir = (
            Path(state_dir).expanduser() / "projects"
            / project_slug(self._cwd)
        )

    @property
    def path(self) -> Path:
        return self._dir / _FILENAME

    # ------------------------------------------------------------------
    # Save / load / clear
    # ------------------------------------------------------------------

    def save(self, messages: list[dict[str, Any]]) -> None:
        """Persist the conversation (minus system messages).  Never raises."""
        if not self._enabled:
            return
        keep = [
            m for m in messages
            if isinstance(m, dict) and m.get("role") != "system"
        ][-_MAX_MESSAGES:]
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "_meta": True,
                    "saved_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds",
                    ),
                    "cwd": self._cwd,
                }, ensure_ascii=False) + "\n")
                for message in keep:
                    fh.write(
                        json.dumps(message, ensure_ascii=False, default=str)
                        + "\n",
                    )
            os.replace(tmp_path, self.path)
        except Exception:
            pass  # fail-open: autosave must never break the session

    def load(self) -> list[dict[str, Any]]:
        """Return the saved messages ([] if absent or unreadable)."""
        if not self._enabled:
            return []
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return []
        messages: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if not isinstance(obj, dict) or obj.get("_meta"):
                continue
            if "role" in obj:
                messages.append(obj)
        return messages

    def info(self) -> dict[str, Any] | None:
        """Summary for a resume prompt: message count, time, preview."""
        messages = self.load()
        if not messages:
            return None
        saved_at = ""
        try:
            first_line = self.path.read_text(
                encoding="utf-8",
            ).splitlines()[0]
            meta = json.loads(first_line)
            if isinstance(meta, dict) and meta.get("_meta"):
                saved_at = str(meta.get("saved_at", ""))
        except (OSError, ValueError, IndexError):
            pass
        preview = ""
        for message in messages:
            if message.get("role") == "user":
                content = str(message.get("content", ""))
                preview = content[:80]
                break
        return {
            "count": len(messages),
            "saved_at": saved_at,
            "preview": preview,
        }

    def clear(self) -> None:
        """Discard the saved conversation (e.g. on /clear).  Never raises."""
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass
