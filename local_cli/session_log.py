"""Always-on session transcript logging — the flight recorder.

Claude Code records every session as JSONL under
``~/.claude/projects/<cwd-slug>/``.  local-cli mirrors that: the moment
a frontend starts in a folder, a :class:`SessionLogger` appends one
JSON object per event to
``<state_dir>/projects/<cwd-slug>/<session-id>.jsonl``.

Design constraints:

- **Fail-open** — logging must never break or slow the agent.  The
  first write error marks the logger broken and it goes silent.
- **Bounded** — tool args/outputs are clipped per event; streaming
  deltas are counted, not stored (the ``assistant`` event carries the
  final content anyway).
- **Layer diagnosis** — every ``turn_end`` records ``visible_chars`` /
  ``thinking_chars`` / ``tool_calls``: the exact numbers that separate
  "the model was silent" from "the reply was swallowed into thinking"
  after the fact, without a live repro.

Disable with ``LOCAL_CLI_SESSION_LOG=0`` (also ``false``/``off``/``no``).
"""

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from local_cli.harness import AgentEvent

# Per-event clip for tool args / outputs / assistant content.  Long
# enough to diagnose, short enough that a runaway bash dump cannot
# balloon the log.
_CLIP_CHARS = 16_000

_DISABLE_VALUES = frozenset({"0", "false", "off", "no"})

# Harness interventions worth a line of their own (same set the server
# forwards to GUIs).
_INTERVENTION_KINDS = frozenset({
    "rescue", "loop_warning", "loop_break", "limit",
    "reminder", "verify_warning", "compaction", "retry",
    "nudge", "error_stop", "empty_response",
    "tools_fallback", "write_deferred", "deliverable_nudge",
    "read_gate",
})

_PROJECTS_SUBDIR = "projects"
_SLUG_MAX = 150


def session_log_enabled() -> bool:
    """Whether the LOCAL_CLI_SESSION_LOG env var allows logging."""
    value = os.environ.get("LOCAL_CLI_SESSION_LOG", "").strip().lower()
    return value not in _DISABLE_VALUES


def project_slug(cwd: str) -> str:
    """Map an absolute path to a filesystem-safe directory name.

    Mirrors the Claude Code convention: ``/Users/x/proj`` becomes
    ``-Users-x-proj``.  Overlong slugs keep their tail — the most
    specific (and therefore distinguishing) part of the path.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(cwd)).rstrip("-")
    if len(slug) > _SLUG_MAX:
        slug = slug[-_SLUG_MAX:].lstrip("-")
    return slug or "-"


def _new_session_id() -> str:
    """Timestamped unique id, same shape SessionManager uses."""
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _clip(text: str, limit: int = _CLIP_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


class SessionLogger:
    """Append-only JSONL transcript for one session in one project.

    Args:
        state_dir: Base state directory (e.g. ``~/.local/state/local-cli``).
        cwd: Project directory the session runs in; defaults to the
            process working directory.
        enabled: Force on/off; ``None`` reads LOCAL_CLI_SESSION_LOG.
        session_id: Fixed id (tests); ``None`` generates one.
    """

    def __init__(
        self,
        state_dir: str,
        cwd: str | None = None,
        enabled: bool | None = None,
        session_id: str | None = None,
    ) -> None:
        self._enabled = session_log_enabled() if enabled is None else bool(enabled)
        self._broken = False
        self._fh: Any = None
        self._lock = threading.Lock()
        try:
            self._cwd = str(Path(cwd or os.getcwd()).resolve())
        except OSError:
            # cwd can vanish under the process (deleted temp dir).
            self._cwd = str(cwd or "?")
        self._dir = (
            Path(state_dir).expanduser() / _PROJECTS_SUBDIR
            / project_slug(self._cwd)
        )
        self._session_id = session_id or _new_session_id()
        self._visible_chars = 0
        self._thinking_chars = 0
        self._tool_calls = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._broken

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def path(self) -> Path:
        """Where this session's transcript lives (or would live)."""
        return self._dir / f"{self._session_id}.jsonl"

    # ------------------------------------------------------------------
    # Core write path
    # ------------------------------------------------------------------

    def log(self, type_: str, **fields: Any) -> None:
        """Append one event line.  Never raises."""
        if not self.enabled:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "type": type_,
            **fields,
        }
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock:
                if self._fh is None:
                    self._dir.mkdir(parents=True, exist_ok=True)
                    self._fh = open(self.path, "a", encoding="utf-8")
                self._fh.write(line + "\n")
                self._fh.flush()
        except Exception:
            # Fail-open: one broken write silences the logger for good
            # rather than erroring on every subsequent event.
            self._broken = True
            self._close_quietly()

    def close(self) -> None:
        with self._lock:
            self._close_quietly()

    def _close_quietly(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def log_session_start(
        self,
        model: str = "",
        provider: str = "",
        app_version: str = "",
        frontend: str = "",
        reason: str = "start",
    ) -> None:
        self.log(
            "session_start",
            cwd=self._cwd,
            model=model,
            provider=provider,
            app_version=app_version,
            frontend=frontend,
            reason=reason,
        )

    def log_user(self, text: str) -> None:
        self.log("user", content=_clip(text))

    def log_turn_end(self, error: bool = False) -> None:
        """Close out one user turn with the layer-diagnosis counters."""
        self.log(
            "turn_end",
            visible_chars=self._visible_chars,
            thinking_chars=self._thinking_chars,
            tool_calls=self._tool_calls,
            error=error,
        )
        self._visible_chars = 0
        self._thinking_chars = 0
        self._tool_calls = 0

    def rotate(self) -> str:
        """Start a new transcript file (e.g. after /clear).

        Returns the new session id.  The caller logs the next
        ``session_start`` with whatever context it has.
        """
        with self._lock:
            self._close_quietly()
        self._session_id = _new_session_id()
        self._visible_chars = 0
        self._thinking_chars = 0
        self._tool_calls = 0
        return self._session_id

    # ------------------------------------------------------------------
    # Agent event tee
    # ------------------------------------------------------------------

    def emit(self, event: AgentEvent) -> None:
        """Logging-only EmitFn: map agent events to transcript lines.

        Streaming deltas are accumulated into the turn counters instead
        of being written (one line per token would dwarf the log).
        Never raises.
        """
        if not self.enabled:
            return
        try:
            kind = event.kind
            data = event.data
            if kind == "content_delta":
                self._visible_chars += len(data.get("text", "") or "")
            elif kind == "thinking_delta":
                self._thinking_chars += len(data.get("text", "") or "")
            elif kind == "llm_start":
                self.log("llm_start", iteration=data.get("iteration"))
            elif kind == "assistant_message":
                message = data.get("message") or {}
                self.log(
                    "assistant",
                    content=_clip(str(message.get("content", "") or "")),
                    thinking_chars=len(data.get("thinking", "") or ""),
                )
            elif kind == "tool_start":
                args = data.get("arguments", {})
                self.log(
                    "tool_start",
                    tool=data.get("tool_name", ""),
                    args=_clip(json.dumps(args, ensure_ascii=False, default=str)),
                )
            elif kind == "tool_result":
                self._tool_calls += 1
                self.log(
                    "tool_result",
                    tool=data.get("tool_name", ""),
                    output=_clip(str(data.get("result", "") or "")),
                )
            elif kind == "error":
                self.log(
                    "error",
                    source=data.get("source", ""),
                    message=str(data.get("detail") or data.get("message", "")),
                )
            elif kind in _INTERVENTION_KINDS:
                self.log("harness", event=kind)
        except Exception:
            self._broken = True
            self._close_quietly()

    def wrap_emit(
        self, inner: Callable[[AgentEvent], None],
    ) -> Callable[[AgentEvent], None]:
        """Tee: forward each event to *inner*, then record it here."""
        def tee(event: AgentEvent) -> None:
            inner(event)
            self.emit(event)
        return tee
