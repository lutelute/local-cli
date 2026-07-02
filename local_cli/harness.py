"""Deterministic harness components for the agent loop.

The goal of this module is "frontier-grade actions without a
frontier-grade model": each component is a deterministic intervention
that detects and repairs a known failure mode of small local models
(4B-9B), so that the agent loop can sustain long multi-step sessions
that would otherwise require a much larger model.

Components (all self-contained; no imports from the rest of local_cli):

- :class:`AgentEvent` / :class:`HarnessConfig` — the event and
  configuration types shared by the unified agent loop
  (:func:`local_cli.agent.run_agent`) and its front-end emitters.
- :func:`extract_text_tool_calls` — rescues tool calls that the model
  wrote as *text* (``<tool_call>`` tags, fenced JSON, bare JSON) instead
  of emitting structured ``tool_calls``.  Small models frequently have
  broken or absent function-calling templates; without this rescue the
  agent silently stops after printing the call it meant to make.
- :class:`LoopDetector` — detects the same tool call repeating without
  progress (the classic small-model infinite loop) and escalates from a
  corrective reminder to a forced stop.
- :func:`verify_file_write` — syntax-checks files right after a
  write/edit (``ast.parse`` for Python, ``json.loads`` for JSON) so the
  model hears about a broken file immediately, not three steps later.
- :class:`TodoReminder` — re-injects the todo state when the list has
  gone stale, the way Claude Code nudges with system reminders, so a
  small model does not abandon a half-finished task list.
- :func:`build_summary_request` / :func:`compaction_bounds` /
  :func:`apply_summary` — helpers for LLM-summarized context compaction
  (with the caller falling back to plain truncation on failure).
"""

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Event and configuration types
# ---------------------------------------------------------------------------


@dataclass
class AgentEvent:
    """A single event emitted by the unified agent loop.

    The loop itself performs no I/O; every observable moment is emitted
    as an event and each front-end (CLI stdout, JSON-line server, web
    monitor SSE, silent sub-agent) renders the events it cares about.

    Attributes:
        kind: The event type.  One of: ``llm_start``, ``thinking_delta``,
            ``content_delta``, ``assistant_message``, ``rescue``,
            ``nudge``, ``tool_start``, ``tool_result``, ``verify_warning``,
            ``loop_warning``, ``loop_break``, ``reminder``, ``compaction``,
            ``limit``, ``retry``, ``error``, ``interrupted``, ``debug``.
        data: Event-specific payload.
    """

    kind: str
    data: dict[str, Any] = field(default_factory=dict)


# Type alias for the emitter callback front-ends provide.
EmitFn = Callable[[AgentEvent], None]


def null_emit(event: AgentEvent) -> None:
    """An emitter that discards every event (silent execution)."""


@dataclass
class HarnessConfig:
    """Tunable switches for the deterministic harness interventions.

    Attributes:
        max_iterations: Hard cap on agent-loop iterations.  ``0`` means
            unlimited.  When the cap is reached the loop makes one final
            tool-free LLM call so the model can summarize its progress.
        text_tool_rescue: Parse tool calls the model wrote as text when
            no structured ``tool_calls`` were emitted.
        loop_detection: Detect repeated identical tool calls and
            intervene (reminder first, forced stop second).
        verify_writes: Syntax-check ``.py`` / ``.json`` files right
            after a successful write/edit and append a warning to the
            tool result on failure.
        todo_reminders: Re-inject the todo list state when it has gone
            stale for several iterations.
        error_stop_guard: When the model tries to finish right after a
            failed tool call, push back once — small models routinely
            ignore the error and declare the task done.
        defer_writes_after_search: When one assistant turn mixes
            discovery calls (grep/glob) with mutations (write/edit),
            defer the mutations — they were decided before the search
            results existed, and small models write "fixes" into brand
            new files while the search correctly locates the real one.
        compact_mode: ``"truncate"`` (classic in-place truncation) or
            ``"summarize"`` (ask the model to summarize older history,
            falling back to truncation on failure).
        keep_recent: Number of recent messages preserved verbatim by
            summary compaction.
        retry_on_overload: Retry the LLM request on provider overload
            (HTTP 503) with exponential backoff.
    """

    max_iterations: int = 40
    text_tool_rescue: bool = True
    loop_detection: bool = True
    verify_writes: bool = True
    todo_reminders: bool = True
    error_stop_guard: bool = True
    defer_writes_after_search: bool = True
    compact_mode: str = "truncate"
    keep_recent: int = 10
    retry_on_overload: bool = True


# ---------------------------------------------------------------------------
# Text-based tool-call rescue
# ---------------------------------------------------------------------------

# Qwen-style chat-template leak: the model prints the structured call as
# literal <tool_call> tags in its content instead of the tool_calls field.
_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE
)

# Fenced code blocks that plausibly contain a tool call.  Only fences
# explicitly labelled as JSON / tool blocks are scanned — a plain ``` fence
# is usually code the user asked for, not a call.
_FENCED_BLOCK_RE = re.compile(
    r"```(?:json|tool_call|tool_code|tool)[ \t]*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Key names models use for the tool name and its arguments.
_NAME_KEYS = ("name", "tool", "tool_name")
_ARGS_KEYS = ("arguments", "args", "parameters", "params", "input")

# Maximum number of rescued calls per assistant message.
_RESCUE_LIMIT = 4


def _scan_json_objects(text: str, limit: int = 8) -> list[Any]:
    """Parse up to *limit* top-level ``{...}`` JSON objects embedded in *text*.

    Scans for balanced braces (string-aware, so braces inside JSON strings
    do not confuse the depth counter) and attempts ``json.loads`` on each
    balanced span.  Spans that fail to parse are skipped.  Scanning stops
    at the first unbalanced brace to keep the pass linear.

    Args:
        text: Arbitrary text possibly containing JSON objects.
        limit: Maximum number of parsed objects to return.

    Returns:
        The parsed objects, in order of appearance.
    """
    objects: list[Any] = []
    i = 0
    n = len(text)
    while i < n and len(objects) < limit:
        start = text.find("{", i)
        if start == -1:
            break
        depth = 0
        in_str = False
        esc = False
        end = -1
        for j in range(start, n):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = j
                        break
        if end == -1:
            # Unbalanced from here on — nothing more can parse.
            break
        try:
            objects.append(json.loads(text[start : end + 1]))
        except (json.JSONDecodeError, ValueError):
            pass
        i = end + 1
    return objects


def _coerce_tool_call(obj: Any) -> tuple[str, dict[str, Any]] | None:
    """Interpret a parsed JSON object as a ``(name, arguments)`` tool call.

    Accepts the common shapes small models emit::

        {"name": "write", "arguments": {...}}
        {"tool": "bash", "args": {...}}
        {"function": {"name": "read", "arguments": "{...json...}"}}

    Args:
        obj: A parsed JSON value.

    Returns:
        ``(name, arguments)`` when *obj* looks like a tool call,
        ``None`` otherwise.  A missing/unparsable arguments value
        degrades to ``{}`` (the tool itself rejects bad args).
    """
    if not isinstance(obj, dict):
        return None

    # Unwrap {"function": {...}} (the OpenAI structured-call shape).
    func = obj.get("function")
    if isinstance(func, dict):
        return _coerce_tool_call(func)

    name: str | None = None
    for key in _NAME_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            name = value.strip()
            break
    if name is None:
        return None

    args: Any = None
    for key in _ARGS_KEYS:
        if key in obj:
            args = obj[key]
            break
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            args = None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return name, args


def extract_text_tool_calls(
    content: str,
    resolver: Callable[[str], str | None],
    limit: int = _RESCUE_LIMIT,
) -> list[dict[str, Any]]:
    """Rescue tool calls the model wrote as text instead of structured calls.

    Small local models frequently *intend* a tool call but emit it as
    plain text — a ``<tool_call>`` tag pair (Qwen chat-template leak), a
    fenced ```` ```json ```` block, or a bare JSON object.  Without rescue
    the agent loop sees "no tool calls" and stops mid-task.

    False-positive guards: only objects that coerce to a tool-call shape
    (a name key plus dict-like arguments) *and* whose name resolves to a
    real tool via *resolver* are accepted; duplicates are collapsed.

    Args:
        content: The assistant message content.
        resolver: Maps a (possibly near-miss) tool name to a real tool
            name, returning ``None`` for unknown names — typically
            ``lambda n: resolve_tool_name(n, tool_map)``.
        limit: Maximum number of calls to rescue.

    Returns:
        A list of tool-call dicts in the standard
        ``{"function": {"name", "arguments"}, "id"}`` shape (ids are
        synthetic: ``text_rescue_N``), or an empty list.
    """
    if not content or "{" not in content:
        return []

    candidates: list[Any] = []
    for regex in (_TOOL_CALL_TAG_RE, _FENCED_BLOCK_RE):
        for match in regex.finditer(content):
            candidates.extend(_scan_json_objects(match.group(1)))

    # Only when no tagged/fenced call was found, scan the whole content
    # for bare JSON objects (models often print the call with no fence).
    if not candidates:
        candidates = _scan_json_objects(content)

    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for obj in candidates:
        if len(calls) >= limit:
            break
        coerced = _coerce_tool_call(obj)
        if coerced is None:
            continue
        name, args = coerced
        if resolver(name) is None:
            continue
        try:
            key = name + "\x00" + json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            key = name + "\x00" + str(args)
        if key in seen:
            continue
        seen.add(key)
        calls.append(
            {
                "function": {"name": name, "arguments": args},
                "id": f"text_rescue_{len(calls)}",
            }
        )
    return calls


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------


class LoopDetector:
    """Detect repeated tool calls that indicate the model is stuck.

    Small models loop in two shapes: the *same* call repeated verbatim
    (retrying a failing edit with the same wrong ``old_text``), and a
    short alternation between two calls (read → failing edit → read …).
    Both are caught by fingerprinting each ``(tool, arguments)`` pair and
    checking the recent history:

    - **warn** (once per session): 3 identical calls in a row, or the
      last 6 calls contain at most 2 distinct fingerprints.
    - **break**: 5 identical calls in a row, or the last 10 calls
      contain at most 2 distinct fingerprints.

    The caller responds to ``warn`` by injecting a corrective reminder
    and to ``break`` by pausing tool access for a final summary turn.
    """

    def __init__(
        self,
        warn_repeats: int = 3,
        break_repeats: int = 5,
        warn_window: int = 6,
        break_window: int = 10,
        window_distinct: int = 2,
    ) -> None:
        self._warn_repeats = warn_repeats
        self._break_repeats = break_repeats
        self._warn_window = warn_window
        self._break_window = break_window
        self._window_distinct = window_distinct
        self._history: list[str] = []
        self._warned = False

    @staticmethod
    def _fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
        try:
            args_key = json.dumps(arguments, sort_keys=True, default=str)
        except (TypeError, ValueError):
            args_key = str(arguments)
        return f"{tool_name}\x00{args_key}"

    def record(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Record a tool call and classify the current repetition state.

        Args:
            tool_name: The resolved tool name.
            arguments: The (normalized) tool arguments.

        Returns:
            ``"break"`` when the loop should be force-stopped,
            ``"warn"`` when a one-time corrective reminder is due,
            ``"ok"`` otherwise.
        """
        self._history.append(self._fingerprint(tool_name, arguments))

        tail = self._history[-self._break_repeats :]
        if len(tail) == self._break_repeats and len(set(tail)) == 1:
            return "break"
        window = self._history[-self._break_window :]
        if len(window) == self._break_window and len(set(window)) <= self._window_distinct:
            return "break"

        if not self._warned:
            tail = self._history[-self._warn_repeats :]
            if len(tail) == self._warn_repeats and len(set(tail)) == 1:
                self._warned = True
                return "warn"
            window = self._history[-self._warn_window :]
            if len(window) == self._warn_window and len(set(window)) <= self._window_distinct:
                self._warned = True
                return "warn"

        return "ok"


def loop_warning_message(tool_name: str) -> dict[str, Any]:
    """Build the corrective reminder injected on a loop *warn* verdict."""
    return {
        "role": "user",
        "content": (
            "<system-reminder>You have repeated the same tool call "
            f"('{tool_name}') several times without progress. Do NOT "
            "repeat it again. Re-read the last error message carefully, "
            "then try a genuinely different approach (different "
            "arguments, a different tool, or read the file first). If "
            "you are blocked, say so briefly instead of retrying."
            "</system-reminder>"
        ),
    }


def loop_break_message() -> dict[str, Any]:
    """Build the final-turn message injected on a loop *break* verdict."""
    return {
        "role": "user",
        "content": (
            "<system-reminder>Tool loop detected: the same calls kept "
            "repeating without progress, so tool access has been paused. "
            "Summarize what you accomplished, what failed and why, and "
            "what you would try next. Do not attempt further tool calls."
            "</system-reminder>"
        ),
    }


def last_tool_result_errored(messages: list[dict[str, Any]]) -> bool:
    """Whether the most recent tool result in this turn demands follow-up.

    Walks backwards from just before the final assistant message,
    skipping harness-injected ``<system-reminder>`` user messages, and
    reports whether the first tool message found carries either an
    ``Error:``-prefixed result or a post-write verification ``WARNING:``
    (only for write/edit results, so a stray "WARNING" in build output
    never triggers it).  A real user message ends the walk — failures
    from a previous turn are not this turn's business.

    Args:
        messages: The conversation history, whose last entry is the
            assistant message that carried no tool calls.

    Returns:
        ``True`` when the model is about to finish on a failed tool
        call or an unaddressed syntax warning.
    """
    for msg in reversed(messages[:-1]):
        role = msg.get("role")
        if role == "tool":
            content = msg.get("content") or ""
            if content.startswith("Error:"):
                return True
            # The verification gate appends "WARNING: ..." to write/edit
            # results when the file it produced fails a syntax check.
            if (
                msg.get("tool_name") in ("write", "edit")
                and "WARNING:" in content
            ):
                return True
            return False
        if role == "user":
            content = msg.get("content") or ""
            if content.startswith("<system-reminder>"):
                continue
            return False
    return False


def error_stop_message() -> dict[str, Any]:
    """Build the push-back injected when finishing on a failed tool call.

    Small models routinely ignore a tool error — or a syntax WARNING on
    the file they just wrote — and declare the task done (both observed
    live on qwen3:0.6b).  One deterministic push-back makes the model
    either actually recover or explain the blocker.
    """
    return {
        "role": "user",
        "content": (
            "<system-reminder>Your last tool call FAILED (or left a "
            "syntax WARNING) and you have not fixed it. Do not stop "
            "here. Re-read the error message, read the file to see its "
            "current exact content, fix it, and complete the task. Only "
            "if the task is truly impossible, explain exactly why."
            "</system-reminder>"
        ),
    }


# Discovery tools whose results should be seen before a same-turn write.
DISCOVERY_TOOLS = frozenset({"grep", "glob"})

# Mutation tools deferred when issued alongside discovery calls.
MUTATION_TOOLS = frozenset({"write", "edit"})


def deferred_write_result(tool_name: str) -> str:
    """Result string for a mutation deferred by the ordering guard.

    Observed live (qwen3:0.6b): one turn issued ``grep("calc_total")``
    plus a ``write`` of a brand-new file — the "fix" was decided before
    the search results existed, while grep correctly located the real
    function.  Deferring the mutation makes the model re-issue it with
    the search results in hand.
    """
    return (
        f"Error: {tool_name} deferred: it was issued in the same turn "
        "as a search (grep/glob), so it could not have used the search "
        "results. Read the search results above, then re-issue the "
        f"{tool_name} against the correct existing file."
    )


def text_tool_nudge_message() -> dict[str, Any]:
    """Build the "use the tools" nudge for text-driven tool mode.

    The standard nudge says "call the write or edit tool", which is
    meaningless to a model running on the no-tool-support fallback — it
    needs the fenced-JSON call shape restated instead.
    """
    return {
        "role": "user",
        "content": (
            "You printed code but did not actually create or modify any "
            "file. To apply it, reply with ONLY one fenced JSON tool "
            "call, exactly like:\n"
            "```json\n"
            '{"name": "write", "arguments": {"file_path": "a.py", '
            '"content": "print(1)"}}\n'
            "```\n"
            "If no file change is needed, say so briefly."
        ),
    }


def is_tools_unsupported_error(exc: Exception) -> bool:
    """Whether *exc* means the model/endpoint rejects structured tool calls.

    Ollama returns HTTP 400 with a message like ``"<model>" does not
    support tools`` for models whose chat template has no tool-calling
    support (many small or Japanese-specialized models).  Detecting this
    lets the loop fall back to text-driven tool calls instead of dying —
    combined with :func:`extract_text_tool_calls`, such models can still
    act as agents.

    Args:
        exc: The provider request error.

    Returns:
        ``True`` when the error indicates missing tool support.
    """
    msg = str(exc).lower()
    return (
        "does not support tools" in msg
        or "tool use is not supported" in msg
        or "tools are not supported" in msg
    )


def text_tools_fallback_message(tools: list[Any]) -> dict[str, Any]:
    """Build the instruction injected when falling back to text tools.

    Teaches the model the exact fenced-JSON shape that
    :func:`extract_text_tool_calls` rescues, and lists each tool with
    its argument names so calls can be produced without a schema.

    Args:
        tools: The active tool instances (name + parameters are read).

    Returns:
        A user-role message dict to append to the conversation.
    """
    lines: list[str] = []
    for tool in tools:
        name = getattr(tool, "name", "?")
        try:
            props = tool.parameters.get("properties", {})
            args = ", ".join(f'"{key}": ...' for key in props)
            lines.append(f"- {name}: {{{args}}}")
        except Exception:
            lines.append(f"- {name}")
    return {
        "role": "user",
        "content": (
            "<system-reminder>This model endpoint rejected structured "
            "tool calls, so tools are now driven by TEXT. To call a "
            "tool, reply with ONLY one JSON object in a fenced block, "
            "exactly like:\n"
            "```json\n"
            '{"name": "write", "arguments": {"file_path": "a.py", '
            '"content": "print(1)"}}\n'
            "```\n"
            "Available tools and their arguments:\n"
            + "\n".join(lines)
            + "\nMake one tool call per reply. The result will come "
            "back as the next message; continue until the task is done, "
            "then answer normally with no JSON.</system-reminder>"
        ),
    }


def step_limit_message(limit: int) -> dict[str, Any]:
    """Build the final-turn message injected when the step cap is reached."""
    return {
        "role": "user",
        "content": (
            f"<system-reminder>You have reached the {limit}-step limit "
            "for this turn, so tool access has been paused. Summarize "
            "what you accomplished and what remains to be done, so the "
            "user can decide how to continue. Do not attempt further "
            "tool calls.</system-reminder>"
        ),
    }


# ---------------------------------------------------------------------------
# Post-write verification gate
# ---------------------------------------------------------------------------

# Do not verify files larger than this (reading them back costs memory).
_VERIFY_MAX_BYTES = 1_000_000

# Tools whose successful results describe a file mutation worth verifying.
_VERIFIABLE_TOOLS = frozenset({"write", "edit"})


def verify_file_write(
    tool_name: str,
    arguments: dict[str, Any],
    result: str,
) -> str | None:
    """Syntax-check a file right after a successful write/edit.

    Small models routinely produce a file with a syntax error, declare
    the task done, and never re-read the file.  Checking immediately and
    appending a warning to the tool result puts the error in front of
    the model while it still has the file in context.

    Only ``.py`` (via ``ast.parse``) and ``.json`` (via ``json.loads``)
    are checked — both come with the stdlib and are cheap.

    Args:
        tool_name: The resolved tool name that just ran.
        arguments: The tool's (normalized) arguments.
        result: The tool's result string.

    Returns:
        A warning string to append to the tool result, or ``None`` when
        the write is fine / not verifiable.
    """
    if tool_name not in _VERIFIABLE_TOOLS:
        return None
    if result.startswith("Error:"):
        return None
    file_path = arguments.get("file_path")
    if not isinstance(file_path, str) or not file_path.strip():
        return None

    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in (".py", ".json"):
        return None

    try:
        if not path.is_file() or path.stat().st_size > _VERIFY_MAX_BYTES:
            return None
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    if suffix == ".py":
        try:
            ast.parse(text)
        except SyntaxError as exc:
            return (
                f"WARNING: {file_path} now contains a Python syntax error "
                f"at line {exc.lineno}: {exc.msg}. Read the surrounding "
                "lines and fix it before finishing."
            )
    else:  # .json
        try:
            json.loads(text)
        except ValueError as exc:
            return (
                f"WARNING: {file_path} is not valid JSON after this "
                f"change: {exc}. Fix it before finishing."
            )
    return None


# ---------------------------------------------------------------------------
# Todo staleness reminders
# ---------------------------------------------------------------------------


class TodoReminder:
    """Re-inject the todo state when the list has gone stale.

    Small models create a todo list, finish two items, and then wander
    off or stop.  This tracker watches how many loop iterations have
    passed since the last ``todo_write`` call; when unfinished items
    exist and the list is stale, it produces a system-reminder string
    for the caller to inject as a user message (mirroring Claude Code's
    reminder mechanism).

    Reminders are capped per session so a genuinely stuck model is not
    nagged forever.
    """

    def __init__(self, stale_after: int = 4, max_reminders: int = 2) -> None:
        self._stale_after = stale_after
        self._max_reminders = max_reminders
        self._last_update = 0
        self._sent = 0

    def note_tool_use(self, iteration: int, tool_name: str) -> None:
        """Record that *tool_name* ran during *iteration*."""
        if tool_name == "todo_write":
            self._last_update = iteration

    def check(self, iteration: int, tools: list[Any]) -> str | None:
        """Return a reminder string when the todo list is stale.

        Args:
            iteration: The current loop iteration (1-based).
            tools: The active tool instances; the todo tool is located
                by its ``name`` and read via its ``current_todos``
                property.

        Returns:
            The reminder text, or ``None`` when no reminder is due.
        """
        if self._sent >= self._max_reminders:
            return None

        todos: list[dict[str, str]] | None = None
        for tool in tools:
            if getattr(tool, "name", "") == "todo_write":
                todos = getattr(tool, "current_todos", None)
                break
        if not todos:
            return None

        pending = [t for t in todos if t.get("status") == "pending"]
        in_progress = [t for t in todos if t.get("status") == "in_progress"]
        if not pending and not in_progress:
            return None
        if iteration - self._last_update < self._stale_after:
            return None

        self._sent += 1
        self._last_update = iteration  # restart the grace period
        current = (in_progress or pending)[0].get("content", "")
        return (
            f"<system-reminder>Your todo list still has {len(pending)} "
            f"pending and {len(in_progress)} in-progress item(s), but it "
            "has not been updated for several steps. Current task: "
            f"'{current}'. If it is done, mark it completed with "
            "todo_write and start the next pending item. Do not stop "
            "until every todo is completed.</system-reminder>"
        )


# ---------------------------------------------------------------------------
# LLM-summarized compaction helpers
# ---------------------------------------------------------------------------

# Cap on the transcript text handed to the summarizer.
_SUMMARY_MAX_SOURCE_CHARS = 24_000

# Per-message snippet length in the summarizer transcript.
_SUMMARY_SNIPPET_CHARS = 600


def compaction_bounds(
    messages: list[dict[str, Any]],
    keep_recent: int = 10,
) -> tuple[int, int] | None:
    """Locate the ``[system_end, recent_start)`` span eligible for summary.

    Leading system messages are always preserved, the *keep_recent* most
    recent messages are preserved verbatim, and the boundary is walked
    backwards so an assistant ``tool_calls`` message is never separated
    from its tool results (providers reject orphaned tool results).

    Args:
        messages: The conversation history.
        keep_recent: Number of trailing messages to preserve.

    Returns:
        ``(system_end, recent_start)`` when at least two messages are
        eligible for summarization, else ``None``.
    """
    total = len(messages)
    system_end = 0
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            system_end = i + 1
        else:
            break

    recent_start = max(system_end, total - keep_recent)
    while recent_start > system_end and messages[recent_start].get("role") == "tool":
        recent_start -= 1

    if recent_start <= system_end + 1:
        return None
    return system_end, recent_start


def build_summary_request(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the one-shot message list asking the model to summarize.

    Each source message is flattened to ``role: snippet`` (tool-call
    names are noted inline) and the whole transcript is capped so the
    summarizer request itself cannot blow the context window.

    Args:
        messages: The message span to summarize.

    Returns:
        A two-message list (system + user) for a tool-free chat call.
    """
    parts: list[str] = []
    total = 0
    for msg in messages:
        content = (msg.get("content") or "")[:_SUMMARY_SNIPPET_CHARS]
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            names = ", ".join(
                tc.get("function", {}).get("name", "?") for tc in tool_calls
            )
            content = f"{content}\n[called tools: {names}]".strip()
        entry = f"{msg.get('role', '?')}: {content}"
        parts.append(entry)
        total += len(entry)
        if total > _SUMMARY_MAX_SOURCE_CHARS:
            break

    transcript = "\n\n".join(parts)
    return [
        {
            "role": "system",
            "content": (
                "You compress the transcript of an AI coding-agent "
                "session so the agent can continue with less context."
            ),
        },
        {
            "role": "user",
            "content": (
                "Summarize the session transcript below in at most 300 "
                "words. Preserve exactly: the user's task and "
                "constraints, file paths touched, commands run, key "
                "decisions, errors encountered, current progress, and "
                "the immediate next step. Output only the summary.\n\n"
                "---\n"
                f"{transcript}"
            ),
        },
    ]


def apply_summary(
    messages: list[dict[str, Any]],
    summary: str,
    system_end: int,
    recent_start: int,
) -> None:
    """Replace ``messages[system_end:recent_start]`` with one summary message.

    Args:
        messages: The conversation history (mutated in place).
        summary: The model-produced summary text.
        system_end: Start of the summarized span (from
            :func:`compaction_bounds`).
        recent_start: End of the summarized span (exclusive).
    """
    summary_msg = {
        "role": "user",
        "content": (
            "[Earlier conversation was summarized to save context]\n"
            + summary.strip()
        ),
    }
    messages[system_end:recent_start] = [summary_msg]
