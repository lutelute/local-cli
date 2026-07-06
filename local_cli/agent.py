"""Agent loop for local-cli.

Implements the core agent loop: send messages to the LLM, collect the
streaming response, execute any tool calls, and loop until the LLM
responds without requesting tools.  All output is streamed to stdout as
tokens arrive.
"""

import json
import re
import sys
import time
from typing import Any, Callable, Generator

from local_cli.harness import (
    DISCOVERY_TOOLS,
    MUTATION_TOOLS,
    AgentEvent,
    EmitFn,
    HarnessConfig,
    LoopDetector,
    TodoReminder,
    apply_summary,
    build_summary_request,
    compaction_bounds,
    deferred_write_result,
    deliverable_missing_message,
    empty_response_message,
    error_stop_message,
    extract_text_tool_calls,
    is_tools_unsupported_error,
    last_tool_result_errored,
    loop_break_message,
    loop_warning_message,
    mentions_file_deliverable,
    null_emit,
    step_limit_message,
    text_tool_nudge_message,
    text_tools_fallback_message,
    verify_file_write,
)
from local_cli.ollama_client import OllamaClient, OllamaStreamError
from local_cli.providers.base import (
    LLMProvider,
    ProviderConnectionError,
    ProviderRequestError,
    ProviderStreamError,
)
from local_cli.spinner import Spinner
from local_cli.token_tracker import TokenTracker
from local_cli.tool_cache import ToolCache
from local_cli.tools.base import Tool

# ---------------------------------------------------------------------------
# Result truncation for tool output displayed to the user
# ---------------------------------------------------------------------------

# Maximum characters of a tool result to print to the console.
_MAX_DISPLAY_RESULT = 200

# ---------------------------------------------------------------------------
# Context compaction thresholds
# ---------------------------------------------------------------------------

# Compact when message count exceeds this threshold.
_COMPACT_MESSAGE_THRESHOLD = 50

# Approximate characters-per-token estimate (conservative).
_CHARS_PER_TOKEN = 4

# Compact when estimated token count exceeds this threshold.
_COMPACT_TOKEN_THRESHOLD = 24_000

# Maximum characters to keep from a compacted tool result.
_COMPACT_TOOL_RESULT_MAX = 200

# Maximum characters to keep from a compacted assistant message.
_COMPACT_ASSISTANT_MAX = 500

# Number of recent messages to preserve uncompacted.
_COMPACT_KEEP_RECENT = 10

# ---------------------------------------------------------------------------
# Fast-mode heuristic constants
# ---------------------------------------------------------------------------

# Default word-count threshold for complexity detection.
_COMPLEX_WORD_THRESHOLD = 50

# Keywords that suggest a request is complex / plan-worthy.
_COMPLEX_KEYWORDS = frozenset({
    "refactor",
    "migrate",
    "redesign",
    "architecture",
    "integrate",
    "implement",
    "restructure",
    "multi-step",
    "multiple files",
    "across files",
    "step by step",
    "plan",
    "break down",
    "phases",
})


# ---------------------------------------------------------------------------
# Streaming response collector
# ---------------------------------------------------------------------------


def collect_streaming_response(
    stream: Generator[dict[str, Any], None, None],
    spinner: Spinner | None = None,
    tracker: TokenTracker | None = None,
) -> dict[str, Any]:
    """Accumulate a streaming chat response and print tokens as they arrive.

    Iterates over NDJSON chunks from :meth:`OllamaClient.chat_stream`,
    concatenating content deltas and accumulating ``tool_calls`` across
    chunks.  Content tokens are printed to stdout immediately for a
    responsive user experience.

    When a *tracker* is provided, token usage is extracted from the
    final response and recorded.  For Ollama, ``prompt_eval_count``
    and ``eval_count`` are read from the final chunk (``done: true``).
    For Claude, the ``usage`` metadata dict is used instead.

    Thinking content (``message.thinking``) from models that support
    thinking mode (e.g. Qwen3) is accumulated separately and returned
    as a ``"thinking"`` key in the result dict.  It is **not** included
    in the assembled ``message.content`` field, keeping the conversation
    history free of internal reasoning tokens.

    Args:
        stream: A generator yielding parsed NDJSON chunks from the Ollama
            streaming chat API.
        spinner: Optional spinner to stop once the first content arrives.
        tracker: Optional :class:`TokenTracker` for recording token
            usage from this response.

    Returns:
        A dictionary representing the full response, structured as::

            {
                "message": {
                    "role": "assistant",
                    "content": "<accumulated text>",
                    "tool_calls": [...]  # only if present
                },
                "thinking": "<accumulated thinking>",  # only if present
                "done": True,
                ...  # other fields from the final chunk
            }

    Raises:
        ProviderStreamError: If the stream yields an error chunk (already
            handled by the provider, but re-raised here for clarity).
            This catches provider-specific subclasses (e.g.
            :class:`OllamaStreamError`) via inheritance.
        KeyboardInterrupt: If the user presses Ctrl+C during streaming.
            Partial content accumulated so far is returned.
    """
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    last_chunk: dict[str, Any] = {}
    spinner_stopped = False

    try:
        for chunk in stream:
            last_chunk = chunk

            message = chunk.get("message", {})

            # Handle thinking field (chain-of-thought reasoning).
            # When `think: true` is enabled, models may include a
            # "thinking" field with internal reasoning.  Display it in
            # a dimmed style (prefixed with "> ") before main content.
            thinking_delta = message.get("thinking", "")
            if thinking_delta:
                if spinner and not spinner_stopped:
                    spinner.stop()
                    spinner_stopped = True
                thinking_parts.append(thinking_delta)
                # Display thinking text line-by-line with "> " prefix.
                for line in thinking_delta.splitlines(keepends=True):
                    sys.stdout.write(f"> {line}")
                sys.stdout.flush()

            # Accumulate content deltas and print to stdout.
            delta = message.get("content", "")
            if delta:
                # Stop the spinner on first content token.
                if spinner and not spinner_stopped:
                    spinner.stop()
                    spinner_stopped = True
                # If transitioning from thinking to content, add a
                # separator newline for readability.
                if thinking_parts and not content_parts:
                    sys.stdout.write("\n")
                content_parts.append(delta)
                sys.stdout.write(delta)
                sys.stdout.flush()

            # Accumulate tool calls (typically in the final chunk, but we
            # handle them appearing in any chunk for robustness).
            chunk_tool_calls = message.get("tool_calls")
            if chunk_tool_calls:
                # Stop the spinner when tool calls arrive (no content yet).
                if spinner and not spinner_stopped:
                    spinner.stop()
                    spinner_stopped = True
                tool_calls.extend(chunk_tool_calls)

    except KeyboardInterrupt:
        # User interrupted streaming.  Return what we have so far.
        if spinner and not spinner_stopped:
            spinner.stop()
        sys.stdout.write("\n")
        sys.stdout.flush()

    except ProviderStreamError:
        # Mid-stream error from the provider.  Print a newline to cleanly
        # separate any partial output, then re-raise so the caller can
        # decide how to handle it.  Catches both OllamaStreamError and
        # other provider-specific stream errors via inheritance.
        if spinner and not spinner_stopped:
            spinner.stop()
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise

    # Ensure spinner is stopped after stream completes.
    if spinner and not spinner_stopped:
        spinner.stop()

    # Print a trailing newline after streamed content (if any was printed).
    if content_parts:
        sys.stdout.write("\n")
        sys.stdout.flush()

    # Build the assembled response.
    assembled_message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    if tool_calls:
        assembled_message["tool_calls"] = tool_calls

    # Merge the final chunk's top-level fields (model, done, metrics, etc.)
    # with our assembled message.
    result: dict[str, Any] = dict(last_chunk)
    result["message"] = assembled_message

    # Include accumulated thinking content as a separate top-level key
    # for debug display.  This is intentionally NOT part of the message
    # dict so it does not pollute the conversation history.
    if thinking_parts:
        result["thinking"] = "".join(thinking_parts)

    # Record token usage if a tracker is provided.  For Ollama the
    # final chunk carries ``prompt_eval_count`` / ``eval_count`` at
    # the top level; for Claude the ``usage`` dict contains the counts.
    if tracker is not None:
        if "usage" in result:
            tracker.record_from_claude(result)
        else:
            tracker.record_from_ollama(result)

    return result


# ---------------------------------------------------------------------------
# Tool execution helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int = _MAX_DISPLAY_RESULT) -> str:
    """Truncate text to *max_len* characters, appending '...' if needed."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _execute_tool(
    tool: Tool,
    arguments: dict[str, Any],
    debug: bool = False,
) -> str:
    """Execute a single tool and return its result string.

    Wraps the tool's ``execute`` method in a try/except so that exceptions
    are converted to error strings rather than crashing the agent loop.

    Args:
        tool: The tool instance to execute.
        arguments: Keyword arguments to pass to the tool.
        debug: If True, print the full arguments before execution.

    Returns:
        The string result from the tool, or an error message.
    """
    if debug:
        sys.stderr.write(f"  [debug] {tool.name} args: {arguments}\n")

    try:
        return tool.execute(**arguments)
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


def parse_tool_call(
    tc: dict[str, Any],
) -> tuple[str, dict[str, Any], str | None]:
    """Extract the name, arguments, and id from a raw tool call.

    Normalizes the shapes local models emit: ``arguments`` may already be a
    dict, or a JSON-encoded string (Ollama frequently returns the latter).
    A string that fails to parse — or any non-dict value — degrades to an
    empty dict rather than raising, so a malformed tool call becomes a
    no-arg call the tool itself can reject, instead of crashing the loop.

    This is the single tool-call parser shared by the CLI agent loop, the
    sub-agent loop, the JSON-line server, and the web monitor, so all four
    repair malformed calls identically.

    Args:
        tc: A single tool-call dict, shaped like
            ``{"function": {"name": ..., "arguments": ...}, "id": ...}``.

    Returns:
        A ``(tool_name, arguments, tool_call_id)`` tuple.  ``tool_name`` is
        ``""`` and ``arguments`` is ``{}`` when absent; ``tool_call_id`` is
        ``None`` when the call carries no id.
    """
    func = tc.get("function", {})
    tool_name = func.get("name", "")
    arguments = func.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return tool_name, arguments, tc.get("id")


# Aliases local models commonly emit for the built-in tools.  Small models
# frequently call a tool by a near-miss name (write_file vs write, run vs
# bash); resolving these instead of erroring keeps the agent moving.  Keys
# are normalized (lowercased, separators stripped) — see resolve_tool_name.
_TOOL_ALIASES: dict[str, str] = {
    "writefile": "write", "createfile": "write", "newfile": "write",
    "savefile": "write", "putfile": "write",
    "readfile": "read", "openfile": "read", "viewfile": "read",
    "cat": "read", "view": "read", "open": "read",
    "editfile": "edit", "replace": "edit", "strreplace": "edit",
    "modify": "edit", "update": "edit", "replaceinfile": "edit",
    "run": "bash", "shell": "bash", "exec": "bash", "execute": "bash",
    "runcommand": "bash", "bashcommand": "bash", "command": "bash",
    "sh": "bash", "terminal": "bash", "runshell": "bash",
    "search": "grep", "searchfiles": "grep", "findinfiles": "grep",
    "ripgrep": "grep", "rg": "grep", "searchcode": "grep",
    "find": "glob", "findfiles": "glob", "listfiles": "glob",
    "ls": "glob", "globfiles": "glob", "listdir": "glob",
    "fetch": "web_fetch", "fetchurl": "web_fetch", "curl": "web_fetch",
    "wget": "web_fetch", "httpget": "web_fetch", "geturl": "web_fetch",
    "todo": "todo_write", "todos": "todo_write", "tasklist": "todo_write",
    "writetodo": "todo_write", "updatetodo": "todo_write",
    "ask": "ask_user", "question": "ask_user", "prompt": "ask_user",
    "askquestion": "ask_user",
    "spawn": "agent", "subagent": "agent", "spawnagent": "agent",
    "delegate": "agent",
}


def _normalize_tool_name(name: str) -> str:
    """Lowercase *name* and strip non-alphanumerics (``write_file`` -> ``writefile``)."""
    return "".join(c for c in name.lower() if c.isalnum())


def resolve_tool_name(
    tool_name: str, tool_map: dict[str, Tool],
) -> str | None:
    """Resolve a possibly-misnamed tool to a real key in *tool_map*.

    Local models routinely emit near-miss tool names.  This tries, in order:
    an exact match; a known alias; a match after normalizing case and
    separators (so ``write_file`` / ``WriteFile`` map to ``write``); and an
    alias of the normalized form.  Returns ``None`` when nothing matches, so
    a genuinely unknown tool still surfaces as an error.

    Args:
        tool_name: The (possibly imperfect) name the model emitted.
        tool_map: Mapping of real tool name to :class:`Tool` instance.

    Returns:
        A key present in *tool_map*, or ``None``.
    """
    if tool_name in tool_map:
        return tool_name

    norm = _normalize_tool_name(tool_name)

    # A direct alias on the raw-lowered name or the normalized form.
    for candidate in (_TOOL_ALIASES.get(tool_name.lower()), _TOOL_ALIASES.get(norm)):
        if candidate and candidate in tool_map:
            return candidate

    # Match by normalized form against the real tool names.
    for real in tool_map:
        if _normalize_tool_name(real) == norm:
            return real

    return None


# Per-tool argument-key aliases.  Small models often pass a correct value
# under a near-miss key (``path`` for ``file_path``, ``text`` for
# ``content``).  Resolving the tool name (above) is only half the fix — the
# call still fails if the keys don't match the tool's schema.  Keys here are
# alias -> canonical, applied only when the canonical key is absent.
_ARG_ALIASES: dict[str, dict[str, str]] = {
    "write": {
        "path": "file_path", "filepath": "file_path", "file": "file_path",
        "filename": "file_path", "text": "content", "data": "content",
        "body": "content", "contents": "content",
    },
    "read": {
        "path": "file_path", "filepath": "file_path", "file": "file_path",
        "filename": "file_path",
    },
    "edit": {
        "path": "file_path", "filepath": "file_path", "file": "file_path",
        "old": "old_text", "old_string": "old_text", "oldstr": "old_text",
        "search": "old_text", "find": "old_text", "target": "old_text",
        "new": "new_text", "new_string": "new_text", "newstr": "new_text",
        "replace": "new_text", "replacement": "new_text",
    },
    "bash": {
        "cmd": "command", "script": "command", "shell_command": "command",
        "shell": "command", "cmdline": "command",
    },
    "grep": {
        "query": "pattern", "regex": "pattern", "search": "pattern",
        "directory": "path", "dir": "path",
    },
    "glob": {"glob": "pattern", "query": "pattern"},
    "web_fetch": {"uri": "url", "link": "url", "address": "url"},
    "todo_write": {"tasks": "todos", "items": "todos", "list": "todos"},
}


def normalize_arguments(
    tool_name: str, arguments: dict[str, Any],
) -> dict[str, Any]:
    """Rename near-miss argument keys to a tool's canonical schema keys.

    Renames an alias key to its canonical name only when the canonical key
    is not already present, so a correct call is never disturbed and a
    correct key always wins over an alias.

    Args:
        tool_name: The (already resolved) real tool name.
        arguments: The tool-call arguments.

    Returns:
        A new dict with alias keys renamed, or *arguments* unchanged when
        the tool has no aliases or *arguments* is not a dict.
    """
    aliases = _ARG_ALIASES.get(tool_name)
    if not aliases or not isinstance(arguments, dict):
        return arguments
    result = dict(arguments)
    for alias, canonical in aliases.items():
        if alias in result and canonical not in result:
            result[canonical] = result.pop(alias)
    return result


def run_tool(
    tool_name: str,
    arguments: dict[str, Any],
    tool_map: dict[str, Tool],
    debug: bool = False,
) -> str:
    """Resolve *tool_name* (alias/fuzzy), normalize its args, and execute it.

    Returns an ``Error: unknown tool`` string only for a name that can't be
    resolved even after alias/normalization fallback, and routes execution
    through :func:`_execute_tool`, so exceptions are converted to error
    strings uniformly across every front-end (CLI agent loop, sub-agent
    loop, JSON-line server, web monitor).

    Args:
        tool_name: The name of the tool to run (may be a near-miss).
        arguments: Keyword arguments for the tool (keys may be near-misses).
        tool_map: Mapping of tool name to :class:`Tool` instance.
        debug: Forwarded to :func:`_execute_tool`.

    Returns:
        The tool result string, or an ``Error: ...`` message.
    """
    resolved = resolve_tool_name(tool_name, tool_map)
    if resolved is None:
        return f"Error: unknown tool '{tool_name}'"
    arguments = normalize_arguments(resolved, arguments)
    return _execute_tool(tool_map[resolved], arguments, debug=debug)


# Max characters of a tool result forwarded to a GUI front-end (the
# JSON-line server and the web monitor) for display.  The full result is
# always kept in the message history; only the on-screen copy is capped.
_GUI_TOOL_RESULT_MAX = 10_000


def truncate_tool_output(
    result: str, max_len: int = _GUI_TOOL_RESULT_MAX,
) -> str:
    """Cap a tool result for GUI display, marking when it was truncated.

    Used by the server and web monitor so both surface the same amount of
    a long tool result (they previously truncated at 10,000 and 5,000
    characters respectively).

    Args:
        result: The full tool result string.
        max_len: Maximum characters to keep.

    Returns:
        The result unchanged if within *max_len*, otherwise the first
        *max_len* characters with a truncation marker appended.
    """
    if len(result) <= max_len:
        return result
    return result[:max_len] + "\n...(truncated)"


def _extract_file_path(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Extract the primary file path from tool arguments for cache tracking.

    For tools that operate on a single file (e.g. ``read``), returns the
    file path argument so that the cache can track the file's ``mtime``
    for invalidation.  For tools that operate on directories or multiple
    files (e.g. ``glob``, ``grep``), returns ``None``.

    Args:
        tool_name: Name of the tool.
        arguments: The tool's argument dictionary.

    Returns:
        The file path string, or ``None`` if no single file path applies.
    """
    if tool_name == "read":
        fp = arguments.get("file_path")
        return fp if isinstance(fp, str) else None
    return None


# ---------------------------------------------------------------------------
# Context compaction
# ---------------------------------------------------------------------------


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate the number of tokens in a message list.

    Uses a rough characters-per-token heuristic.  This is intentionally
    conservative (under-estimates tokens) so that compaction triggers
    before actually hitting the model's context window limit.

    Args:
        messages: The conversation message list.

    Returns:
        Estimated token count.
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if content:
            total_chars += len(content)
        # Account for tool call arguments (they consume tokens too).
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            args = func.get("arguments", {})
            total_chars += len(str(args))
    return total_chars // _CHARS_PER_TOKEN


def _compact_message(message: dict[str, Any]) -> dict[str, Any]:
    """Compact a single message by truncating its content.

    Preserves the message role and structure but reduces content length.
    System messages are never compacted.  Tool results are truncated
    aggressively.  Assistant messages are truncated moderately.

    Args:
        message: A conversation message dict.

    Returns:
        A new dict with truncated content (or the original if no
        compaction was needed).
    """
    role = message.get("role", "")

    # Never compact system messages -- they contain the system prompt.
    if role == "system":
        return message

    content = message.get("content", "")

    if role == "tool":
        max_len = _COMPACT_TOOL_RESULT_MAX
    elif role == "assistant":
        max_len = _COMPACT_ASSISTANT_MAX
    else:
        # User messages: keep them intact (they're typically short).
        return message

    if len(content) <= max_len:
        return message

    # Build a compacted copy.
    compacted = dict(message)
    compacted["content"] = content[:max_len] + "\n... [truncated for context]"

    # Strip tool_calls from old assistant messages to save space.
    # The tool results are already recorded in subsequent tool messages.
    if role == "assistant" and "tool_calls" in compacted:
        tc_count = len(compacted["tool_calls"])
        compacted.pop("tool_calls")
        compacted["content"] += f"\n[{tc_count} tool call(s) omitted]"

    return compacted


def _needs_compaction(
    messages: list[dict[str, Any]],
    token_threshold: int | None = None,
) -> bool:
    """Check whether the message list should be compacted.

    Returns ``True`` if either the message count or estimated token
    count exceeds their respective thresholds.

    Args:
        messages: The conversation message list.
        token_threshold: Optional override for the token threshold.
            When provided (e.g. derived from ``num_ctx``), this value
            is used instead of the module-level default.

    Returns:
        True if compaction is warranted.
    """
    if len(messages) > _COMPACT_MESSAGE_THRESHOLD:
        return True
    threshold = token_threshold if token_threshold is not None else _COMPACT_TOKEN_THRESHOLD
    if _estimate_tokens(messages) > threshold:
        return True
    return False


def compact_messages(
    messages: list[dict[str, Any]],
    debug: bool = False,
) -> None:
    """Compact the conversation history in place to preserve context space.

    Keeps the system message(s) at the start and the most recent
    ``_COMPACT_KEEP_RECENT`` messages intact.  Older messages in between
    are truncated to reduce token usage.

    This is called automatically by :func:`agent_loop` when thresholds
    are exceeded.

    Args:
        messages: The conversation history (mutated in place).
        debug: If True, print compaction details to stderr.
    """
    total = len(messages)
    if total <= _COMPACT_KEEP_RECENT:
        return

    # Identify the boundary between "old" and "recent" messages.
    # System messages at the start are always preserved fully.
    system_end = 0
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            system_end = i + 1
        else:
            break

    # Recent messages to keep intact.
    recent_start = max(system_end, total - _COMPACT_KEEP_RECENT)

    if recent_start <= system_end:
        # Not enough old messages to compact.
        return

    old_tokens = _estimate_tokens(messages[system_end:recent_start])

    compacted_count = 0
    for i in range(system_end, recent_start):
        original = messages[i]
        compacted = _compact_message(original)
        if compacted is not original:
            messages[i] = compacted
            compacted_count += 1

    if debug and compacted_count > 0:
        new_tokens = _estimate_tokens(messages[system_end:recent_start])
        sys.stderr.write(
            f"[debug] Compacted {compacted_count} messages "
            f"(~{old_tokens} -> ~{new_tokens} tokens)\n"
        )


# ---------------------------------------------------------------------------
# "Use the tools" nudge
# ---------------------------------------------------------------------------

# English build/edit verbs, matched as whole words so "add" does not fire on
# "address" nor "fix" on "prefix".
_BUILD_KEYWORDS_EN: frozenset[str] = frozenset({
    "create", "write", "make", "build", "implement", "generate",
    "add", "fix", "refactor", "edit", "modify", "save", "rename",
})

# Japanese stems, matched as substrings (Japanese is not whitespace-delimited).
_BUILD_KEYWORDS_JA: tuple[str, ...] = (
    "作", "書", "実装", "追加", "修正", "生成", "直",
)


def _mentions_build_intent(user_text: str) -> bool:
    """Whether the user's request implies a file should be written or changed.

    English verbs match on whole words to avoid false positives like
    "address" (contains "add") or "prefix" (contains "fix"); Japanese stems
    match as substrings since Japanese has no word boundaries.

    Args:
        user_text: The most recent user message text.

    Returns:
        True if the request mentions a build/edit intent.
    """
    if any(stem in user_text for stem in _BUILD_KEYWORDS_JA):
        return True
    words = set(re.findall(r"[a-z]+", user_text.lower()))
    return bool(words & _BUILD_KEYWORDS_EN)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the content of the most recent user message (or '')."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "") or ""
    return ""


def _wrote_file_this_turn(messages: list[dict[str, Any]]) -> bool:
    """Whether a write/edit tool ran since the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return False
        if msg.get("role") == "tool" and msg.get("tool_name") in ("write", "edit"):
            return True
    return False


def _should_nudge_to_use_tools(
    messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    already_nudged: bool,
) -> bool:
    """Decide whether to nudge the model to use tools instead of printing code.

    Conservative on purpose, to avoid nagging on legitimate "explain this
    code" answers.  Fires only when *all* hold: we have not nudged yet this
    turn; the assistant's answer contains a code fence; the user's request
    contained a build/edit keyword; and no write/edit tool ran this turn.

    Args:
        messages: The conversation so far.
        assistant_message: The assistant message that carried no tool calls.
        already_nudged: Whether a nudge was already issued this turn.

    Returns:
        True if a single nudge is warranted.
    """
    if already_nudged:
        return False
    content = assistant_message.get("content", "") or ""
    if "```" not in content:
        return False
    if not _mentions_build_intent(_last_user_text(messages)):
        return False
    if _wrote_file_this_turn(messages):
        return False
    return True


# Injected (once per turn) when the model answered with code but called no
# tool on a request that asked for a file change.
_TOOL_NUDGE_MESSAGE: dict[str, Any] = {
    "role": "user",
    "content": (
        "You printed code but did not create or modify any file. If the "
        "task needs a file written or changed, call the write or edit tool "
        "now to actually apply it. If no file change is needed, say so "
        "briefly."
    ),
}


# ---------------------------------------------------------------------------
# Unified agent loop (the harness core)
# ---------------------------------------------------------------------------

# Maximum number of retries on provider overload (HTTP 503).
_RETRY_MAX_ATTEMPTS = 3

# Base delay in seconds for exponential backoff (1s, 2s, 4s).
_RETRY_BASE_DELAY = 1.0


def _is_overloaded_error(exc: Exception) -> bool:
    """Check if an exception indicates a provider overload (HTTP 503).

    Inspects the exception message for common overload indicators
    such as HTTP 503 status codes and "Service Unavailable" messages.
    Used to decide whether to retry a failed provider request in
    :func:`run_agent`.

    Args:
        exc: The exception to check.

    Returns:
        ``True`` if the error indicates a 503 or overload condition.
    """
    msg = str(exc).lower()
    return "503" in msg or "service unavailable" in msg or "overloaded" in msg


def _collect_stream_emitting(
    stream: Generator[dict[str, Any], None, None],
    emit: EmitFn,
    tracker: TokenTracker | None = None,
    should_stop: Callable[[], bool] | None = None,
    reraise_interrupt: bool = False,
) -> dict[str, Any]:
    """Accumulate a streaming chat response, emitting deltas as events.

    Event-driven variant of :func:`collect_streaming_response`: instead
    of writing tokens to stdout, each thinking/content delta is emitted
    as an :class:`~local_cli.harness.AgentEvent` so the front-end decides
    how (and whether) to render it.

    A ``KeyboardInterrupt`` during streaming emits an ``interrupted``
    event and returns the partial response, mirroring the classic
    collector's behaviour.

    Args:
        stream: A generator yielding parsed streaming chunks.
        emit: The event callback.
        tracker: Optional :class:`TokenTracker` for usage recording.
        should_stop: Optional callback checked per chunk; when it
            returns ``True`` streaming stops and the partial response
            is returned (used by the JSON-line server's stop button).

    Returns:
        The assembled response dict, in the same shape as
        :func:`collect_streaming_response` (including a top-level
        ``"thinking"`` key when the model produced any).

    Raises:
        ProviderStreamError: Re-raised on a mid-stream provider error.
    """
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    last_chunk: dict[str, Any] = {}

    try:
        for chunk in stream:
            if should_stop is not None and should_stop():
                break
            last_chunk = chunk
            message = chunk.get("message", {})

            thinking_delta = message.get("thinking", "")
            if thinking_delta:
                thinking_parts.append(thinking_delta)
                emit(AgentEvent("thinking_delta", {"text": thinking_delta}))

            delta = message.get("content", "")
            if delta:
                content_parts.append(delta)
                emit(AgentEvent("content_delta", {"text": delta}))

            chunk_tool_calls = message.get("tool_calls")
            if chunk_tool_calls:
                tool_calls.extend(chunk_tool_calls)

    except KeyboardInterrupt:
        if reraise_interrupt:
            # Sub-agent mode: the caller owns interrupt semantics.
            raise
        # User interrupted streaming.  Keep what we have so far.
        emit(AgentEvent("interrupted", {"where": "stream"}))

    assembled_message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    if tool_calls:
        assembled_message["tool_calls"] = tool_calls

    result: dict[str, Any] = dict(last_chunk)
    result["message"] = assembled_message
    if thinking_parts:
        result["thinking"] = "".join(thinking_parts)

    if tracker is not None:
        if "usage" in result:
            tracker.record_from_claude(result)
        else:
            tracker.record_from_ollama(result)

    return result


def _stream_with_retry(
    provider: LLMProvider,
    model: str,
    messages: list[dict[str, Any]],
    chat_kwargs: dict[str, Any],
    emit: EmitFn,
    retry_on_overload: bool,
    tracker: TokenTracker | None,
    should_stop: Callable[[], bool] | None = None,
    reraise_interrupt: bool = False,
) -> dict[str, Any]:
    """Call ``provider.chat_stream`` with retry on overload (HTTP 503).

    Retries up to ``_RETRY_MAX_ATTEMPTS`` times with exponential backoff
    (1s, 2s, 4s) when the provider raises an overload error *before*
    streaming starts.  Mid-stream errors are never retried (partial
    output may already have been emitted).

    Args:
        provider: The LLM provider (or duck-typed client).
        model: Model name for the chat request.
        messages: Conversation history.
        chat_kwargs: Extra keyword arguments for ``chat_stream``.
        emit: The event callback (``retry`` events are emitted here).
        retry_on_overload: When ``False``, the first error propagates.
        tracker: Optional token tracker, forwarded to the collector.

    Returns:
        The assembled response dict.

    Raises:
        ProviderRequestError: If retries are exhausted or the error is
            not an overload condition.
        ProviderStreamError: On a mid-stream error (never retried).
    """
    attempts = _RETRY_MAX_ATTEMPTS if retry_on_overload else 0
    for attempt in range(attempts + 1):
        if attempt > 0:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            emit(AgentEvent("retry", {"attempt": attempt, "delay": delay}))
            time.sleep(delay)
        try:
            stream = provider.chat_stream(model, messages, **chat_kwargs)
            return _collect_stream_emitting(
                stream, emit, tracker=tracker, should_stop=should_stop,
                reraise_interrupt=reraise_interrupt,
            )
        except ProviderRequestError as exc:
            if not _is_overloaded_error(exc) or attempt == attempts:
                raise
            # Will retry on the next iteration.

    raise ProviderRequestError(  # pragma: no cover -- unreachable
        "Provider overloaded after retries"
    )


def _summary_compact(
    provider: LLMProvider,
    model: str,
    messages: list[dict[str, Any]],
    keep_recent: int,
    emit: EmitFn,
    debug: bool = False,
) -> None:
    """Compact history by asking the model to summarize the older span.

    Replaces everything between the leading system messages and the
    *keep_recent* most recent messages with a single summary message.
    Falls back to the classic in-place truncation
    (:func:`compact_messages`) when there is nothing to summarize, the
    summarizer call fails, or it returns an empty summary — compaction
    must never leave the history over-threshold.

    Args:
        provider: The LLM provider used for the summarizer call.
        model: Model name for the summarizer call.
        messages: The conversation history (mutated in place).
        keep_recent: Number of trailing messages preserved verbatim.
        emit: The event callback (``compaction`` events).
        debug: Forwarded to the truncation fallback.
    """
    before_tokens = _estimate_tokens(messages)
    bounds = compaction_bounds(messages, keep_recent=keep_recent)
    summary = ""
    if bounds is not None:
        system_end, recent_start = bounds
        try:
            request = build_summary_request(messages[system_end:recent_start])
            stream = provider.chat_stream(model, request)
            response = _collect_stream_emitting(stream, null_emit)
            summary = (response.get("message") or {}).get("content", "").strip()
        except Exception:
            summary = ""

    if bounds is not None and summary:
        system_end, recent_start = bounds
        apply_summary(messages, summary, system_end, recent_start)
        mode = "summarize"
    else:
        compact_messages(messages, debug=debug)
        mode = "truncate"

    emit(AgentEvent("compaction", {
        "mode": mode,
        "before_tokens": before_tokens,
        "after_tokens": _estimate_tokens(messages),
    }))


def _tool_message(
    tool_name: str,
    result: str,
    tool_call_id: str | None,
) -> dict[str, Any]:
    """Build a ``role: "tool"`` message, attaching the id when present.

    The ``tool_call_id`` is critical for providers like Claude that
    require ``tool_use_id`` on tool results.
    """
    msg: dict[str, Any] = {
        "role": "tool",
        "tool_name": tool_name,
        "content": result,
    }
    if tool_call_id is not None:
        msg["tool_call_id"] = tool_call_id
    return msg


def run_agent(
    provider: LLMProvider,
    model: str,
    tools: list[Tool],
    messages: list[dict[str, Any]],
    *,
    emit: EmitFn = null_emit,
    harness: HarnessConfig | None = None,
    cache: ToolCache | None = None,
    tracker: TokenTracker | None = None,
    options: dict[str, Any] | None = None,
    think: bool | None = None,
    chat_extra: dict[str, Any] | None = None,
    should_stop: Callable[[], bool] | None = None,
    raise_provider_errors: bool = False,
    debug: bool = False,
) -> str:
    """The unified agent loop: prompt the LLM, run tools, repeat.

    This is the single core loop shared by every front-end — the CLI
    REPL, the JSON-line server, the web monitor, and sub-agents.  The
    loop performs no I/O of its own; every observable moment is emitted
    as an :class:`~local_cli.harness.AgentEvent` through *emit*, and each
    front-end renders the events it cares about.

    On top of the classic loop it layers the deterministic harness
    interventions that let small local models sustain long agentic
    sessions (see :mod:`local_cli.harness`):

    - **Text tool-call rescue** — when the model wrote its tool call as
      text (``<tool_call>`` tags, fenced JSON, bare JSON) instead of a
      structured call, the call is parsed and executed anyway.
    - **Loop detection** — repeated identical calls first draw a
      corrective reminder, then pause tool access for a final summary.
    - **Post-write verification** — ``.py``/``.json`` files are
      syntax-checked immediately after write/edit and a warning is
      appended to the tool result on failure.
    - **Todo staleness reminders** — a stale todo list is re-surfaced
      so multi-step work is not silently abandoned.
    - **Step limit** — after ``max_iterations`` the model gets one final
      tool-free turn to summarize.
    - **Overload retry** — HTTP 503s are retried with backoff.
    - **Context compaction** — truncation (classic) or LLM
      summarization with truncation fallback.

    Args:
        provider: An :class:`~local_cli.providers.base.LLMProvider`
            instance (or any object with a compatible
            ``chat_stream(model, messages, ...)`` method — including
            :class:`OllamaClient`, accepted via duck typing).  Providers
            without ``format_tools`` fall back to the Ollama tool format.
        model: The model name to use (e.g. ``"qwen3:8b"``).
        tools: The :class:`Tool` instances available to the LLM.
        messages: The conversation history (mutated in place).
        emit: Event callback; defaults to a no-op (silent execution).
        harness: Harness intervention switches; defaults to
            :class:`HarnessConfig`'s defaults.
        cache: Optional :class:`ToolCache` for idempotent tools.
        tracker: Optional :class:`TokenTracker` for usage recording.
        options: Optional inference parameters forwarded to the
            provider.  A ``num_ctx`` key dynamically sets the compaction
            threshold to 75% of the context window.
        think: Optional thinking-mode flag forwarded to the provider.
        chat_extra: Extra keyword arguments merged into every
            ``chat_stream`` call (e.g. ``keep_alive`` for Ollama).
        should_stop: Optional callback polled at every loop checkpoint
            (per stream chunk, per iteration, per tool call); when it
            returns ``True`` the loop stops gracefully, emitting a
            ``stopped`` event.  Used by GUI front-ends' stop buttons.
        raise_provider_errors: When ``True``, provider errors propagate
            to the caller instead of being recorded in the history —
            sub-agents use this so their runner can report an error
            status.
        debug: Emit extra ``debug`` events (rendered on stderr by the
            console emitter).

    Returns:
        The final assistant message content (empty string when the
        loop produced none).

    Raises:
        KeyboardInterrupt: Propagated if the user interrupts tool
            execution (streaming interrupts are handled gracefully).
        ProviderRequestError: Only when *raise_provider_errors* is set.
        ProviderConnectionError: Only when *raise_provider_errors* is set.
        ProviderStreamError: Only when *raise_provider_errors* is set.
    """
    hc = harness or HarnessConfig()
    tool_map: dict[str, Tool] = {t.name: t for t in tools}
    if isinstance(provider, LLMProvider):
        tool_defs: list[dict[str, Any]] = provider.format_tools(tools)
    else:
        # Duck-typed clients (e.g. OllamaClient, mocks) lack
        # format_tools; fall back to the Ollama function-call format.
        tool_defs = [t.to_ollama_tool() for t in tools]

    compact_token_threshold: int | None = None
    if options and "num_ctx" in options:
        compact_token_threshold = int(options["num_ctx"] * 0.75)
        if debug:
            emit(AgentEvent("debug", {"text": (
                f"[debug] Dynamic compaction threshold: "
                f"{compact_token_threshold} tokens "
                f"(75% of num_ctx={options['num_ctx']})"
            )}))

    detector = LoopDetector() if hc.loop_detection else None
    reminder = TodoReminder() if hc.todo_reminders else None
    nudged = False  # whether we've already nudged "use the tools" this turn
    error_nudged = False  # one push-back per turn for finishing on an error
    empty_nudged = False  # one push-back per turn for an empty reply
    deliverable_nudged = False  # one push-back for finishing without the asked file
    force_final = False  # when True, the next LLM call gets no tools
    tools_disabled = False  # endpoint rejected tools; text-driven fallback
    final_content = ""
    iteration = 0

    while True:
        iteration += 1

        # ---------------------------------------------------------------
        # 0. Front-end stop request (GUI stop button).
        # ---------------------------------------------------------------
        if should_stop is not None and should_stop():
            emit(AgentEvent("stopped", {"where": "loop"}))
            break

        # ---------------------------------------------------------------
        # 0a. Step limit: pause tools and ask the model to wrap up.
        # ---------------------------------------------------------------
        if (
            hc.max_iterations
            and iteration > hc.max_iterations
            and not force_final
        ):
            force_final = True
            emit(AgentEvent("limit", {"iterations": hc.max_iterations}))
            messages.append(step_limit_message(hc.max_iterations))

        # ---------------------------------------------------------------
        # 0b. Compact conversation history if thresholds are exceeded.
        # ---------------------------------------------------------------
        if _needs_compaction(messages, token_threshold=compact_token_threshold):
            if hc.compact_mode == "summarize":
                _summary_compact(
                    provider, model, messages, hc.keep_recent, emit,
                    debug=debug,
                )
            else:
                compact_messages(messages, debug=debug)

        # ---------------------------------------------------------------
        # 0c. Todo staleness reminder.
        # ---------------------------------------------------------------
        if reminder is not None and not force_final:
            reminder_text = reminder.check(iteration, tools)
            if reminder_text:
                messages.append({"role": "user", "content": reminder_text})
                emit(AgentEvent("reminder", {"text": reminder_text}))

        # ---------------------------------------------------------------
        # 1. Send messages to the LLM and stream the response.
        # ---------------------------------------------------------------
        emit(AgentEvent("llm_start", {
            "iteration": iteration,
            "message_count": len(messages),
            "model": model,
        }))

        chat_kwargs: dict[str, Any] = {}
        if not force_final and not tools_disabled:
            chat_kwargs["tools"] = tool_defs
        if options is not None:
            chat_kwargs["options"] = options
        if think is not None:
            chat_kwargs["think"] = think
        if chat_extra:
            chat_kwargs.update(chat_extra)

        try:
            full_response = _stream_with_retry(
                provider, model, messages, chat_kwargs, emit,
                retry_on_overload=hc.retry_on_overload, tracker=tracker,
                should_stop=should_stop,
                reraise_interrupt=raise_provider_errors,
            )
        except ProviderStreamError as exc:
            if raise_provider_errors:
                raise
            # Use provider-specific prefix when possible for backward
            # compatibility (existing tests assert "Error from Ollama").
            if isinstance(exc, OllamaStreamError):
                prefix = "Error from Ollama"
            else:
                prefix = "Error from provider"
            emit(AgentEvent("error", {
                "message": f"{prefix}: {exc}",
                "source": "stream",
                "detail": str(exc),
            }))
            messages.append({
                "role": "assistant",
                "content": f"[{prefix}: {exc}]",
            })
            break
        except ProviderConnectionError as exc:
            if raise_provider_errors:
                raise
            emit(AgentEvent("error", {
                "message": f"Error from provider: {exc}",
                "source": "connection",
                "detail": str(exc),
            }))
            messages.append({
                "role": "assistant",
                "content": f"[Error from provider: {exc}]",
            })
            break
        except ProviderRequestError as exc:
            # A model whose chat template has no tool support rejects
            # the request outright.  Fall back to text-driven tools:
            # drop the tools parameter, teach the model the fenced-JSON
            # call shape, and let extract_text_tool_calls rescue its
            # calls — such models can still act as agents.
            if (
                not tools_disabled
                and "tools" in chat_kwargs
                and hc.text_tool_rescue
                and is_tools_unsupported_error(exc)
            ):
                tools_disabled = True
                messages.append(text_tools_fallback_message(tools))
                emit(AgentEvent("tools_fallback", {"model": model}))
                continue
            if raise_provider_errors:
                raise
            emit(AgentEvent("error", {
                "message": f"Error from provider: {exc}",
                "source": "request",
                "detail": str(exc),
            }))
            messages.append({
                "role": "assistant",
                "content": f"[Error from provider: {exc}]",
            })
            break
        except KeyboardInterrupt:
            if raise_provider_errors:
                raise
            emit(AgentEvent("interrupted", {"where": "llm"}))
            break

        # ---------------------------------------------------------------
        # 2. Append the assistant message to the conversation history.
        #    Thinking content is intentionally excluded — it lives in
        #    full_response["thinking"] and must NOT be appended, to avoid
        #    wasting context window space on internal reasoning.
        # ---------------------------------------------------------------
        assistant_message = full_response["message"]
        if "thinking" in assistant_message:
            assistant_message = {
                k: v for k, v in assistant_message.items() if k != "thinking"
            }
        messages.append(assistant_message)
        final_content = assistant_message.get("content", "")

        emit(AgentEvent("assistant_message", {
            "message": assistant_message,
            "thinking": full_response.get("thinking", ""),
        }))

        # ---------------------------------------------------------------
        # 3. Determine tool calls: structured first, then text rescue.
        # ---------------------------------------------------------------
        tool_calls = assistant_message.get("tool_calls", [])

        if not tool_calls and not force_final and hc.text_tool_rescue:
            rescued = extract_text_tool_calls(
                assistant_message.get("content", "") or "",
                lambda name: resolve_tool_name(name, tool_map),
            )
            if rescued:
                # Attach to the (already appended) assistant message so
                # the history stays consistent with the tool results.
                assistant_message["tool_calls"] = rescued
                tool_calls = rescued
                emit(AgentEvent("rescue", {"count": len(rescued)}))

        if not tool_calls or force_final:
            if not force_final and _should_nudge_to_use_tools(
                messages, assistant_message, nudged,
            ):
                # On the no-tool-support fallback the standard "call the
                # write tool" wording is meaningless — restate the
                # fenced-JSON call shape instead.
                if tools_disabled:
                    messages.append(text_tool_nudge_message())
                else:
                    messages.append(dict(_TOOL_NUDGE_MESSAGE))
                nudged = True
                emit(AgentEvent("nudge", {}))
                continue
            # Empty-response guard: the model said nothing at all
            # (observed on small models after a tool result — the task
            # is half done and the turn would just end).  Push back once.
            if (
                not force_final
                and hc.empty_response_guard
                and not empty_nudged
                and not (assistant_message.get("content") or "").strip()
            ):
                messages.append(empty_response_message())
                empty_nudged = True
                emit(AgentEvent("empty_response", {}))
                continue
            # Error-stop guard: the model is finishing right after a
            # failed tool call (small models routinely ignore the error
            # and declare success).  Push back once.
            if (
                not force_final
                and hc.error_stop_guard
                and not error_nudged
                and last_tool_result_errored(messages)
            ):
                messages.append(error_stop_message())
                error_nudged = True
                emit(AgentEvent("error_stop", {}))
                continue
            # Deliverable guard: the request named a file/document to
            # produce, nothing was written this turn, and the model is
            # about to finish — typically after printing the whole
            # report into chat (observed live on the desktop app).
            if (
                not force_final
                and hc.deliverable_guard
                and not deliverable_nudged
                and mentions_file_deliverable(_last_user_text(messages))
                and _mentions_build_intent(_last_user_text(messages))
                and not _wrote_file_this_turn(messages)
            ):
                messages.append(deliverable_missing_message())
                deliverable_nudged = True
                emit(AgentEvent("deliverable_nudge", {}))
                continue
            break

        # ---------------------------------------------------------------
        # 4. Execute each tool call and append results.
        # ---------------------------------------------------------------
        # Search-then-write ordering guard: when this turn mixes
        # discovery calls with mutations, the mutations were decided
        # before the discovery results existed — defer them so the
        # model re-issues them with the results in hand.
        defer_mutations = False
        if hc.defer_writes_after_search and len(tool_calls) > 1:
            turn_names = set()
            for tc in tool_calls:
                name, _, _ = parse_tool_call(tc)
                resolved_name = resolve_tool_name(name, tool_map)
                if resolved_name is not None:
                    turn_names.add(resolved_name)
            defer_mutations = bool(turn_names & DISCOVERY_TOOLS) and bool(
                turn_names & MUTATION_TOOLS
            )

        loop_broken = False
        stopped = False
        for tc in tool_calls:
            raw_name, arguments, tool_call_id = parse_tool_call(tc)
            resolved = resolve_tool_name(raw_name, tool_map)
            tool_name = resolved or raw_name
            if resolved is not None:
                arguments = normalize_arguments(resolved, arguments)
            tool = tool_map.get(tool_name)

            # Front-end stop request between tool calls.  Remaining
            # calls still get a result message so no tool_call is left
            # unanswered in the history.
            if not stopped and should_stop is not None and should_stop():
                stopped = True
                emit(AgentEvent("stopped", {"where": "tools"}))
            if stopped:
                result = "Error: stopped by user."
                messages.append(_tool_message(tool_name, result, tool_call_id))
                continue

            # Loop detection: classify before executing, so a run that
            # crossed the break threshold is not executed yet again.
            verdict = "ok"
            if detector is not None and not loop_broken:
                verdict = detector.record(tool_name, arguments)

            if loop_broken or verdict == "break":
                if not loop_broken:
                    loop_broken = True
                    force_final = True
                    messages.append(loop_break_message())
                    emit(AgentEvent("loop_break", {"tool_name": tool_name}))
                result = "Error: skipped (tool loop detected)."
                emit(AgentEvent("tool_result", {
                    "tool_name": tool_name,
                    "result": result,
                    "cached": False,
                    "skipped": True,
                    "tool_call_id": tool_call_id,
                }))
                messages.append(_tool_message(tool_name, result, tool_call_id))
                continue

            if tool is None:
                result = f"Error: unknown tool '{tool_name}'"
                emit(AgentEvent("tool_result", {
                    "tool_name": tool_name,
                    "result": result,
                    "cached": False,
                    "unknown": True,
                    "tool_call_id": tool_call_id,
                }))
                messages.append(_tool_message(tool_name, result, tool_call_id))
                continue

            if defer_mutations and tool_name in MUTATION_TOOLS:
                result = deferred_write_result(tool_name)
                emit(AgentEvent("write_deferred", {
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                }))
                emit(AgentEvent("tool_result", {
                    "tool_name": tool_name,
                    "result": result,
                    "cached": False,
                    "deferred": True,
                    "tool_call_id": tool_call_id,
                }))
                messages.append(_tool_message(tool_name, result, tool_call_id))
                continue

            # Check cache for cacheable tools before execution.
            cached = False
            result = None
            if cache is not None and tool.cacheable:
                hit = cache.get(tool_name, arguments)
                if hit is not None:
                    result = hit
                    cached = True
                    if debug:
                        emit(AgentEvent("debug", {
                            "text": f"  [debug] cache hit: {tool_name}",
                        }))
                elif debug:
                    emit(AgentEvent("debug", {
                        "text": f"  [debug] cache miss: {tool_name}",
                    }))

            if result is None:
                emit(AgentEvent("tool_start", {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "tool_call_id": tool_call_id,
                }))
                try:
                    result = _execute_tool(tool, arguments, debug=debug)
                except KeyboardInterrupt:
                    result = "Error: tool execution interrupted by user."
                    emit(AgentEvent("interrupted", {
                        "where": "tool",
                        "tool_name": tool_name,
                    }))
                    messages.append(
                        _tool_message(tool_name, result, tool_call_id)
                    )
                    raise

                # Store successful results in cache for cacheable tools.
                if (
                    cache is not None
                    and tool.cacheable
                    and not result.startswith("Error:")
                ):
                    cache.put(
                        tool_name, arguments, result,
                        file_path=_extract_file_path(tool_name, arguments),
                    )

                # Invalidate cache when a mutating tool modifies a file.
                if (
                    cache is not None
                    and not tool.cacheable
                    and not result.startswith("Error:")
                ):
                    fp = arguments.get("file_path")
                    if isinstance(fp, str):
                        cache.invalidate_file(fp)

            # Post-write verification gate: surface syntax errors in the
            # file the model just produced while it still has context.
            if hc.verify_writes:
                warning = verify_file_write(tool_name, arguments, result)
                if warning is not None:
                    result = f"{result}\n\n{warning}"
                    emit(AgentEvent("verify_warning", {
                        "file_path": arguments.get("file_path", ""),
                        "message": warning,
                    }))

            emit(AgentEvent("tool_result", {
                "tool_name": tool_name,
                "result": result,
                "cached": cached,
                "tool_call_id": tool_call_id,
            }))
            messages.append(_tool_message(tool_name, result, tool_call_id))

            if reminder is not None:
                reminder.note_tool_use(iteration, tool_name)

            if verdict == "warn":
                messages.append(loop_warning_message(tool_name))
                emit(AgentEvent("loop_warning", {"tool_name": tool_name}))

        if stopped:
            break

        # ---------------------------------------------------------------
        # 5. Loop back to send tool results to the LLM.
        # ---------------------------------------------------------------

    return final_content


# ---------------------------------------------------------------------------
# Console emitter (classic CLI presentation)
# ---------------------------------------------------------------------------


class _ConsoleEmitter:
    """Render agent events with the classic CLI presentation.

    Reproduces exactly the stdout/stderr behaviour the REPL always had:
    streamed content on stdout, ``> ``-prefixed thinking lines, spinners
    while waiting, indented tool-result previews on stderr, and
    ``[debug]`` diagnostics when debug mode is on.  Harness events that
    did not exist before (rescue, loop detection, verification, retries)
    are surfaced as brief ``[harness]`` lines on stderr.
    """

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug
        self._spinner: Spinner | None = None
        self._had_thinking = False
        self._had_content = False

    def _start_spinner(self, message: str) -> None:
        self._stop_spinner()
        self._spinner = Spinner(message)
        self._spinner.start()

    def _stop_spinner(self) -> None:
        if self._spinner is not None:
            self._spinner.stop()
            self._spinner = None

    def close(self) -> None:
        """Stop any live spinner (called when the loop exits)."""
        self._stop_spinner()

    def __call__(self, event: AgentEvent) -> None:
        kind = event.kind
        data = event.data

        if kind == "llm_start":
            self._had_thinking = False
            self._had_content = False
            if self._debug:
                sys.stderr.write(
                    f"[debug] Sending {data['message_count']} messages "
                    f"to {data['model']}\n"
                )
            self._start_spinner("Thinking")

        elif kind == "thinking_delta":
            self._stop_spinner()
            self._had_thinking = True
            for line in data["text"].splitlines(keepends=True):
                sys.stdout.write(f"> {line}")
            sys.stdout.flush()

        elif kind == "content_delta":
            self._stop_spinner()
            if self._had_thinking and not self._had_content:
                sys.stdout.write("\n")
            self._had_content = True
            sys.stdout.write(data["text"])
            sys.stdout.flush()

        elif kind == "assistant_message":
            self._stop_spinner()
            if self._had_content:
                sys.stdout.write("\n")
                sys.stdout.flush()
            if self._debug:
                message = data["message"]
                thinking = data.get("thinking", "")
                if thinking:
                    preview = _truncate(thinking, max_len=500)
                    sys.stderr.write(f"[debug] Thinking: {preview}\n")
                tc_count = len(message.get("tool_calls", []))
                sys.stderr.write(
                    f"[debug] Assistant responded: "
                    f"{len(message.get('content', ''))} chars, "
                    f"{tc_count} tool call(s)\n"
                )

        elif kind == "tool_start":
            self._start_spinner(f"Running {data['tool_name']}")

        elif kind == "tool_result":
            self._stop_spinner()
            if data.get("unknown"):
                sys.stderr.write(f"  Unknown tool: {data['tool_name']}\n")
            else:
                preview = _truncate(
                    data["result"].replace("\n", " ").strip(),
                )
                sys.stderr.write(f"  Result: {preview}\n")

        elif kind == "interrupted":
            self._stop_spinner()
            where = data.get("where", "")
            if where == "stream":
                sys.stdout.write("\n")
                sys.stdout.flush()
            elif where == "tool":
                sys.stderr.write(
                    f"  Tool {data.get('tool_name', '?')} interrupted.\n"
                )
            else:
                sys.stderr.write("\nInterrupted.\n")

        elif kind == "error":
            self._stop_spinner()
            sys.stderr.write(f"{data['message']}\n")

        elif kind == "rescue":
            sys.stderr.write(
                f"  [harness] rescued {data['count']} tool call(s) "
                "written as text\n"
            )

        elif kind == "tools_fallback":
            sys.stderr.write(
                f"  [harness] {data.get('model', 'model')} does not "
                "support structured tool calls — switching to "
                "text-driven tools\n"
            )

        elif kind == "error_stop":
            sys.stderr.write(
                "  [harness] model tried to finish on a failed tool "
                "call — pushing back\n"
            )

        elif kind == "write_deferred":
            sys.stderr.write(
                f"  [harness] {data['tool_name']} deferred — issued "
                "alongside a search; the model should use the search "
                "results first\n"
            )

        elif kind == "empty_response":
            sys.stderr.write(
                "  [harness] model replied with nothing — pushing back\n"
            )

        elif kind == "verify_warning":
            sys.stderr.write(
                f"  [harness] post-write check failed: "
                f"{data['file_path']}\n"
            )

        elif kind == "loop_warning":
            sys.stderr.write(
                f"  [harness] repeated tool call detected "
                f"({data['tool_name']}) — nudging the model\n"
            )

        elif kind == "loop_break":
            sys.stderr.write(
                "  [harness] tool loop detected — asking the model "
                "to wrap up\n"
            )

        elif kind == "limit":
            sys.stderr.write(
                f"  [harness] step limit reached "
                f"({data['iterations']} iterations) — wrapping up\n"
            )

        elif kind == "retry":
            sys.stderr.write(
                f"  [harness] provider overloaded — retrying in "
                f"{data['delay']:.0f}s (attempt {data['attempt']})\n"
            )

        elif kind == "debug":
            if self._debug:
                sys.stderr.write(data["text"] + "\n")


# ---------------------------------------------------------------------------
# Classic CLI agent loop (thin wrapper over run_agent)
# ---------------------------------------------------------------------------


def agent_loop(
    client: LLMProvider,
    model: str,
    tools: list[Tool],
    messages: list[dict[str, Any]],
    debug: bool = False,
    cache: ToolCache | None = None,
    tracker: TokenTracker | None = None,
    options: dict[str, Any] | None = None,
    think: bool | None = None,
    harness: HarnessConfig | None = None,
    tee: EmitFn | None = None,
) -> None:
    """Core agent loop with the classic CLI presentation.

    Thin wrapper over :func:`run_agent` using :class:`_ConsoleEmitter`,
    preserving the historical stdout/stderr behaviour (streamed tokens,
    spinners, indented result previews).  All loop logic — including the
    deterministic harness interventions — lives in :func:`run_agent`.

    Args:
        client: An :class:`~local_cli.providers.base.LLMProvider`
            instance (or any object with a compatible ``chat_stream``
            method — including :class:`OllamaClient`, accepted via duck
            typing for backward compatibility).
        model: The model name to use (e.g. ``"qwen3:8b"``).
        tools: A list of :class:`Tool` instances available for the LLM.
        messages: The conversation history (mutated in place).
        debug: If True, print extra diagnostic information to stderr.
        cache: Optional :class:`ToolCache` for cacheable tools.
        tracker: Optional :class:`TokenTracker` for usage recording.
        options: Optional inference parameters (``num_ctx`` also tunes
            the compaction threshold).
        think: Optional thinking-mode flag forwarded to the provider.
        harness: Optional harness intervention switches; defaults to
            :class:`HarnessConfig`'s defaults.
        tee: Optional second emit sink (e.g. a session logger) that
            receives every event after the console emitter.

    Raises:
        KeyboardInterrupt: Propagated if the user presses Ctrl+C during
            tool execution (streaming interrupts are handled gracefully).
    """
    emitter = _ConsoleEmitter(debug=debug)
    emit_fn: EmitFn = emitter
    if tee is not None:
        def emit_fn(event: AgentEvent) -> None:
            emitter(event)
            tee(event)
    try:
        run_agent(
            client, model, tools, messages,
            emit=emit_fn,
            harness=harness,
            cache=cache,
            tracker=tracker,
            options=options,
            think=think,
            debug=debug,
        )
    finally:
        emitter.close()


# ---------------------------------------------------------------------------
# Plan context injection
# ---------------------------------------------------------------------------


def build_plan_context(plan_content: str) -> dict[str, Any]:
    """Build a system message containing plan context for injection.

    When a plan is active, this message is inserted into the conversation
    so the LLM is aware of the plan structure and progress.  The content
    is wrapped with clear delimiters so the model can distinguish plan
    context from the main system prompt.

    Args:
        plan_content: The raw markdown content of the active plan.

    Returns:
        A system-role message dict suitable for insertion into the
        messages list.
    """
    return {
        "role": "system",
        "content": (
            "--- ACTIVE PLAN ---\n"
            f"{plan_content}\n"
            "--- END PLAN ---\n\n"
            "You are working on the plan above. Execute the next "
            "incomplete step and update the plan as you make progress."
        ),
    }


# ---------------------------------------------------------------------------
# Fast-mode heuristic
# ---------------------------------------------------------------------------


def _is_complex_request(
    prompt: str,
    threshold: int = _COMPLEX_WORD_THRESHOLD,
) -> bool:
    """Determine whether a user prompt represents a complex request.

    Used as a fast-mode heuristic: simple requests (short prompts, quick
    questions) skip planning overhead and go directly to the agent loop.
    Complex requests (long prompts, multi-file operations, architectural
    changes) may benefit from plan mode.

    A request is considered complex if:

    * The word count exceeds *threshold*, **or**
    * The prompt contains one or more plan-related keywords.

    Args:
        prompt: The user's input text.
        threshold: Word-count threshold above which a prompt is
            considered complex.  Defaults to ``_COMPLEX_WORD_THRESHOLD``.

    Returns:
        ``True`` if the prompt appears complex; ``False`` otherwise.
    """
    # Check word count.
    words = prompt.split()
    if len(words) > threshold:
        return True

    # Check for complexity-indicating keywords (case-insensitive).
    prompt_lower = prompt.lower()
    for keyword in _COMPLEX_KEYWORDS:
        if keyword in prompt_lower:
            return True

    return False


# ---------------------------------------------------------------------------
# Ideation loop (tool-free brainstorming)
# ---------------------------------------------------------------------------


def ideation_loop(
    client: OllamaClient,
    model: str,
    messages: list[dict[str, Any]],
    think: bool | None = True,
) -> None:
    """Tool-free chat loop for brainstorming / ideation mode.

    Similar to :func:`agent_loop` but calls ``chat_stream`` **without**
    the ``tools`` parameter.  There is no tool execution step — the
    function sends the messages, streams the response, appends it to
    the conversation history, and returns.  This is a single-turn
    interaction; the caller (REPL) manages the multi-turn loop.

    The ``think`` parameter enables chain-of-thought reasoning when the
    model supports it.  If the model does not support it, the request
    is retried transparently without ``think``.

    Args:
        client: An :class:`OllamaClient` or :class:`LLMProvider` instance.
            Any object with a ``chat_stream(model, messages, ...)`` method
            is accepted.
        model: The model name to use (e.g. ``"qwen3:8b"``).
        messages: The ideation conversation history (mutated in place).
        think: Enable chain-of-thought reasoning.  Defaults to ``True``.
            Set to ``None`` to omit the parameter entirely.

    Raises:
        KeyboardInterrupt: Propagated if the user presses Ctrl+C before
            streaming starts.
    """
    thinking_spinner = Spinner("Thinking")
    thinking_spinner.start()

    try:
        kwargs: dict[str, Any] = {}
        if think is not None:
            kwargs["think"] = think

        stream = client.chat_stream(model, messages, **kwargs)
        full_response = collect_streaming_response(
            stream, spinner=thinking_spinner,
        )
    except ProviderStreamError as exc:
        thinking_spinner.stop()
        if isinstance(exc, OllamaStreamError):
            prefix = "Error from Ollama"
        else:
            prefix = "Error from provider"
        sys.stderr.write(f"{prefix}: {exc}\n")
        messages.append({
            "role": "assistant",
            "content": f"[{prefix}: {exc}]",
        })
        return
    except ProviderRequestError:
        # Model may not support `think` parameter — retry without.
        thinking_spinner.stop()
        if think is not None:
            sys.stderr.write(
                "Model does not support thinking mode, "
                "falling back to standard generation\n"
            )
            thinking_spinner = Spinner("Thinking")
            thinking_spinner.start()
            try:
                stream = client.chat_stream(model, messages)
                full_response = collect_streaming_response(
                    stream, spinner=thinking_spinner,
                )
            except ProviderStreamError as exc2:
                thinking_spinner.stop()
                if isinstance(exc2, OllamaStreamError):
                    prefix = "Error from Ollama"
                else:
                    prefix = "Error from provider"
                sys.stderr.write(f"{prefix}: {exc2}\n")
                messages.append({
                    "role": "assistant",
                    "content": f"[{prefix}: {exc2}]",
                })
                return
            except (ProviderRequestError, KeyboardInterrupt):
                thinking_spinner.stop()
                sys.stderr.write("\nIdeation request failed.\n")
                return
        else:
            sys.stderr.write("\nIdeation request failed.\n")
            return
    except KeyboardInterrupt:
        thinking_spinner.stop()
        sys.stderr.write("\nInterrupted.\n")
        return

    # Append the assistant message to the conversation history.
    assistant_message = full_response["message"]
    messages.append(assistant_message)
