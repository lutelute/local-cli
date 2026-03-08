"""Agent loop for local-cli.

Implements the core agent loop: send messages to the LLM, collect the
streaming response, execute any tool calls, and loop until the LLM
responds without requesting tools.  All output is streamed to stdout as
tokens arrive.
"""

import json
import sys
from typing import Any, Generator

from local_cli.ollama_client import OllamaClient, OllamaStreamError
from local_cli.providers.base import ProviderRequestError, ProviderStreamError
from local_cli.spinner import Spinner
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
) -> dict[str, Any]:
    """Accumulate a streaming chat response and print tokens as they arrive.

    Iterates over NDJSON chunks from :meth:`OllamaClient.chat_stream`,
    concatenating content deltas and accumulating ``tool_calls`` across
    chunks.  Content tokens are printed to stdout immediately for a
    responsive user experience.

    Args:
        stream: A generator yielding parsed NDJSON chunks from the Ollama
            streaming chat API.
        spinner: Optional spinner to stop once the first content arrives.

    Returns:
        A dictionary representing the full response, structured as::

            {
                "message": {
                    "role": "assistant",
                    "content": "<accumulated text>",
                    "tool_calls": [...]  # only if present
                },
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
    if thinking_parts:
        assembled_message["thinking"] = "".join(thinking_parts)
    if tool_calls:
        assembled_message["tool_calls"] = tool_calls

    # Merge the final chunk's top-level fields (model, done, metrics, etc.)
    # with our assembled message.
    result: dict[str, Any] = dict(last_chunk)
    result["message"] = assembled_message

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


def _needs_compaction(messages: list[dict[str, Any]]) -> bool:
    """Check whether the message list should be compacted.

    Returns ``True`` if either the message count or estimated token
    count exceeds their respective thresholds.

    Args:
        messages: The conversation message list.

    Returns:
        True if compaction is warranted.
    """
    if len(messages) > _COMPACT_MESSAGE_THRESHOLD:
        return True
    if _estimate_tokens(messages) > _COMPACT_TOKEN_THRESHOLD:
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
# Agent loop
# ---------------------------------------------------------------------------


def agent_loop(
    client: OllamaClient,
    model: str,
    tools: list[Tool],
    messages: list[dict[str, Any]],
    debug: bool = False,
) -> None:
    """Core agent loop: prompt LLM, execute tool calls, repeat.

    Sends the conversation *messages* to the LLM via streaming chat.  If
    the LLM responds with tool calls, each tool is executed and the results
    are appended to *messages* as ``role: "tool"`` entries.  The loop
    continues until the LLM responds without any tool calls.

    The *messages* list is mutated in place -- each assistant response and
    tool result is appended so the caller retains the full conversation
    history.

    Tool result messages include a ``tool_call_id`` field when the tool
    call has an ``id`` (critical for providers like Claude that require
    ``tool_use_id`` in subsequent tool result messages).

    Args:
        client: An :class:`OllamaClient` or :class:`LLMProvider` instance.
            Any object with a ``chat_stream(model, messages, tools=...)``
            method is accepted.
        model: The model name to use (e.g. ``"qwen3:8b"``).
        tools: A list of :class:`Tool` instances available for the LLM.
        messages: The conversation history (mutated in place).
        debug: If True, print extra diagnostic information to stderr.

    Raises:
        KeyboardInterrupt: Propagated if the user presses Ctrl+C during
            tool execution (streaming interrupts are handled gracefully).
    """
    tool_map: dict[str, Tool] = {t.name: t for t in tools}
    tool_defs: list[dict[str, Any]] = [t.to_ollama_tool() for t in tools]

    while True:
        # ---------------------------------------------------------------
        # 0. Compact conversation history if thresholds are exceeded.
        # ---------------------------------------------------------------
        if _needs_compaction(messages):
            compact_messages(messages, debug=debug)

        # ---------------------------------------------------------------
        # 1. Send messages to the LLM and collect the streaming response.
        # ---------------------------------------------------------------
        if debug:
            sys.stderr.write(
                f"[debug] Sending {len(messages)} messages to {model}\n"
            )

        thinking_spinner = Spinner("Thinking")
        thinking_spinner.start()

        try:
            stream = client.chat_stream(model, messages, tools=tool_defs)
            full_response = collect_streaming_response(
                stream, spinner=thinking_spinner,
            )
        except ProviderStreamError as exc:
            thinking_spinner.stop()
            # Use provider-specific prefix when possible for backward
            # compatibility (existing tests assert "Error from Ollama").
            if isinstance(exc, OllamaStreamError):
                prefix = "Error from Ollama"
            else:
                prefix = "Error from provider"
            sys.stderr.write(f"{prefix}: {exc}\n")
            messages.append({
                "role": "assistant",
                "content": f"[{prefix}: {exc}]",
            })
            break
        except KeyboardInterrupt:
            thinking_spinner.stop()
            sys.stderr.write("\nInterrupted.\n")
            break

        # ---------------------------------------------------------------
        # 2. Append the assistant message to the conversation history.
        # ---------------------------------------------------------------
        assistant_message = full_response["message"]
        messages.append(assistant_message)

        if debug:
            tc_count = len(assistant_message.get("tool_calls", []))
            sys.stderr.write(
                f"[debug] Assistant responded: "
                f"{len(assistant_message.get('content', ''))} chars, "
                f"{tc_count} tool call(s)\n"
            )

        # ---------------------------------------------------------------
        # 3. Check for tool calls.  If none, we're done.
        # ---------------------------------------------------------------
        tool_calls = assistant_message.get("tool_calls", [])
        if not tool_calls:
            break

        # ---------------------------------------------------------------
        # 4. Execute each tool call and append results.
        # ---------------------------------------------------------------
        for tc in tool_calls:
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            arguments = func.get("arguments", {})
            # Ollama sometimes returns arguments as a JSON string.
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, ValueError):
                    arguments = {}
            tool_call_id = tc.get("id")

            tool = tool_map.get(tool_name)
            if tool is None:
                result = f"Error: unknown tool '{tool_name}'"
                sys.stderr.write(f"  Unknown tool: {tool_name}\n")
            else:
                tool_spinner = Spinner(f"Running {tool_name}")
                tool_spinner.start()

                try:
                    result = _execute_tool(tool, arguments, debug=debug)
                    tool_spinner.stop()
                except KeyboardInterrupt:
                    tool_spinner.stop()
                    result = "Error: tool execution interrupted by user."
                    sys.stderr.write(f"  Tool {tool_name} interrupted.\n")
                    tool_msg: dict[str, Any] = {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": result,
                    }
                    if tool_call_id is not None:
                        tool_msg["tool_call_id"] = tool_call_id
                    messages.append(tool_msg)
                    raise

                # Show a truncated preview of the result to the user.
                preview = _truncate(
                    result.replace("\n", " ").strip(),
                )
                sys.stderr.write(f"  Result: {preview}\n")

            # Append tool result as a 'tool' role message.
            # Include tool_call_id when provided (critical for Claude
            # provider which requires tool_use_id in tool results).
            tool_msg = {
                "role": "tool",
                "tool_name": tool_name,
                "content": result,
            }
            if tool_call_id is not None:
                tool_msg["tool_call_id"] = tool_call_id
            messages.append(tool_msg)

        # ---------------------------------------------------------------
        # 5. Loop back to send tool results to the LLM.
        # ---------------------------------------------------------------


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
