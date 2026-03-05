"""Agent loop for local-cli.

Implements the core agent loop: send messages to the LLM, collect the
streaming response, execute any tool calls, and loop until the LLM
responds without requesting tools.  All output is streamed to stdout as
tokens arrive.
"""

import sys
from typing import Any, Generator

from local_cli.ollama_client import OllamaClient, OllamaStreamError
from local_cli.tools.base import Tool

# ---------------------------------------------------------------------------
# Result truncation for tool output displayed to the user
# ---------------------------------------------------------------------------

# Maximum characters of a tool result to print to the console.
_MAX_DISPLAY_RESULT = 200


# ---------------------------------------------------------------------------
# Streaming response collector
# ---------------------------------------------------------------------------


def collect_streaming_response(
    stream: Generator[dict[str, Any], None, None],
) -> dict[str, Any]:
    """Accumulate a streaming chat response and print tokens as they arrive.

    Iterates over NDJSON chunks from :meth:`OllamaClient.chat_stream`,
    concatenating content deltas and accumulating ``tool_calls`` across
    chunks.  Content tokens are printed to stdout immediately for a
    responsive user experience.

    Args:
        stream: A generator yielding parsed NDJSON chunks from the Ollama
            streaming chat API.

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
        OllamaStreamError: If the stream yields an error chunk (already
            handled by :class:`OllamaClient`, but re-raised here for
            clarity).
        KeyboardInterrupt: If the user presses Ctrl+C during streaming.
            Partial content accumulated so far is returned.
    """
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    last_chunk: dict[str, Any] = {}

    try:
        for chunk in stream:
            last_chunk = chunk

            message = chunk.get("message", {})

            # Accumulate content deltas and print to stdout.
            delta = message.get("content", "")
            if delta:
                content_parts.append(delta)
                sys.stdout.write(delta)
                sys.stdout.flush()

            # Accumulate tool calls (typically in the final chunk, but we
            # handle them appearing in any chunk for robustness).
            chunk_tool_calls = message.get("tool_calls")
            if chunk_tool_calls:
                tool_calls.extend(chunk_tool_calls)

    except KeyboardInterrupt:
        # User interrupted streaming.  Return what we have so far.
        sys.stdout.write("\n")
        sys.stdout.flush()

    except OllamaStreamError:
        # Mid-stream error from Ollama.  Print a newline to cleanly
        # separate any partial output, then re-raise so the caller can
        # decide how to handle it.
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise

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

    Args:
        client: An :class:`OllamaClient` instance.
        model: The Ollama model name to use (e.g. ``"qwen3:8b"``).
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
        # 1. Send messages to the LLM and collect the streaming response.
        # ---------------------------------------------------------------
        if debug:
            sys.stderr.write(
                f"[debug] Sending {len(messages)} messages to {model}\n"
            )

        try:
            stream = client.chat_stream(model, messages, tools=tool_defs)
            full_response = collect_streaming_response(stream)
        except OllamaStreamError as exc:
            sys.stderr.write(f"Error from Ollama: {exc}\n")
            break
        except KeyboardInterrupt:
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

            tool = tool_map.get(tool_name)
            if tool is None:
                result = f"Error: unknown tool '{tool_name}'"
                sys.stderr.write(f"  Unknown tool: {tool_name}\n")
            else:
                sys.stderr.write(f"  Running tool: {tool_name}\n")

                try:
                    result = _execute_tool(tool, arguments, debug=debug)
                except KeyboardInterrupt:
                    result = "Error: tool execution interrupted by user."
                    sys.stderr.write(f"  Tool {tool_name} interrupted.\n")

                # Show a truncated preview of the result to the user.
                preview = _truncate(
                    result.replace("\n", " ").strip(),
                )
                sys.stderr.write(f"  Result: {preview}\n")

            # Append tool result as a 'tool' role message.
            messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": result,
            })

        # ---------------------------------------------------------------
        # 5. Loop back to send tool results to the LLM.
        # ---------------------------------------------------------------
