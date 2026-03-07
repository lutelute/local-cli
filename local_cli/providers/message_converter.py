"""Bidirectional message format conversion between Ollama and Claude.

The internal (normalized) message format follows the Ollama / OpenAI
convention with ``role``, ``content``, and ``tool_calls`` keys.  The
Claude Messages API uses a different structure: system prompts are a
separate request field, content is an array of typed blocks, and tool
results are embedded as content blocks inside ``user`` messages.

This module provides conversion functions at the provider boundary so
that the agent loop, session storage, and conversation history all use
the single normalized format.

Format summary::

    Ollama / Normalized (internal)
    ──────────────────────────────
    system:      {"role": "system", "content": "..."}
    user:        {"role": "user", "content": "..."}
    assistant:   {"role": "assistant", "content": "...",
                  "tool_calls": [{"function": {"name": ..., "arguments": {...}}, "id": ...}]}
    tool result: {"role": "tool", "tool_name": "...", "tool_call_id": "...", "content": "..."}

    Claude (external)
    ─────────────────
    system:      separate ``system`` field in request body
    user:        {"role": "user", "content": [{"type": "text", "text": "..."}]}
    assistant:   {"role": "assistant", "content": [
                     {"type": "text", "text": "..."},
                     {"type": "tool_use", "id": "toolu_...", "name": "...", "input": {...}}
                 ]}
    tool result: {"role": "user", "content": [
                     {"type": "tool_result", "tool_use_id": "toolu_...", "content": "..."}
                 ]}
"""

from typing import Any


# ---------------------------------------------------------------------------
# Normalized -> Claude
# ---------------------------------------------------------------------------


def messages_to_claude(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert normalized messages to Claude API format.

    Extracts system messages from the list (Claude requires a separate
    ``system`` parameter) and converts each remaining message to Claude's
    content-block format.  Adjacent ``tool`` role messages are merged into
    a single ``user`` message with multiple ``tool_result`` content blocks.

    Args:
        messages: Conversation history in normalized format.

    Returns:
        A 2-tuple of ``(system_text, claude_messages)`` where
        *system_text* is the concatenated system prompt (or ``None`` if
        there are no system messages) and *claude_messages* is the list
        of Claude-formatted message dicts.
    """
    system_parts: list[str] = []
    claude_messages: list[dict[str, Any]] = []

    # Pending tool_result blocks waiting to be flushed into a user message.
    tool_result_blocks: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            content = msg.get("content", "")
            if content:
                system_parts.append(content)
            continue

        if role == "tool":
            # Accumulate tool results; they will be flushed as a user
            # message when a non-tool message appears (or at the end).
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id") or "",
                "content": msg.get("content", ""),
            }
            tool_result_blocks.append(block)
            continue

        # Before processing a non-tool message, flush any pending
        # tool_result blocks.
        if tool_result_blocks:
            claude_messages.append({
                "role": "user",
                "content": list(tool_result_blocks),
            })
            tool_result_blocks.clear()

        if role == "user":
            claude_messages.append(_user_to_claude(msg))
        elif role == "assistant":
            claude_messages.append(_assistant_to_claude(msg))
        else:
            # Unknown role -- pass through with content blocks.
            claude_messages.append({
                "role": role,
                "content": [{"type": "text", "text": msg.get("content", "")}],
            })

    # Flush any remaining tool_result blocks at the end.
    if tool_result_blocks:
        claude_messages.append({
            "role": "user",
            "content": list(tool_result_blocks),
        })

    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, claude_messages


def _user_to_claude(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a normalized user message to Claude format.

    Args:
        msg: A normalized ``role: "user"`` message.

    Returns:
        A Claude-formatted user message with content blocks.
    """
    content = msg.get("content", "")
    return {
        "role": "user",
        "content": [{"type": "text", "text": content}],
    }


def _assistant_to_claude(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a normalized assistant message to Claude format.

    Text content becomes a ``text`` block.  Tool calls become
    ``tool_use`` blocks.

    Args:
        msg: A normalized ``role: "assistant"`` message.

    Returns:
        A Claude-formatted assistant message with content blocks.
    """
    blocks: list[dict[str, Any]] = []

    content = msg.get("content", "")
    if content:
        blocks.append({"type": "text", "text": content})

    for tc in msg.get("tool_calls", []):
        func = tc.get("function", {})
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or "",
            "name": func.get("name", ""),
            "input": func.get("arguments", {}),
        })

    # Claude requires at least one content block.
    if not blocks:
        blocks.append({"type": "text", "text": ""})

    return {
        "role": "assistant",
        "content": blocks,
    }


# ---------------------------------------------------------------------------
# Claude -> Normalized
# ---------------------------------------------------------------------------


def claude_response_to_normalized(
    response: dict[str, Any],
) -> dict[str, Any]:
    """Convert a Claude Messages API response to the normalized format.

    Parses the ``content`` array from a Claude response, extracting text
    and tool-use blocks into the normalized ``content`` string and
    ``tool_calls`` list.

    Args:
        response: A Claude Messages API response dict.  Expected keys
            include ``content`` (list of content blocks), ``role``,
            ``model``, ``stop_reason``, and optionally ``usage``.

    Returns:
        A normalized response dict matching the :class:`LLMProvider`
        interface::

            {
                "message": {
                    "role": "assistant",
                    "content": "...",
                    "tool_calls": [...]  # only if tool_use blocks present
                }
            }
    """
    content_blocks = response.get("content", [])

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in content_blocks:
        block_type = block.get("type", "")

        if block_type == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)

        elif block_type == "tool_use":
            tool_calls.append({
                "function": {
                    "name": block.get("name", ""),
                    "arguments": block.get("input", {}),
                },
                "id": block.get("id", ""),
            })

    assembled_message: dict[str, Any] = {
        "role": response.get("role", "assistant"),
        "content": "\n\n".join(text_parts),
    }
    if tool_calls:
        assembled_message["tool_calls"] = tool_calls

    return {"message": assembled_message}


def claude_stream_to_normalized(
    event_type: str,
    data: dict[str, Any],
) -> dict[str, Any] | None:
    """Convert a single Claude SSE event to a normalized stream chunk.

    This is a stateless helper for the most common streaming events.
    The caller (typically :class:`ClaudeProvider`) is responsible for
    accumulating tool input JSON deltas across events.

    Args:
        event_type: The SSE event type (e.g. ``"content_block_delta"``).
        data: The parsed JSON data payload for the event.

    Returns:
        A normalized chunk dict, or ``None`` if the event should be
        skipped (e.g. ``ping``, ``message_start``).
    """
    if event_type == "content_block_delta":
        delta = data.get("delta", {})
        delta_type = delta.get("type", "")

        if delta_type == "text_delta":
            return {
                "message": {"content": delta.get("text", "")},
                "done": False,
            }
        # input_json_delta is accumulated by the provider, not emitted
        # as a normalized chunk.
        return None

    if event_type == "message_stop":
        return {
            "message": {"content": ""},
            "done": True,
        }

    if event_type == "message_delta":
        # message_delta carries stop_reason and usage; emit as done hint.
        return {
            "message": {"content": ""},
            "done": data.get("delta", {}).get("stop_reason") is not None,
        }

    # All other events (message_start, content_block_start,
    # content_block_stop, ping) are informational.
    return None


# ---------------------------------------------------------------------------
# Tool definition conversion
# ---------------------------------------------------------------------------


def tools_to_claude(
    tool_defs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Ollama-format tool definitions to Claude format.

    Ollama / OpenAI tool format::

        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

    Claude tool format::

        {"name": ..., "description": ..., "input_schema": ...}

    Args:
        tool_defs: A list of tool definitions in Ollama / OpenAI format.

    Returns:
        A list of tool definitions in Claude format.
    """
    claude_tools: list[dict[str, Any]] = []
    for td in tool_defs:
        func = td.get("function", {})
        claude_tools.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {}),
        })
    return claude_tools


def tools_from_claude(
    claude_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Claude-format tool definitions to Ollama / OpenAI format.

    Args:
        claude_tools: A list of tool definitions in Claude format.

    Returns:
        A list of tool definitions in Ollama / OpenAI format.
    """
    ollama_tools: list[dict[str, Any]] = []
    for ct in claude_tools:
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": ct.get("name", ""),
                "description": ct.get("description", ""),
                "parameters": ct.get("input_schema", {}),
            },
        })
    return ollama_tools
