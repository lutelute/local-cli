"""Server-Sent Events (SSE) line parser for Claude API streaming.

The Claude Messages API streams responses as SSE events.  Each event
consists of an ``event:`` line, a ``data:`` line (containing JSON), and
a trailing blank line.  This module provides a generator that reads raw
HTTP response lines and yields structured :class:`SSEEvent` objects.

SSE event format::

    event: content_block_delta
    data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

    event: message_stop
    data: {"type":"message_stop"}

Special lines:

- ``: ping`` comments are sent as keep-alive signals and are skipped.
- ``data: [DONE]`` markers indicate the end of the stream and are skipped.
- Empty lines delimit event groups.

Event types emitted by the Claude Messages API:

- ``message_start`` -- initial message metadata (role, model, usage).
- ``content_block_start`` -- start of a new content block (text or tool_use).
- ``content_block_delta`` -- incremental update (``text_delta`` or
  ``input_json_delta``).
- ``content_block_stop`` -- end of a content block.
- ``message_delta`` -- final message-level updates (``stop_reason``,
  ``usage``).
- ``message_stop`` -- stream is complete.
- ``ping`` -- keep-alive signal.
- ``error`` -- server-side error.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Generator


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SSEEvent:
    """A parsed Server-Sent Event.

    Attributes:
        event_type: The event type string (e.g. ``"content_block_delta"``).
            Defaults to ``"message"`` if no ``event:`` line preceded the
            ``data:`` line (per the SSE specification).
        data: The parsed JSON data payload, or an empty dict if the data
            could not be parsed.
    """

    event_type: str = "message"
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_sse_stream(
    response: Any,
) -> Generator[SSEEvent, None, None]:
    """Parse an HTTP response as a Server-Sent Events stream.

    Reads lines from *response* (any iterable of ``bytes`` or ``str``),
    groups them into SSE events, and yields :class:`SSEEvent` instances
    with the ``event_type`` and parsed ``data`` payload.

    Lines beginning with ``:`` (comments, including ``: ping``) are
    silently skipped.  ``data: [DONE]`` markers are also skipped.
    Multi-line ``data:`` fields are accumulated and concatenated before
    JSON parsing.

    Args:
        response: An iterable of lines (bytes or str) from an HTTP
            response.  Typically ``urllib.request.urlopen(...)`` which
            yields ``bytes`` lines.

    Yields:
        :class:`SSEEvent` instances for each complete event group.
    """
    current_event_type: str | None = None
    data_lines: list[str] = []

    for raw_line in response:
        # Decode bytes to str if necessary.
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8")
        else:
            line = raw_line

        # Strip trailing newline / carriage-return.
        line = line.rstrip("\r\n")

        # Empty line: emit the accumulated event (if any).
        if not line:
            if data_lines:
                yield _build_event(current_event_type, data_lines)
                current_event_type = None
                data_lines = []
            continue

        # Comment lines (including `: ping`) are skipped.
        if line.startswith(":"):
            continue

        # Parse the field name and value.
        if ":" in line:
            field_name, _, field_value = line.partition(":")
            # SSE spec: strip a single leading space from the value.
            if field_value.startswith(" "):
                field_value = field_value[1:]
        else:
            # Lines without a colon use the whole line as the field name
            # with an empty value (per SSE spec).
            field_name = line
            field_value = ""

        if field_name == "event":
            current_event_type = field_value

        elif field_name == "data":
            # Skip [DONE] markers.
            if field_value == "[DONE]":
                continue
            data_lines.append(field_value)

        # Other field names (``id:``, ``retry:``) are ignored.

    # Flush any remaining event at end-of-stream.
    if data_lines:
        yield _build_event(current_event_type, data_lines)


def _build_event(
    event_type: str | None,
    data_lines: list[str],
) -> SSEEvent:
    """Assemble an :class:`SSEEvent` from accumulated fields.

    Multi-line data values are concatenated with newlines before JSON
    parsing, per the SSE specification.

    Args:
        event_type: The event type, or ``None`` for the SSE default
            (``"message"``).
        data_lines: Accumulated ``data:`` field values.

    Returns:
        A fully constructed :class:`SSEEvent`.
    """
    raw_data = "\n".join(data_lines)

    data: dict[str, Any] = {}
    if raw_data:
        try:
            parsed = json.loads(raw_data)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            # Non-JSON data is stored under a "raw" key so that the
            # caller can still inspect it.
            data = {"raw": raw_data}

    return SSEEvent(
        event_type=event_type if event_type is not None else "message",
        data=data,
    )
