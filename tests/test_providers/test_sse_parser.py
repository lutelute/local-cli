"""Tests for local_cli.providers.sse_parser module.

Verifies that the SSE parser correctly handles all event types emitted
by the Claude Messages API, including text deltas, tool use streaming,
ping keep-alive signals, multi-line data fields, and edge cases.
"""

import json
import unittest
from typing import Any

from local_cli.providers.sse_parser import SSEEvent, parse_sse_stream


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sse_lines(
    events: list[tuple[str, dict[str, Any]]],
    *,
    as_bytes: bool = False,
) -> list[str] | list[bytes]:
    """Build a list of SSE lines from (event_type, data) pairs.

    Each pair becomes an ``event: ...`` line, a ``data: ...`` line, and
    a trailing blank line.  If *as_bytes* is True the lines are returned
    as UTF-8 bytes (simulating ``urllib.request.urlopen``).
    """
    lines: list[str] = []
    for event_type, data in events:
        lines.append(f"event: {event_type}\n")
        lines.append(f"data: {json.dumps(data)}\n")
        lines.append("\n")
    if as_bytes:
        return [line.encode("utf-8") for line in lines]
    return lines


# ---------------------------------------------------------------------------
# SSEEvent dataclass
# ---------------------------------------------------------------------------


class TestSSEEventDataclass(unittest.TestCase):
    """SSEEvent dataclass construction and defaults."""

    def test_default_event_type(self) -> None:
        event = SSEEvent()
        self.assertEqual(event.event_type, "message")

    def test_default_data(self) -> None:
        event = SSEEvent()
        self.assertEqual(event.data, {})

    def test_custom_values(self) -> None:
        event = SSEEvent(event_type="ping", data={"status": "ok"})
        self.assertEqual(event.event_type, "ping")
        self.assertEqual(event.data, {"status": "ok"})

    def test_data_default_not_shared(self) -> None:
        """Each SSEEvent instance gets its own dict for data."""
        event1 = SSEEvent()
        event2 = SSEEvent()
        event1.data["key"] = "value"
        self.assertNotIn("key", event2.data)


# ---------------------------------------------------------------------------
# Basic parsing: event/data pairs
# ---------------------------------------------------------------------------


class TestParseBasicEvents(unittest.TestCase):
    """Basic event + data pair parsing."""

    def test_single_text_delta_event(self) -> None:
        lines = _make_sse_lines([
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            }),
        ])
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "content_block_delta")
        self.assertEqual(events[0].data["delta"]["text"], "Hello")

    def test_multiple_events(self) -> None:
        lines = _make_sse_lines([
            ("message_start", {"type": "message_start", "message": {"role": "assistant"}}),
            ("content_block_start", {"type": "content_block_start", "index": 0}),
            ("content_block_delta", {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_stop", {"type": "message_stop"}),
        ])
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 5)
        self.assertEqual(events[0].event_type, "message_start")
        self.assertEqual(events[1].event_type, "content_block_start")
        self.assertEqual(events[2].event_type, "content_block_delta")
        self.assertEqual(events[3].event_type, "content_block_stop")
        self.assertEqual(events[4].event_type, "message_stop")

    def test_message_delta_with_stop_reason(self) -> None:
        lines = _make_sse_lines([
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 42},
            }),
        ])
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "message_delta")
        self.assertEqual(events[0].data["delta"]["stop_reason"], "end_turn")
        self.assertEqual(events[0].data["usage"]["output_tokens"], 42)


# ---------------------------------------------------------------------------
# Bytes input (simulating urllib.request.urlopen)
# ---------------------------------------------------------------------------


class TestParseFromBytes(unittest.TestCase):
    """Parser handles byte input from HTTP responses."""

    def test_bytes_input(self) -> None:
        lines = _make_sse_lines([
            ("content_block_delta", {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "World"},
            }),
        ], as_bytes=True)
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data["delta"]["text"], "World")

    def test_mixed_bytes_and_str_not_expected(self) -> None:
        """Parser handles each line independently (bytes or str)."""
        lines: list[str | bytes] = [
            b"event: ping\n",
            "data: {}\n".encode("utf-8"),
            b"\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "ping")


# ---------------------------------------------------------------------------
# Ping / comment handling
# ---------------------------------------------------------------------------


class TestPingAndComments(unittest.TestCase):
    """Comment lines (starting with ':') are skipped."""

    def test_ping_comment_skipped(self) -> None:
        lines = [
            ": ping\n",
            "event: message_stop\n",
            "data: {}\n",
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "message_stop")

    def test_arbitrary_comment_skipped(self) -> None:
        lines = [
            ": this is a comment\n",
            ":another comment no space\n",
            "event: ping\n",
            "data: {}\n",
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)

    def test_ping_between_events(self) -> None:
        lines = [
            "event: message_start\n",
            'data: {"type": "message_start"}\n',
            "\n",
            ": ping\n",
            ": ping\n",
            "event: message_stop\n",
            "data: {}\n",
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, "message_start")
        self.assertEqual(events[1].event_type, "message_stop")

    def test_only_pings_produces_no_events(self) -> None:
        lines = [": ping\n", ": ping\n", ": ping\n"]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 0)


# ---------------------------------------------------------------------------
# [DONE] marker handling
# ---------------------------------------------------------------------------


class TestDoneMarker(unittest.TestCase):
    """data: [DONE] markers are skipped."""

    def test_done_marker_skipped(self) -> None:
        lines = [
            "event: message_stop\n",
            "data: {}\n",
            "\n",
            "data: [DONE]\n",
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        # Only message_stop event; [DONE] produces no event since
        # data_lines is empty after skipping.
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "message_stop")

    def test_done_marker_standalone(self) -> None:
        lines = ["data: [DONE]\n", "\n"]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 0)


# ---------------------------------------------------------------------------
# Empty lines and edge cases
# ---------------------------------------------------------------------------


class TestEmptyLinesAndEdgeCases(unittest.TestCase):
    """Empty lines, missing event types, and other edge cases."""

    def test_empty_input(self) -> None:
        events = list(parse_sse_stream([]))
        self.assertEqual(len(events), 0)

    def test_only_empty_lines(self) -> None:
        events = list(parse_sse_stream(["\n", "\n", "\n"]))
        self.assertEqual(len(events), 0)

    def test_multiple_consecutive_empty_lines(self) -> None:
        """Extra empty lines between events do not create extra events."""
        lines = [
            "event: message_start\n",
            'data: {"type": "message_start"}\n',
            "\n",
            "\n",
            "\n",
            "event: message_stop\n",
            "data: {}\n",
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 2)

    def test_data_without_event_line(self) -> None:
        """Data without a preceding event: line uses default 'message' type."""
        lines = [
            'data: {"key": "value"}\n',
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "message")
        self.assertEqual(events[0].data, {"key": "value"})

    def test_event_at_end_of_stream_without_trailing_newline(self) -> None:
        """Events at end of stream are flushed even without trailing blank."""
        lines = [
            "event: message_stop\n",
            "data: {}\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "message_stop")

    def test_event_line_without_data(self) -> None:
        """Event type set but no data line produces no event."""
        lines = [
            "event: ping\n",
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        # No data lines -> no event emitted on blank line.
        self.assertEqual(len(events), 0)

    def test_line_without_colon(self) -> None:
        """Lines without a colon are treated as field name with empty value."""
        lines = [
            "event: test\n",
            "data: {}\n",
            "someunknownfield\n",
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "test")


# ---------------------------------------------------------------------------
# Multi-line data fields
# ---------------------------------------------------------------------------


class TestMultiLineData(unittest.TestCase):
    """Multiple data: lines are concatenated before JSON parsing."""

    def test_two_data_lines(self) -> None:
        """Adjacent data lines are joined with newlines."""
        # JSON split across two data: lines.
        lines = [
            "event: content_block_delta\n",
            'data: {"type": "content_block_delta",\n',
            'data: "delta": {"type": "text_delta", "text": "hi"}}\n',
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "content_block_delta")
        self.assertEqual(events[0].data["delta"]["text"], "hi")

    def test_non_json_multi_line_data(self) -> None:
        """Non-JSON multi-line data is stored under 'raw' key."""
        lines = [
            "event: error\n",
            "data: something went wrong\n",
            "data: very badly\n",
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data["raw"], "something went wrong\nvery badly")


# ---------------------------------------------------------------------------
# Tool use streaming (input_json_delta)
# ---------------------------------------------------------------------------


class TestToolUseStreaming(unittest.TestCase):
    """Parsing of tool_use related SSE events."""

    def test_content_block_start_tool_use(self) -> None:
        lines = _make_sse_lines([
            ("content_block_start", {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_abc123",
                    "name": "read_file",
                    "input": {},
                },
            }),
        ])
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "content_block_start")
        block = events[0].data["content_block"]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["id"], "toolu_abc123")
        self.assertEqual(block["name"], "read_file")

    def test_input_json_delta_events(self) -> None:
        """Multiple input_json_delta events carry partial JSON strings."""
        lines = _make_sse_lines([
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"pa'},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": 'th": "/tmp/test.txt"}'},
            }),
        ])
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 2)

        # Both are content_block_delta with input_json_delta.
        for ev in events:
            self.assertEqual(ev.event_type, "content_block_delta")
            self.assertEqual(ev.data["delta"]["type"], "input_json_delta")

        # Accumulate the partial JSON (this is the caller's responsibility).
        accumulated = "".join(
            ev.data["delta"]["partial_json"] for ev in events
        )
        self.assertEqual(json.loads(accumulated), {"path": "/tmp/test.txt"})

    def test_full_tool_use_event_sequence(self) -> None:
        """Complete tool use streaming sequence."""
        lines = _make_sse_lines([
            ("content_block_start", {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_xyz",
                    "name": "bash",
                    "input": {},
                },
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"comma'},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": 'nd": "ls -la"}'},
            }),
            ("content_block_stop", {
                "type": "content_block_stop",
                "index": 1,
            }),
        ])
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 4)

        # Verify event types in order.
        event_types = [e.event_type for e in events]
        self.assertEqual(event_types, [
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
        ])

        # Verify tool block ID from content_block_start.
        self.assertEqual(
            events[0].data["content_block"]["id"],
            "toolu_xyz",
        )

        # Accumulate input_json_delta.
        json_parts = [
            e.data["delta"]["partial_json"]
            for e in events
            if e.data.get("delta", {}).get("type") == "input_json_delta"
        ]
        full_input = json.loads("".join(json_parts))
        self.assertEqual(full_input, {"command": "ls -la"})


# ---------------------------------------------------------------------------
# Full stream simulation (realistic Claude API response)
# ---------------------------------------------------------------------------


class TestFullStreamSimulation(unittest.TestCase):
    """Simulates a realistic Claude API streaming response."""

    def test_simple_text_response_stream(self) -> None:
        """Typical text-only streaming response."""
        lines = _make_sse_lines([
            ("message_start", {
                "type": "message_start",
                "message": {
                    "id": "msg_abc",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250514",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            }),
            ("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " world!"},
            }),
            ("content_block_stop", {
                "type": "content_block_stop",
                "index": 0,
            }),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 3},
            }),
            ("message_stop", {"type": "message_stop"}),
        ])
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 7)

        # Extract text from all text_delta events.
        text = "".join(
            e.data["delta"]["text"]
            for e in events
            if e.data.get("delta", {}).get("type") == "text_delta"
        )
        self.assertEqual(text, "Hello world!")

    def test_stream_with_pings_between_events(self) -> None:
        """Pings interspersed do not interfere."""
        lines = [
            "event: message_start\n",
            'data: {"type": "message_start"}\n',
            "\n",
            ": ping\n",
            "event: content_block_start\n",
            'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}\n',
            "\n",
            ": ping\n",
            ": ping\n",
            "event: content_block_delta\n",
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "test"}}\n',
            "\n",
            "event: message_stop\n",
            "data: {}\n",
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 4)

    def test_tool_use_and_text_mixed_response(self) -> None:
        """Response with both text and tool_use content blocks."""
        lines = _make_sse_lines([
            ("message_start", {
                "type": "message_start",
                "message": {"role": "assistant", "content": []},
            }),
            # Text block
            ("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Let me read that."},
            }),
            ("content_block_stop", {
                "type": "content_block_stop",
                "index": 0,
            }),
            # Tool use block
            ("content_block_start", {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_read",
                    "name": "read_file",
                    "input": {},
                },
            }),
            ("content_block_delta", {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"path": "a.txt"}'},
            }),
            ("content_block_stop", {
                "type": "content_block_stop",
                "index": 1,
            }),
            ("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
            }),
            ("message_stop", {"type": "message_stop"}),
        ])
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 9)

        # Verify we can find the tool_use block start.
        tool_start = [
            e for e in events
            if e.event_type == "content_block_start"
            and e.data.get("content_block", {}).get("type") == "tool_use"
        ]
        self.assertEqual(len(tool_start), 1)
        self.assertEqual(tool_start[0].data["content_block"]["name"], "read_file")


# ---------------------------------------------------------------------------
# Error event handling
# ---------------------------------------------------------------------------


class TestErrorEvents(unittest.TestCase):
    """Error events from the Claude API."""

    def test_error_event(self) -> None:
        lines = _make_sse_lines([
            ("error", {
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "Overloaded",
                },
            }),
        ])
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "error")
        self.assertEqual(events[0].data["error"]["type"], "overloaded_error")


# ---------------------------------------------------------------------------
# SSE spec compliance
# ---------------------------------------------------------------------------


class TestSSESpecCompliance(unittest.TestCase):
    """Compliance with the SSE specification details."""

    def test_leading_space_stripped_from_value(self) -> None:
        """SSE spec: a single leading space after ':' is stripped."""
        lines = [
            "event:  double_space\n",  # two spaces: first stripped, 'd' starts with space
            "data: {}\n",
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        # Only one space is stripped, so event_type is " double_space"
        self.assertEqual(events[0].event_type, " double_space")

    def test_no_space_after_colon(self) -> None:
        """If no space after colon, value is taken as-is."""
        lines = [
            "event:message_stop\n",
            "data:{}\n",
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "message_stop")
        self.assertEqual(events[0].data, {})

    def test_carriage_return_line_endings(self) -> None:
        """Lines with \\r\\n endings are handled correctly."""
        lines = [
            "event: message_stop\r\n",
            "data: {}\r\n",
            "\r\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "message_stop")

    def test_id_and_retry_fields_ignored(self) -> None:
        """id: and retry: fields are silently ignored."""
        lines = [
            "event: message_stop\n",
            "id: 123\n",
            "retry: 5000\n",
            "data: {}\n",
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "message_stop")

    def test_data_empty_string_produces_empty_dict(self) -> None:
        """Empty data value produces empty dict (can't parse empty JSON)."""
        lines = [
            "event: test\n",
            "data: \n",
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        # Empty string after stripping is still empty, can't JSON-parse.
        # Non-JSON data goes to {"raw": ""} but empty string is falsy so data={}
        self.assertEqual(events[0].data, {})

    def test_non_json_data_stored_as_raw(self) -> None:
        """Non-JSON data payloads are stored under a 'raw' key."""
        lines = [
            "event: error\n",
            "data: Internal server error\n",
            "\n",
        ]
        events = list(parse_sse_stream(lines))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data["raw"], "Internal server error")


# ---------------------------------------------------------------------------
# Generator behavior
# ---------------------------------------------------------------------------


class TestGeneratorBehavior(unittest.TestCase):
    """parse_sse_stream is a proper generator."""

    def test_returns_generator(self) -> None:
        import types
        result = parse_sse_stream([])
        self.assertIsInstance(result, types.GeneratorType)

    def test_yields_events_lazily(self) -> None:
        """Events are yielded as they complete, not all at once."""
        lines = _make_sse_lines([
            ("message_start", {"type": "message_start"}),
            ("message_stop", {"type": "message_stop"}),
        ])
        gen = parse_sse_stream(lines)
        first = next(gen)
        self.assertEqual(first.event_type, "message_start")
        second = next(gen)
        self.assertEqual(second.event_type, "message_stop")
        with self.assertRaises(StopIteration):
            next(gen)


if __name__ == "__main__":
    unittest.main()
