"""Tests for local_cli.web_monitor — the SSE agent runner.

The web monitor was previously untested.  These exercise _run_agent through
its injected dependencies (a mock provider, a real Queue, a stub tool), so
no HTTP server or Ollama connection is needed.
"""

import json
import queue
import unittest
from unittest.mock import MagicMock

from local_cli.config import Config
from local_cli.tools.base import Tool
from local_cli.web_monitor import _run_agent


class _EchoTool(Tool):
    """Minimal tool that echoes its ``text`` argument."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo the text argument."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    def execute(self, **kwargs: object) -> str:
        return f"echoed: {kwargs.get('text', '')}"


def _drain(eq: "queue.Queue") -> list:
    """Collect JSON events from the queue up to the None sentinel."""
    events = []
    while True:
        item = eq.get_nowait()
        if item is None:
            break
        events.append(json.loads(item))
    return events


class TestRunAgent(unittest.TestCase):
    """Tests for _run_agent — the web monitor's agent loop."""

    def _make_provider(self, turns: list) -> MagicMock:
        provider = MagicMock()
        provider.name = "test"  # not "ollama" — skips ollama-only kwargs
        provider.chat_stream.side_effect = [iter(t) for t in turns]
        return provider

    def test_executes_tool_and_emits_events(self) -> None:
        """A tool call is executed, recorded with tool_name, events emitted."""
        turn1 = [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "function": {"name": "echo", "arguments": {"text": "hi"}},
                    "id": "c1",
                }],
            },
            "done": True,
        }]
        turn2 = [{"message": {"content": "all done"}, "done": True}]
        provider = self._make_provider([turn1, turn2])

        tool = _EchoTool()
        eq: queue.Queue = queue.Queue()
        messages: list = []
        _run_agent(
            provider, Config(), [tool], [], {"echo": tool},
            "say hi", eq, messages,
        )

        # The tool-result message carries tool_name (the phase-1b bug fix:
        # this path used to omit it, breaking Claude tool-result matching).
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_name"], "echo")
        self.assertIn("echoed: hi", tool_msgs[0]["content"])

        types = [e["type"] for e in _drain(eq)]
        self.assertIn("tool_call", types)
        self.assertIn("tool_result", types)
        self.assertIn("done", types)

    def test_plain_answer_no_tool(self) -> None:
        """A turn with no tool calls ends the loop with just text + done."""
        turn1 = [{"message": {"content": "hello there"}, "done": True}]
        provider = self._make_provider([turn1])

        eq: queue.Queue = queue.Queue()
        messages: list = []
        _run_agent(provider, Config(), [], [], {}, "hi", eq, messages)

        self.assertFalse(any(m.get("role") == "tool" for m in messages))
        assistant = [m for m in messages if m.get("role") == "assistant"]
        self.assertEqual(assistant[-1]["content"], "hello there")
        types = [e["type"] for e in _drain(eq)]
        self.assertIn("done", types)
        self.assertNotIn("tool_call", types)

    def test_unknown_tool_returns_error_result(self) -> None:
        """An unknown tool name yields an error tool-result, not a crash."""
        turn1 = [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "function": {"name": "nope", "arguments": {}},
                    "id": "c1",
                }],
            },
            "done": True,
        }]
        turn2 = [{"message": {"content": "ok"}, "done": True}]
        provider = self._make_provider([turn1, turn2])

        eq: queue.Queue = queue.Queue()
        messages: list = []
        _run_agent(provider, Config(), [], [], {}, "use nope", eq, messages)

        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("unknown tool", tool_msgs[0]["content"])


if __name__ == "__main__":
    unittest.main()
