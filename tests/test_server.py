"""Tests for local_cli.server — the JSON-line server's chat handler.

JsonLineServer.__init__ builds an Ollama client, a provider and several
managers, so these tests construct the instance via __new__ and inject only
the attributes _handle_chat touches.  _send is patched to capture the
JSON-line events instead of writing to stdout.
"""

import threading
import unittest
from unittest.mock import MagicMock, patch

from local_cli.config import Config
from local_cli.session_log import SessionLogger
from local_cli.server import JsonLineServer
from local_cli.token_tracker import TokenTracker
from local_cli.tool_cache import ToolCache
from local_cli.tools.base import Tool


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


def _make_server(provider: MagicMock, tools: list) -> JsonLineServer:
    """Build a JsonLineServer with only the attributes _handle_chat needs."""
    server = JsonLineServer.__new__(JsonLineServer)
    server._config = Config()
    server._provider = provider
    server._tools = tools
    server._tool_map = {t.name: t for t in tools}
    server._tool_defs = []
    server._messages = [{"role": "system", "content": "sys"}]
    server._stop_flag = threading.Event()
    server._pending_switch = None
    server._tool_cache = ToolCache()
    server._token_tracker = TokenTracker()
    server._ideation_active = False
    server._skills_loader = None
    server._session_log = SessionLogger(".", enabled=False)
    server._instruction_source = None
    server._instruction_message = None
    return server


class TestHandleChat(unittest.TestCase):
    """Tests for JsonLineServer._handle_chat."""

    def _provider(self, turns: list) -> MagicMock:
        provider = MagicMock()
        provider.name = "test"  # not "ollama" — skips ollama-only kwargs
        provider.chat_stream.side_effect = [iter(t) for t in turns]
        return provider

    def test_chat_executes_tool_with_tool_name(self) -> None:
        """A tool call runs and its result message carries tool_name."""
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
        turn2 = [{"message": {"content": "done"}, "done": True}]
        tool = _EchoTool()
        server = _make_server(self._provider([turn1, turn2]), [tool])

        sent: list = []
        with patch("local_cli.server._send", side_effect=sent.append):
            server._handle_chat(1, "say hi")

        tool_msgs = [m for m in server._messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_name"], "echo")
        self.assertIn("echoed: hi", tool_msgs[0]["content"])

        types = [o.get("type") for o in sent]
        self.assertIn("tool_call", types)
        self.assertIn("tool_result", types)
        self.assertIn("done", types)

    def test_empty_message_returns_error(self) -> None:
        """An empty/whitespace message produces an error event, no call."""
        server = _make_server(self._provider([]), [])
        sent: list = []
        with patch("local_cli.server._send", side_effect=sent.append):
            server._handle_chat(1, "   ")
        self.assertTrue(any(o.get("type") == "error" for o in sent))

    def test_plain_answer_records_assistant(self) -> None:
        """A no-tool turn streams text, records it, and emits done."""
        turn1 = [{"message": {"content": "hi there"}, "done": True}]
        server = _make_server(self._provider([turn1]), [])
        sent: list = []
        with patch("local_cli.server._send", side_effect=sent.append):
            server._handle_chat(1, "hello")

        types = [o.get("type") for o in sent]
        self.assertIn("done", types)
        self.assertNotIn("tool_call", types)
        self.assertTrue(
            any(m.get("role") == "assistant" for m in server._messages)
        )

    def test_matching_skills_injected_before_user_message(self) -> None:
        """Server chat injects matching skills like the CLI does."""

        class _Skill:
            name = "deploy-guide"
            content = "Always run the smoke test first."

        class _Loader:
            def get_matching_skills(self, text):
                return [_Skill()] if "deploy" in text else []

        turn1 = [{"message": {"content": "done"}, "done": True}]
        server = _make_server(self._provider([turn1]), [])
        server._skills_loader = _Loader()

        with patch("local_cli.server._send"):
            server._handle_chat(1, "deploy the app")

        roles = [m.get("role") for m in server._messages]
        skill_idx = next(
            i for i, m in enumerate(server._messages)
            if "deploy-guide" in m.get("content", "")
        )
        user_idx = next(
            i for i, m in enumerate(server._messages)
            if m.get("role") == "user"
        )
        self.assertEqual(server._messages[skill_idx]["role"], "system")
        self.assertLess(skill_idx, user_idx)
        self.assertIn("smoke test", server._messages[skill_idx]["content"])

    def test_nudges_on_code_only_build_answer(self) -> None:
        """A code-only answer to a build request triggers one nudge."""
        turn1 = [{"message": {"content": "```python\nprint(1)\n```"}, "done": True}]
        turn2 = [{"message": {"content": "no file needed"}, "done": True}]
        server = _make_server(self._provider([turn1, turn2]), [])
        sent: list = []
        with patch("local_cli.server._send", side_effect=sent.append):
            server._handle_chat(1, "create a script")

        nudges = [
            m for m in server._messages
            if m.get("role") == "user" and "did not create" in m.get("content", "")
        ]
        self.assertEqual(len(nudges), 1)
        self.assertEqual(server._provider.chat_stream.call_count, 2)


if __name__ == "__main__":
    unittest.main()
