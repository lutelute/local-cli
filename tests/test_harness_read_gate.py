"""Tests for the read-before-edit gate.

Small models edit files they have never looked at; the invented
``old_text`` never matches and the edit fails (or a fuzzy variant would
corrupt the file).  The gate defers the first blind edit of an existing
file so the model reads it and copies exact text — a file the model has
read (or itself wrote) passes straight through.
"""

import tempfile
import unittest
from pathlib import Path
from typing import Any

from local_cli.agent import run_agent
from local_cli.harness import (
    HarnessConfig,
    files_known_to_conversation,
)
from local_cli.tools.base import Tool
from tests.test_run_agent import (
    _FileWriteTool,
    _ScriptedClient,
    _call,
    _kinds,
    _recorder,
    _turn,
)


class _ReadTool(Tool):
    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        return "Read a file."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs: object) -> str:
        return Path(str(kwargs["file_path"])).read_text(encoding="utf-8")


class _EditTool(Tool):
    def __init__(self) -> None:
        self.executed: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "edit"

    @property
    def description(self) -> str:
        return "Replace text in a file."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs: object) -> str:
        self.executed.append(dict(kwargs))
        path = Path(str(kwargs["file_path"]))
        text = path.read_text(encoding="utf-8")
        old = str(kwargs.get("old_text", ""))
        if old not in text:
            return "Error: old_text not found"
        path.write_text(
            text.replace(old, str(kwargs.get("new_text", "")), 1),
            encoding="utf-8",
        )
        return "1 occurrence replaced"


def _tools() -> tuple[_ReadTool, _EditTool, _FileWriteTool]:
    return _ReadTool(), _EditTool(), _FileWriteTool()


class TestReadBeforeEditGate(unittest.TestCase):
    def _existing(self, tmp: str) -> str:
        path = Path(tmp) / "app.py"
        path.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        return str(path)

    def test_blind_edit_deferred_then_read_then_edit_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._existing(tmp)
            read, edit, write = _tools()
            client = _ScriptedClient([
                _turn("", [_call("edit", {
                    "file_path": path,
                    "old_text": "return a - b",
                    "new_text": "return a + b",
                })]),
                _turn("", [_call("read", {"file_path": path})]),
                _turn("", [_call("edit", {
                    "file_path": path,
                    "old_text": "return a - b",
                    "new_text": "return a + b",
                })]),
                _turn("fixed."),
            ])
            events, emit = _recorder()
            messages = [{"role": "user", "content": "fix add()"}]
            result = run_agent(
                client, "m", [read, edit, write], messages, emit=emit,
            )

            self.assertEqual(result, "fixed.")
            self.assertIn("read_gate", _kinds(events))
            # The first (blind) edit never executed; the post-read one did.
            self.assertEqual(len(edit.executed), 1)
            self.assertIn(
                "return a + b",
                Path(path).read_text(encoding="utf-8"),
            )

    def test_edit_after_read_same_turn_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._existing(tmp)
            read, edit, write = _tools()
            client = _ScriptedClient([
                _turn("", [
                    _call("read", {"file_path": path}),
                    _call("edit", {
                        "file_path": path,
                        "old_text": "return a - b",
                        "new_text": "return a + b",
                    }),
                ]),
                _turn("done."),
            ])
            events, emit = _recorder()
            run_agent(
                client, "m", [read, edit, write],
                [{"role": "user", "content": "fix"}], emit=emit,
            )
            self.assertNotIn("read_gate", _kinds(events))
            self.assertEqual(len(edit.executed), 1)

    def test_file_known_from_history_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._existing(tmp)
            read, edit, write = _tools()
            history = [
                {"role": "user", "content": "look at app.py"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {
                        "name": "read",
                        "arguments": {"file_path": path},
                    }},
                ]},
                {"role": "tool", "content": "def add...",
                 "tool_name": "read"},
                {"role": "assistant", "content": "it subtracts."},
                {"role": "user", "content": "fix it"},
            ]
            client = _ScriptedClient([
                _turn("", [_call("edit", {
                    "file_path": path,
                    "old_text": "return a - b",
                    "new_text": "return a + b",
                })]),
                _turn("done."),
            ])
            events, emit = _recorder()
            run_agent(client, "m", [read, edit, write], history, emit=emit)
            self.assertNotIn("read_gate", _kinds(events))
            self.assertEqual(len(edit.executed), 1)

    def test_second_blind_edit_passes(self) -> None:
        """One deferral per file — an insistent model is not looped."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._existing(tmp)
            read, edit, write = _tools()
            client = _ScriptedClient([
                _turn("", [_call("edit", {
                    "file_path": path, "old_text": "x", "new_text": "y",
                })]),
                _turn("", [_call("edit", {
                    "file_path": path, "old_text": "x", "new_text": "y",
                })]),
                _turn("gave up."),
            ])
            events, emit = _recorder()
            run_agent(
                client, "m", [read, edit, write],
                [{"role": "user", "content": "edit"}], emit=emit,
            )
            self.assertEqual(_kinds(events).count("read_gate"), 1)
            self.assertEqual(len(edit.executed), 1)

    def test_missing_file_not_gated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ghost = str(Path(tmp) / "ghost.py")
            read, edit, write = _tools()
            client = _ScriptedClient([
                _turn("", [_call("edit", {
                    "file_path": ghost, "old_text": "a", "new_text": "b",
                })]),
                _turn("no such file."),
            ])
            events, emit = _recorder()
            run_agent(
                client, "m", [read, edit, write],
                [{"role": "user", "content": "edit"}], emit=emit,
            )
            self.assertNotIn("read_gate", _kinds(events))
            self.assertEqual(len(edit.executed), 1)

    def test_disabled_by_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._existing(tmp)
            read, edit, write = _tools()
            client = _ScriptedClient([
                _turn("", [_call("edit", {
                    "file_path": path,
                    "old_text": "return a - b",
                    "new_text": "return a + b",
                })]),
                _turn("done."),
            ])
            events, emit = _recorder()
            run_agent(
                client, "m", [read, edit, write],
                [{"role": "user", "content": "edit"}], emit=emit,
                harness=HarnessConfig(read_before_edit=False),
            )
            self.assertNotIn("read_gate", _kinds(events))
            self.assertEqual(len(edit.executed), 1)


class TestFilesKnownToConversation(unittest.TestCase):
    def test_collects_paths_from_history(self) -> None:
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "read",
                              "arguments": {"file_path": "/a/one.py"}}},
                {"function": {"name": "read_file",
                              "arguments": '{"path": "/a/two.py"}'}},
                {"function": {"name": "grep",
                              "arguments": {"pattern": "x"}}},
            ]},
        ]
        known = files_known_to_conversation(messages)
        self.assertIn("/a/one.py", known)
        self.assertIn("/a/two.py", known)
        self.assertEqual(len(known), 2)

    def test_ignores_malformed_entries(self) -> None:
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                "not a dict",
                {"function": {"name": "edit", "arguments": "{broken"}},
                {"function": {"name": "edit", "arguments": {"file_path": 3}}},
            ]},
        ]
        self.assertEqual(files_known_to_conversation(messages), set())


if __name__ == "__main__":
    unittest.main()
