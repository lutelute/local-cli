"""Tests for the last-conversation autosave store (conversation_store.py).

Quitting the app used to lose the whole conversation.  The store saves
the message history after every turn and restores it on demand (CLI
/resume, desktop restore button, server "resume" request).
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_cli.conversation_store import _MAX_MESSAGES, ConversationStore
from local_cli.server import JsonLineServer
from tests.test_server import _make_server


class TestSaveLoad(unittest.TestCase):
    def test_roundtrip_excludes_system(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(tmp, cwd=tmp)
            store.save([
                {"role": "system", "content": "sys prompt"},
                {"role": "user", "content": "こんにちは"},
                {"role": "assistant", "content": "hi",
                 "tool_calls": [{"function": {"name": "read"}}]},
                {"role": "tool", "content": "data", "tool_name": "read"},
            ])
            loaded = store.load()
            self.assertEqual(len(loaded), 3)
            self.assertEqual(
                [m["role"] for m in loaded], ["user", "assistant", "tool"],
            )
            self.assertEqual(loaded[0]["content"], "こんにちは")
            self.assertIn("tool_calls", loaded[1])
            self.assertEqual(loaded[2]["tool_name"], "read")

    def test_second_save_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(tmp, cwd=tmp)
            store.save([{"role": "user", "content": "one"}])
            store.save([{"role": "user", "content": "two"}])
            loaded = store.load()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["content"], "two")

    def test_load_absent_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(tmp, cwd=tmp)
            self.assertEqual(store.load(), [])

    def test_corrupt_lines_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(tmp, cwd=tmp)
            store.save([{"role": "user", "content": "keep"}])
            with open(store.path, "a", encoding="utf-8") as fh:
                fh.write("{broken json\n")
            loaded = store.load()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["content"], "keep")

    def test_caps_to_newest_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(tmp, cwd=tmp)
            many = [
                {"role": "user", "content": f"m{i}"}
                for i in range(_MAX_MESSAGES + 50)
            ]
            store.save(many)
            loaded = store.load()
            self.assertEqual(len(loaded), _MAX_MESSAGES)
            self.assertEqual(loaded[-1]["content"], f"m{_MAX_MESSAGES + 49}")

    def test_info_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(tmp, cwd=tmp)
            self.assertIsNone(store.info())
            store.save([
                {"role": "user", "content": "レポートを書いて"},
                {"role": "assistant", "content": "done"},
            ])
            info = store.info()
            self.assertEqual(info["count"], 2)
            self.assertEqual(info["preview"], "レポートを書いて")
            self.assertTrue(info["saved_at"])

    def test_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(tmp, cwd=tmp)
            store.save([{"role": "user", "content": "x"}])
            store.clear()
            self.assertEqual(store.load(), [])
            store.clear()  # missing file: must not raise

    def test_disabled_store_inert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ConversationStore(tmp, cwd=tmp, enabled=False)
            store.save([{"role": "user", "content": "x"}])
            self.assertFalse(store.path.exists())
            self.assertEqual(store.load(), [])

    def test_fail_open_on_unwritable_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "blocker"
            blocker.write_text("file", encoding="utf-8")
            store = ConversationStore(str(blocker / "sub"), cwd=tmp)
            store.save([{"role": "user", "content": "x"}])  # must not raise
            self.assertEqual(store.load(), [])


class TestServerResume(unittest.TestCase):
    """The server "resume" request rebuilds history and replays it."""

    def _server(self, tmp: str) -> JsonLineServer:
        server = _make_server(provider=None, tools=[])
        server._conversation_store = ConversationStore(tmp, cwd=tmp)
        server._system_prompt = "sys"
        return server

    def test_resume_restores_messages_and_replies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = self._server(tmp)
            server._conversation_store.save([
                {"role": "user", "content": "前回の質問"},
                {"role": "assistant", "content": "前回の答え"},
                {"role": "tool", "content": "internal", "tool_name": "read"},
            ])
            sent: list[dict] = []
            with patch("local_cli.server._send", side_effect=sent.append), \
                 patch("local_cli.server.project_map_message",
                       return_value=None):
                server._handle_resume(7)

            self.assertEqual(
                [m["role"] for m in server._messages],
                ["system", "user", "assistant", "tool"],
            )
            (reply,) = sent
            self.assertEqual(reply["type"], "restored")
            self.assertEqual(reply["count"], 3)
            # Display replay: user/assistant text only, no tool messages.
            self.assertEqual(
                [m["role"] for m in reply["messages"]],
                ["user", "assistant"],
            )

    def test_resume_reinjects_instruction_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = self._server(tmp)
            server._instruction_message = {
                "role": "system", "content": "PROJECT INSTRUCTIONS",
            }
            server._conversation_store.save(
                [{"role": "user", "content": "q"}],
            )
            with patch("local_cli.server._send"), \
                 patch("local_cli.server.project_map_message",
                       return_value=None):
                server._handle_resume(1)
            self.assertEqual(server._messages[1]["content"],
                             "PROJECT INSTRUCTIONS")

    def test_resume_without_saved_conversation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = self._server(tmp)
            sent: list[dict] = []
            with patch("local_cli.server._send", side_effect=sent.append):
                server._handle_resume(3)
            (reply,) = sent
            self.assertEqual(reply["type"], "error")


if __name__ == "__main__":
    unittest.main()
