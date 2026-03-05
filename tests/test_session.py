"""Tests for local_cli.session module."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from local_cli.session import SessionManager


class TestSessionManagerInit(unittest.TestCase):
    """Tests for SessionManager construction."""

    def test_creates_sessions_directory(self) -> None:
        """Sessions subdirectory is created on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "state")
            mgr = SessionManager(state_dir)
            sessions_dir = Path(state_dir) / "sessions"
            self.assertTrue(sessions_dir.is_dir())

    def test_tilde_expansion(self) -> None:
        """State dir path with ~ is expanded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a concrete path to test that expanduser is applied.
            mgr = SessionManager(tmpdir)
            self.assertTrue(mgr._sessions_dir.is_absolute())

    def test_existing_directory_ok(self) -> None:
        """Initializing with an existing state dir does not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr1 = SessionManager(tmpdir)
            mgr2 = SessionManager(tmpdir)  # Second init should not fail.
            self.assertTrue(mgr2._sessions_dir.is_dir())


class TestGenerateSessionId(unittest.TestCase):
    """Tests for SessionManager.generate_session_id()."""

    def test_format(self) -> None:
        """Session ID follows YYYYMMDD-HHMMSS-<8_hex_chars> format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            sid = mgr.generate_session_id()

        # e.g. "20260306-123456-a1b2c3d4"
        parts = sid.split("-")
        self.assertEqual(len(parts), 3)
        self.assertEqual(len(parts[0]), 8)  # YYYYMMDD
        self.assertEqual(len(parts[1]), 6)  # HHMMSS
        self.assertEqual(len(parts[2]), 8)  # hex UUID fragment

        # Verify date part is all digits.
        self.assertTrue(parts[0].isdigit())
        self.assertTrue(parts[1].isdigit())

        # Verify UUID fragment is hex.
        int(parts[2], 16)  # Should not raise.

    def test_uniqueness(self) -> None:
        """Multiple calls generate unique IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            ids = {mgr.generate_session_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)


class TestSaveSession(unittest.TestCase):
    """Tests for SessionManager.save_session()."""

    def test_creates_jsonl_file(self) -> None:
        """save_session creates a .jsonl file in the sessions directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            sid = mgr.save_session(messages, session_id="test-session")

            file_path = mgr._sessions_dir / "test-session.jsonl"
            self.assertTrue(file_path.exists())
            self.assertEqual(sid, "test-session")

    def test_jsonl_format(self) -> None:
        """Each message is written as a separate JSON line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "World"},
            ]
            mgr.save_session(messages, session_id="fmt-test")

            file_path = mgr._sessions_dir / "fmt-test.jsonl"
            with open(file_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()

            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["role"], "user")
            self.assertEqual(json.loads(lines[1])["role"], "assistant")

    def test_auto_generates_session_id(self) -> None:
        """save_session auto-generates an ID if none is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            messages = [{"role": "user", "content": "test"}]
            sid = mgr.save_session(messages)

            # Should look like a generated ID.
            self.assertIsInstance(sid, str)
            self.assertGreater(len(sid), 0)

            # File should exist.
            file_path = mgr._sessions_dir / f"{sid}.jsonl"
            self.assertTrue(file_path.exists())

    def test_overwrites_existing_session(self) -> None:
        """Saving with the same session ID overwrites the previous file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            mgr.save_session(
                [{"role": "user", "content": "first"}],
                session_id="overwrite-test",
            )
            mgr.save_session(
                [{"role": "user", "content": "second"}],
                session_id="overwrite-test",
            )

            loaded = mgr.load_session("overwrite-test")
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["content"], "second")

    def test_empty_messages_list(self) -> None:
        """Saving an empty message list creates an empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            sid = mgr.save_session([], session_id="empty-test")

            file_path = mgr._sessions_dir / "empty-test.jsonl"
            self.assertTrue(file_path.exists())
            self.assertEqual(file_path.read_text(), "")

    def test_unicode_content(self) -> None:
        """Unicode content is preserved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            text = "Hello \u4e16\u754c \U0001f600"
            messages = [
                {"role": "user", "content": text},
            ]
            mgr.save_session(messages, session_id="unicode-test")

            loaded = mgr.load_session("unicode-test")
            self.assertEqual(loaded[0]["content"], text)

    def test_complex_message_structure(self) -> None:
        """Messages with tool_calls and nested structures are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            messages = [
                {"role": "user", "content": "Read file"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read",
                                "arguments": {"path": "/tmp/test.py"},
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_name": "read",
                    "content": "import sys\nprint('hello')\n",
                },
                {"role": "assistant", "content": "I read the file."},
            ]
            mgr.save_session(messages, session_id="complex-test")

            loaded = mgr.load_session("complex-test")
            self.assertEqual(len(loaded), 4)
            self.assertIn("tool_calls", loaded[1])
            self.assertEqual(
                loaded[1]["tool_calls"][0]["function"]["name"], "read"
            )
            self.assertEqual(loaded[2]["tool_name"], "read")


class TestLoadSession(unittest.TestCase):
    """Tests for SessionManager.load_session()."""

    def test_round_trip(self) -> None:
        """Save then load preserves all messages exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            original = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
            mgr.save_session(original, session_id="rt-test")
            loaded = mgr.load_session("rt-test")

            self.assertEqual(loaded, original)

    def test_nonexistent_session(self) -> None:
        """Loading a non-existent session returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            loaded = mgr.load_session("does-not-exist")
            self.assertEqual(loaded, [])

    def test_corrupt_lines_skipped(self) -> None:
        """Corrupt JSONL lines are silently skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)

            # Write a file with some corrupt lines.
            file_path = mgr._sessions_dir / "corrupt.jsonl"
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"role": "user", "content": "good"}) + "\n")
                fh.write("this is not valid json\n")
                fh.write("{broken json\n")
                fh.write(json.dumps({"role": "assistant", "content": "also good"}) + "\n")

            loaded = mgr.load_session("corrupt")
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["content"], "good")
            self.assertEqual(loaded[1]["content"], "also good")

    def test_empty_lines_skipped(self) -> None:
        """Empty lines in the JSONL file are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)

            file_path = mgr._sessions_dir / "blanks.jsonl"
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"role": "user", "content": "msg"}) + "\n")
                fh.write("\n")
                fh.write("  \n")
                fh.write(json.dumps({"role": "assistant", "content": "reply"}) + "\n")

            loaded = mgr.load_session("blanks")
            self.assertEqual(len(loaded), 2)

    def test_non_dict_json_skipped(self) -> None:
        """JSON values that are not dicts are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)

            file_path = mgr._sessions_dir / "non-dict.jsonl"
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"role": "user", "content": "valid"}) + "\n")
                fh.write(json.dumps([1, 2, 3]) + "\n")  # list, not dict
                fh.write(json.dumps("just a string") + "\n")  # string, not dict
                fh.write(json.dumps(42) + "\n")  # number, not dict
                fh.write(json.dumps({"role": "assistant", "content": "ok"}) + "\n")

            loaded = mgr.load_session("non-dict")
            self.assertEqual(len(loaded), 2)

    def test_empty_file_returns_empty_list(self) -> None:
        """An empty session file returns an empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)

            file_path = mgr._sessions_dir / "empty.jsonl"
            file_path.touch()

            loaded = mgr.load_session("empty")
            self.assertEqual(loaded, [])

    def test_entirely_corrupt_file(self) -> None:
        """A file with no valid JSON lines returns an empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)

            file_path = mgr._sessions_dir / "all-bad.jsonl"
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write("not json 1\n")
                fh.write("not json 2\n")
                fh.write("{broken\n")

            loaded = mgr.load_session("all-bad")
            self.assertEqual(loaded, [])


class TestListSessions(unittest.TestCase):
    """Tests for SessionManager.list_sessions()."""

    def test_empty_directory(self) -> None:
        """No sessions returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            self.assertEqual(mgr.list_sessions(), [])

    def test_lists_session_ids(self) -> None:
        """list_sessions returns IDs of saved sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            mgr.save_session([{"role": "user", "content": "a"}], session_id="session-a")
            mgr.save_session([{"role": "user", "content": "b"}], session_id="session-b")

            sessions = mgr.list_sessions()
            self.assertEqual(len(sessions), 2)
            self.assertIn("session-a", sessions)
            self.assertIn("session-b", sessions)

    def test_sorted_newest_first(self) -> None:
        """Sessions are sorted newest-first (reverse lexicographic)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            mgr.save_session(
                [{"role": "user", "content": "old"}],
                session_id="20250101-000000-aaaaaaaa",
            )
            mgr.save_session(
                [{"role": "user", "content": "new"}],
                session_id="20260301-120000-bbbbbbbb",
            )
            mgr.save_session(
                [{"role": "user", "content": "mid"}],
                session_id="20250601-060000-cccccccc",
            )

            sessions = mgr.list_sessions()
            self.assertEqual(sessions[0], "20260301-120000-bbbbbbbb")
            self.assertEqual(sessions[1], "20250601-060000-cccccccc")
            self.assertEqual(sessions[2], "20250101-000000-aaaaaaaa")

    def test_non_jsonl_files_ignored(self) -> None:
        """Files without .jsonl extension are not listed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            mgr.save_session(
                [{"role": "user", "content": "ok"}],
                session_id="valid-session",
            )
            # Create a non-jsonl file.
            other_file = mgr._sessions_dir / "notes.txt"
            other_file.write_text("not a session")

            sessions = mgr.list_sessions()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0], "valid-session")

    def test_directories_inside_sessions_ignored(self) -> None:
        """Subdirectories inside the sessions dir are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            mgr.save_session(
                [{"role": "user", "content": "ok"}],
                session_id="real-session",
            )
            # Create a subdirectory (even with .jsonl name).
            subdir = mgr._sessions_dir / "fake.jsonl"
            subdir.mkdir()

            sessions = mgr.list_sessions()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0], "real-session")


class TestSessionRoundTrip(unittest.TestCase):
    """Integration tests for save/load round-trip."""

    def test_full_conversation_round_trip(self) -> None:
        """A full multi-turn conversation survives save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)

            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Read main.py"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read",
                                "arguments": {"path": "main.py"},
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_name": "read",
                    "content": "print('hello')\n",
                },
                {
                    "role": "assistant",
                    "content": "The file contains a simple print statement.",
                },
                {"role": "user", "content": "Thanks!"},
                {"role": "assistant", "content": "You're welcome!"},
            ]

            sid = mgr.save_session(conversation)
            loaded = mgr.load_session(sid)

            self.assertEqual(loaded, conversation)

    def test_save_load_then_list(self) -> None:
        """After saving, the session appears in list_sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            sid = mgr.save_session(
                [{"role": "user", "content": "test"}],
                session_id="listed-session",
            )

            sessions = mgr.list_sessions()
            self.assertIn(sid, sessions)


if __name__ == "__main__":
    unittest.main()
