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


class TestSaveSessionWithTokenUsage(unittest.TestCase):
    """Tests for SessionManager.save_session() with token_tracker parameter."""

    def _make_tracker(self, records: list[dict]) -> "TokenTracker":
        """Create a TokenTracker with pre-populated records."""
        from local_cli.token_tracker import TokenTracker, TokenUsage

        tracker = TokenTracker()
        for rec in records:
            usage = TokenUsage(
                input_tokens=rec.get("input_tokens", 0),
                output_tokens=rec.get("output_tokens", 0),
                provider=rec.get("provider", "ollama"),
            )
            tracker.record(usage)
        return tracker

    def test_token_usage_embedded_in_assistant_messages(self) -> None:
        """Assistant messages get token_usage when tracker is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            tracker = self._make_tracker([
                {"input_tokens": 100, "output_tokens": 50, "provider": "ollama"},
            ])
            mgr.save_session(messages, session_id="tok-test", token_tracker=tracker)

            loaded = mgr.load_session("tok-test")
            # The assistant message should have token_usage.
            assistant_msg = loaded[2]
            self.assertIn("token_usage", assistant_msg)
            self.assertEqual(assistant_msg["token_usage"]["input_tokens"], 100)
            self.assertEqual(assistant_msg["token_usage"]["output_tokens"], 50)
            self.assertEqual(assistant_msg["token_usage"]["total_tokens"], 150)
            self.assertEqual(assistant_msg["token_usage"]["provider"], "ollama")

    def test_system_and_user_messages_not_enriched(self) -> None:
        """Non-assistant messages are not modified by token tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            messages = [
                {"role": "system", "content": "Sys prompt"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
            tracker = self._make_tracker([
                {"input_tokens": 10, "output_tokens": 5},
            ])
            mgr.save_session(messages, session_id="no-enrich", token_tracker=tracker)

            loaded = mgr.load_session("no-enrich")
            self.assertNotIn("token_usage", loaded[0])  # system
            self.assertNotIn("token_usage", loaded[1])  # user

    def test_multiple_assistant_messages_matched_in_order(self) -> None:
        """Each assistant message gets the corresponding tracker record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            messages = [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ]
            tracker = self._make_tracker([
                {"input_tokens": 10, "output_tokens": 20, "provider": "ollama"},
                {"input_tokens": 30, "output_tokens": 40, "provider": "claude"},
            ])
            mgr.save_session(messages, session_id="multi", token_tracker=tracker)

            loaded = mgr.load_session("multi")
            self.assertEqual(loaded[1]["token_usage"]["input_tokens"], 10)
            self.assertEqual(loaded[1]["token_usage"]["provider"], "ollama")
            self.assertEqual(loaded[3]["token_usage"]["input_tokens"], 30)
            self.assertEqual(loaded[3]["token_usage"]["provider"], "claude")

    def test_more_assistant_messages_than_records(self) -> None:
        """Extra assistant messages beyond tracker records have no token_usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            messages = [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ]
            tracker = self._make_tracker([
                {"input_tokens": 5, "output_tokens": 10},
            ])
            mgr.save_session(messages, session_id="extra", token_tracker=tracker)

            loaded = mgr.load_session("extra")
            self.assertIn("token_usage", loaded[1])  # First assistant has it
            self.assertNotIn("token_usage", loaded[3])  # Second does not

    def test_no_tracker_means_no_token_usage(self) -> None:
        """Without a tracker, messages are saved without token_usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
            mgr.save_session(messages, session_id="no-tracker")

            loaded = mgr.load_session("no-tracker")
            self.assertNotIn("token_usage", loaded[1])

    def test_caller_messages_not_mutated(self) -> None:
        """The caller's message list is not modified by save_session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
            tracker = self._make_tracker([
                {"input_tokens": 100, "output_tokens": 50},
            ])

            # Save a copy to verify the original is unchanged.
            original_assistant = dict(messages[1])
            mgr.save_session(messages, session_id="no-mutate", token_tracker=tracker)

            # Original message should NOT have token_usage.
            self.assertEqual(messages[1], original_assistant)
            self.assertNotIn("token_usage", messages[1])

    def test_backward_compat_old_sessions_load_without_token_usage(self) -> None:
        """Sessions saved without token_usage load fine (backward compat)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            # Save without tracker.
            messages = [
                {"role": "user", "content": "Old format"},
                {"role": "assistant", "content": "Old response"},
            ]
            mgr.save_session(messages, session_id="old-format")

            # Load should work with all fields intact, no errors.
            loaded = mgr.load_session("old-format")
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[1]["content"], "Old response")
            self.assertNotIn("token_usage", loaded[1])

    def test_token_usage_round_trip_with_tool_messages(self) -> None:
        """Token usage is correctly embedded even with tool messages present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            messages = [
                {"role": "system", "content": "Sys"},
                {"role": "user", "content": "Read file"},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read"}}]},
                {"role": "tool", "tool_name": "read", "content": "data"},
                {"role": "assistant", "content": "Here is the file."},
                {"role": "user", "content": "Thanks"},
                {"role": "assistant", "content": "You're welcome!"},
            ]
            tracker = self._make_tracker([
                {"input_tokens": 50, "output_tokens": 10, "provider": "ollama"},
                {"input_tokens": 80, "output_tokens": 30, "provider": "ollama"},
                {"input_tokens": 90, "output_tokens": 15, "provider": "ollama"},
            ])
            mgr.save_session(messages, session_id="tool-round", token_tracker=tracker)

            loaded = mgr.load_session("tool-round")
            # 3 assistant messages, each gets a token_usage.
            self.assertEqual(loaded[2]["token_usage"]["input_tokens"], 50)
            self.assertEqual(loaded[4]["token_usage"]["input_tokens"], 80)
            self.assertEqual(loaded[6]["token_usage"]["input_tokens"], 90)

            # Non-assistant messages should not have token_usage.
            self.assertNotIn("token_usage", loaded[0])  # system
            self.assertNotIn("token_usage", loaded[1])  # user
            self.assertNotIn("token_usage", loaded[3])  # tool

    def test_empty_tracker_no_enrichment(self) -> None:
        """An empty tracker does not add token_usage to any message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager(tmpdir)
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
            tracker = self._make_tracker([])
            mgr.save_session(messages, session_id="empty-tracker", token_tracker=tracker)

            loaded = mgr.load_session("empty-tracker")
            self.assertNotIn("token_usage", loaded[1])


class TestReplContextTokenTrackerAndToolCache(unittest.TestCase):
    """Tests for _ReplContext token_tracker and tool_cache attributes."""

    def test_repl_context_has_token_tracker_slot(self) -> None:
        """_ReplContext includes token_tracker in __slots__."""
        from local_cli.cli import _ReplContext

        self.assertIn("token_tracker", _ReplContext.__slots__)

    def test_repl_context_has_tool_cache_slot(self) -> None:
        """_ReplContext includes tool_cache in __slots__."""
        from local_cli.cli import _ReplContext

        self.assertIn("tool_cache", _ReplContext.__slots__)

    def test_repl_context_defaults_to_none(self) -> None:
        """token_tracker and tool_cache default to None."""
        from unittest.mock import MagicMock

        from local_cli.cli import _ReplContext

        ctx = _ReplContext(
            config=MagicMock(),
            client=MagicMock(),
            tools=[],
            messages=[],
            session_manager=MagicMock(),
            system_prompt="",
        )
        self.assertIsNone(ctx.token_tracker)
        self.assertIsNone(ctx.tool_cache)

    def test_repl_context_accepts_token_tracker(self) -> None:
        """_ReplContext can be constructed with a token_tracker."""
        from unittest.mock import MagicMock

        from local_cli.cli import _ReplContext
        from local_cli.token_tracker import TokenTracker

        tracker = TokenTracker()
        ctx = _ReplContext(
            config=MagicMock(),
            client=MagicMock(),
            tools=[],
            messages=[],
            session_manager=MagicMock(),
            system_prompt="",
            token_tracker=tracker,
        )
        self.assertIs(ctx.token_tracker, tracker)

    def test_repl_context_accepts_tool_cache(self) -> None:
        """_ReplContext can be constructed with a tool_cache."""
        from unittest.mock import MagicMock

        from local_cli.cli import _ReplContext
        from local_cli.tool_cache import ToolCache

        cache = ToolCache()
        ctx = _ReplContext(
            config=MagicMock(),
            client=MagicMock(),
            tools=[],
            messages=[],
            session_manager=MagicMock(),
            system_prompt="",
            tool_cache=cache,
        )
        self.assertIs(ctx.tool_cache, cache)


if __name__ == "__main__":
    unittest.main()
