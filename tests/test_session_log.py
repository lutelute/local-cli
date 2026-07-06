"""Tests for the session flight recorder (session_log.py).

Field gap (desktop, 0.12.x): sessions left no trace on disk, so a
failure report ("it did nothing") arrived with no transcript to
diagnose.  The SessionLogger writes one JSONL line per event under
``<state_dir>/projects/<cwd-slug>/`` from the moment a session starts,
fails open on any write error, and closes each turn with the
layer-diagnosis counters (visible/thinking chars, tool calls).
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_cli.agent import run_agent
from local_cli.harness import AgentEvent
from local_cli.session_log import (
    _CLIP_CHARS,
    SessionLogger,
    _clip,
    project_slug,
    session_log_enabled,
)
from tests.test_run_agent import _DummyTool, _ScriptedClient, _call, _turn


def _lines(logger: SessionLogger) -> list[dict]:
    text = logger.path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestProjectSlug(unittest.TestCase):
    def test_path_maps_to_dashes(self) -> None:
        self.assertEqual(project_slug("/Users/x/proj"), "-Users-x-proj")

    def test_special_chars_collapse(self) -> None:
        self.assertEqual(
            project_slug("/tmp/my proj (v2)"), "-tmp-my-proj-v2",
        )

    def test_overlong_keeps_tail(self) -> None:
        slug = project_slug("/x" * 120)
        self.assertLessEqual(len(slug), 150)
        self.assertTrue(slug.endswith("x"))

    def test_empty_path(self) -> None:
        self.assertEqual(project_slug(""), "-")


class TestEnabledEnv(unittest.TestCase):
    def test_default_on(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("LOCAL_CLI_SESSION_LOG", None)
            self.assertTrue(session_log_enabled())

    def test_disable_values(self) -> None:
        for value in ("0", "false", "OFF", "no"):
            with patch.dict(
                os.environ, {"LOCAL_CLI_SESSION_LOG": value},
            ):
                self.assertFalse(session_log_enabled())

    def test_one_is_on(self) -> None:
        with patch.dict(os.environ, {"LOCAL_CLI_SESSION_LOG": "1"}):
            self.assertTrue(session_log_enabled())


class TestSessionLoggerWrites(unittest.TestCase):
    def test_lines_land_in_project_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / "myproj"
            proj.mkdir()
            logger = SessionLogger(tmp, cwd=str(proj), enabled=True)
            logger.log_session_start(model="m1", frontend="cli")
            logger.log_user("こんにちは")

            self.assertIn("projects", str(logger.path))
            self.assertIn(project_slug(str(proj.resolve())), str(logger.path))
            records = _lines(logger)
            self.assertEqual(
                [r["type"] for r in records], ["session_start", "user"],
            )
            self.assertEqual(records[0]["model"], "m1")
            self.assertEqual(records[1]["content"], "こんにちは")
            for record in records:
                self.assertIn("ts", record)

    def test_disabled_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = SessionLogger(tmp, cwd=tmp, enabled=False)
            logger.log_user("hi")
            self.assertFalse(logger.path.exists())

    def test_env_kill_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"LOCAL_CLI_SESSION_LOG": "0"}):
                logger = SessionLogger(tmp, cwd=tmp)
            self.assertFalse(logger.enabled)

    def test_fail_open_on_unwritable_state_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "blocker"
            blocker.write_text("a file, not a dir", encoding="utf-8")
            logger = SessionLogger(
                str(blocker / "sub"), cwd=tmp, enabled=True,
            )
            logger.log_user("never raises")  # mkdir fails inside
            self.assertFalse(logger.enabled)
            logger.log_user("still silent")  # broken logger stays silent

    def test_clip_marker(self) -> None:
        clipped = _clip("a" * (_CLIP_CHARS + 500))
        self.assertLess(len(clipped), _CLIP_CHARS + 100)
        self.assertIn("truncated 500 chars", clipped)

    def test_rotate_starts_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = SessionLogger(tmp, cwd=tmp, enabled=True)
            logger.log("cleared")
            first_path = logger.path
            first_id = logger.session_id
            logger.rotate()
            logger.log_session_start(reason="clear")
            self.assertNotEqual(logger.session_id, first_id)
            self.assertNotEqual(logger.path, first_path)
            self.assertTrue(first_path.exists())
            self.assertTrue(logger.path.exists())


class TestEmitMapping(unittest.TestCase):
    def _logger(self, tmp: str) -> SessionLogger:
        return SessionLogger(tmp, cwd=tmp, enabled=True)

    def test_deltas_counted_not_stored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(tmp)
            logger.emit(AgentEvent("content_delta", {"text": "abc"}))
            logger.emit(AgentEvent("content_delta", {"text": "def"}))
            logger.emit(AgentEvent("thinking_delta", {"text": "xy"}))
            logger.emit(AgentEvent("tool_start", {
                "tool_name": "dummy", "arguments": {"arg": "v"},
            }))
            logger.emit(AgentEvent("tool_result", {
                "tool_name": "dummy", "result": "R",
            }))
            logger.emit(AgentEvent("deliverable_nudge", {}))
            logger.log_turn_end()

            records = _lines(logger)
            types = [r["type"] for r in records]
            self.assertEqual(
                types, ["tool_start", "tool_result", "harness", "turn_end"],
            )
            self.assertEqual(records[2]["event"], "deliverable_nudge")
            turn = records[-1]
            self.assertEqual(turn["visible_chars"], 6)
            self.assertEqual(turn["thinking_chars"], 2)
            self.assertEqual(turn["tool_calls"], 1)

    def test_turn_end_resets_counters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(tmp)
            logger.emit(AgentEvent("content_delta", {"text": "abc"}))
            logger.log_turn_end()
            logger.log_turn_end()
            records = _lines(logger)
            self.assertEqual(records[0]["visible_chars"], 3)
            self.assertEqual(records[1]["visible_chars"], 0)

    def test_assistant_message_logged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(tmp)
            logger.emit(AgentEvent("assistant_message", {
                "message": {"role": "assistant", "content": "hello"},
                "thinking": "tt",
            }))
            (record,) = _lines(logger)
            self.assertEqual(record["type"], "assistant")
            self.assertEqual(record["content"], "hello")
            self.assertEqual(record["thinking_chars"], 2)

    def test_error_event_logged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(tmp)
            logger.emit(AgentEvent("error", {
                "source": "request", "detail": "boom",
            }))
            (record,) = _lines(logger)
            self.assertEqual(record["type"], "error")
            self.assertEqual(record["message"], "boom")

    def test_tool_output_clipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(tmp)
            logger.emit(AgentEvent("tool_result", {
                "tool_name": "bash", "result": "x" * (_CLIP_CHARS + 999),
            }))
            (record,) = _lines(logger)
            self.assertIn("truncated 999 chars", record["output"])

    def test_wrap_emit_tees(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._logger(tmp)
            seen: list[str] = []
            tee = logger.wrap_emit(lambda ev: seen.append(ev.kind))
            tee(AgentEvent("tool_start", {
                "tool_name": "dummy", "arguments": {},
            }))
            self.assertEqual(seen, ["tool_start"])
            (record,) = _lines(logger)
            self.assertEqual(record["type"], "tool_start")


class TestRunAgentIntegration(unittest.TestCase):
    """The tee records a real run_agent turn end to end."""

    def test_full_turn_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = SessionLogger(tmp, cwd=tmp, enabled=True)
            client = _ScriptedClient([
                _turn("", [_call("dummy", {"arg": "go"})]),
                _turn("done!"),
            ])
            messages = [{"role": "user", "content": "do the thing"}]
            logger.log_user("do the thing")
            run_agent(
                client, "m", [_DummyTool()], messages,
                emit=logger.wrap_emit(lambda ev: None),
            )
            logger.log_turn_end()

            records = _lines(logger)
            types = [r["type"] for r in records]
            self.assertEqual(types[0], "user")
            self.assertIn("llm_start", types)
            self.assertIn("tool_start", types)
            self.assertIn("tool_result", types)
            self.assertIn("assistant", types)
            turn = records[-1]
            self.assertEqual(turn["type"], "turn_end")
            self.assertEqual(turn["tool_calls"], 1)
            self.assertEqual(turn["visible_chars"], len("done!"))
            self.assertFalse(turn["error"])


if __name__ == "__main__":
    unittest.main()
