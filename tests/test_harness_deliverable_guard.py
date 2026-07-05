"""Tests for the file-deliverable completion guard.

Field failure (desktop app, 0.12.0): "このフォルダを読んで報告書をmdで
作成して" — the model read the files, printed the whole report as chat
prose, and finished without writing any file.  The code-print nudge never
fired because a prose report has no code fence.  The deliverable guard
catches the shape directly: the request named a document, nothing was
written this turn, and the model is about to finish → one push-back.
"""

import unittest

from local_cli.agent import run_agent
from local_cli.harness import HarnessConfig, mentions_file_deliverable
from tests.test_run_agent import (
    _DummyTool,
    _FileWriteTool,
    _ScriptedClient,
    _call,
    _kinds,
    _recorder,
    _turn,
)


class TestMentionsFileDeliverable(unittest.TestCase):
    """Detector for document/file deliverables in the user request."""

    def test_japanese_report_words(self) -> None:
        self.assertTrue(mentions_file_deliverable("フォルダを読んで報告書を書いて"))
        self.assertTrue(mentions_file_deliverable("内容のレポートを作成して"))
        self.assertTrue(mentions_file_deliverable("仕様書にまとめて"))

    def test_english_report_words(self) -> None:
        self.assertTrue(mentions_file_deliverable("read the folder and write a report"))
        self.assertTrue(mentions_file_deliverable("produce a summary of the code"))

    def test_explicit_extension(self) -> None:
        self.assertTrue(mentions_file_deliverable("結果を notes.md に保存して"))
        self.assertTrue(mentions_file_deliverable("dump it into out.csv"))

    def test_plain_questions_not_matched(self) -> None:
        self.assertFalse(mentions_file_deliverable("このコードを説明して"))
        self.assertFalse(mentions_file_deliverable("what does app.py do?"))
        self.assertFalse(mentions_file_deliverable("fix the bug in add()"))


class TestDeliverableGuard(unittest.TestCase):
    """Finishing without the requested file draws one push-back."""

    _PROMPT = "このフォルダを読んで報告書をmdファイルで作成して"

    def test_prose_report_pushed_back_then_written(self) -> None:
        tool = _DummyTool()
        client = _ScriptedClient([
            _turn("", [_call("dummy", {"arg": "look"})]),
            _turn("# 報告書\n内容はこの通りです。"),   # prose into chat, no file
            _turn("", [_call("dummy", {"arg": "write-it"})]),  # reacts to push-back
            _turn("report.md を作成しました。"),
        ])
        events, emit = _recorder()
        messages = [{"role": "user", "content": self._PROMPT}]
        result = run_agent(client, "m", [tool], messages, emit=emit)

        self.assertEqual(result, "report.md を作成しました。")
        self.assertIn("deliverable_nudge", _kinds(events))
        pushbacks = [
            m for m in messages
            if m.get("role") == "user"
            and "file deliverable" in m.get("content", "")
        ]
        self.assertEqual(len(pushbacks), 1)

    def test_fires_once_per_turn(self) -> None:
        client = _ScriptedClient([
            _turn("here is the report in chat"),
            _turn("still just chat"),      # ignores the push-back
        ])
        events, emit = _recorder()
        run_agent(client, "m", [],
                  [{"role": "user", "content": self._PROMPT}], emit=emit)
        self.assertEqual(_kinds(events).count("deliverable_nudge"), 1)

    def test_silent_when_file_was_written(self) -> None:
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.md")
            client = _ScriptedClient([
                _turn("", [_call("write", {
                    "file_path": path, "content": "# report\n",
                })]),
                _turn("報告書を作成しました。"),
            ])
            events, emit = _recorder()
            run_agent(
                client, "m", [_FileWriteTool()],
                [{"role": "user", "content": self._PROMPT}], emit=emit,
            )
        self.assertNotIn("deliverable_nudge", _kinds(events))

    def test_silent_without_deliverable_intent(self) -> None:
        client = _ScriptedClient([_turn("add() は2数を足す関数です。")])
        events, emit = _recorder()
        run_agent(client, "m", [],
                  [{"role": "user", "content": "add() を説明して"}], emit=emit)
        self.assertNotIn("deliverable_nudge", _kinds(events))

    def test_disabled_by_config(self) -> None:
        client = _ScriptedClient([_turn("chat only")])
        events, emit = _recorder()
        run_agent(
            client, "m", [],
            [{"role": "user", "content": self._PROMPT}], emit=emit,
            harness=HarnessConfig(deliverable_guard=False),
        )
        self.assertNotIn("deliverable_nudge", _kinds(events))


if __name__ == "__main__":
    unittest.main()
