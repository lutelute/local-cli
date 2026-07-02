"""Tests for local_cli.harness — deterministic harness components."""

import json
import tempfile
import unittest
from pathlib import Path

from local_cli.harness import (
    AgentEvent,
    HarnessConfig,
    LoopDetector,
    TodoReminder,
    _coerce_tool_call,
    _scan_json_objects,
    apply_summary,
    build_summary_request,
    compaction_bounds,
    error_stop_message,
    extract_text_tool_calls,
    last_tool_result_errored,
    loop_break_message,
    loop_warning_message,
    null_emit,
    step_limit_message,
    verify_file_write,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A resolver that accepts the built-in tool names used in these tests.
_KNOWN_TOOLS = {"write", "read", "edit", "bash", "todo_write"}


def _resolver(name: str) -> str | None:
    return name if name in _KNOWN_TOOLS else None


# ---------------------------------------------------------------------------
# AgentEvent / HarnessConfig / null_emit
# ---------------------------------------------------------------------------


class TestEventTypes(unittest.TestCase):
    """Basic behaviour of the event and config types."""

    def test_agent_event_defaults(self) -> None:
        """AgentEvent data defaults to an empty dict."""
        event = AgentEvent("nudge")
        self.assertEqual(event.kind, "nudge")
        self.assertEqual(event.data, {})

    def test_agent_event_data_not_shared(self) -> None:
        """Each event gets its own data dict (no shared mutable default)."""
        a = AgentEvent("x")
        b = AgentEvent("y")
        a.data["k"] = 1
        self.assertEqual(b.data, {})

    def test_harness_config_defaults(self) -> None:
        """Defaults enable every intervention with sane limits."""
        hc = HarnessConfig()
        self.assertEqual(hc.max_iterations, 40)
        self.assertTrue(hc.text_tool_rescue)
        self.assertTrue(hc.loop_detection)
        self.assertTrue(hc.verify_writes)
        self.assertTrue(hc.todo_reminders)
        self.assertEqual(hc.compact_mode, "truncate")
        self.assertTrue(hc.retry_on_overload)

    def test_null_emit_swallows_events(self) -> None:
        """null_emit accepts any event and returns None."""
        self.assertIsNone(null_emit(AgentEvent("anything", {"a": 1})))


# ---------------------------------------------------------------------------
# _scan_json_objects
# ---------------------------------------------------------------------------


class TestScanJsonObjects(unittest.TestCase):
    """Tests for the string-aware balanced-brace JSON scanner."""

    def test_single_object(self) -> None:
        objs = _scan_json_objects('prefix {"a": 1} suffix')
        self.assertEqual(objs, [{"a": 1}])

    def test_nested_object(self) -> None:
        """Nested braces are balanced, not cut at the first '}'."""
        objs = _scan_json_objects('{"a": {"b": {"c": 1}}}')
        self.assertEqual(objs, [{"a": {"b": {"c": 1}}}])

    def test_braces_inside_strings_ignored(self) -> None:
        """Braces inside JSON strings do not confuse the depth counter."""
        objs = _scan_json_objects('{"code": "if x { y }"}')
        self.assertEqual(objs, [{"code": "if x { y }"}])

    def test_escaped_quotes_inside_strings(self) -> None:
        objs = _scan_json_objects('{"s": "say \\"hi\\" {ok}"}')
        self.assertEqual(objs, [{"s": 'say "hi" {ok}'}])

    def test_multiple_objects(self) -> None:
        objs = _scan_json_objects('{"a": 1} and {"b": 2}')
        self.assertEqual(objs, [{"a": 1}, {"b": 2}])

    def test_invalid_json_skipped(self) -> None:
        """A balanced but unparsable span is skipped, later ones kept."""
        objs = _scan_json_objects("{not json} {\"ok\": true}")
        self.assertEqual(objs, [{"ok": True}])

    def test_unbalanced_stops_scanning(self) -> None:
        objs = _scan_json_objects('{"a": 1} {"never closes": ')
        self.assertEqual(objs, [{"a": 1}])

    def test_limit_respected(self) -> None:
        text = " ".join('{"i": %d}' % i for i in range(20))
        objs = _scan_json_objects(text, limit=3)
        self.assertEqual(len(objs), 3)

    def test_no_braces(self) -> None:
        self.assertEqual(_scan_json_objects("plain text"), [])


# ---------------------------------------------------------------------------
# _coerce_tool_call
# ---------------------------------------------------------------------------


class TestCoerceToolCall(unittest.TestCase):
    """Tests for interpreting parsed JSON as a tool call."""

    def test_standard_shape(self) -> None:
        result = _coerce_tool_call(
            {"name": "write", "arguments": {"file_path": "a.py"}}
        )
        self.assertEqual(result, ("write", {"file_path": "a.py"}))

    def test_tool_key_variant(self) -> None:
        result = _coerce_tool_call({"tool": "bash", "args": {"command": "ls"}})
        self.assertEqual(result, ("bash", {"command": "ls"}))

    def test_tool_name_key_variant(self) -> None:
        result = _coerce_tool_call(
            {"tool_name": "read", "parameters": {"file_path": "x"}}
        )
        self.assertEqual(result, ("read", {"file_path": "x"}))

    def test_function_wrapper_unwrapped(self) -> None:
        """The OpenAI {"function": {...}} wrapper is unwrapped."""
        result = _coerce_tool_call(
            {"function": {"name": "write", "arguments": {"a": 1}}}
        )
        self.assertEqual(result, ("write", {"a": 1}))

    def test_string_arguments_parsed(self) -> None:
        """A JSON-encoded arguments string is parsed to a dict."""
        result = _coerce_tool_call(
            {"name": "write", "arguments": '{"file_path": "a.py"}'}
        )
        self.assertEqual(result, ("write", {"file_path": "a.py"}))

    def test_unparsable_string_arguments_degrade_to_empty(self) -> None:
        result = _coerce_tool_call({"name": "write", "arguments": "not json"})
        self.assertEqual(result, ("write", {}))

    def test_missing_arguments_degrade_to_empty(self) -> None:
        result = _coerce_tool_call({"name": "read"})
        self.assertEqual(result, ("read", {}))

    def test_non_dict_arguments_rejected(self) -> None:
        self.assertIsNone(_coerce_tool_call({"name": "read", "arguments": [1]}))

    def test_missing_name_rejected(self) -> None:
        self.assertIsNone(_coerce_tool_call({"arguments": {"a": 1}}))

    def test_non_dict_rejected(self) -> None:
        self.assertIsNone(_coerce_tool_call(["name", "write"]))
        self.assertIsNone(_coerce_tool_call("write"))
        self.assertIsNone(_coerce_tool_call(None))


# ---------------------------------------------------------------------------
# extract_text_tool_calls
# ---------------------------------------------------------------------------


class TestExtractTextToolCalls(unittest.TestCase):
    """Tests for the text-based tool-call rescue parser."""

    def test_qwen_tool_call_tags(self) -> None:
        """Qwen-style <tool_call> tags are rescued."""
        content = (
            "I'll write the file.\n<tool_call>\n"
            '{"name": "write", "arguments": {"file_path": "a.py", '
            '"content": "x = 1"}}\n</tool_call>'
        )
        calls = extract_text_tool_calls(content, _resolver)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "write")
        self.assertEqual(
            calls[0]["function"]["arguments"]["file_path"], "a.py"
        )
        self.assertEqual(calls[0]["id"], "text_rescue_0")

    def test_fenced_json_block(self) -> None:
        content = (
            "Let me run this:\n```json\n"
            '{"name": "bash", "arguments": {"command": "ls"}}\n```\n'
        )
        calls = extract_text_tool_calls(content, _resolver)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "bash")

    def test_fenced_tool_call_block(self) -> None:
        content = (
            "```tool_call\n"
            '{"name": "read", "arguments": {"file_path": "x.py"}}\n```'
        )
        calls = extract_text_tool_calls(content, _resolver)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read")

    def test_bare_json_content(self) -> None:
        """A bare JSON object (no fence, no tag) is rescued."""
        content = '{"name": "write", "arguments": {"file_path": "b.py", "content": ""}}'
        calls = extract_text_tool_calls(content, _resolver)
        self.assertEqual(len(calls), 1)

    def test_unknown_tool_rejected(self) -> None:
        """A JSON object naming an unknown tool is not rescued."""
        content = '{"name": "launch_rocket", "arguments": {}}'
        self.assertEqual(extract_text_tool_calls(content, _resolver), [])

    def test_plain_code_not_rescued(self) -> None:
        """Ordinary code with braces is not mistaken for a tool call."""
        content = "```python\ndef f():\n    return {'a': 1}\n```"
        self.assertEqual(extract_text_tool_calls(content, _resolver), [])

    def test_plain_text_not_rescued(self) -> None:
        self.assertEqual(extract_text_tool_calls("hello", _resolver), [])
        self.assertEqual(extract_text_tool_calls("", _resolver), [])

    def test_duplicates_collapsed(self) -> None:
        """The same call repeated in the content is rescued once."""
        call = '{"name": "read", "arguments": {"file_path": "a.py"}}'
        content = f"<tool_call>{call}</tool_call>\n<tool_call>{call}</tool_call>"
        calls = extract_text_tool_calls(content, _resolver)
        self.assertEqual(len(calls), 1)

    def test_multiple_distinct_calls(self) -> None:
        content = (
            '<tool_call>{"name": "read", "arguments": {"file_path": "a"}}'
            "</tool_call>"
            '<tool_call>{"name": "read", "arguments": {"file_path": "b"}}'
            "</tool_call>"
        )
        calls = extract_text_tool_calls(content, _resolver)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["id"], "text_rescue_1")

    def test_limit_respected(self) -> None:
        parts = [
            '<tool_call>{"name": "read", "arguments": {"file_path": "%d"}}</tool_call>'
            % i
            for i in range(10)
        ]
        calls = extract_text_tool_calls("".join(parts), _resolver, limit=3)
        self.assertEqual(len(calls), 3)

    def test_tagged_call_takes_priority_over_bare_scan(self) -> None:
        """When a tagged call exists, prose JSON is not also scanned."""
        content = (
            'Example: {"name": "bash", "arguments": {"command": "rm -rf /"}}\n'
            '<tool_call>{"name": "read", "arguments": {"file_path": "a"}}'
            "</tool_call>"
        )
        calls = extract_text_tool_calls(content, _resolver)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read")

    def test_resolver_receives_near_miss_names(self) -> None:
        """Near-miss names are accepted when the resolver resolves them."""
        content = '{"name": "write_file", "arguments": {"file_path": "a"}}'
        calls = extract_text_tool_calls(
            content, lambda n: "write" if n == "write_file" else None
        )
        self.assertEqual(len(calls), 1)
        # The original name is preserved; run_agent resolves it again.
        self.assertEqual(calls[0]["function"]["name"], "write_file")


# ---------------------------------------------------------------------------
# LoopDetector
# ---------------------------------------------------------------------------


class TestLoopDetector(unittest.TestCase):
    """Tests for repeated-call loop detection."""

    def test_distinct_calls_ok(self) -> None:
        det = LoopDetector()
        self.assertEqual(det.record("read", {"file_path": "a"}), "ok")
        self.assertEqual(det.record("read", {"file_path": "b"}), "ok")
        self.assertEqual(det.record("write", {"file_path": "a"}), "ok")

    def test_three_identical_warns_once(self) -> None:
        det = LoopDetector()
        args = {"file_path": "a", "old_text": "x", "new_text": "y"}
        self.assertEqual(det.record("edit", args), "ok")
        self.assertEqual(det.record("edit", args), "ok")
        self.assertEqual(det.record("edit", args), "warn")
        # Fourth identical call: already warned, not yet at break.
        self.assertEqual(det.record("edit", args), "ok")

    def test_five_identical_breaks(self) -> None:
        det = LoopDetector()
        args = {"command": "ls"}
        verdicts = [det.record("bash", args) for _ in range(5)]
        self.assertEqual(verdicts[-1], "break")

    def test_alternating_pair_warns_then_breaks(self) -> None:
        """A two-call alternation (read → edit → read …) is caught."""
        det = LoopDetector()
        a = ("read", {"file_path": "a"})
        b = ("edit", {"file_path": "a", "old_text": "x", "new_text": "y"})
        verdicts = []
        for i in range(10):
            call = a if i % 2 == 0 else b
            verdicts.append(det.record(*call))
        self.assertIn("warn", verdicts)  # at the 6-call window
        self.assertEqual(verdicts[-1], "break")  # at the 10-call window

    def test_different_args_do_not_trigger(self) -> None:
        det = LoopDetector()
        for i in range(12):
            verdict = det.record("read", {"file_path": f"file_{i}.py"})
            self.assertEqual(verdict, "ok")

    def test_unserializable_args_do_not_crash(self) -> None:
        det = LoopDetector()
        # json.dumps with default=str handles this; ensure no exception.
        verdict = det.record("bash", {"x": object()})
        self.assertEqual(verdict, "ok")

    def test_messages_have_system_reminder_shape(self) -> None:
        warn = loop_warning_message("edit")
        brk = loop_break_message()
        limit = step_limit_message(40)
        for msg in (warn, brk, limit):
            self.assertEqual(msg["role"], "user")
            self.assertIn("<system-reminder>", msg["content"])
        self.assertIn("edit", warn["content"])
        self.assertIn("40", limit["content"])


# ---------------------------------------------------------------------------
# verify_file_write
# ---------------------------------------------------------------------------


class TestVerifyFileWrite(unittest.TestCase):
    """Tests for the post-write syntax verification gate."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, text: str) -> str:
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_broken_python_flagged(self) -> None:
        path = self._write("bad.py", "def f(:\n    pass\n")
        warning = verify_file_write("write", {"file_path": path}, "Wrote file")
        self.assertIsNotNone(warning)
        self.assertIn("syntax error", warning)
        self.assertIn("line 1", warning)

    def test_valid_python_passes(self) -> None:
        path = self._write("ok.py", "def f():\n    return 1\n")
        self.assertIsNone(
            verify_file_write("write", {"file_path": path}, "Wrote file")
        )

    def test_broken_json_flagged(self) -> None:
        path = self._write("bad.json", '{"a": 1,}')
        warning = verify_file_write("edit", {"file_path": path}, "ok")
        self.assertIsNotNone(warning)
        self.assertIn("not valid JSON", warning)

    def test_valid_json_passes(self) -> None:
        path = self._write("ok.json", '{"a": 1}')
        self.assertIsNone(verify_file_write("edit", {"file_path": path}, "ok"))

    def test_other_extensions_skipped(self) -> None:
        path = self._write("notes.txt", "anything {[(")
        self.assertIsNone(verify_file_write("write", {"file_path": path}, "ok"))

    def test_error_results_skipped(self) -> None:
        path = self._write("bad.py", "def f(:\n")
        self.assertIsNone(
            verify_file_write(
                "write", {"file_path": path}, "Error: permission denied"
            )
        )

    def test_non_write_tools_skipped(self) -> None:
        path = self._write("bad.py", "def f(:\n")
        self.assertIsNone(verify_file_write("read", {"file_path": path}, "ok"))

    def test_missing_file_skipped(self) -> None:
        self.assertIsNone(
            verify_file_write(
                "write", {"file_path": str(self.dir / "nope.py")}, "ok"
            )
        )

    def test_missing_file_path_skipped(self) -> None:
        self.assertIsNone(verify_file_write("write", {}, "ok"))
        self.assertIsNone(verify_file_write("write", {"file_path": 3}, "ok"))


# ---------------------------------------------------------------------------
# TodoReminder
# ---------------------------------------------------------------------------


class _FakeTodoTool:
    """Stands in for TodoWriteTool: a name and a current_todos property."""

    name = "todo_write"

    def __init__(self, todos: list[dict[str, str]]) -> None:
        self.current_todos = todos


class TestTodoReminder(unittest.TestCase):
    """Tests for todo staleness reminders."""

    def test_stale_unfinished_todos_remind(self) -> None:
        reminder = TodoReminder(stale_after=4)
        tools = [_FakeTodoTool([
            {"content": "task A", "status": "completed"},
            {"content": "task B", "status": "pending"},
        ])]
        self.assertIsNone(reminder.check(3, tools))  # not stale yet
        text = reminder.check(5, tools)
        self.assertIsNotNone(text)
        self.assertIn("task B", text)
        self.assertIn("<system-reminder>", text)

    def test_all_completed_never_reminds(self) -> None:
        reminder = TodoReminder(stale_after=1)
        tools = [_FakeTodoTool([{"content": "a", "status": "completed"}])]
        self.assertIsNone(reminder.check(100, tools))

    def test_todo_write_use_resets_grace_period(self) -> None:
        reminder = TodoReminder(stale_after=4)
        tools = [_FakeTodoTool([{"content": "a", "status": "in_progress"}])]
        reminder.note_tool_use(4, "todo_write")
        self.assertIsNone(reminder.check(6, tools))  # 6 - 4 < 4
        self.assertIsNotNone(reminder.check(8, tools))  # 8 - 4 >= 4

    def test_other_tools_do_not_reset(self) -> None:
        reminder = TodoReminder(stale_after=4)
        tools = [_FakeTodoTool([{"content": "a", "status": "pending"}])]
        reminder.note_tool_use(4, "bash")
        self.assertIsNotNone(reminder.check(5, tools))

    def test_max_reminders_cap(self) -> None:
        reminder = TodoReminder(stale_after=1, max_reminders=2)
        tools = [_FakeTodoTool([{"content": "a", "status": "pending"}])]
        self.assertIsNotNone(reminder.check(2, tools))
        self.assertIsNotNone(reminder.check(10, tools))
        self.assertIsNone(reminder.check(100, tools))

    def test_no_todo_tool_never_reminds(self) -> None:
        reminder = TodoReminder(stale_after=1)
        self.assertIsNone(reminder.check(100, []))

    def test_empty_todos_never_remind(self) -> None:
        reminder = TodoReminder(stale_after=1)
        self.assertIsNone(reminder.check(100, [_FakeTodoTool([])]))

    def test_in_progress_named_before_pending(self) -> None:
        reminder = TodoReminder(stale_after=1)
        tools = [_FakeTodoTool([
            {"content": "pending task", "status": "pending"},
            {"content": "active task", "status": "in_progress"},
        ])]
        text = reminder.check(5, tools)
        self.assertIn("active task", text)


# ---------------------------------------------------------------------------
# last_tool_result_errored
# ---------------------------------------------------------------------------


class TestLastToolResultErrored(unittest.TestCase):
    """Tests for the error-stop guard's finishing-on-failure detector."""

    @staticmethod
    def _history(*tail: dict) -> list[dict]:
        base = [
            {"role": "user", "content": "fix app.py"},
        ]
        return base + list(tail) + [
            {"role": "assistant", "content": "done!"},
        ]

    def test_error_result_detected(self) -> None:
        messages = self._history(
            {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "tool_name": "edit",
             "content": "Error: old_text not found in app.py"},
        )
        self.assertTrue(last_tool_result_errored(messages))

    def test_verify_warning_on_write_detected(self) -> None:
        messages = self._history(
            {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "tool_name": "write",
             "content": "Wrote app.py\n\nWARNING: app.py now contains a "
                        "Python syntax error at line 1: invalid syntax."},
        )
        self.assertTrue(last_tool_result_errored(messages))

    def test_warning_in_bash_output_not_detected(self) -> None:
        """A stray WARNING in build output must not trigger the guard."""
        messages = self._history(
            {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "tool_name": "bash",
             "content": "gcc: WARNING: deprecated flag\nbuild ok"},
        )
        self.assertFalse(last_tool_result_errored(messages))

    def test_successful_result_not_detected(self) -> None:
        messages = self._history(
            {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "tool_name": "edit", "content": "--- diff ok"},
        )
        self.assertFalse(last_tool_result_errored(messages))

    def test_reminder_messages_are_skipped(self) -> None:
        """Harness-injected reminders between tool and answer are skipped."""
        messages = self._history(
            {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "tool_name": "edit", "content": "Error: nope"},
            {"role": "user",
             "content": "<system-reminder>update your todos</system-reminder>"},
            {"role": "assistant", "content": "hmm"},
        )
        self.assertTrue(last_tool_result_errored(messages))

    def test_real_user_message_ends_the_walk(self) -> None:
        """Errors from a previous turn are not this turn's business."""
        messages = [
            {"role": "user", "content": "first task"},
            {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "tool_name": "edit", "content": "Error: nope"},
            {"role": "user", "content": "never mind, what is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        self.assertFalse(last_tool_result_errored(messages))

    def test_no_tools_this_turn(self) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        self.assertFalse(last_tool_result_errored(messages))

    def test_error_stop_message_shape(self) -> None:
        msg = error_stop_message()
        self.assertEqual(msg["role"], "user")
        self.assertIn("<system-reminder>", msg["content"])
        self.assertIn("Do not stop", msg["content"])


# ---------------------------------------------------------------------------
# Summary-compaction helpers
# ---------------------------------------------------------------------------


def _msg(role: str, content: str, **extra) -> dict:
    return {"role": role, "content": content, **extra}


class TestCompactionBounds(unittest.TestCase):
    """Tests for the summarizable-span calculation."""

    def test_basic_bounds(self) -> None:
        messages = [_msg("system", "sys")] + [
            _msg("user" if i % 2 == 0 else "assistant", f"m{i}")
            for i in range(20)
        ]
        bounds = compaction_bounds(messages, keep_recent=10)
        self.assertIsNotNone(bounds)
        system_end, recent_start = bounds
        self.assertEqual(system_end, 1)
        self.assertEqual(recent_start, len(messages) - 10)

    def test_boundary_never_splits_tool_pair(self) -> None:
        """recent_start walks back over tool results to the assistant."""
        messages = [_msg("system", "sys")]
        for i in range(8):
            messages.append(_msg("user", f"u{i}"))
            messages.append(_msg("assistant", "", tool_calls=[{}]))
            messages.append(_msg("tool", f"r{i}", tool_name="read"))
        bounds = compaction_bounds(messages, keep_recent=4)
        self.assertIsNotNone(bounds)
        _, recent_start = bounds
        # The message at the boundary must not be a tool result.
        self.assertNotEqual(messages[recent_start]["role"], "tool")

    def test_short_history_returns_none(self) -> None:
        messages = [_msg("system", "s"), _msg("user", "u"), _msg("assistant", "a")]
        self.assertIsNone(compaction_bounds(messages, keep_recent=10))

    def test_no_system_message(self) -> None:
        messages = [_msg("user", f"m{i}") for i in range(20)]
        bounds = compaction_bounds(messages, keep_recent=5)
        self.assertIsNotNone(bounds)
        self.assertEqual(bounds[0], 0)


class TestBuildSummaryRequest(unittest.TestCase):
    """Tests for the summarizer request builder."""

    def test_includes_roles_and_content(self) -> None:
        request = build_summary_request([
            _msg("user", "fix the bug in foo.py"),
            _msg("assistant", "reading the file"),
        ])
        self.assertEqual(len(request), 2)
        self.assertEqual(request[0]["role"], "system")
        self.assertIn("fix the bug in foo.py", request[1]["content"])
        self.assertIn("assistant:", request[1]["content"])

    def test_tool_call_names_noted(self) -> None:
        request = build_summary_request([
            _msg("assistant", "", tool_calls=[
                {"function": {"name": "write", "arguments": {}}},
            ]),
        ])
        self.assertIn("called tools: write", request[1]["content"])

    def test_long_messages_snipped(self) -> None:
        request = build_summary_request([_msg("tool", "x" * 10_000)])
        self.assertLess(len(request[1]["content"]), 5_000)


class TestApplySummary(unittest.TestCase):
    """Tests for splicing the summary back into the history."""

    def test_replaces_span_with_single_message(self) -> None:
        messages = [_msg("system", "sys")] + [
            _msg("user", f"m{i}") for i in range(10)
        ]
        apply_summary(messages, "the summary", 1, 8)
        self.assertEqual(len(messages), 1 + 1 + 3)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("the summary", messages[1]["content"])
        self.assertIn("summarized to save context", messages[1]["content"])
        self.assertEqual(messages[2]["content"], "m7")


if __name__ == "__main__":
    unittest.main()
