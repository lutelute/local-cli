"""Integration tests for local_cli.agent.run_agent — the unified loop.

These tests drive run_agent with scripted duck-typed clients and verify
the harness interventions end to end: text tool-call rescue, loop
detection, the step limit, the post-write verification gate, todo
staleness reminders, overload retry, and summary compaction.
"""

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from local_cli.agent import run_agent
from local_cli.harness import AgentEvent, HarnessConfig
from local_cli.providers.base import ProviderRequestError
from local_cli.tools.base import Tool
from local_cli.tools.todo_tool import TodoWriteTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyTool(Tool):
    """Minimal concrete tool that records its calls."""

    def __init__(self, name: str = "dummy", result: str = "ok") -> None:
        self._name = name
        self._result = result
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A dummy tool for testing."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs: object) -> str:
        self.calls.append(dict(kwargs))
        return self._result


class _FileWriteTool(Tool):
    """A 'write' tool that actually writes files (for the verify gate)."""

    @property
    def name(self) -> str:
        return "write"

    @property
    def description(self) -> str:
        return "Write a file."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, **kwargs: object) -> str:
        path = Path(str(kwargs["file_path"]))
        path.write_text(str(kwargs.get("content", "")), encoding="utf-8")
        return f"Wrote {path}"


def _turn(content: str, tool_calls: list[dict] | None = None) -> list[dict]:
    """Build the chunk list for one scripted assistant turn."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return [{"message": message, "done": True}]


def _call(name: str, arguments: dict | None = None) -> dict:
    return {"function": {"name": name, "arguments": arguments or {}}}


class _ScriptedClient:
    """Duck-typed client returning scripted turns in order.

    Records every chat_stream invocation (kwargs included) so tests can
    assert on what the loop sent.  When the script runs out, a plain
    'done' turn is returned so tests cannot hang.
    """

    def __init__(self, turns: list[list[dict]]) -> None:
        self._turns = list(turns)
        self.requests: list[dict[str, Any]] = []

    def chat_stream(self, model: str, messages: list[dict], **kwargs: Any):
        self.requests.append({
            "model": model,
            "message_count": len(messages),
            "kwargs": kwargs,
        })
        if self._turns:
            return iter(self._turns.pop(0))
        return iter(_turn("done"))


def _recorder() -> tuple[list[AgentEvent], Any]:
    events: list[AgentEvent] = []
    return events, events.append


def _kinds(events: list[AgentEvent]) -> list[str]:
    return [e.kind for e in events]


# ---------------------------------------------------------------------------
# Basic loop behaviour
# ---------------------------------------------------------------------------


class TestRunAgentBasics(unittest.TestCase):
    """Core loop behaviour through the event interface."""

    def test_returns_final_content(self) -> None:
        client = _ScriptedClient([_turn("all done")])
        result = run_agent(client, "m", [], [{"role": "user", "content": "hi"}])
        self.assertEqual(result, "all done")

    def test_executes_structured_tool_calls(self) -> None:
        tool = _DummyTool()
        client = _ScriptedClient([
            _turn("", [_call("dummy", {"arg": "1"})]),
            _turn("finished"),
        ])
        messages = [{"role": "user", "content": "go"}]
        result = run_agent(client, "m", [tool], messages)

        self.assertEqual(result, "finished")
        self.assertEqual(tool.calls, [{"arg": "1"}])
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["content"], "ok")

    def test_emits_lifecycle_events(self) -> None:
        tool = _DummyTool()
        client = _ScriptedClient([
            _turn("", [_call("dummy")]),
            _turn("finished"),
        ])
        events, emit = _recorder()
        run_agent(client, "m", [tool], [{"role": "user", "content": "go"}],
                  emit=emit)

        kinds = _kinds(events)
        self.assertEqual(kinds.count("llm_start"), 2)
        self.assertIn("tool_start", kinds)
        self.assertIn("tool_result", kinds)
        self.assertEqual(kinds.count("assistant_message"), 2)
        # tool_start must come after the first assistant_message and
        # before the second llm_start.
        self.assertLess(kinds.index("assistant_message"),
                        kinds.index("tool_start"))

    def test_silent_by_default(self) -> None:
        """run_agent with the default null emitter performs no I/O."""
        client = _ScriptedClient([_turn("quiet")])
        # Would raise if any event handler tried to touch a spinner etc.
        result = run_agent(client, "m", [], [{"role": "user", "content": "x"}])
        self.assertEqual(result, "quiet")


# ---------------------------------------------------------------------------
# Text-based tool-call rescue
# ---------------------------------------------------------------------------


class TestTextRescue(unittest.TestCase):
    """Models that print their tool call as text still act."""

    def test_qwen_tag_rescued_and_executed(self) -> None:
        tool = _DummyTool()
        content = (
            "I'll do it now.\n<tool_call>\n"
            '{"name": "dummy", "arguments": {"arg": "z"}}\n</tool_call>'
        )
        client = _ScriptedClient([_turn(content), _turn("finished")])
        events, emit = _recorder()
        messages = [{"role": "user", "content": "go"}]
        result = run_agent(client, "m", [tool], messages, emit=emit)

        self.assertEqual(result, "finished")
        self.assertEqual(tool.calls, [{"arg": "z"}])
        self.assertIn("rescue", _kinds(events))
        # The rescued call is attached to the assistant message so the
        # history stays consistent with the tool result that follows.
        assistant = next(
            m for m in messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        )
        self.assertEqual(
            assistant["tool_calls"][0]["function"]["name"], "dummy"
        )

    def test_near_miss_name_in_text_rescued(self) -> None:
        """A text call with an alias name (run -> bash) is rescued."""
        tool = _DummyTool(name="bash")
        content = '{"name": "run", "arguments": {"command": "ls"}}'
        client = _ScriptedClient([_turn(content), _turn("ok done")])
        run_agent(client, "m", [tool], [{"role": "user", "content": "go"}])
        self.assertEqual(tool.calls, [{"command": "ls"}])

    def test_rescue_disabled_by_config(self) -> None:
        tool = _DummyTool()
        content = '<tool_call>{"name": "dummy", "arguments": {}}</tool_call>'
        client = _ScriptedClient([_turn(content)])
        run_agent(
            client, "m", [tool], [{"role": "user", "content": "go"}],
            harness=HarnessConfig(text_tool_rescue=False),
        )
        self.assertEqual(tool.calls, [])

    def test_prose_answer_not_rescued(self) -> None:
        tool = _DummyTool()
        client = _ScriptedClient([_turn("The answer is 42.")])
        result = run_agent(client, "m", [tool],
                           [{"role": "user", "content": "what is 6*7"}])
        self.assertEqual(result, "The answer is 42.")
        self.assertEqual(tool.calls, [])


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------


class TestLoopDetection(unittest.TestCase):
    """Repeated identical calls draw a warning, then a forced stop."""

    def test_warning_injected_on_third_repeat(self) -> None:
        tool = _DummyTool(result="Error: old_text not found in a.py")
        same = [_call("dummy", {"arg": "same"})]
        client = _ScriptedClient([
            _turn("", same), _turn("", same), _turn("", same),
            _turn("giving a summary"),
        ])
        events, emit = _recorder()
        messages = [{"role": "user", "content": "go"}]
        run_agent(client, "m", [tool], messages, emit=emit)

        self.assertIn("loop_warning", _kinds(events))
        warnings = [
            m for m in messages
            if m.get("role") == "user"
            and "repeated the same tool call" in m.get("content", "")
        ]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(len(tool.calls), 3)  # warn does not block execution

    def test_fifth_repeat_forces_final_turn(self) -> None:
        same = [_call("dummy", {"arg": "same"})]
        tool = _DummyTool(result="Error: still failing")
        client = _ScriptedClient([
            _turn("", same), _turn("", same), _turn("", same),
            _turn("", same), _turn("", same),
            _turn("here is my summary"),
        ])
        events, emit = _recorder()
        messages = [{"role": "user", "content": "go"}]
        result = run_agent(client, "m", [tool], messages, emit=emit)

        self.assertEqual(result, "here is my summary")
        self.assertIn("loop_break", _kinds(events))
        # The 5th call is skipped, not executed.
        self.assertEqual(len(tool.calls), 4)
        # The final request must not offer tools.
        self.assertNotIn("tools", client.requests[-1]["kwargs"])
        # The skipped call still gets a tool result (id pairing).
        skipped = [
            m for m in messages
            if m.get("role") == "tool" and "skipped" in m.get("content", "")
        ]
        self.assertEqual(len(skipped), 1)

    def test_detection_disabled_by_config(self) -> None:
        same = [_call("dummy", {"arg": "same"})]
        tool = _DummyTool()
        client = _ScriptedClient(
            [_turn("", same)] * 6 + [_turn("done")]
        )
        events, emit = _recorder()
        run_agent(
            client, "m", [tool], [{"role": "user", "content": "go"}],
            emit=emit, harness=HarnessConfig(loop_detection=False),
        )
        self.assertNotIn("loop_warning", _kinds(events))
        self.assertNotIn("loop_break", _kinds(events))
        self.assertEqual(len(tool.calls), 6)


# ---------------------------------------------------------------------------
# Step limit
# ---------------------------------------------------------------------------


class TestStepLimit(unittest.TestCase):
    """After max_iterations the model gets one tool-free wrap-up turn."""

    def test_limit_forces_tool_free_final_turn(self) -> None:
        tool = _DummyTool()
        # Distinct args each turn so loop detection stays quiet.  Three
        # tool turns exhaust the limit; the fourth (tool-free) turn is
        # the scripted wrap-up answer.
        turns = [
            _turn("", [_call("dummy", {"arg": str(i)})]) for i in range(3)
        ]
        turns.append(_turn("wrap-up"))
        client = _ScriptedClient(turns)
        events, emit = _recorder()
        messages = [{"role": "user", "content": "go"}]
        result = run_agent(
            client, "m", [tool], messages, emit=emit,
            harness=HarnessConfig(max_iterations=3),
        )

        self.assertEqual(result, "wrap-up")
        self.assertIn("limit", _kinds(events))
        self.assertEqual(len(tool.calls), 3)  # stopped after 3 iterations
        self.assertNotIn("tools", client.requests[-1]["kwargs"])
        limit_msgs = [
            m for m in messages
            if m.get("role") == "user" and "step limit" in m.get("content", "")
        ]
        self.assertEqual(len(limit_msgs), 1)

    def test_zero_means_unlimited(self) -> None:
        tool = _DummyTool()
        turns = [
            _turn("", [_call("dummy", {"arg": str(i)})]) for i in range(6)
        ]
        turns.append(_turn("done"))
        client = _ScriptedClient(turns)
        events, emit = _recorder()
        run_agent(
            client, "m", [tool], [{"role": "user", "content": "go"}],
            emit=emit, harness=HarnessConfig(max_iterations=0),
        )
        self.assertNotIn("limit", _kinds(events))
        self.assertEqual(len(tool.calls), 6)


# ---------------------------------------------------------------------------
# Post-write verification gate
# ---------------------------------------------------------------------------


class TestVerifyGate(unittest.TestCase):
    """Syntax errors in a just-written file are fed straight back."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_broken_python_write_gets_warning(self) -> None:
        path = str(self.dir / "bad.py")
        client = _ScriptedClient([
            _turn("", [_call("write", {
                "file_path": path, "content": "def f(:\n    pass\n",
            })]),
            _turn("finished"),
        ])
        events, emit = _recorder()
        messages = [{"role": "user", "content": "write it"}]
        run_agent(client, "m", [_FileWriteTool()], messages, emit=emit)

        self.assertIn("verify_warning", _kinds(events))
        tool_msg = next(m for m in messages if m.get("role") == "tool")
        self.assertIn("WARNING", tool_msg["content"])
        self.assertIn("syntax error", tool_msg["content"])

    def test_valid_python_write_unchanged(self) -> None:
        path = str(self.dir / "ok.py")
        client = _ScriptedClient([
            _turn("", [_call("write", {
                "file_path": path, "content": "x = 1\n",
            })]),
            _turn("finished"),
        ])
        events, emit = _recorder()
        messages = [{"role": "user", "content": "write it"}]
        run_agent(client, "m", [_FileWriteTool()], messages, emit=emit)

        self.assertNotIn("verify_warning", _kinds(events))
        tool_msg = next(m for m in messages if m.get("role") == "tool")
        self.assertNotIn("WARNING", tool_msg["content"])

    def test_gate_disabled_by_config(self) -> None:
        path = str(self.dir / "bad.py")
        client = _ScriptedClient([
            _turn("", [_call("write", {
                "file_path": path, "content": "def f(:\n",
            })]),
            _turn("finished"),
        ])
        events, emit = _recorder()
        run_agent(
            client, "m", [_FileWriteTool()],
            [{"role": "user", "content": "write it"}],
            emit=emit, harness=HarnessConfig(verify_writes=False),
        )
        self.assertNotIn("verify_warning", _kinds(events))


# ---------------------------------------------------------------------------
# Todo staleness reminder
# ---------------------------------------------------------------------------


class TestTodoReminderIntegration(unittest.TestCase):
    """A stale todo list is re-surfaced as a system reminder."""

    def test_reminder_injected_when_stale(self) -> None:
        todo_tool = TodoWriteTool()
        todo_tool.execute(todos=[
            {"content": "build the parser", "status": "in_progress"},
            {"content": "write tests", "status": "pending"},
        ])
        work = _DummyTool()
        # 4 tool turns with distinct args, then a final answer.  The
        # reminder fires at iteration 5 (stale_after=4, never updated).
        turns = [
            _turn("", [_call("dummy", {"arg": str(i)})]) for i in range(4)
        ]
        turns.append(_turn("finished"))
        client = _ScriptedClient(turns)
        events, emit = _recorder()
        messages = [{"role": "user", "content": "do the work"}]
        run_agent(client, "m", [work, todo_tool], messages, emit=emit)

        self.assertIn("reminder", _kinds(events))
        reminders = [
            m for m in messages
            if m.get("role") == "user"
            and "todo list" in m.get("content", "")
        ]
        self.assertEqual(len(reminders), 1)
        self.assertIn("build the parser", reminders[0]["content"])

    def test_no_reminder_when_all_done(self) -> None:
        todo_tool = TodoWriteTool()
        todo_tool.execute(todos=[{"content": "a", "status": "completed"}])
        work = _DummyTool()
        turns = [
            _turn("", [_call("dummy", {"arg": str(i)})]) for i in range(6)
        ]
        turns.append(_turn("finished"))
        client = _ScriptedClient(turns)
        events, emit = _recorder()
        run_agent(client, "m", [work, todo_tool],
                  [{"role": "user", "content": "go"}], emit=emit)
        self.assertNotIn("reminder", _kinds(events))


# ---------------------------------------------------------------------------
# Overload retry
# ---------------------------------------------------------------------------


class _FlakyClient:
    """Fails the first N chat_stream calls with a 503, then succeeds."""

    def __init__(self, fail_times: int) -> None:
        self._fail_times = fail_times
        self.attempts = 0

    def chat_stream(self, model: str, messages: list[dict], **kwargs: Any):
        self.attempts += 1
        if self.attempts <= self._fail_times:
            raise ProviderRequestError("HTTP 503 Service Unavailable")
        return iter(_turn("recovered"))


class TestOverloadRetry(unittest.TestCase):
    """HTTP 503s are retried with backoff instead of aborting the turn."""

    def test_retries_on_503_and_recovers(self) -> None:
        client = _FlakyClient(fail_times=2)
        events, emit = _recorder()
        with patch("local_cli.agent.time.sleep") as mock_sleep:
            result = run_agent(
                client, "m", [], [{"role": "user", "content": "hi"}],
                emit=emit,
            )
        self.assertEqual(result, "recovered")
        self.assertEqual(client.attempts, 3)
        self.assertEqual(_kinds(events).count("retry"), 2)
        # Exponential backoff: 1s then 2s.
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        self.assertEqual(delays, [1.0, 2.0])

    def test_gives_up_after_max_attempts(self) -> None:
        client = _FlakyClient(fail_times=99)
        events, emit = _recorder()
        messages = [{"role": "user", "content": "hi"}]
        with patch("local_cli.agent.time.sleep"):
            result = run_agent(client, "m", [], messages, emit=emit)
        self.assertEqual(result, "")
        self.assertEqual(client.attempts, 4)  # 1 + 3 retries
        self.assertIn("error", _kinds(events))
        self.assertIn("Error from provider", messages[-1]["content"])

    def test_non_overload_errors_not_retried(self) -> None:
        class _BadRequestClient:
            attempts = 0

            def chat_stream(self, model, messages, **kwargs):
                self.attempts += 1
                raise ProviderRequestError("HTTP 400 Bad Request")

        client = _BadRequestClient()
        with patch("local_cli.agent.time.sleep"):
            run_agent(client, "m", [], [{"role": "user", "content": "hi"}])
        self.assertEqual(client.attempts, 1)

    def test_retry_disabled_by_config(self) -> None:
        client = _FlakyClient(fail_times=1)
        with patch("local_cli.agent.time.sleep"):
            run_agent(
                client, "m", [], [{"role": "user", "content": "hi"}],
                harness=HarnessConfig(retry_on_overload=False),
            )
        self.assertEqual(client.attempts, 1)


# ---------------------------------------------------------------------------
# Text-driven fallback for models without tool support
# ---------------------------------------------------------------------------


class _NoToolSupportClient:
    """Rejects requests that carry tools; then answers with text calls."""

    def __init__(self, turns: list[list[dict]]) -> None:
        self._turns = list(turns)
        self.requests: list[dict[str, Any]] = []

    def chat_stream(self, model: str, messages: list[dict], **kwargs: Any):
        self.requests.append({"kwargs": kwargs, "message_count": len(messages)})
        if "tools" in kwargs:
            raise ProviderRequestError(
                'registry.ollama.ai/library/tiny "tiny" does not support tools'
            )
        if self._turns:
            return iter(self._turns.pop(0))
        return iter(_turn("done"))


class TestToolsFallback(unittest.TestCase):
    """Models whose endpoint rejects tools switch to text-driven calls."""

    def test_fallback_teaches_format_and_rescues_calls(self) -> None:
        tool = _DummyTool()
        # After the fallback, the model "calls" the tool as fenced JSON.
        client = _NoToolSupportClient([
            _turn(
                "```json\n"
                '{"name": "dummy", "arguments": {"arg": "via-text"}}\n'
                "```"
            ),
            _turn("task complete"),
        ])
        events, emit = _recorder()
        messages = [{"role": "user", "content": "do it"}]
        result = run_agent(client, "tiny", [tool], messages, emit=emit)

        self.assertEqual(result, "task complete")
        self.assertEqual(tool.calls, [{"arg": "via-text"}])
        kinds = _kinds(events)
        self.assertIn("tools_fallback", kinds)
        self.assertIn("rescue", kinds)
        # First request carried tools; all later ones must not.
        self.assertIn("tools", client.requests[0]["kwargs"])
        for req in client.requests[1:]:
            self.assertNotIn("tools", req["kwargs"])
        # The format instruction was injected for the model.
        instructions = [
            m for m in messages
            if m.get("role") == "user"
            and "driven by TEXT" in m.get("content", "")
        ]
        self.assertEqual(len(instructions), 1)
        self.assertIn("dummy", instructions[0]["content"])

    def test_other_request_errors_still_fail(self) -> None:
        class _BadModelClient:
            def chat_stream(self, model, messages, **kwargs):
                raise ProviderRequestError("model not found")

        events, emit = _recorder()
        messages = [{"role": "user", "content": "hi"}]
        result = run_agent(_BadModelClient(), "m", [], messages, emit=emit)
        self.assertEqual(result, "")
        self.assertIn("error", _kinds(events))

    def test_fallback_disabled_with_rescue_off(self) -> None:
        tool = _DummyTool()
        client = _NoToolSupportClient([_turn("never reached")])
        events, emit = _recorder()
        run_agent(
            client, "tiny", [tool], [{"role": "user", "content": "x"}],
            emit=emit, harness=HarnessConfig(text_tool_rescue=False),
        )
        self.assertNotIn("tools_fallback", _kinds(events))
        self.assertIn("error", _kinds(events))

    def test_nudge_in_text_mode_teaches_json_shape(self) -> None:
        """In fallback mode the nudge restates the fenced-JSON format."""
        tool = _DummyTool(name="write")
        client = _NoToolSupportClient([
            # Prints code (no JSON call) on a build request -> nudge.
            _turn("```python\nprint('hi')\n```"),
            # Answers the nudge with a proper fenced JSON call.
            _turn(
                "```json\n"
                '{"name": "write", "arguments": {"file_path": "a.py", '
                '"content": "print(1)"}}\n'
                "```"
            ),
            _turn("created it"),
        ])
        events, emit = _recorder()
        messages = [{"role": "user", "content": "create a.py that prints hi"}]
        result = run_agent(client, "tiny", [tool], messages, emit=emit)

        self.assertEqual(result, "created it")
        self.assertIn("nudge", _kinds(events))
        self.assertEqual(len(tool.calls), 1)
        nudges = [
            m for m in messages
            if m.get("role") == "user"
            and "fenced JSON tool" in m.get("content", "")
        ]
        self.assertEqual(len(nudges), 1)


# ---------------------------------------------------------------------------
# Error-stop guard
# ---------------------------------------------------------------------------


class TestErrorStopGuard(unittest.TestCase):
    """Finishing right after a failed tool call draws one push-back."""

    def test_pushback_on_finishing_after_error(self) -> None:
        tool = _DummyTool(result="Error: old_text not found in app.py")
        client = _ScriptedClient([
            _turn("", [_call("dummy", {"arg": "bad"})]),
            _turn("I fixed app.py!"),          # finishing on an error
            _turn("Sorry — actually retrying"),  # reply to the push-back
        ])
        events, emit = _recorder()
        messages = [{"role": "user", "content": "fix app.py"}]
        result = run_agent(client, "m", [tool], messages, emit=emit)

        self.assertEqual(result, "Sorry — actually retrying")
        self.assertIn("error_stop", _kinds(events))
        pushbacks = [
            m for m in messages
            if m.get("role") == "user"
            and "tool call FAILED" in m.get("content", "")
        ]
        self.assertEqual(len(pushbacks), 1)

    def test_pushback_only_once_per_turn(self) -> None:
        tool = _DummyTool(result="Error: still failing")
        client = _ScriptedClient([
            _turn("", [_call("dummy", {"arg": "a"})]),
            _turn("done!"),      # push-back 1 fires
            _turn("done again!"),  # still ends on error, but no 2nd push
        ])
        events, emit = _recorder()
        run_agent(client, "m", [tool],
                  [{"role": "user", "content": "go"}], emit=emit)
        self.assertEqual(_kinds(events).count("error_stop"), 1)

    def test_no_pushback_after_successful_tool(self) -> None:
        tool = _DummyTool(result="wrote file ok")
        client = _ScriptedClient([
            _turn("", [_call("dummy", {"arg": "a"})]),
            _turn("all done"),
        ])
        events, emit = _recorder()
        run_agent(client, "m", [tool],
                  [{"role": "user", "content": "go"}], emit=emit)
        self.assertNotIn("error_stop", _kinds(events))

    def test_no_pushback_when_error_was_recovered(self) -> None:
        """An error followed by a successful call is a recovery, not a stop."""
        fail = _DummyTool(name="edit", result="Error: not found")
        ok = _DummyTool(name="write", result="Wrote file")
        client = _ScriptedClient([
            _turn("", [_call("edit", {"arg": "x"})]),
            _turn("", [_call("write", {"arg": "y"})]),
            _turn("all done"),
        ])
        events, emit = _recorder()
        run_agent(client, "m", [fail, ok],
                  [{"role": "user", "content": "go"}], emit=emit)
        self.assertNotIn("error_stop", _kinds(events))

    def test_no_pushback_on_plain_answer_turn(self) -> None:
        client = _ScriptedClient([_turn("The answer is 42.")])
        events, emit = _recorder()
        run_agent(client, "m", [],
                  [{"role": "user", "content": "what is 6*7?"}], emit=emit)
        self.assertNotIn("error_stop", _kinds(events))

    def test_guard_disabled_by_config(self) -> None:
        tool = _DummyTool(result="Error: nope")
        client = _ScriptedClient([
            _turn("", [_call("dummy", {"arg": "a"})]),
            _turn("done!"),
        ])
        events, emit = _recorder()
        run_agent(
            client, "m", [tool], [{"role": "user", "content": "go"}],
            emit=emit, harness=HarnessConfig(error_stop_guard=False),
        )
        self.assertNotIn("error_stop", _kinds(events))

    def test_pushback_on_unaddressed_verify_warning(self) -> None:
        """Finishing on a syntax WARNING from the verify gate pushes back."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "bad.py")
            client = _ScriptedClient([
                _turn("", [_call("write", {
                    "file_path": path, "content": "def f(:\n",
                })]),
                _turn("wrote it, done!"),   # ignoring the WARNING
                _turn("let me fix that"),   # reply to the push-back
            ])
            events, emit = _recorder()
            result = run_agent(
                client, "m", [_FileWriteTool()],
                [{"role": "user", "content": "write bad.py"}], emit=emit,
            )
        self.assertEqual(result, "let me fix that")
        kinds = _kinds(events)
        self.assertIn("verify_warning", kinds)
        self.assertIn("error_stop", kinds)


# ---------------------------------------------------------------------------
# Search-then-write ordering guard
# ---------------------------------------------------------------------------


class TestSearchThenWriteGuard(unittest.TestCase):
    """Mutations issued alongside discovery calls are deferred."""

    def test_mixed_turn_defers_write_but_runs_search(self) -> None:
        grep = _DummyTool(name="grep", result="src/utils.py:5:def calc_total")
        write = _DummyTool(name="write", result="wrote")
        client = _ScriptedClient([
            _turn("", [
                _call("grep", {"pattern": "calc_total"}),
                _call("write", {"file_path": "new.py", "content": "x"}),
            ]),
            # Re-issues the write alone after seeing the search results.
            _turn("", [_call("write", {
                "file_path": "src/utils.py", "content": "fixed",
            })]),
            _turn("done"),
        ])
        events, emit = _recorder()
        messages = [{"role": "user", "content": "find and fix calc_total"}]
        result = run_agent(client, "m", [grep, write], messages, emit=emit)

        self.assertEqual(result, "done")
        self.assertEqual(len(grep.calls), 1)      # search ran
        self.assertEqual(len(write.calls), 1)     # only the re-issue ran
        self.assertEqual(write.calls[0]["file_path"], "src/utils.py")
        self.assertIn("write_deferred", _kinds(events))
        deferred = [
            m for m in messages
            if m.get("role") == "tool" and "deferred" in m.get("content", "")
        ]
        self.assertEqual(len(deferred), 1)

    def test_write_only_turn_not_deferred(self) -> None:
        write = _DummyTool(name="write", result="wrote")
        client = _ScriptedClient([
            _turn("", [_call("write", {"file_path": "a.py", "content": "x"})]),
            _turn("done"),
        ])
        events, emit = _recorder()
        run_agent(client, "m", [write],
                  [{"role": "user", "content": "write a.py"}], emit=emit)
        self.assertEqual(len(write.calls), 1)
        self.assertNotIn("write_deferred", _kinds(events))

    def test_read_plus_write_not_deferred(self) -> None:
        """Only grep/glob defer writes; read+write stays allowed."""
        read = _DummyTool(name="read", result="contents")
        write = _DummyTool(name="write", result="wrote")
        client = _ScriptedClient([
            _turn("", [
                _call("read", {"file_path": "a.py"}),
                _call("write", {"file_path": "b.py", "content": "x"}),
            ]),
            _turn("done"),
        ])
        events, emit = _recorder()
        run_agent(client, "m", [read, write],
                  [{"role": "user", "content": "copy a to b"}], emit=emit)
        self.assertEqual(len(write.calls), 1)
        self.assertNotIn("write_deferred", _kinds(events))

    def test_guard_disabled_by_config(self) -> None:
        grep = _DummyTool(name="grep", result="hit")
        write = _DummyTool(name="write", result="wrote")
        client = _ScriptedClient([
            _turn("", [
                _call("grep", {"pattern": "x"}),
                _call("write", {"file_path": "a.py", "content": "x"}),
            ]),
            _turn("done"),
        ])
        events, emit = _recorder()
        run_agent(
            client, "m", [grep, write],
            [{"role": "user", "content": "go"}], emit=emit,
            harness=HarnessConfig(defer_writes_after_search=False),
        )
        self.assertEqual(len(write.calls), 1)
        self.assertNotIn("write_deferred", _kinds(events))


# ---------------------------------------------------------------------------
# Summary compaction
# ---------------------------------------------------------------------------


def _long_history(n: int = 60) -> list[dict[str, Any]]:
    """A history long enough to trip the message-count threshold."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"message {i}"})
    return messages


class TestSummaryCompaction(unittest.TestCase):
    """compact_mode='summarize' replaces old history with a summary."""

    def test_summary_replaces_old_span(self) -> None:
        messages = _long_history()
        original_len = len(messages)
        client = _ScriptedClient([
            _turn("SUMMARY: user wants X; wrote a.py; next: tests"),
            _turn("continuing"),
        ])
        events, emit = _recorder()
        result = run_agent(
            client, "m", [], messages, emit=emit,
            harness=HarnessConfig(compact_mode="summarize"),
        )

        self.assertEqual(result, "continuing")
        compactions = [e for e in events if e.kind == "compaction"]
        self.assertEqual(len(compactions), 1)
        self.assertEqual(compactions[0].data["mode"], "summarize")
        self.assertLess(len(messages), original_len)
        summary_msgs = [
            m for m in messages
            if "summarized to save context" in m.get("content", "")
        ]
        self.assertEqual(len(summary_msgs), 1)
        self.assertIn("SUMMARY: user wants X", summary_msgs[0]["content"])
        # System prompt survives compaction.
        self.assertEqual(messages[0]["role"], "system")

    def test_falls_back_to_truncation_on_failure(self) -> None:
        messages = _long_history()

        class _FailingThenOkClient:
            """Summarizer call raises; the following real call works."""

            def __init__(self) -> None:
                self.calls = 0

            def chat_stream(self, model, msgs, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise ProviderRequestError("boom")
                return iter(_turn("continuing"))

        events, emit = _recorder()
        result = run_agent(
            _FailingThenOkClient(), "m", [], messages, emit=emit,
            harness=HarnessConfig(compact_mode="summarize"),
        )
        self.assertEqual(result, "continuing")
        compactions = [e for e in events if e.kind == "compaction"]
        self.assertEqual(len(compactions), 1)
        self.assertEqual(compactions[0].data["mode"], "truncate")

    def test_truncate_mode_never_calls_summarizer(self) -> None:
        messages = _long_history()
        client = _ScriptedClient([_turn("continuing")])
        run_agent(client, "m", [], messages,
                  harness=HarnessConfig(compact_mode="truncate"))
        # Only the actual conversation call happened.
        self.assertEqual(len(client.requests), 1)


if __name__ == "__main__":
    unittest.main()
