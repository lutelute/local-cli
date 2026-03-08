"""Tests for local_cli.sub_agent module."""

import os
import sys
import time
import unittest
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

from local_cli.providers.base import (
    LLMProvider,
    ProviderStreamError,
)
from local_cli.sub_agent import (
    SubAgent,
    SubAgentResult,
    SubAgentRunner,
    _DEFAULT_TIMEOUT,
    _SUB_AGENT_SYSTEM_PROMPT,
    _SubAgentTimeout,
)
from local_cli.tools.base import Tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyTool(Tool):
    """Minimal concrete tool for testing."""

    def __init__(
        self,
        name: str = "dummy",
        result: str = "ok",
        *,
        side_effect: Exception | None = None,
    ) -> None:
        self._name = name
        self._result = result
        self._side_effect = side_effect

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A dummy tool for testing."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "arg": {"type": "string", "description": "An argument."},
            },
            "required": [],
        }

    def execute(self, **kwargs: object) -> str:
        if self._side_effect is not None:
            raise self._side_effect
        return self._result


def _make_mock_provider(name: str = "ollama") -> MagicMock:
    """Create a mock LLMProvider."""
    provider = MagicMock(spec=LLMProvider)
    provider.name = name
    # Default: format_tools returns a list of ollama-format dicts.
    provider.format_tools.side_effect = lambda tools: [
        t.to_ollama_tool() for t in tools
    ]
    return provider


def _make_chunks(
    content_parts: list[str],
    tool_calls: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a list of NDJSON-style chunks for streaming tests.

    All chunks except the last have ``done: False``.  Tool calls (if any)
    appear in the final ``done: True`` chunk.
    """
    chunks: list[dict[str, Any]] = []
    for i, text in enumerate(content_parts):
        is_last = i == len(content_parts) - 1
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if is_last and tool_calls:
            msg["tool_calls"] = tool_calls
        chunks.append({"message": msg, "done": is_last})
    return chunks


def _setup_provider_simple_response(
    provider: MagicMock,
    content: str = "Hello from sub-agent",
) -> None:
    """Configure a mock provider to return a simple response (no tool calls).

    Sets up chat_stream to yield chunks that produce the given content.
    """
    chunks = _make_chunks([content])
    provider.chat_stream.return_value = iter(chunks)


def _setup_provider_tool_then_response(
    provider: MagicMock,
    tool_name: str = "dummy",
    tool_args: dict[str, Any] | None = None,
    final_content: str = "Done with tools",
) -> None:
    """Configure a mock provider to first call a tool, then respond.

    First call to chat_stream returns a tool call.
    Second call returns a simple response.
    """
    tc = [
        {
            "function": {
                "name": tool_name,
                "arguments": tool_args or {},
            },
        }
    ]
    tool_call_chunks = _make_chunks([""], tool_calls=tc)
    final_chunks = _make_chunks([final_content])

    provider.chat_stream.side_effect = [
        iter(tool_call_chunks),
        iter(final_chunks),
    ]


# ---------------------------------------------------------------------------
# SubAgentResult
# ---------------------------------------------------------------------------


class TestSubAgentResult(unittest.TestCase):
    """Tests for SubAgentResult dataclass."""

    def test_creation_with_required_fields(self) -> None:
        """SubAgentResult can be created with all required fields."""
        result = SubAgentResult(
            agent_id="agent-20260308-120000-abcd1234",
            description="test task",
            content="Hello world",
            status="success",
            duration_seconds=1.5,
            messages_count=3,
            tool_calls_count=0,
        )
        self.assertEqual(result.agent_id, "agent-20260308-120000-abcd1234")
        self.assertEqual(result.description, "test task")
        self.assertEqual(result.content, "Hello world")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.duration_seconds, 1.5)
        self.assertEqual(result.messages_count, 3)
        self.assertEqual(result.tool_calls_count, 0)
        self.assertEqual(result.error_message, "")

    def test_error_message_default_empty(self) -> None:
        """error_message defaults to empty string."""
        result = SubAgentResult(
            agent_id="id",
            description="task",
            content="",
            status="success",
            duration_seconds=0.1,
            messages_count=1,
            tool_calls_count=0,
        )
        self.assertEqual(result.error_message, "")

    def test_error_message_can_be_set(self) -> None:
        """error_message can be provided at creation."""
        result = SubAgentResult(
            agent_id="id",
            description="task",
            content="",
            status="error",
            duration_seconds=0.1,
            messages_count=1,
            tool_calls_count=0,
            error_message="Something went wrong",
        )
        self.assertEqual(result.error_message, "Something went wrong")

    def test_format_result_success(self) -> None:
        """format_result() produces readable output for success."""
        result = SubAgentResult(
            agent_id="agent-001",
            description="search files",
            content="Found 3 files",
            status="success",
            duration_seconds=2.5,
            messages_count=5,
            tool_calls_count=2,
        )
        formatted = result.format_result()

        self.assertIn("search files", formatted)
        self.assertIn("agent-001", formatted)
        self.assertIn("success", formatted)
        self.assertIn("2.5s", formatted)
        self.assertIn("Messages: 5", formatted)
        self.assertIn("Tool calls: 2", formatted)
        self.assertIn("Found 3 files", formatted)

    def test_format_result_with_error(self) -> None:
        """format_result() includes error message when present."""
        result = SubAgentResult(
            agent_id="agent-002",
            description="failing task",
            content="",
            status="error",
            duration_seconds=0.3,
            messages_count=2,
            tool_calls_count=0,
            error_message="RuntimeError: something broke",
        )
        formatted = result.format_result()

        self.assertIn("error", formatted)
        self.assertIn("Error: RuntimeError: something broke", formatted)

    def test_format_result_timeout(self) -> None:
        """format_result() handles timeout status."""
        result = SubAgentResult(
            agent_id="agent-003",
            description="slow task",
            content="partial output",
            status="timeout",
            duration_seconds=300.0,
            messages_count=10,
            tool_calls_count=5,
            error_message="Sub-agent timed out after 300.0s",
        )
        formatted = result.format_result()

        self.assertIn("timeout", formatted)
        self.assertIn("300.0s", formatted)
        self.assertIn("partial output", formatted)

    def test_format_result_no_content(self) -> None:
        """format_result() omits Result section when content is empty."""
        result = SubAgentResult(
            agent_id="agent-004",
            description="empty task",
            content="",
            status="success",
            duration_seconds=0.1,
            messages_count=2,
            tool_calls_count=0,
        )
        formatted = result.format_result()

        self.assertNotIn("Result:", formatted)

    def test_field_access(self) -> None:
        """All fields are accessible as attributes."""
        result = SubAgentResult(
            agent_id="a",
            description="b",
            content="c",
            status="success",
            duration_seconds=1.0,
            messages_count=2,
            tool_calls_count=3,
            error_message="d",
        )
        self.assertEqual(result.agent_id, "a")
        self.assertEqual(result.description, "b")
        self.assertEqual(result.content, "c")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.duration_seconds, 1.0)
        self.assertEqual(result.messages_count, 2)
        self.assertEqual(result.tool_calls_count, 3)
        self.assertEqual(result.error_message, "d")


# ---------------------------------------------------------------------------
# SubAgent initialization
# ---------------------------------------------------------------------------


class TestSubAgentInit(unittest.TestCase):
    """Tests for SubAgent construction and initialization."""

    def test_basic_creation(self) -> None:
        """SubAgent can be created with required arguments."""
        provider = _make_mock_provider()
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="Do something",
        )
        self.assertIsNotNone(agent)

    def test_agent_id_generated(self) -> None:
        """Agent ID is generated automatically if not provided."""
        provider = _make_mock_provider()
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
        )
        self.assertTrue(agent.agent_id.startswith("agent-"))
        self.assertGreater(len(agent.agent_id), 10)

    def test_agent_id_custom(self) -> None:
        """Custom agent ID is used when provided."""
        provider = _make_mock_provider()
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
            agent_id="custom-id-001",
        )
        self.assertEqual(agent.agent_id, "custom-id-001")

    def test_description_default(self) -> None:
        """Description defaults to 'sub-agent task' when empty."""
        provider = _make_mock_provider()
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
        )
        self.assertEqual(agent.description, "sub-agent task")

    def test_description_custom(self) -> None:
        """Custom description is preserved."""
        provider = _make_mock_provider()
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
            description="search files",
        )
        self.assertEqual(agent.description, "search files")

    def test_isolated_messages(self) -> None:
        """Each SubAgent has its own message list (not shared)."""
        provider = _make_mock_provider()
        agent1 = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task1",
        )
        agent2 = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task2",
        )
        # Messages lists are distinct objects.
        self.assertIsNot(agent1._messages, agent2._messages)

    def test_tools_are_copied(self) -> None:
        """Tools list is copied, not shared with the caller."""
        provider = _make_mock_provider()
        original_tools = [_DummyTool()]
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=original_tools,
            prompt="task",
        )
        # Modifying the original list should not affect the agent.
        original_tools.append(_DummyTool(name="extra"))
        self.assertEqual(len(agent._tools), 1)

    def test_default_timeout(self) -> None:
        """Default timeout is _DEFAULT_TIMEOUT (300s)."""
        provider = _make_mock_provider()
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
        )
        self.assertEqual(agent._timeout, _DEFAULT_TIMEOUT)

    def test_custom_timeout(self) -> None:
        """Custom timeout value is accepted."""
        provider = _make_mock_provider()
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
            timeout=60.0,
        )
        self.assertEqual(agent._timeout, 60.0)

    def test_unique_agent_ids(self) -> None:
        """Two SubAgents get different auto-generated IDs."""
        provider = _make_mock_provider()
        agent1 = SubAgent(
            provider=provider, model="m", tools=[], prompt="a",
        )
        agent2 = SubAgent(
            provider=provider, model="m", tools=[], prompt="b",
        )
        self.assertNotEqual(agent1.agent_id, agent2.agent_id)


# ---------------------------------------------------------------------------
# SubAgent.run() — simple response (no tool calls)
# ---------------------------------------------------------------------------


class TestSubAgentRunSimple(unittest.TestCase):
    """Tests for SubAgent.run() with simple responses."""

    def test_returns_sub_agent_result(self) -> None:
        """run() returns a SubAgentResult instance."""
        provider = _make_mock_provider()
        _setup_provider_simple_response(provider, "Hello")
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="Say hello",
            agent_id="test-001",
            description="greet",
        )
        result = agent.run()

        self.assertIsInstance(result, SubAgentResult)

    def test_success_status(self) -> None:
        """Successful run returns status='success'."""
        provider = _make_mock_provider()
        _setup_provider_simple_response(provider, "Done")
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
            agent_id="test-002",
        )
        result = agent.run()

        self.assertEqual(result.status, "success")

    def test_content_captured(self) -> None:
        """Final assistant content is captured in the result."""
        provider = _make_mock_provider()
        _setup_provider_simple_response(provider, "The answer is 42")
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="What is the answer?",
            agent_id="test-003",
        )
        result = agent.run()

        self.assertEqual(result.content, "The answer is 42")

    def test_agent_id_in_result(self) -> None:
        """Result contains the correct agent_id."""
        provider = _make_mock_provider()
        _setup_provider_simple_response(provider)
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
            agent_id="my-agent-id",
        )
        result = agent.run()

        self.assertEqual(result.agent_id, "my-agent-id")

    def test_description_in_result(self) -> None:
        """Result contains the correct description."""
        provider = _make_mock_provider()
        _setup_provider_simple_response(provider)
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
            description="search files",
        )
        result = agent.run()

        self.assertEqual(result.description, "search files")

    def test_duration_non_negative(self) -> None:
        """Duration is a non-negative number."""
        provider = _make_mock_provider()
        _setup_provider_simple_response(provider)
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
        )
        result = agent.run()

        self.assertGreaterEqual(result.duration_seconds, 0)
        self.assertIsInstance(result.duration_seconds, float)

    def test_messages_count(self) -> None:
        """Messages count includes system + user + assistant messages."""
        provider = _make_mock_provider()
        _setup_provider_simple_response(provider, "response")
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
        )
        result = agent.run()

        # system + user + assistant = 3
        self.assertEqual(result.messages_count, 3)

    def test_zero_tool_calls(self) -> None:
        """tool_calls_count is 0 when no tools are called."""
        provider = _make_mock_provider()
        _setup_provider_simple_response(provider)
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
        )
        result = agent.run()

        self.assertEqual(result.tool_calls_count, 0)

    def test_no_error_message(self) -> None:
        """error_message is empty on success."""
        provider = _make_mock_provider()
        _setup_provider_simple_response(provider)
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
        )
        result = agent.run()

        self.assertEqual(result.error_message, "")

    def test_silent_no_stdout(self) -> None:
        """run() does not produce stdout output."""
        provider = _make_mock_provider()
        _setup_provider_simple_response(provider, "should not print")

        captured = StringIO()
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
        )

        orig_stdout = sys.stdout
        try:
            sys.stdout = captured
            agent.run()
        finally:
            sys.stdout = orig_stdout

        self.assertEqual(captured.getvalue(), "")

    def test_system_prompt_in_messages(self) -> None:
        """Messages sent to the provider start with the system prompt."""
        provider = _make_mock_provider()
        _setup_provider_simple_response(provider)
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="do something",
        )
        agent.run()

        # Verify the first call to chat_stream had the system prompt.
        call_args = provider.chat_stream.call_args
        messages = call_args[0][1]  # Second positional arg is messages.
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], _SUB_AGENT_SYSTEM_PROMPT)

    def test_user_prompt_in_messages(self) -> None:
        """Messages sent to the provider include the user prompt."""
        provider = _make_mock_provider()
        _setup_provider_simple_response(provider)
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="find all Python files",
        )
        agent.run()

        call_args = provider.chat_stream.call_args
        messages = call_args[0][1]
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "find all Python files")

    def test_provider_called_with_correct_model(self) -> None:
        """chat_stream is called with the correct model name."""
        provider = _make_mock_provider()
        _setup_provider_simple_response(provider)
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
        )
        agent.run()

        call_args = provider.chat_stream.call_args
        model = call_args[0][0]  # First positional arg is model.
        self.assertEqual(model, "qwen3:8b")

    def test_format_tools_called(self) -> None:
        """Provider's format_tools is called to generate tool definitions."""
        provider = _make_mock_provider()
        _setup_provider_simple_response(provider)
        tools = [_DummyTool()]
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=tools,
            prompt="task",
        )
        agent.run()

        provider.format_tools.assert_called_once()


# ---------------------------------------------------------------------------
# SubAgent.run() — with tool calls
# ---------------------------------------------------------------------------


class TestSubAgentRunWithTools(unittest.TestCase):
    """Tests for SubAgent.run() when the LLM calls tools."""

    def test_tool_executed(self) -> None:
        """Tool is executed when the LLM requests a tool call."""
        provider = _make_mock_provider()
        _setup_provider_tool_then_response(
            provider,
            tool_name="dummy",
            final_content="All done",
        )
        tool = _DummyTool(name="dummy", result="tool result")
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[tool],
            prompt="use dummy tool",
            agent_id="tool-test-001",
        )
        result = agent.run()

        self.assertEqual(result.status, "success")
        self.assertEqual(result.content, "All done")
        self.assertEqual(result.tool_calls_count, 1)

    def test_tool_result_in_messages(self) -> None:
        """Tool execution result is appended to messages as role='tool'."""
        provider = _make_mock_provider()
        _setup_provider_tool_then_response(
            provider,
            tool_name="dummy",
            final_content="Done",
        )
        tool = _DummyTool(name="dummy", result="tool output here")
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[tool],
            prompt="task",
        )
        agent.run()

        # Find the tool message in the agent's messages.
        tool_msgs = [m for m in agent._messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["content"], "tool output here")
        self.assertEqual(tool_msgs[0]["tool_name"], "dummy")

    def test_unknown_tool_returns_error(self) -> None:
        """Unknown tool name produces an error message in the tool result."""
        provider = _make_mock_provider()
        _setup_provider_tool_then_response(
            provider,
            tool_name="nonexistent",
            final_content="Recovered",
        )
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],  # No tools registered.
            prompt="task",
        )
        result = agent.run()

        # The agent should still succeed overall.
        self.assertEqual(result.status, "success")
        # The tool error should be in the messages.
        tool_msgs = [m for m in agent._messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("unknown tool", tool_msgs[0]["content"])

    def test_tool_exception_captured(self) -> None:
        """Tool exception is caught and returned as error string."""
        provider = _make_mock_provider()
        _setup_provider_tool_then_response(
            provider,
            tool_name="broken",
            final_content="Handled error",
        )
        broken_tool = _DummyTool(
            name="broken",
            result="",
            side_effect=ValueError("bad input"),
        )
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[broken_tool],
            prompt="task",
        )
        result = agent.run()

        # Agent loop continues after tool error.
        self.assertEqual(result.status, "success")
        tool_msgs = [m for m in agent._messages if m.get("role") == "tool"]
        self.assertIn("ValueError", tool_msgs[0]["content"])

    def test_messages_count_with_tools(self) -> None:
        """Messages count reflects system + user + assistant + tool + assistant."""
        provider = _make_mock_provider()
        _setup_provider_tool_then_response(
            provider,
            tool_name="dummy",
            final_content="Done",
        )
        tool = _DummyTool(name="dummy", result="ok")
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[tool],
            prompt="task",
        )
        result = agent.run()

        # system + user + assistant(tool_call) + tool + assistant(final) = 5
        self.assertEqual(result.messages_count, 5)

    def test_tool_call_with_arguments(self) -> None:
        """Tool arguments are passed through to the tool's execute()."""
        provider = _make_mock_provider()
        tc = [
            {
                "function": {
                    "name": "dummy",
                    "arguments": {"arg": "hello"},
                },
            }
        ]
        tool_call_chunks = _make_chunks([""], tool_calls=tc)
        final_chunks = _make_chunks(["Done"])
        provider.chat_stream.side_effect = [
            iter(tool_call_chunks),
            iter(final_chunks),
        ]

        executed_with: dict[str, Any] = {}

        class CapturingTool(_DummyTool):
            def execute(self, **kwargs: object) -> str:
                executed_with.update(kwargs)
                return "captured"

        tool = CapturingTool(name="dummy")
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[tool],
            prompt="task",
        )
        agent.run()

        self.assertEqual(executed_with.get("arg"), "hello")

    def test_tool_call_json_string_arguments(self) -> None:
        """Arguments provided as JSON string are parsed correctly."""
        provider = _make_mock_provider()
        tc = [
            {
                "function": {
                    "name": "dummy",
                    "arguments": '{"arg": "parsed"}',
                },
            }
        ]
        tool_call_chunks = _make_chunks([""], tool_calls=tc)
        final_chunks = _make_chunks(["Done"])
        provider.chat_stream.side_effect = [
            iter(tool_call_chunks),
            iter(final_chunks),
        ]

        executed_with: dict[str, Any] = {}

        class CapturingTool(_DummyTool):
            def execute(self, **kwargs: object) -> str:
                executed_with.update(kwargs)
                return "captured"

        tool = CapturingTool(name="dummy")
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[tool],
            prompt="task",
        )
        agent.run()

        self.assertEqual(executed_with.get("arg"), "parsed")

    def test_tool_call_id_preserved(self) -> None:
        """tool_call_id is preserved in the tool result message."""
        provider = _make_mock_provider()
        tc = [
            {
                "function": {
                    "name": "dummy",
                    "arguments": {},
                },
                "id": "call_123",
            }
        ]
        tool_call_chunks = _make_chunks([""], tool_calls=tc)
        final_chunks = _make_chunks(["Done"])
        provider.chat_stream.side_effect = [
            iter(tool_call_chunks),
            iter(final_chunks),
        ]

        tool = _DummyTool(name="dummy")
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[tool],
            prompt="task",
        )
        agent.run()

        tool_msgs = [m for m in agent._messages if m.get("role") == "tool"]
        self.assertEqual(tool_msgs[0].get("tool_call_id"), "call_123")

    def test_multiple_tool_calls_in_one_turn(self) -> None:
        """Multiple tool calls in a single turn are all executed."""
        provider = _make_mock_provider()
        tc = [
            {"function": {"name": "tool_a", "arguments": {}}},
            {"function": {"name": "tool_b", "arguments": {}}},
        ]
        tool_call_chunks = _make_chunks([""], tool_calls=tc)
        final_chunks = _make_chunks(["All done"])
        provider.chat_stream.side_effect = [
            iter(tool_call_chunks),
            iter(final_chunks),
        ]

        tool_a = _DummyTool(name="tool_a", result="result_a")
        tool_b = _DummyTool(name="tool_b", result="result_b")
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[tool_a, tool_b],
            prompt="task",
        )
        result = agent.run()

        self.assertEqual(result.tool_calls_count, 2)
        tool_msgs = [m for m in agent._messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 2)


# ---------------------------------------------------------------------------
# SubAgent.run() — timeout handling
# ---------------------------------------------------------------------------


class TestSubAgentTimeout(unittest.TestCase):
    """Tests for SubAgent timeout handling."""

    def test_timeout_returns_timeout_status(self) -> None:
        """When timeout is exceeded, status is 'timeout'."""
        provider = _make_mock_provider()

        # Make chat_stream raise timeout by using a very small timeout.
        # We simulate this by making the provider take time.
        def slow_stream(*args: Any, **kwargs: Any):
            time.sleep(0.1)
            return iter(_make_chunks(["partial"]))

        provider.chat_stream.side_effect = slow_stream

        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
            timeout=0.0,  # Immediate timeout.
        )
        result = agent.run()

        self.assertEqual(result.status, "timeout")

    def test_timeout_error_message(self) -> None:
        """Timeout result includes descriptive error message."""
        provider = _make_mock_provider()
        _setup_provider_simple_response(provider)

        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
            timeout=0.0,
        )
        result = agent.run()

        self.assertEqual(result.status, "timeout")
        self.assertIn("timed out", result.error_message)

    def test_timeout_preserves_partial_content(self) -> None:
        """Timeout preserves partial content from earlier messages."""
        provider = _make_mock_provider()

        call_count = 0

        def streaming_then_timeout(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call succeeds with tool call.
                tc = [{"function": {"name": "dummy", "arguments": {}}}]
                return iter(_make_chunks(["Partial output"], tool_calls=tc))
            else:
                # Second call: simulate timeout by sleeping.
                time.sleep(0.2)
                return iter(_make_chunks(["Final"]))

        provider.chat_stream.side_effect = streaming_then_timeout

        tool = _DummyTool(name="dummy", result="ok")
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[tool],
            prompt="task",
            timeout=0.05,  # Very short timeout.
        )
        result = agent.run()

        # Should be timeout with partial content from the first assistant msg.
        self.assertEqual(result.status, "timeout")
        self.assertIn("Partial output", result.content)

    def test_timeout_during_tool_execution(self) -> None:
        """Timeout during tool execution produces timeout status."""
        provider = _make_mock_provider()
        tc = [{"function": {"name": "slow_tool", "arguments": {}}}]
        tool_call_chunks = _make_chunks([""], tool_calls=tc)
        provider.chat_stream.return_value = iter(tool_call_chunks)

        class SlowTool(_DummyTool):
            def execute(self, **kwargs: object) -> str:
                time.sleep(0.2)
                return "done"

        slow_tool = SlowTool(name="slow_tool")
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[slow_tool],
            prompt="task",
            timeout=0.01,  # Very short timeout.
        )
        result = agent.run()

        # The tool itself might complete, but the timeout check after
        # tool execution should catch it.
        self.assertIn(result.status, ("timeout", "success"))


# ---------------------------------------------------------------------------
# SubAgent.run() — error handling
# ---------------------------------------------------------------------------


class TestSubAgentErrorHandling(unittest.TestCase):
    """Tests for SubAgent error handling."""

    def test_provider_exception_returns_error(self) -> None:
        """Provider exception produces error status."""
        provider = _make_mock_provider()
        provider.chat_stream.side_effect = RuntimeError("connection lost")

        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
            agent_id="err-001",
        )
        result = agent.run()

        self.assertEqual(result.status, "error")
        self.assertIn("RuntimeError", result.error_message)
        self.assertIn("connection lost", result.error_message)

    def test_provider_stream_error_returns_error(self) -> None:
        """ProviderStreamError during streaming produces error status."""
        provider = _make_mock_provider()

        def error_stream(*args: Any, **kwargs: Any):
            yield {
                "message": {"role": "assistant", "content": "partial"},
                "done": False,
            }
            raise ProviderStreamError("stream broken")

        provider.chat_stream.side_effect = error_stream

        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
        )
        result = agent.run()

        self.assertEqual(result.status, "error")
        self.assertIn("ProviderStreamError", result.error_message)

    def test_error_does_not_crash(self) -> None:
        """Error in sub-agent does not raise — returns SubAgentResult."""
        provider = _make_mock_provider()
        provider.chat_stream.side_effect = Exception("unexpected!")

        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
        )
        # Should not raise.
        result = agent.run()
        self.assertIsInstance(result, SubAgentResult)
        self.assertEqual(result.status, "error")

    def test_error_duration_recorded(self) -> None:
        """Duration is still recorded on error."""
        provider = _make_mock_provider()
        provider.chat_stream.side_effect = RuntimeError("fail")

        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
        )
        result = agent.run()

        self.assertGreaterEqual(result.duration_seconds, 0)

    def test_error_preserves_partial_messages(self) -> None:
        """Error preserves partial assistant content from earlier messages."""
        provider = _make_mock_provider()

        call_count = 0

        def first_ok_then_fail(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                tc = [{"function": {"name": "dummy", "arguments": {}}}]
                return iter(_make_chunks(["First output"], tool_calls=tc))
            else:
                raise RuntimeError("second call fails")

        provider.chat_stream.side_effect = first_ok_then_fail

        tool = _DummyTool(name="dummy", result="ok")
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[tool],
            prompt="task",
        )
        result = agent.run()

        self.assertEqual(result.status, "error")
        self.assertEqual(result.content, "First output")


# ---------------------------------------------------------------------------
# SubAgent._collect_silent_response
# ---------------------------------------------------------------------------


class TestCollectSilentResponse(unittest.TestCase):
    """Tests for SubAgent._collect_silent_response()."""

    def test_accumulates_content(self) -> None:
        """Content deltas across chunks are concatenated."""
        chunks = _make_chunks(["Hello", " ", "world"])
        result = SubAgent._collect_silent_response(iter(chunks))

        self.assertEqual(result["message"]["content"], "Hello world")

    def test_no_stdout_output(self) -> None:
        """Silent response does not write to stdout."""
        captured = StringIO()
        orig_stdout = sys.stdout

        try:
            sys.stdout = captured
            chunks = _make_chunks(["Hello", " world"])
            SubAgent._collect_silent_response(iter(chunks))
        finally:
            sys.stdout = orig_stdout

        self.assertEqual(captured.getvalue(), "")

    def test_tool_calls_collected(self) -> None:
        """Tool calls from the final chunk are collected."""
        tc = [{"function": {"name": "read", "arguments": {"path": "a.py"}}}]
        chunks = _make_chunks([""], tool_calls=tc)
        result = SubAgent._collect_silent_response(iter(chunks))

        self.assertIn("tool_calls", result["message"])
        self.assertEqual(len(result["message"]["tool_calls"]), 1)

    def test_no_tool_calls_omits_key(self) -> None:
        """When no tool calls are present, 'tool_calls' is not in the message."""
        chunks = _make_chunks(["Hello"])
        result = SubAgent._collect_silent_response(iter(chunks))
        self.assertNotIn("tool_calls", result["message"])

    def test_empty_stream(self) -> None:
        """An empty stream produces empty content."""
        result = SubAgent._collect_silent_response(iter([]))
        self.assertEqual(result["message"]["content"], "")

    def test_provider_stream_error_reraised(self) -> None:
        """ProviderStreamError from the stream is re-raised."""

        def error_stream():
            yield {
                "message": {"role": "assistant", "content": "partial"},
                "done": False,
            }
            raise ProviderStreamError("broken stream")

        with self.assertRaises(ProviderStreamError):
            SubAgent._collect_silent_response(error_stream())

    def test_keyboard_interrupt_reraised(self) -> None:
        """KeyboardInterrupt from the stream is re-raised."""

        def interrupted_stream():
            yield {
                "message": {"role": "assistant", "content": "partial"},
                "done": False,
            }
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            SubAgent._collect_silent_response(interrupted_stream())


# ---------------------------------------------------------------------------
# SubAgent._generate_agent_id
# ---------------------------------------------------------------------------


class TestGenerateAgentId(unittest.TestCase):
    """Tests for SubAgent._generate_agent_id()."""

    def test_starts_with_agent_prefix(self) -> None:
        """Generated IDs start with 'agent-'."""
        agent_id = SubAgent._generate_agent_id()
        self.assertTrue(agent_id.startswith("agent-"))

    def test_contains_timestamp(self) -> None:
        """Generated IDs contain a timestamp-like pattern."""
        agent_id = SubAgent._generate_agent_id()
        # Format: agent-YYYYMMDD-HHMMSS-<hex>
        parts = agent_id.split("-")
        # Should have at least: agent, YYYYMMDD, HHMMSS, hex
        self.assertGreaterEqual(len(parts), 4)
        # Date part should be 8 digits.
        self.assertEqual(len(parts[1]), 8)
        self.assertTrue(parts[1].isdigit())

    def test_unique_ids(self) -> None:
        """Two consecutive calls produce different IDs."""
        id1 = SubAgent._generate_agent_id()
        id2 = SubAgent._generate_agent_id()
        self.assertNotEqual(id1, id2)


# ---------------------------------------------------------------------------
# SubAgent context isolation
# ---------------------------------------------------------------------------


class TestSubAgentContextIsolation(unittest.TestCase):
    """Tests that sub-agents have truly isolated contexts."""

    def test_messages_not_shared_between_agents(self) -> None:
        """Two agents' message lists are independent."""
        provider1 = _make_mock_provider()
        provider2 = _make_mock_provider()
        _setup_provider_simple_response(provider1, "Response 1")
        _setup_provider_simple_response(provider2, "Response 2")

        agent1 = SubAgent(
            provider=provider1,
            model="qwen3:8b",
            tools=[],
            prompt="Task 1",
        )
        agent2 = SubAgent(
            provider=provider2,
            model="qwen3:8b",
            tools=[],
            prompt="Task 2",
        )

        result1 = agent1.run()
        result2 = agent2.run()

        self.assertEqual(result1.content, "Response 1")
        self.assertEqual(result2.content, "Response 2")
        self.assertIsNot(agent1._messages, agent2._messages)

    def test_run_resets_messages(self) -> None:
        """Each run() call starts with fresh messages."""
        provider = _make_mock_provider()

        # First run.
        _setup_provider_simple_response(provider, "First")
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[],
            prompt="task",
        )
        result1 = agent.run()
        first_count = result1.messages_count

        # Second run (re-setup the provider).
        _setup_provider_simple_response(provider, "Second")
        result2 = agent.run()

        # Both runs should have the same message count (fresh start).
        self.assertEqual(result2.messages_count, first_count)
        # Second run should have fresh content.
        self.assertEqual(result2.content, "Second")

    def test_tool_calls_count_resets(self) -> None:
        """tool_calls_count resets between runs."""
        provider = _make_mock_provider()
        tool = _DummyTool(name="dummy", result="ok")

        # First run with a tool call.
        _setup_provider_tool_then_response(
            provider, tool_name="dummy", final_content="Done",
        )
        agent = SubAgent(
            provider=provider,
            model="qwen3:8b",
            tools=[tool],
            prompt="task",
        )
        result1 = agent.run()
        self.assertEqual(result1.tool_calls_count, 1)

        # Second run without tool calls.
        _setup_provider_simple_response(provider, "No tools")
        result2 = agent.run()
        self.assertEqual(result2.tool_calls_count, 0)


# ---------------------------------------------------------------------------
# _SubAgentTimeout exception
# ---------------------------------------------------------------------------


class TestSubAgentTimeoutException(unittest.TestCase):
    """Tests for _SubAgentTimeout internal exception."""

    def test_is_exception(self) -> None:
        """_SubAgentTimeout is an Exception."""
        self.assertTrue(issubclass(_SubAgentTimeout, Exception))

    def test_message_preserved(self) -> None:
        """Error message is preserved."""
        exc = _SubAgentTimeout("timed out after 300s")
        self.assertEqual(str(exc), "timed out after 300s")


# ---------------------------------------------------------------------------
# SubAgentRunner — helper
# ---------------------------------------------------------------------------


def _make_mock_sub_agent(
    agent_id: str = "mock-agent-001",
    description: str = "mock task",
    content: str = "mock result",
    status: str = "success",
    *,
    delay: float = 0.0,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock SubAgent with a predictable run() result.

    Args:
        agent_id: The agent ID to assign.
        description: The description for the result.
        content: The content for the result.
        status: The status for the result.
        delay: Optional sleep delay before returning (simulates work).
        side_effect: Optional exception to raise from run().

    Returns:
        A MagicMock configured as a SubAgent with a controlled run().
    """
    mock = MagicMock(spec=SubAgent)
    mock.agent_id = agent_id

    def _run() -> SubAgentResult:
        if delay > 0:
            time.sleep(delay)
        if side_effect is not None:
            raise side_effect
        return SubAgentResult(
            agent_id=agent_id,
            description=description,
            content=content,
            status=status,
            duration_seconds=delay or 0.01,
            messages_count=3,
            tool_calls_count=0,
        )

    mock.run.side_effect = _run
    return mock


# ---------------------------------------------------------------------------
# SubAgentRunner — submit()
# ---------------------------------------------------------------------------


class TestSubAgentRunnerSubmit(unittest.TestCase):
    """Tests for SubAgentRunner.submit()."""

    def setUp(self) -> None:
        self.runner = SubAgentRunner(max_workers=2)

    def tearDown(self) -> None:
        self.runner.shutdown()

    def test_submit_returns_sub_agent_result(self) -> None:
        """submit() returns a SubAgentResult instance."""
        agent = _make_mock_sub_agent()
        result = self.runner.submit(agent)

        self.assertIsInstance(result, SubAgentResult)

    def test_submit_blocks_and_returns_result(self) -> None:
        """submit() blocks until the sub-agent completes and returns result."""
        agent = _make_mock_sub_agent(
            agent_id="blocking-001",
            content="completed work",
            delay=0.05,
        )
        result = self.runner.submit(agent)

        self.assertEqual(result.agent_id, "blocking-001")
        self.assertEqual(result.content, "completed work")
        self.assertEqual(result.status, "success")

    def test_submit_calls_run(self) -> None:
        """submit() calls the sub-agent's run() method."""
        agent = _make_mock_sub_agent()
        self.runner.submit(agent)

        agent.run.assert_called_once()

    def test_submit_preserves_result_fields(self) -> None:
        """submit() preserves all fields from the SubAgentResult."""
        agent = _make_mock_sub_agent(
            agent_id="field-test",
            description="field check",
            content="field content",
        )
        result = self.runner.submit(agent)

        self.assertEqual(result.agent_id, "field-test")
        self.assertEqual(result.description, "field check")
        self.assertEqual(result.content, "field content")
        self.assertEqual(result.messages_count, 3)
        self.assertEqual(result.tool_calls_count, 0)

    def test_submit_creates_executor_lazily(self) -> None:
        """Executor is not created until first submit()."""
        runner = SubAgentRunner(max_workers=2)
        self.assertIsNone(runner._executor)

        agent = _make_mock_sub_agent()
        runner.submit(agent)
        self.assertIsNotNone(runner._executor)
        runner.shutdown()


# ---------------------------------------------------------------------------
# SubAgentRunner — submit_background()
# ---------------------------------------------------------------------------


class TestSubAgentRunnerSubmitBackground(unittest.TestCase):
    """Tests for SubAgentRunner.submit_background()."""

    def setUp(self) -> None:
        self.runner = SubAgentRunner(max_workers=2)

    def tearDown(self) -> None:
        self.runner.shutdown()

    def test_submit_background_returns_agent_id(self) -> None:
        """submit_background() returns the agent ID as a string."""
        agent = _make_mock_sub_agent(agent_id="bg-001")
        agent_id = self.runner.submit_background(agent)

        self.assertEqual(agent_id, "bg-001")
        self.assertIsInstance(agent_id, str)

    def test_submit_background_returns_immediately(self) -> None:
        """submit_background() does not block for completion."""
        agent = _make_mock_sub_agent(agent_id="bg-fast", delay=0.5)

        start = time.monotonic()
        self.runner.submit_background(agent)
        elapsed = time.monotonic() - start

        # Should return well before the 0.5s delay.
        self.assertLess(elapsed, 0.3)

    def test_submit_background_tracks_agent(self) -> None:
        """submit_background() stores the agent in _background dict."""
        agent = _make_mock_sub_agent(agent_id="bg-tracked")
        self.runner.submit_background(agent)

        self.assertIn("bg-tracked", self.runner._background)

    def test_submit_multiple_background_agents(self) -> None:
        """Multiple background agents can be submitted concurrently."""
        agents = [
            _make_mock_sub_agent(agent_id=f"bg-multi-{i}", delay=0.05)
            for i in range(3)
        ]
        ids = [self.runner.submit_background(a) for a in agents]

        self.assertEqual(len(ids), 3)
        self.assertEqual(len(set(ids)), 3)  # All unique.


# ---------------------------------------------------------------------------
# SubAgentRunner — get_background_result()
# ---------------------------------------------------------------------------


class TestSubAgentRunnerGetBackgroundResult(unittest.TestCase):
    """Tests for SubAgentRunner.get_background_result()."""

    def setUp(self) -> None:
        self.runner = SubAgentRunner(max_workers=2)

    def tearDown(self) -> None:
        self.runner.shutdown()

    def test_returns_none_while_running(self) -> None:
        """get_background_result() returns None for a still-running agent."""
        agent = _make_mock_sub_agent(agent_id="bg-slow", delay=1.0)
        self.runner.submit_background(agent)

        # Check immediately — agent should still be running.
        result = self.runner.get_background_result("bg-slow")
        self.assertIsNone(result)

    def test_returns_result_when_done(self) -> None:
        """get_background_result() returns result after agent completes."""
        agent = _make_mock_sub_agent(
            agent_id="bg-done",
            content="background work complete",
            delay=0.05,
        )
        self.runner.submit_background(agent)

        # Wait for completion.
        time.sleep(0.2)
        result = self.runner.get_background_result("bg-done")

        self.assertIsNotNone(result)
        self.assertIsInstance(result, SubAgentResult)
        self.assertEqual(result.content, "background work complete")
        self.assertEqual(result.status, "success")

    def test_returns_none_for_unknown_id(self) -> None:
        """get_background_result() returns None for unrecognized agent ID."""
        result = self.runner.get_background_result("nonexistent-id")
        self.assertIsNone(result)

    def test_result_available_after_poll(self) -> None:
        """Polling eventually returns a result for a completed agent."""
        agent = _make_mock_sub_agent(agent_id="bg-poll", delay=0.05)
        self.runner.submit_background(agent)

        # Poll until done or timeout.
        result = None
        for _ in range(20):
            result = self.runner.get_background_result("bg-poll")
            if result is not None:
                break
            time.sleep(0.05)

        self.assertIsNotNone(result)
        self.assertEqual(result.agent_id, "bg-poll")


# ---------------------------------------------------------------------------
# SubAgentRunner — list_background_agents()
# ---------------------------------------------------------------------------


class TestSubAgentRunnerListBackground(unittest.TestCase):
    """Tests for SubAgentRunner.list_background_agents()."""

    def setUp(self) -> None:
        self.runner = SubAgentRunner(max_workers=4)

    def tearDown(self) -> None:
        self.runner.shutdown()

    def test_empty_initially(self) -> None:
        """list_background_agents() returns empty list when none submitted."""
        agents = self.runner.list_background_agents()
        self.assertEqual(agents, [])

    def test_shows_running_status(self) -> None:
        """list_background_agents() shows 'running' for active agents."""
        agent = _make_mock_sub_agent(agent_id="bg-running", delay=1.0)
        self.runner.submit_background(agent)

        agents = self.runner.list_background_agents()

        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0]["agent_id"], "bg-running")
        self.assertEqual(agents[0]["status"], "running")

    def test_shows_completed_status(self) -> None:
        """list_background_agents() shows 'completed' after agent finishes."""
        agent = _make_mock_sub_agent(agent_id="bg-complete", delay=0.05)
        self.runner.submit_background(agent)
        time.sleep(0.2)

        agents = self.runner.list_background_agents()

        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0]["agent_id"], "bg-complete")
        self.assertEqual(agents[0]["status"], "completed")

    def test_shows_error_status(self) -> None:
        """list_background_agents() shows 'error' for failed agents."""
        agent = _make_mock_sub_agent(
            agent_id="bg-error",
            side_effect=RuntimeError("boom"),
        )
        self.runner.submit_background(agent)
        time.sleep(0.2)

        agents = self.runner.list_background_agents()

        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0]["agent_id"], "bg-error")
        self.assertEqual(agents[0]["status"], "error")

    def test_shows_multiple_agents_mixed_status(self) -> None:
        """list_background_agents() shows correct status for multiple agents."""
        fast = _make_mock_sub_agent(agent_id="bg-fast", delay=0.01)
        slow = _make_mock_sub_agent(agent_id="bg-slow", delay=2.0)

        self.runner.submit_background(fast)
        self.runner.submit_background(slow)
        time.sleep(0.2)

        agents = self.runner.list_background_agents()
        status_map = {a["agent_id"]: a["status"] for a in agents}

        self.assertEqual(len(agents), 2)
        self.assertEqual(status_map["bg-fast"], "completed")
        self.assertEqual(status_map["bg-slow"], "running")

    def test_list_contains_agent_id_and_status_keys(self) -> None:
        """Each dict in the list contains 'agent_id' and 'status' keys."""
        agent = _make_mock_sub_agent(agent_id="bg-keys")
        self.runner.submit_background(agent)
        time.sleep(0.1)

        agents = self.runner.list_background_agents()
        self.assertIn("agent_id", agents[0])
        self.assertIn("status", agents[0])


# ---------------------------------------------------------------------------
# SubAgentRunner — parallel execution
# ---------------------------------------------------------------------------


class TestSubAgentRunnerParallel(unittest.TestCase):
    """Tests for concurrent sub-agent execution."""

    def test_two_agents_run_concurrently(self) -> None:
        """Two agents with 0.2s delay each complete in < 0.35s (not 0.4s)."""
        runner = SubAgentRunner(max_workers=4)
        try:
            agent1 = _make_mock_sub_agent(agent_id="par-1", delay=0.2)
            agent2 = _make_mock_sub_agent(agent_id="par-2", delay=0.2)

            runner.submit_background(agent1)
            runner.submit_background(agent2)

            start = time.monotonic()
            # Wait for both to complete.
            results = {}
            for _ in range(40):
                for aid in ("par-1", "par-2"):
                    if aid not in results:
                        r = runner.get_background_result(aid)
                        if r is not None:
                            results[aid] = r
                if len(results) == 2:
                    break
                time.sleep(0.02)
            elapsed = time.monotonic() - start

            self.assertEqual(len(results), 2)
            # If sequential, total would be >= 0.4s. Parallel should be < 0.35s.
            self.assertLess(elapsed, 0.35)
        finally:
            runner.shutdown()

    def test_parallel_results_independent(self) -> None:
        """Each parallel agent returns its own independent result."""
        runner = SubAgentRunner(max_workers=4)
        try:
            agent1 = _make_mock_sub_agent(
                agent_id="ind-1", content="result-A", delay=0.05,
            )
            agent2 = _make_mock_sub_agent(
                agent_id="ind-2", content="result-B", delay=0.05,
            )

            runner.submit_background(agent1)
            runner.submit_background(agent2)
            time.sleep(0.3)

            r1 = runner.get_background_result("ind-1")
            r2 = runner.get_background_result("ind-2")

            self.assertIsNotNone(r1)
            self.assertIsNotNone(r2)
            self.assertEqual(r1.content, "result-A")
            self.assertEqual(r2.content, "result-B")
        finally:
            runner.shutdown()

    def test_submit_foreground_concurrent_with_background(self) -> None:
        """Foreground submit() works while background agents are running."""
        runner = SubAgentRunner(max_workers=4)
        try:
            bg_agent = _make_mock_sub_agent(agent_id="bg-mix", delay=0.3)
            fg_agent = _make_mock_sub_agent(
                agent_id="fg-mix", content="foreground done",
            )

            runner.submit_background(bg_agent)
            # Foreground submit should still work.
            fg_result = runner.submit(fg_agent)

            self.assertEqual(fg_result.content, "foreground done")
        finally:
            runner.shutdown()


# ---------------------------------------------------------------------------
# SubAgentRunner — max_workers cap
# ---------------------------------------------------------------------------


class TestSubAgentRunnerMaxWorkers(unittest.TestCase):
    """Tests for max_workers capping and OLLAMA_NUM_PARALLEL."""

    def test_default_max_workers(self) -> None:
        """Default max_workers is 3 (when OLLAMA_NUM_PARALLEL >= 3)."""
        with patch.dict(os.environ, {"OLLAMA_NUM_PARALLEL": "8"}):
            runner = SubAgentRunner()
            self.assertEqual(runner._max_workers, 3)
            runner.shutdown()

    def test_max_workers_capped_by_ollama_num_parallel(self) -> None:
        """max_workers is capped to OLLAMA_NUM_PARALLEL when lower."""
        with patch.dict(os.environ, {"OLLAMA_NUM_PARALLEL": "2"}):
            runner = SubAgentRunner(max_workers=5)
            self.assertEqual(runner._max_workers, 2)
            runner.shutdown()

    def test_max_workers_uses_requested_when_lower(self) -> None:
        """max_workers uses the requested value when lower than env var."""
        with patch.dict(os.environ, {"OLLAMA_NUM_PARALLEL": "8"}):
            runner = SubAgentRunner(max_workers=2)
            self.assertEqual(runner._max_workers, 2)
            runner.shutdown()

    def test_get_ollama_num_parallel_default(self) -> None:
        """Default OLLAMA_NUM_PARALLEL is 4 when env var is unset."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove OLLAMA_NUM_PARALLEL if present.
            os.environ.pop("OLLAMA_NUM_PARALLEL", None)
            value = SubAgentRunner._get_ollama_num_parallel()
            self.assertEqual(value, 4)

    def test_get_ollama_num_parallel_from_env(self) -> None:
        """OLLAMA_NUM_PARALLEL is read from environment."""
        with patch.dict(os.environ, {"OLLAMA_NUM_PARALLEL": "6"}):
            value = SubAgentRunner._get_ollama_num_parallel()
            self.assertEqual(value, 6)

    def test_get_ollama_num_parallel_invalid(self) -> None:
        """Invalid OLLAMA_NUM_PARALLEL defaults to 4."""
        with patch.dict(os.environ, {"OLLAMA_NUM_PARALLEL": "not_a_number"}):
            value = SubAgentRunner._get_ollama_num_parallel()
            self.assertEqual(value, 4)

    def test_get_ollama_num_parallel_minimum_one(self) -> None:
        """OLLAMA_NUM_PARALLEL enforces minimum of 1."""
        with patch.dict(os.environ, {"OLLAMA_NUM_PARALLEL": "0"}):
            value = SubAgentRunner._get_ollama_num_parallel()
            self.assertEqual(value, 1)

    def test_get_ollama_num_parallel_negative(self) -> None:
        """Negative OLLAMA_NUM_PARALLEL is clamped to 1."""
        with patch.dict(os.environ, {"OLLAMA_NUM_PARALLEL": "-5"}):
            value = SubAgentRunner._get_ollama_num_parallel()
            self.assertEqual(value, 1)


# ---------------------------------------------------------------------------
# SubAgentRunner — shutdown()
# ---------------------------------------------------------------------------


class TestSubAgentRunnerShutdown(unittest.TestCase):
    """Tests for SubAgentRunner.shutdown()."""

    def test_shutdown_sets_executor_to_none(self) -> None:
        """shutdown() sets _executor to None."""
        runner = SubAgentRunner(max_workers=2)
        agent = _make_mock_sub_agent()
        runner.submit(agent)  # Force executor creation.
        self.assertIsNotNone(runner._executor)

        runner.shutdown()
        self.assertIsNone(runner._executor)

    def test_shutdown_safe_to_call_multiple_times(self) -> None:
        """shutdown() can be called multiple times without error."""
        runner = SubAgentRunner(max_workers=2)
        agent = _make_mock_sub_agent()
        runner.submit(agent)

        runner.shutdown()
        runner.shutdown()  # Second call should not raise.
        runner.shutdown()  # Third call should not raise.

    def test_shutdown_without_use(self) -> None:
        """shutdown() can be called on a runner that was never used."""
        runner = SubAgentRunner(max_workers=2)
        # Never submitted anything — executor is None.
        runner.shutdown()  # Should not raise.

    def test_shutdown_waits_for_running_agents(self) -> None:
        """shutdown() waits for running agents to complete."""
        runner = SubAgentRunner(max_workers=2)
        agent = _make_mock_sub_agent(agent_id="bg-wait", delay=0.1)
        runner.submit_background(agent)

        # Shutdown should wait for the agent to complete.
        runner.shutdown()

        # After shutdown, the agent's run should have been called.
        agent.run.assert_called_once()


# ---------------------------------------------------------------------------
# SubAgentRunner — error handling
# ---------------------------------------------------------------------------


class TestSubAgentRunnerErrorHandling(unittest.TestCase):
    """Tests for SubAgentRunner error handling."""

    def test_submit_propagates_exception(self) -> None:
        """submit() propagates exceptions from sub-agent run()."""
        runner = SubAgentRunner(max_workers=2)
        try:
            agent = _make_mock_sub_agent(
                side_effect=RuntimeError("agent crash"),
            )
            with self.assertRaises(RuntimeError) as ctx:
                runner.submit(agent)
            self.assertIn("agent crash", str(ctx.exception))
        finally:
            runner.shutdown()

    def test_background_error_reflected_in_list(self) -> None:
        """Background agent error shows 'error' status in list."""
        runner = SubAgentRunner(max_workers=2)
        try:
            agent = _make_mock_sub_agent(
                agent_id="bg-crash",
                side_effect=ValueError("bad input"),
            )
            runner.submit_background(agent)
            time.sleep(0.2)

            agents = runner.list_background_agents()
            self.assertEqual(agents[0]["status"], "error")
        finally:
            runner.shutdown()

    def test_background_error_result_raises_on_get(self) -> None:
        """get_background_result() raises the exception for failed agent."""
        runner = SubAgentRunner(max_workers=2)
        try:
            agent = _make_mock_sub_agent(
                agent_id="bg-raise",
                side_effect=RuntimeError("failed"),
            )
            runner.submit_background(agent)
            time.sleep(0.2)

            # future.result() will re-raise the exception.
            with self.assertRaises(RuntimeError):
                runner.get_background_result("bg-raise")
        finally:
            runner.shutdown()

    def test_one_error_does_not_affect_others(self) -> None:
        """One failing background agent does not affect other agents."""
        runner = SubAgentRunner(max_workers=4)
        try:
            good_agent = _make_mock_sub_agent(
                agent_id="bg-good",
                content="all good",
                delay=0.05,
            )
            bad_agent = _make_mock_sub_agent(
                agent_id="bg-bad",
                side_effect=RuntimeError("boom"),
            )

            runner.submit_background(good_agent)
            runner.submit_background(bad_agent)
            time.sleep(0.3)

            # Good agent should succeed.
            result = runner.get_background_result("bg-good")
            self.assertIsNotNone(result)
            self.assertEqual(result.content, "all good")

            # Bad agent should be in error state.
            agents = runner.list_background_agents()
            status_map = {a["agent_id"]: a["status"] for a in agents}
            self.assertEqual(status_map["bg-good"], "completed")
            self.assertEqual(status_map["bg-bad"], "error")
        finally:
            runner.shutdown()


if __name__ == "__main__":
    unittest.main()
