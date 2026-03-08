"""Tests for local_cli.tools.agent_tool module."""

import unittest
from unittest.mock import MagicMock, patch

from local_cli.sub_agent import SubAgentResult, SubAgentRunner
from local_cli.tools.agent_tool import AgentTool
from local_cli.tools.base import Tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyTool(Tool):
    """Minimal concrete tool for testing."""

    def __init__(self, name: str = "dummy") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A dummy tool."

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
        return "ok"


def _make_agent_tool(
    runner: SubAgentRunner | None = None,
    provider: MagicMock | None = None,
    model: str = "qwen3:8b",
    sub_agent_tools: list[Tool] | None = None,
) -> AgentTool:
    """Create an AgentTool with sensible mock defaults."""
    if runner is None:
        runner = MagicMock(spec=SubAgentRunner)
    if provider is None:
        provider = MagicMock()
        provider.name = "ollama"
    if sub_agent_tools is None:
        sub_agent_tools = [_DummyTool("bash"), _DummyTool("read")]
    return AgentTool(
        runner=runner,
        provider=provider,
        model=model,
        sub_agent_tools=sub_agent_tools,
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestAgentToolMetadata(unittest.TestCase):
    """Tests for AgentTool metadata properties."""

    def setUp(self) -> None:
        self.tool = _make_agent_tool()

    def test_name(self) -> None:
        """Tool name is 'agent'."""
        self.assertEqual(self.tool.name, "agent")

    def test_description_is_nonempty(self) -> None:
        """Description is a non-empty string."""
        self.assertIsInstance(self.tool.description, str)
        self.assertTrue(len(self.tool.description) > 0)

    def test_parameters_schema_type(self) -> None:
        """Parameters schema has type 'object'."""
        params = self.tool.parameters
        self.assertEqual(params["type"], "object")

    def test_parameters_has_description_field(self) -> None:
        """Parameters schema includes 'description' property."""
        params = self.tool.parameters
        self.assertIn("description", params["properties"])
        self.assertEqual(params["properties"]["description"]["type"], "string")

    def test_parameters_has_prompt_field(self) -> None:
        """Parameters schema includes 'prompt' property."""
        params = self.tool.parameters
        self.assertIn("prompt", params["properties"])
        self.assertEqual(params["properties"]["prompt"]["type"], "string")

    def test_parameters_has_run_in_background_field(self) -> None:
        """Parameters schema includes 'run_in_background' property."""
        params = self.tool.parameters
        self.assertIn("run_in_background", params["properties"])
        self.assertEqual(
            params["properties"]["run_in_background"]["type"], "boolean"
        )

    def test_required_fields(self) -> None:
        """'description' and 'prompt' are required."""
        params = self.tool.parameters
        self.assertIn("description", params["required"])
        self.assertIn("prompt", params["required"])

    def test_run_in_background_not_required(self) -> None:
        """'run_in_background' is NOT required."""
        params = self.tool.parameters
        self.assertNotIn("run_in_background", params["required"])


# ---------------------------------------------------------------------------
# to_ollama_tool / to_claude_tool
# ---------------------------------------------------------------------------


class TestAgentToolFormats(unittest.TestCase):
    """Tests for to_ollama_tool() and to_claude_tool() output formats."""

    def setUp(self) -> None:
        self.tool = _make_agent_tool()

    def test_to_ollama_tool_type(self) -> None:
        """to_ollama_tool returns dict with type 'function'."""
        tool_def = self.tool.to_ollama_tool()
        self.assertEqual(tool_def["type"], "function")

    def test_to_ollama_tool_function_name(self) -> None:
        """to_ollama_tool function.name is 'agent'."""
        tool_def = self.tool.to_ollama_tool()
        self.assertEqual(tool_def["function"]["name"], "agent")

    def test_to_ollama_tool_function_description(self) -> None:
        """to_ollama_tool function has a non-empty description."""
        tool_def = self.tool.to_ollama_tool()
        self.assertIsInstance(tool_def["function"]["description"], str)
        self.assertTrue(len(tool_def["function"]["description"]) > 0)

    def test_to_ollama_tool_function_parameters(self) -> None:
        """to_ollama_tool function.parameters matches self.parameters."""
        tool_def = self.tool.to_ollama_tool()
        self.assertEqual(tool_def["function"]["parameters"], self.tool.parameters)

    def test_to_claude_tool_name(self) -> None:
        """to_claude_tool returns dict with name 'agent'."""
        tool_def = self.tool.to_claude_tool()
        self.assertEqual(tool_def["name"], "agent")

    def test_to_claude_tool_description(self) -> None:
        """to_claude_tool has a non-empty description."""
        tool_def = self.tool.to_claude_tool()
        self.assertIsInstance(tool_def["description"], str)
        self.assertTrue(len(tool_def["description"]) > 0)

    def test_to_claude_tool_input_schema(self) -> None:
        """to_claude_tool uses 'input_schema' (not 'parameters')."""
        tool_def = self.tool.to_claude_tool()
        self.assertIn("input_schema", tool_def)
        self.assertNotIn("parameters", tool_def)
        self.assertEqual(tool_def["input_schema"], self.tool.parameters)

    def test_to_claude_tool_no_function_wrapper(self) -> None:
        """to_claude_tool is a flat dict (no 'function' wrapper)."""
        tool_def = self.tool.to_claude_tool()
        self.assertNotIn("function", tool_def)
        self.assertNotIn("type", tool_def)


# ---------------------------------------------------------------------------
# Execution – validation errors
# ---------------------------------------------------------------------------


class TestAgentToolValidation(unittest.TestCase):
    """Tests for execute() parameter validation."""

    def setUp(self) -> None:
        self.tool = _make_agent_tool()

    def test_empty_prompt_returns_error(self) -> None:
        """Empty prompt returns an error string."""
        result = self.tool.execute(description="test", prompt="")
        self.assertIn("Error", result)
        self.assertIn("prompt", result)

    def test_whitespace_only_prompt_returns_error(self) -> None:
        """Whitespace-only prompt returns an error string."""
        result = self.tool.execute(description="test", prompt="   ")
        self.assertIn("Error", result)
        self.assertIn("prompt", result)

    def test_missing_prompt_returns_error(self) -> None:
        """Missing prompt kwarg returns an error string."""
        result = self.tool.execute(description="test")
        self.assertIn("Error", result)
        self.assertIn("prompt", result)

    def test_non_string_prompt_returns_error(self) -> None:
        """Non-string prompt returns an error string."""
        result = self.tool.execute(description="test", prompt=123)
        self.assertIn("Error", result)
        self.assertIn("prompt", result)

    def test_empty_description_returns_error(self) -> None:
        """Empty description returns an error string."""
        result = self.tool.execute(description="", prompt="do something")
        self.assertIn("Error", result)
        self.assertIn("description", result)

    def test_whitespace_only_description_returns_error(self) -> None:
        """Whitespace-only description returns an error string."""
        result = self.tool.execute(description="  ", prompt="do something")
        self.assertIn("Error", result)
        self.assertIn("description", result)

    def test_missing_description_returns_error(self) -> None:
        """Missing description kwarg returns an error string."""
        result = self.tool.execute(prompt="do something")
        self.assertIn("Error", result)
        self.assertIn("description", result)

    def test_non_string_description_returns_error(self) -> None:
        """Non-string description returns an error string."""
        result = self.tool.execute(description=42, prompt="do something")
        self.assertIn("Error", result)
        self.assertIn("description", result)


# ---------------------------------------------------------------------------
# Execution – foreground mode
# ---------------------------------------------------------------------------


class TestAgentToolForeground(unittest.TestCase):
    """Tests for foreground execution (blocking)."""

    def test_foreground_spawns_sub_agent_and_returns_result(self) -> None:
        """Foreground execute blocks and returns formatted result."""
        mock_result = SubAgentResult(
            agent_id="agent-20260309-010203-abcd1234",
            description="test task",
            content="Task completed successfully.",
            status="success",
            duration_seconds=1.5,
            messages_count=4,
            tool_calls_count=2,
        )
        runner = MagicMock(spec=SubAgentRunner)
        runner.submit.return_value = mock_result

        tool = _make_agent_tool(runner=runner)

        with patch.object(tool, "_create_fresh_provider") as mock_create:
            mock_create.return_value = MagicMock()
            result = tool.execute(
                description="test task",
                prompt="Do the test task.",
            )

        # Runner.submit was called (not submit_background).
        runner.submit.assert_called_once()
        runner.submit_background.assert_not_called()

        # Result contains the formatted output.
        self.assertIn("Task completed successfully.", result)
        self.assertIn("test task", result)

    def test_foreground_default_when_background_not_specified(self) -> None:
        """When run_in_background is omitted, foreground mode is used."""
        mock_result = SubAgentResult(
            agent_id="agent-test",
            description="fg task",
            content="done",
            status="success",
            duration_seconds=0.5,
            messages_count=2,
            tool_calls_count=0,
        )
        runner = MagicMock(spec=SubAgentRunner)
        runner.submit.return_value = mock_result

        tool = _make_agent_tool(runner=runner)

        with patch.object(tool, "_create_fresh_provider") as mock_create:
            mock_create.return_value = MagicMock()
            tool.execute(description="fg task", prompt="some prompt")

        runner.submit.assert_called_once()

    def test_foreground_invalid_background_flag_defaults_to_foreground(self) -> None:
        """Non-bool run_in_background defaults to foreground."""
        mock_result = SubAgentResult(
            agent_id="agent-test",
            description="fg task",
            content="done",
            status="success",
            duration_seconds=0.5,
            messages_count=2,
            tool_calls_count=0,
        )
        runner = MagicMock(spec=SubAgentRunner)
        runner.submit.return_value = mock_result

        tool = _make_agent_tool(runner=runner)

        with patch.object(tool, "_create_fresh_provider") as mock_create:
            mock_create.return_value = MagicMock()
            tool.execute(
                description="fg task",
                prompt="some prompt",
                run_in_background="yes",
            )

        runner.submit.assert_called_once()
        runner.submit_background.assert_not_called()


# ---------------------------------------------------------------------------
# Execution – background mode
# ---------------------------------------------------------------------------


class TestAgentToolBackground(unittest.TestCase):
    """Tests for background execution (non-blocking)."""

    def test_background_returns_agent_id_immediately(self) -> None:
        """Background execute returns agent ID without blocking."""
        runner = MagicMock(spec=SubAgentRunner)
        runner.submit_background.return_value = "agent-20260309-010203-abcd1234"

        tool = _make_agent_tool(runner=runner)

        with patch.object(tool, "_create_fresh_provider") as mock_create:
            mock_create.return_value = MagicMock()
            result = tool.execute(
                description="bg task",
                prompt="Do something in background.",
                run_in_background=True,
            )

        # Runner.submit_background was called (not submit).
        runner.submit_background.assert_called_once()
        runner.submit.assert_not_called()

        # Result contains the agent ID.
        self.assertIn("agent-20260309-010203-abcd1234", result)
        self.assertIn("background", result)

    def test_background_result_contains_description(self) -> None:
        """Background result string includes the task description."""
        runner = MagicMock(spec=SubAgentRunner)
        runner.submit_background.return_value = "agent-test-id"

        tool = _make_agent_tool(runner=runner)

        with patch.object(tool, "_create_fresh_provider") as mock_create:
            mock_create.return_value = MagicMock()
            result = tool.execute(
                description="my bg task",
                prompt="Do bg work.",
                run_in_background=True,
            )

        self.assertIn("my bg task", result)


# ---------------------------------------------------------------------------
# Provider creation failure
# ---------------------------------------------------------------------------


class TestAgentToolProviderError(unittest.TestCase):
    """Tests for error handling when fresh provider creation fails."""

    def test_provider_creation_failure_returns_error(self) -> None:
        """If _create_fresh_provider raises, execute returns error string."""
        tool = _make_agent_tool()

        with patch.object(
            tool,
            "_create_fresh_provider",
            side_effect=RuntimeError("connection refused"),
        ):
            result = tool.execute(
                description="test",
                prompt="do something",
            )

        self.assertIn("Error", result)
        self.assertIn("connection refused", result)


# ---------------------------------------------------------------------------
# Sub-agent tool list exclusions
# ---------------------------------------------------------------------------


class TestSubAgentToolExclusions(unittest.TestCase):
    """Tests that sub-agent tools exclude AgentTool and AskUserTool."""

    def test_get_sub_agent_tools_excludes_ask_user(self) -> None:
        """get_sub_agent_tools() does not include AskUserTool."""
        from local_cli.tools import get_sub_agent_tools

        tools = get_sub_agent_tools()
        tool_names = [t.name for t in tools]
        self.assertNotIn("ask_user", tool_names)

    def test_get_sub_agent_tools_excludes_agent(self) -> None:
        """get_sub_agent_tools() does not include AgentTool."""
        from local_cli.tools import get_sub_agent_tools

        tools = get_sub_agent_tools()
        tool_names = [t.name for t in tools]
        self.assertNotIn("agent", tool_names)

    def test_get_sub_agent_tools_includes_core_tools(self) -> None:
        """get_sub_agent_tools() includes bash, read, write, edit, etc."""
        from local_cli.tools import get_sub_agent_tools

        tools = get_sub_agent_tools()
        tool_names = [t.name for t in tools]
        for expected in ("bash", "read", "write", "edit", "glob", "grep"):
            self.assertIn(expected, tool_names)

    def test_sub_agent_receives_tools_without_agent_or_ask_user(self) -> None:
        """Sub-agent created by execute() gets tools that exclude agent/ask_user."""
        from local_cli.tools import get_sub_agent_tools

        sub_tools = get_sub_agent_tools()
        mock_result = SubAgentResult(
            agent_id="agent-test",
            description="check tools",
            content="done",
            status="success",
            duration_seconds=0.1,
            messages_count=2,
            tool_calls_count=0,
        )
        runner = MagicMock(spec=SubAgentRunner)
        runner.submit.return_value = mock_result

        tool = AgentTool(
            runner=runner,
            provider=MagicMock(),
            model="qwen3:8b",
            sub_agent_tools=sub_tools,
        )

        with patch.object(tool, "_create_fresh_provider") as mock_create:
            mock_create.return_value = MagicMock()
            tool.execute(description="check tools", prompt="verify tools")

        # Inspect the SubAgent that was passed to runner.submit.
        call_args = runner.submit.call_args
        sub_agent = call_args[0][0]
        sub_agent_tool_names = [t.name for t in sub_agent._tools]

        self.assertNotIn("agent", sub_agent_tool_names)
        self.assertNotIn("ask_user", sub_agent_tool_names)


if __name__ == "__main__":
    unittest.main()
