"""Tests for local_cli.tools.base module.

Verifies that the ``Tool`` ABC enforces its interface contract and that
the concrete ``to_ollama_tool()`` and ``to_claude_tool()`` methods produce
the correct provider-specific formats.
"""

import unittest

from local_cli.tools.base import Tool


# ---------------------------------------------------------------------------
# Concrete stub for testing the ABC
# ---------------------------------------------------------------------------

class _StubTool(Tool):
    """Minimal concrete implementation of Tool for testing."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def description(self) -> str:
        return "A stub tool for testing."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "arg1": {
                    "type": "string",
                    "description": "First argument.",
                },
                "arg2": {
                    "type": "integer",
                    "description": "Second argument.",
                },
            },
            "required": ["arg1"],
        }

    def execute(self, **kwargs: object) -> str:
        return "stub result"


class _MinimalTool(Tool):
    """Tool with empty properties and no required fields."""

    @property
    def name(self) -> str:
        return "minimal"

    @property
    def description(self) -> str:
        return "Minimal tool."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {},
        }

    def execute(self, **kwargs: object) -> str:
        return ""


# ---------------------------------------------------------------------------
# Tests: ABC enforcement
# ---------------------------------------------------------------------------

class TestToolABC(unittest.TestCase):
    """Tests that Tool ABC enforces its abstract interface."""

    def test_cannot_instantiate_abstract(self) -> None:
        """Direct instantiation of Tool raises TypeError."""
        with self.assertRaises(TypeError):
            Tool()  # type: ignore[abstract]

    def test_missing_name_raises(self) -> None:
        """Omitting the name property prevents instantiation."""

        class _Incomplete(Tool):
            @property
            def description(self) -> str:
                return "d"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            def execute(self, **kwargs: object) -> str:
                return ""

        with self.assertRaises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_missing_execute_raises(self) -> None:
        """Omitting execute() prevents instantiation."""

        class _Incomplete(Tool):
            @property
            def name(self) -> str:
                return "x"

            @property
            def description(self) -> str:
                return "d"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

        with self.assertRaises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_concrete_stub_instantiates(self) -> None:
        """A fully implemented subclass can be instantiated."""
        tool = _StubTool()
        self.assertIsInstance(tool, Tool)


# ---------------------------------------------------------------------------
# Tests: to_ollama_tool()
# ---------------------------------------------------------------------------

class TestToOllamaTool(unittest.TestCase):
    """Tests for Tool.to_ollama_tool() format conversion."""

    def setUp(self) -> None:
        self.tool = _StubTool()
        self.result = self.tool.to_ollama_tool()

    def test_top_level_type(self) -> None:
        """Top-level 'type' is 'function'."""
        self.assertEqual(self.result["type"], "function")

    def test_has_function_key(self) -> None:
        """Result contains a 'function' key."""
        self.assertIn("function", self.result)

    def test_function_name(self) -> None:
        """Function name matches tool name."""
        self.assertEqual(self.result["function"]["name"], "stub")

    def test_function_description(self) -> None:
        """Function description matches tool description."""
        self.assertEqual(
            self.result["function"]["description"],
            "A stub tool for testing.",
        )

    def test_function_parameters(self) -> None:
        """Function parameters match tool parameters."""
        params = self.result["function"]["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertIn("arg1", params["properties"])
        self.assertIn("arg2", params["properties"])
        self.assertEqual(params["required"], ["arg1"])

    def test_ollama_format_keys(self) -> None:
        """Ollama format has exactly 'type' and 'function' top-level keys."""
        self.assertEqual(set(self.result.keys()), {"type", "function"})

    def test_ollama_function_keys(self) -> None:
        """Function dict has exactly 'name', 'description', 'parameters'."""
        self.assertEqual(
            set(self.result["function"].keys()),
            {"name", "description", "parameters"},
        )

    def test_minimal_tool_ollama(self) -> None:
        """Minimal tool with empty properties produces valid Ollama format."""
        tool = _MinimalTool()
        result = tool.to_ollama_tool()
        self.assertEqual(result["type"], "function")
        self.assertEqual(result["function"]["name"], "minimal")
        self.assertEqual(result["function"]["parameters"]["properties"], {})


# ---------------------------------------------------------------------------
# Tests: to_claude_tool()
# ---------------------------------------------------------------------------

class TestToClaudeTool(unittest.TestCase):
    """Tests for Tool.to_claude_tool() format conversion."""

    def setUp(self) -> None:
        self.tool = _StubTool()
        self.result = self.tool.to_claude_tool()

    def test_has_name(self) -> None:
        """Result contains 'name' key."""
        self.assertEqual(self.result["name"], "stub")

    def test_has_description(self) -> None:
        """Result contains 'description' key."""
        self.assertEqual(
            self.result["description"],
            "A stub tool for testing.",
        )

    def test_has_input_schema(self) -> None:
        """Result contains 'input_schema' key (not 'parameters')."""
        self.assertIn("input_schema", self.result)
        self.assertNotIn("parameters", self.result)

    def test_input_schema_matches_parameters(self) -> None:
        """input_schema contains the same JSON Schema as parameters."""
        schema = self.result["input_schema"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("arg1", schema["properties"])
        self.assertIn("arg2", schema["properties"])
        self.assertEqual(schema["required"], ["arg1"])

    def test_no_type_wrapper(self) -> None:
        """Claude format has no 'type': 'function' wrapper."""
        self.assertNotIn("type", self.result)

    def test_no_function_wrapper(self) -> None:
        """Claude format has no 'function' wrapper."""
        self.assertNotIn("function", self.result)

    def test_claude_format_keys(self) -> None:
        """Claude format has exactly 'name', 'description', 'input_schema'."""
        self.assertEqual(
            set(self.result.keys()),
            {"name", "description", "input_schema"},
        )

    def test_minimal_tool_claude(self) -> None:
        """Minimal tool with empty properties produces valid Claude format."""
        tool = _MinimalTool()
        result = tool.to_claude_tool()
        self.assertEqual(result["name"], "minimal")
        self.assertEqual(result["description"], "Minimal tool.")
        self.assertEqual(result["input_schema"]["properties"], {})

    def test_input_schema_equals_parameters(self) -> None:
        """input_schema value equals the tool's parameters dict."""
        self.assertEqual(self.result["input_schema"], self.tool.parameters)


# ---------------------------------------------------------------------------
# Tests: cross-format consistency
# ---------------------------------------------------------------------------

class TestCrossFormatConsistency(unittest.TestCase):
    """Verify that both format methods produce consistent data."""

    def setUp(self) -> None:
        self.tool = _StubTool()
        self.ollama = self.tool.to_ollama_tool()
        self.claude = self.tool.to_claude_tool()

    def test_names_match(self) -> None:
        """Both formats produce the same tool name."""
        self.assertEqual(
            self.ollama["function"]["name"],
            self.claude["name"],
        )

    def test_descriptions_match(self) -> None:
        """Both formats produce the same description."""
        self.assertEqual(
            self.ollama["function"]["description"],
            self.claude["description"],
        )

    def test_schemas_match(self) -> None:
        """Ollama 'parameters' and Claude 'input_schema' are equivalent."""
        self.assertEqual(
            self.ollama["function"]["parameters"],
            self.claude["input_schema"],
        )

    def test_both_formats_from_real_tool(self) -> None:
        """Verify formats with BashTool as a real-world tool."""
        from local_cli.tools.bash_tool import BashTool

        tool = BashTool()
        ollama = tool.to_ollama_tool()
        claude = tool.to_claude_tool()

        self.assertEqual(ollama["function"]["name"], claude["name"])
        self.assertEqual(ollama["function"]["name"], "bash")
        self.assertEqual(
            ollama["function"]["parameters"],
            claude["input_schema"],
        )
        self.assertIn("command", claude["input_schema"]["properties"])


if __name__ == "__main__":
    unittest.main()
