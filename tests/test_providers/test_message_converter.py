"""Tests for local_cli.providers.message_converter module.

Verifies bidirectional message format conversion between the normalized
(Ollama-style) internal format and the Claude Messages API external format.
"""

import unittest
from typing import Any

from local_cli.providers.message_converter import (
    claude_response_to_normalized,
    claude_stream_to_normalized,
    messages_to_claude,
    tools_from_claude,
    tools_to_claude,
)


# ---------------------------------------------------------------------------
# messages_to_claude: System prompt extraction
# ---------------------------------------------------------------------------


class TestMessagesToClaudeSystemPrompt(unittest.TestCase):
    """System messages are extracted to a separate system string."""

    def test_single_system_message(self) -> None:
        messages = [{"role": "system", "content": "You are helpful."}]
        system, claude_msgs = messages_to_claude(messages)
        self.assertEqual(system, "You are helpful.")
        self.assertEqual(claude_msgs, [])

    def test_multiple_system_messages_concatenated(self) -> None:
        messages = [
            {"role": "system", "content": "First rule."},
            {"role": "system", "content": "Second rule."},
        ]
        system, claude_msgs = messages_to_claude(messages)
        self.assertEqual(system, "First rule.\n\nSecond rule.")
        self.assertEqual(claude_msgs, [])

    def test_no_system_message_returns_none(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        system, _ = messages_to_claude(messages)
        self.assertIsNone(system)

    def test_empty_system_content_skipped(self) -> None:
        messages = [
            {"role": "system", "content": ""},
            {"role": "system", "content": "Actual rule."},
        ]
        system, _ = messages_to_claude(messages)
        self.assertEqual(system, "Actual rule.")

    def test_system_messages_removed_from_output(self) -> None:
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "hello"},
        ]
        _, claude_msgs = messages_to_claude(messages)
        for msg in claude_msgs:
            self.assertNotEqual(msg["role"], "system")


# ---------------------------------------------------------------------------
# messages_to_claude: User message conversion
# ---------------------------------------------------------------------------


class TestMessagesToClaudeUserMessages(unittest.TestCase):
    """User messages are converted to content-block format."""

    def test_simple_user_message(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        _, claude_msgs = messages_to_claude(messages)
        self.assertEqual(len(claude_msgs), 1)
        msg = claude_msgs[0]
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["content"], [{"type": "text", "text": "hello"}])

    def test_empty_user_content(self) -> None:
        messages = [{"role": "user", "content": ""}]
        _, claude_msgs = messages_to_claude(messages)
        self.assertEqual(
            claude_msgs[0]["content"],
            [{"type": "text", "text": ""}],
        )

    def test_user_message_with_special_characters(self) -> None:
        messages = [{"role": "user", "content": 'Say "hello" & <bye>'}]
        _, claude_msgs = messages_to_claude(messages)
        self.assertEqual(
            claude_msgs[0]["content"][0]["text"],
            'Say "hello" & <bye>',
        )


# ---------------------------------------------------------------------------
# messages_to_claude: Assistant message conversion
# ---------------------------------------------------------------------------


class TestMessagesToClaudeAssistantMessages(unittest.TestCase):
    """Assistant messages are converted with text and tool_use blocks."""

    def test_simple_assistant_message(self) -> None:
        messages = [{"role": "assistant", "content": "Hi there!"}]
        _, claude_msgs = messages_to_claude(messages)
        self.assertEqual(len(claude_msgs), 1)
        msg = claude_msgs[0]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(
            msg["content"],
            [{"type": "text", "text": "Hi there!"}],
        )

    def test_assistant_with_tool_calls(self) -> None:
        messages = [{
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [{
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "/tmp/test.txt"},
                },
                "id": "toolu_abc123",
            }],
        }]
        _, claude_msgs = messages_to_claude(messages)
        msg = claude_msgs[0]
        self.assertEqual(len(msg["content"]), 2)
        # First block is text.
        self.assertEqual(msg["content"][0]["type"], "text")
        self.assertEqual(msg["content"][0]["text"], "Let me check.")
        # Second block is tool_use.
        tool_block = msg["content"][1]
        self.assertEqual(tool_block["type"], "tool_use")
        self.assertEqual(tool_block["id"], "toolu_abc123")
        self.assertEqual(tool_block["name"], "read_file")
        self.assertEqual(tool_block["input"], {"path": "/tmp/test.txt"})

    def test_assistant_with_multiple_tool_calls(self) -> None:
        messages = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {"name": "read_file", "arguments": {"path": "a.txt"}},
                    "id": "toolu_001",
                },
                {
                    "function": {"name": "write_file", "arguments": {"path": "b.txt", "content": "x"}},
                    "id": "toolu_002",
                },
            ],
        }]
        _, claude_msgs = messages_to_claude(messages)
        msg = claude_msgs[0]
        # Empty content -> no text block; 2 tool_use blocks.
        self.assertEqual(len(msg["content"]), 2)
        self.assertEqual(msg["content"][0]["type"], "tool_use")
        self.assertEqual(msg["content"][1]["type"], "tool_use")

    def test_assistant_empty_content_no_tool_calls(self) -> None:
        """An empty assistant message still has at least one content block."""
        messages = [{"role": "assistant", "content": ""}]
        _, claude_msgs = messages_to_claude(messages)
        msg = claude_msgs[0]
        self.assertEqual(len(msg["content"]), 1)
        self.assertEqual(msg["content"][0]["type"], "text")
        self.assertEqual(msg["content"][0]["text"], "")

    def test_assistant_tool_call_with_none_id(self) -> None:
        """Tool calls from Ollama may have id=None; should produce ''."""
        messages = [{
            "role": "assistant",
            "content": "thinking...",
            "tool_calls": [{
                "function": {"name": "bash", "arguments": {"cmd": "ls"}},
                "id": None,
            }],
        }]
        _, claude_msgs = messages_to_claude(messages)
        tool_block = claude_msgs[0]["content"][1]
        self.assertEqual(tool_block["id"], "")


# ---------------------------------------------------------------------------
# messages_to_claude: Tool result conversion
# ---------------------------------------------------------------------------


class TestMessagesToClaudeToolResults(unittest.TestCase):
    """Tool-role messages become user messages with tool_result blocks."""

    def test_single_tool_result(self) -> None:
        messages = [{
            "role": "tool",
            "tool_name": "read_file",
            "tool_call_id": "toolu_abc123",
            "content": "file contents here",
        }]
        _, claude_msgs = messages_to_claude(messages)
        self.assertEqual(len(claude_msgs), 1)
        msg = claude_msgs[0]
        self.assertEqual(msg["role"], "user")
        self.assertEqual(len(msg["content"]), 1)
        block = msg["content"][0]
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "toolu_abc123")
        self.assertEqual(block["content"], "file contents here")

    def test_adjacent_tool_results_merged(self) -> None:
        """Multiple consecutive tool results are merged into one user msg."""
        messages = [
            {
                "role": "tool",
                "tool_name": "read_file",
                "tool_call_id": "toolu_001",
                "content": "result 1",
            },
            {
                "role": "tool",
                "tool_name": "write_file",
                "tool_call_id": "toolu_002",
                "content": "result 2",
            },
        ]
        _, claude_msgs = messages_to_claude(messages)
        self.assertEqual(len(claude_msgs), 1)
        msg = claude_msgs[0]
        self.assertEqual(msg["role"], "user")
        self.assertEqual(len(msg["content"]), 2)
        self.assertEqual(msg["content"][0]["tool_use_id"], "toolu_001")
        self.assertEqual(msg["content"][1]["tool_use_id"], "toolu_002")

    def test_tool_results_flushed_before_user_message(self) -> None:
        """Tool results followed by a user message form two messages."""
        messages = [
            {
                "role": "tool",
                "tool_name": "bash",
                "tool_call_id": "toolu_001",
                "content": "ok",
            },
            {"role": "user", "content": "What happened?"},
        ]
        _, claude_msgs = messages_to_claude(messages)
        self.assertEqual(len(claude_msgs), 2)
        self.assertEqual(claude_msgs[0]["role"], "user")
        self.assertEqual(claude_msgs[0]["content"][0]["type"], "tool_result")
        self.assertEqual(claude_msgs[1]["role"], "user")
        self.assertEqual(claude_msgs[1]["content"][0]["type"], "text")

    def test_tool_result_missing_tool_call_id(self) -> None:
        """Ollama tool results (no tool_call_id) should produce ''."""
        messages = [{
            "role": "tool",
            "tool_name": "bash",
            "content": "output",
        }]
        _, claude_msgs = messages_to_claude(messages)
        block = claude_msgs[0]["content"][0]
        self.assertEqual(block["tool_use_id"], "")


# ---------------------------------------------------------------------------
# messages_to_claude: Full conversation round-trip
# ---------------------------------------------------------------------------


class TestMessagesToClaudeFullConversation(unittest.TestCase):
    """Full conversation sequences convert correctly."""

    def test_typical_tool_use_conversation(self) -> None:
        """System + user + assistant(tool_call) + tool_result sequence."""
        messages = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Read my file."},
            {
                "role": "assistant",
                "content": "I'll read it.",
                "tool_calls": [{
                    "function": {"name": "read_file", "arguments": {"path": "a.txt"}},
                    "id": "toolu_1",
                }],
            },
            {
                "role": "tool",
                "tool_name": "read_file",
                "tool_call_id": "toolu_1",
                "content": "Hello world",
            },
            {"role": "assistant", "content": "The file says: Hello world"},
        ]

        system, claude_msgs = messages_to_claude(messages)
        self.assertEqual(system, "You are an assistant.")
        # user, assistant(tool_use), user(tool_result), assistant
        self.assertEqual(len(claude_msgs), 4)

        self.assertEqual(claude_msgs[0]["role"], "user")
        self.assertEqual(claude_msgs[1]["role"], "assistant")
        self.assertEqual(claude_msgs[2]["role"], "user")
        self.assertEqual(claude_msgs[2]["content"][0]["type"], "tool_result")
        self.assertEqual(claude_msgs[3]["role"], "assistant")

    def test_empty_messages_list(self) -> None:
        system, claude_msgs = messages_to_claude([])
        self.assertIsNone(system)
        self.assertEqual(claude_msgs, [])


# ---------------------------------------------------------------------------
# claude_response_to_normalized
# ---------------------------------------------------------------------------


class TestClaudeResponseToNormalized(unittest.TestCase):
    """Claude API responses are converted to the normalized format."""

    def test_text_only_response(self) -> None:
        response = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
        }
        result = claude_response_to_normalized(response)
        self.assertIn("message", result)
        msg = result["message"]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "Hello!")
        self.assertNotIn("tool_calls", msg)

    def test_multiple_text_blocks(self) -> None:
        response = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "First paragraph."},
                {"type": "text", "text": "Second paragraph."},
            ],
        }
        result = claude_response_to_normalized(response)
        self.assertEqual(
            result["message"]["content"],
            "First paragraph.\n\nSecond paragraph.",
        )

    def test_tool_use_response(self) -> None:
        response = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "toolu_abc123",
                    "name": "read_file",
                    "input": {"path": "/tmp/test.txt"},
                },
            ],
        }
        result = claude_response_to_normalized(response)
        msg = result["message"]
        self.assertEqual(msg["content"], "Let me check.")
        self.assertIn("tool_calls", msg)
        self.assertEqual(len(msg["tool_calls"]), 1)

        tc = msg["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "read_file")
        self.assertEqual(tc["function"]["arguments"], {"path": "/tmp/test.txt"})
        self.assertEqual(tc["id"], "toolu_abc123")

    def test_multiple_tool_use_blocks(self) -> None:
        response = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_001",
                    "name": "bash",
                    "input": {"command": "ls"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_002",
                    "name": "read_file",
                    "input": {"path": "a.txt"},
                },
            ],
        }
        result = claude_response_to_normalized(response)
        msg = result["message"]
        self.assertEqual(msg["content"], "")
        self.assertEqual(len(msg["tool_calls"]), 2)
        self.assertEqual(msg["tool_calls"][0]["id"], "toolu_001")
        self.assertEqual(msg["tool_calls"][1]["id"], "toolu_002")

    def test_empty_content_blocks(self) -> None:
        response = {"role": "assistant", "content": []}
        result = claude_response_to_normalized(response)
        self.assertEqual(result["message"]["content"], "")
        self.assertNotIn("tool_calls", result["message"])

    def test_preserves_role_from_response(self) -> None:
        response = {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
        }
        result = claude_response_to_normalized(response)
        self.assertEqual(result["message"]["role"], "assistant")

    def test_defaults_role_to_assistant(self) -> None:
        """If response has no role, defaults to 'assistant'."""
        response = {"content": [{"type": "text", "text": "hi"}]}
        result = claude_response_to_normalized(response)
        self.assertEqual(result["message"]["role"], "assistant")

    def test_unknown_block_type_ignored(self) -> None:
        """Unknown content block types are silently skipped."""
        response = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image", "source": {"url": "..."}},
            ],
        }
        result = claude_response_to_normalized(response)
        self.assertEqual(result["message"]["content"], "hello")
        self.assertNotIn("tool_calls", result["message"])

    def test_empty_text_block_skipped(self) -> None:
        """Empty text blocks do not contribute to content."""
        response = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ""},
                {"type": "text", "text": "actual"},
            ],
        }
        result = claude_response_to_normalized(response)
        self.assertEqual(result["message"]["content"], "actual")

    def test_tool_use_with_empty_input(self) -> None:
        response = {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_x",
                "name": "bash",
                "input": {},
            }],
        }
        result = claude_response_to_normalized(response)
        tc = result["message"]["tool_calls"][0]
        self.assertEqual(tc["function"]["arguments"], {})


# ---------------------------------------------------------------------------
# claude_stream_to_normalized
# ---------------------------------------------------------------------------


class TestClaudeStreamToNormalized(unittest.TestCase):
    """SSE stream events are converted to normalized chunks."""

    def test_text_delta(self) -> None:
        result = claude_stream_to_normalized(
            "content_block_delta",
            {"delta": {"type": "text_delta", "text": "Hello"}},
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["message"]["content"], "Hello")
        self.assertFalse(result["done"])

    def test_input_json_delta_returns_none(self) -> None:
        """input_json_delta events are handled by the provider, not here."""
        result = claude_stream_to_normalized(
            "content_block_delta",
            {"delta": {"type": "input_json_delta", "partial_json": '{"pa'}},
        )
        self.assertIsNone(result)

    def test_message_stop(self) -> None:
        result = claude_stream_to_normalized("message_stop", {})
        self.assertIsNotNone(result)
        self.assertTrue(result["done"])

    def test_message_delta_with_stop_reason(self) -> None:
        result = claude_stream_to_normalized(
            "message_delta",
            {"delta": {"stop_reason": "end_turn"}},
        )
        self.assertIsNotNone(result)
        self.assertTrue(result["done"])

    def test_message_delta_without_stop_reason(self) -> None:
        result = claude_stream_to_normalized(
            "message_delta",
            {"delta": {}},
        )
        self.assertIsNotNone(result)
        self.assertFalse(result["done"])

    def test_message_start_returns_none(self) -> None:
        result = claude_stream_to_normalized(
            "message_start",
            {"message": {"role": "assistant"}},
        )
        self.assertIsNone(result)

    def test_content_block_start_returns_none(self) -> None:
        result = claude_stream_to_normalized(
            "content_block_start",
            {"content_block": {"type": "text", "text": ""}},
        )
        self.assertIsNone(result)

    def test_content_block_stop_returns_none(self) -> None:
        result = claude_stream_to_normalized("content_block_stop", {})
        self.assertIsNone(result)

    def test_ping_returns_none(self) -> None:
        result = claude_stream_to_normalized("ping", {})
        self.assertIsNone(result)

    def test_unknown_event_returns_none(self) -> None:
        result = claude_stream_to_normalized("some_new_event", {"data": 42})
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# tools_to_claude
# ---------------------------------------------------------------------------


class TestToolsToClaude(unittest.TestCase):
    """Ollama-format tool definitions are converted to Claude format."""

    def test_single_tool_conversion(self) -> None:
        ollama_tools = [{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to run."},
                    },
                    "required": ["command"],
                },
            },
        }]
        claude_tools = tools_to_claude(ollama_tools)
        self.assertEqual(len(claude_tools), 1)
        ct = claude_tools[0]
        self.assertEqual(ct["name"], "bash")
        self.assertEqual(ct["description"], "Execute a shell command.")
        self.assertIn("input_schema", ct)
        self.assertEqual(ct["input_schema"]["type"], "object")
        self.assertIn("command", ct["input_schema"]["properties"])

    def test_multiple_tools(self) -> None:
        ollama_tools = [
            {"type": "function", "function": {"name": "tool_a", "description": "A", "parameters": {}}},
            {"type": "function", "function": {"name": "tool_b", "description": "B", "parameters": {}}},
        ]
        claude_tools = tools_to_claude(ollama_tools)
        self.assertEqual(len(claude_tools), 2)
        self.assertEqual(claude_tools[0]["name"], "tool_a")
        self.assertEqual(claude_tools[1]["name"], "tool_b")

    def test_empty_tool_list(self) -> None:
        self.assertEqual(tools_to_claude([]), [])

    def test_no_type_or_function_wrapper(self) -> None:
        """Claude tools should NOT have 'type' or 'function' keys."""
        ollama_tools = [{
            "type": "function",
            "function": {"name": "x", "description": "y", "parameters": {}},
        }]
        ct = tools_to_claude(ollama_tools)[0]
        self.assertNotIn("type", ct)
        self.assertNotIn("function", ct)

    def test_uses_input_schema_not_parameters(self) -> None:
        """Claude uses 'input_schema', not 'parameters'."""
        ollama_tools = [{
            "type": "function",
            "function": {"name": "x", "description": "y", "parameters": {"type": "object"}},
        }]
        ct = tools_to_claude(ollama_tools)[0]
        self.assertIn("input_schema", ct)
        self.assertNotIn("parameters", ct)


# ---------------------------------------------------------------------------
# tools_from_claude
# ---------------------------------------------------------------------------


class TestToolsFromClaude(unittest.TestCase):
    """Claude-format tool definitions are converted to Ollama format."""

    def test_single_tool_conversion(self) -> None:
        claude_tools = [{
            "name": "bash",
            "description": "Execute a shell command.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        }]
        ollama_tools = tools_from_claude(claude_tools)
        self.assertEqual(len(ollama_tools), 1)
        ot = ollama_tools[0]
        self.assertEqual(ot["type"], "function")
        self.assertIn("function", ot)
        self.assertEqual(ot["function"]["name"], "bash")
        self.assertEqual(ot["function"]["description"], "Execute a shell command.")
        self.assertEqual(
            ot["function"]["parameters"]["type"],
            "object",
        )

    def test_empty_tool_list(self) -> None:
        self.assertEqual(tools_from_claude([]), [])

    def test_round_trip_tools(self) -> None:
        """Converting Ollama -> Claude -> Ollama preserves structure."""
        original = [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }]
        round_tripped = tools_from_claude(tools_to_claude(original))
        self.assertEqual(round_tripped, original)


# ---------------------------------------------------------------------------
# Round-trip semantic preservation
# ---------------------------------------------------------------------------


class TestRoundTripPreservation(unittest.TestCase):
    """Converting normalized -> Claude -> normalized preserves meaning."""

    def test_text_content_round_trip(self) -> None:
        """Plain text survives the Ollama -> Claude -> normalized round trip."""
        original_messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"},
        ]
        _, claude_msgs = messages_to_claude(original_messages)

        # Simulate Claude responding with the same content.
        for cm in claude_msgs:
            if cm["role"] == "assistant":
                response = {"role": "assistant", "content": cm["content"]}
                normalized = claude_response_to_normalized(response)
                self.assertEqual(
                    normalized["message"]["content"],
                    "Hi!",
                )

    def test_tool_call_id_preserved_through_conversion(self) -> None:
        """tool_call_id survives: normalized -> claude -> (execute) -> claude tool_result."""
        tool_call_id = "toolu_abc123"

        # 1. Assistant message with tool_call.
        assistant_msg = {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [{
                "function": {"name": "bash", "arguments": {"command": "ls"}},
                "id": tool_call_id,
            }],
        }

        # 2. Tool result.
        tool_result_msg = {
            "role": "tool",
            "tool_name": "bash",
            "tool_call_id": tool_call_id,
            "content": "file.txt",
        }

        _, claude_msgs = messages_to_claude([assistant_msg, tool_result_msg])

        # The assistant message should have a tool_use block with the ID.
        assistant_claude = claude_msgs[0]
        tool_use_block = [
            b for b in assistant_claude["content"]
            if b.get("type") == "tool_use"
        ][0]
        self.assertEqual(tool_use_block["id"], tool_call_id)

        # The tool result message should reference the same ID.
        tool_result_claude = claude_msgs[1]
        result_block = tool_result_claude["content"][0]
        self.assertEqual(result_block["tool_use_id"], tool_call_id)

    def test_system_prompt_round_trip(self) -> None:
        """System prompt extracted and can be passed as separate param."""
        messages = [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": "hello"},
        ]
        system, claude_msgs = messages_to_claude(messages)
        self.assertEqual(system, "You are an AI assistant.")
        # Only user message remains (no system).
        self.assertEqual(len(claude_msgs), 1)
        self.assertEqual(claude_msgs[0]["role"], "user")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_messages_with_missing_content_key(self) -> None:
        """Messages without 'content' key produce empty strings."""
        messages = [{"role": "user"}]
        _, claude_msgs = messages_to_claude(messages)
        self.assertEqual(claude_msgs[0]["content"][0]["text"], "")

    def test_tool_result_without_tool_name(self) -> None:
        """Tool results without tool_name still convert."""
        messages = [{
            "role": "tool",
            "content": "some output",
            "tool_call_id": "id_1",
        }]
        _, claude_msgs = messages_to_claude(messages)
        block = claude_msgs[0]["content"][0]
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "id_1")
        self.assertEqual(block["content"], "some output")

    def test_claude_response_missing_content(self) -> None:
        """Claude response with no content key."""
        response = {"role": "assistant"}
        result = claude_response_to_normalized(response)
        self.assertEqual(result["message"]["content"], "")

    def test_tool_def_with_missing_function_key(self) -> None:
        """Gracefully handle tool def without 'function' key."""
        result = tools_to_claude([{"type": "function"}])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "")
        self.assertEqual(result[0]["description"], "")
        self.assertEqual(result[0]["input_schema"], {})

    def test_tool_results_at_end_of_messages(self) -> None:
        """Tool results at the very end are still flushed."""
        messages = [
            {"role": "user", "content": "do stuff"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {"name": "bash", "arguments": {}},
                    "id": "id_1",
                }],
            },
            {
                "role": "tool",
                "tool_name": "bash",
                "tool_call_id": "id_1",
                "content": "done",
            },
        ]
        _, claude_msgs = messages_to_claude(messages)
        # user, assistant, user(tool_result)
        self.assertEqual(len(claude_msgs), 3)
        last = claude_msgs[-1]
        self.assertEqual(last["role"], "user")
        self.assertEqual(last["content"][0]["type"], "tool_result")

    def test_interleaved_tool_results_and_user(self) -> None:
        """Tool results between user messages are correctly separated."""
        messages = [
            {"role": "tool", "tool_call_id": "id_1", "content": "r1"},
            {"role": "tool", "tool_call_id": "id_2", "content": "r2"},
            {"role": "user", "content": "follow up"},
            {"role": "tool", "tool_call_id": "id_3", "content": "r3"},
        ]
        _, claude_msgs = messages_to_claude(messages)
        # First: user(tool_result x2), then user(text), then user(tool_result)
        self.assertEqual(len(claude_msgs), 3)
        self.assertEqual(len(claude_msgs[0]["content"]), 2)  # 2 tool_results
        self.assertEqual(claude_msgs[1]["content"][0]["type"], "text")
        self.assertEqual(len(claude_msgs[2]["content"]), 1)  # 1 tool_result


if __name__ == "__main__":
    unittest.main()
