"""Tests for local_cli.providers.ollama_provider module.

Verifies that :class:`OllamaProvider` correctly wraps
:class:`OllamaClient` and implements the :class:`LLMProvider` interface.
All tests mock the underlying :class:`OllamaClient` to avoid requiring
a running Ollama server.
"""

import unittest
from typing import Any, Generator
from unittest.mock import MagicMock, patch

from local_cli.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaRequestError,
    OllamaStreamError,
)
from local_cli.providers.base import (
    LLMProvider,
    ProviderConnectionError,
    ProviderRequestError,
    ProviderStreamError,
)
from local_cli.providers.ollama_provider import OllamaProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTool:
    """Minimal stand-in for a Tool instance used by format_tools tests."""

    def __init__(self, name: str, description: str, parameters: dict) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return self._parameters

    def to_ollama_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": self._description,
                "parameters": self._parameters,
            },
        }


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestOllamaProviderInit(unittest.TestCase):
    """Tests for OllamaProvider construction."""

    def test_default_construction(self) -> None:
        """Provider can be created with no arguments."""
        provider = OllamaProvider()
        self.assertIsInstance(provider, OllamaProvider)
        self.assertIsInstance(provider.client, OllamaClient)

    def test_custom_base_url(self) -> None:
        """Provider passes base_url to the underlying client."""
        provider = OllamaProvider(base_url="http://127.0.0.1:11435")
        self.assertEqual(provider.client.base_url, "http://127.0.0.1:11435")

    def test_inject_client(self) -> None:
        """Provider accepts an existing OllamaClient."""
        client = OllamaClient()
        provider = OllamaProvider(client=client)
        self.assertIs(provider.client, client)

    def test_client_takes_precedence_over_base_url(self) -> None:
        """When both client and base_url are given, client wins."""
        client = OllamaClient("http://127.0.0.1:9999")
        provider = OllamaProvider(client=client, base_url="http://localhost:1111")
        self.assertIs(provider.client, client)
        self.assertEqual(provider.client.base_url, "http://127.0.0.1:9999")

    def test_implements_llm_provider(self) -> None:
        """OllamaProvider is an instance of LLMProvider."""
        provider = OllamaProvider()
        self.assertIsInstance(provider, LLMProvider)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestOllamaProviderProperties(unittest.TestCase):
    """Tests for OllamaProvider properties."""

    def setUp(self) -> None:
        self.provider = OllamaProvider()

    def test_name_is_ollama(self) -> None:
        """Provider name is 'ollama'."""
        self.assertEqual(self.provider.name, "ollama")

    def test_name_is_string(self) -> None:
        """Provider name is a string."""
        self.assertIsInstance(self.provider.name, str)

    def test_client_property(self) -> None:
        """client property returns the underlying OllamaClient."""
        self.assertIsInstance(self.provider.client, OllamaClient)


# ---------------------------------------------------------------------------
# chat() tests
# ---------------------------------------------------------------------------


class TestOllamaProviderChat(unittest.TestCase):
    """Tests for OllamaProvider.chat() delegation."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=OllamaClient)
        self.provider = OllamaProvider(client=self.mock_client)

    def test_delegates_to_client_chat(self) -> None:
        """chat() delegates to OllamaClient.chat()."""
        self.mock_client.chat.return_value = {
            "message": {"role": "assistant", "content": "hello"},
        }
        messages = [{"role": "user", "content": "hi"}]
        result = self.provider.chat(model="qwen3:8b", messages=messages)

        self.mock_client.chat.assert_called_once_with(
            model="qwen3:8b", messages=messages, tools=None,
        )
        self.assertEqual(result["message"]["content"], "hello")

    def test_passes_tools_to_client(self) -> None:
        """chat() forwards tools to the client."""
        self.mock_client.chat.return_value = {
            "message": {"role": "assistant", "content": "ok"},
        }
        tool_defs = [{"type": "function", "function": {"name": "bash"}}]
        self.provider.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "run ls"}],
            tools=tool_defs,
        )
        self.mock_client.chat.assert_called_once()
        call_kwargs = self.mock_client.chat.call_args
        self.assertEqual(call_kwargs[1]["tools"], tool_defs)

    def test_max_tokens_ignored(self) -> None:
        """chat() accepts max_tokens but does not forward it."""
        self.mock_client.chat.return_value = {
            "message": {"role": "assistant", "content": "ok"},
        }
        self.provider.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
        )
        # OllamaClient.chat does not accept max_tokens.
        call_kwargs = self.mock_client.chat.call_args[1]
        self.assertNotIn("max_tokens", call_kwargs)

    def test_returns_normalized_response(self) -> None:
        """chat() returns a dict with 'message' key."""
        self.mock_client.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "response text",
                "tool_calls": [
                    {
                        "function": {"name": "bash", "arguments": {"command": "ls"}},
                    }
                ],
            },
        }
        result = self.provider.chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "list files"}],
        )
        self.assertIn("message", result)
        self.assertEqual(result["message"]["role"], "assistant")
        self.assertIn("tool_calls", result["message"])

    def test_connection_error_propagates(self) -> None:
        """OllamaConnectionError propagates as ProviderConnectionError."""
        self.mock_client.chat.side_effect = OllamaConnectionError("offline")
        with self.assertRaises(ProviderConnectionError):
            self.provider.chat(
                model="qwen3:8b",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_request_error_propagates(self) -> None:
        """OllamaRequestError propagates as ProviderRequestError."""
        self.mock_client.chat.side_effect = OllamaRequestError("bad request")
        with self.assertRaises(ProviderRequestError):
            self.provider.chat(
                model="qwen3:8b",
                messages=[{"role": "user", "content": "hi"}],
            )


# ---------------------------------------------------------------------------
# chat_stream() tests
# ---------------------------------------------------------------------------


class TestOllamaProviderChatStream(unittest.TestCase):
    """Tests for OllamaProvider.chat_stream() delegation."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=OllamaClient)
        self.provider = OllamaProvider(client=self.mock_client)

    def test_delegates_to_client_chat_stream(self) -> None:
        """chat_stream() delegates to OllamaClient.chat_stream()."""
        chunks = [
            {"message": {"content": "hel"}, "done": False},
            {"message": {"content": "lo"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
        self.mock_client.chat_stream.return_value = iter(chunks)

        messages = [{"role": "user", "content": "hi"}]
        result = list(self.provider.chat_stream(
            model="qwen3:8b", messages=messages,
        ))

        self.mock_client.chat_stream.assert_called_once_with(
            model="qwen3:8b", messages=messages, tools=None,
        )
        self.assertEqual(len(result), 3)

    def test_returns_generator(self) -> None:
        """chat_stream() returns a generator."""
        self.mock_client.chat_stream.return_value = iter([])
        stream = self.provider.chat_stream(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertIsInstance(stream, Generator)

    def test_passes_tools_to_client(self) -> None:
        """chat_stream() forwards tools to the client."""
        self.mock_client.chat_stream.return_value = iter([
            {"message": {"content": ""}, "done": True},
        ])
        tool_defs = [{"type": "function", "function": {"name": "bash"}}]
        list(self.provider.chat_stream(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "run ls"}],
            tools=tool_defs,
        ))
        call_kwargs = self.mock_client.chat_stream.call_args[1]
        self.assertEqual(call_kwargs["tools"], tool_defs)

    def test_max_tokens_ignored(self) -> None:
        """chat_stream() accepts max_tokens but does not forward it."""
        self.mock_client.chat_stream.return_value = iter([
            {"message": {"content": ""}, "done": True},
        ])
        list(self.provider.chat_stream(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=512,
        ))
        call_kwargs = self.mock_client.chat_stream.call_args[1]
        self.assertNotIn("max_tokens", call_kwargs)

    def test_yields_normalized_chunks(self) -> None:
        """chat_stream() yields dicts with 'message' and 'done' keys."""
        chunks = [
            {"message": {"content": "token"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
        self.mock_client.chat_stream.return_value = iter(chunks)

        result = list(self.provider.chat_stream(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
        ))
        for chunk in result:
            self.assertIn("message", chunk)
            self.assertIn("done", chunk)

    def test_last_chunk_is_done(self) -> None:
        """The final chunk has done=True."""
        chunks = [
            {"message": {"content": "a"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
        self.mock_client.chat_stream.return_value = iter(chunks)

        result = list(self.provider.chat_stream(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
        ))
        self.assertTrue(result[-1]["done"])

    def test_tool_calls_in_final_chunk(self) -> None:
        """Tool calls appear in the final streaming chunk."""
        chunks = [
            {"message": {"content": ""}, "done": False},
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "bash", "arguments": {"command": "ls"}}},
                    ],
                },
                "done": True,
            },
        ]
        self.mock_client.chat_stream.return_value = iter(chunks)

        result = list(self.provider.chat_stream(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "list files"}],
        ))
        self.assertIn("tool_calls", result[-1]["message"])

    def test_connection_error_propagates(self) -> None:
        """OllamaConnectionError propagates as ProviderConnectionError."""
        self.mock_client.chat_stream.side_effect = OllamaConnectionError(
            "cannot connect"
        )
        with self.assertRaises(ProviderConnectionError):
            list(self.provider.chat_stream(
                model="qwen3:8b",
                messages=[{"role": "user", "content": "hi"}],
            ))

    def test_stream_error_propagates(self) -> None:
        """OllamaStreamError propagates as ProviderStreamError."""
        def _failing_stream(*args: Any, **kwargs: Any) -> Generator:
            yield {"message": {"content": "partial"}, "done": False}
            raise OllamaStreamError("mid-stream failure")

        self.mock_client.chat_stream.return_value = _failing_stream()

        with self.assertRaises(ProviderStreamError):
            list(self.provider.chat_stream(
                model="qwen3:8b",
                messages=[{"role": "user", "content": "hi"}],
            ))


# ---------------------------------------------------------------------------
# list_models() tests
# ---------------------------------------------------------------------------


class TestOllamaProviderListModels(unittest.TestCase):
    """Tests for OllamaProvider.list_models() delegation."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=OllamaClient)
        self.provider = OllamaProvider(client=self.mock_client)

    def test_delegates_to_client(self) -> None:
        """list_models() delegates to OllamaClient.list_models()."""
        self.mock_client.list_models.return_value = [
            {"name": "qwen3:8b", "size": 5200000000},
            {"name": "gemma3:4b", "size": 2400000000},
        ]
        result = self.provider.list_models()
        self.mock_client.list_models.assert_called_once()
        self.assertEqual(len(result), 2)

    def test_returns_list(self) -> None:
        """list_models() returns a list."""
        self.mock_client.list_models.return_value = []
        result = self.provider.list_models()
        self.assertIsInstance(result, list)

    def test_items_have_name(self) -> None:
        """Each model dict has a 'name' key."""
        self.mock_client.list_models.return_value = [
            {"name": "qwen3:8b"},
        ]
        result = self.provider.list_models()
        for model in result:
            self.assertIn("name", model)

    def test_connection_error_propagates(self) -> None:
        """OllamaConnectionError propagates as ProviderConnectionError."""
        self.mock_client.list_models.side_effect = OllamaConnectionError(
            "offline"
        )
        with self.assertRaises(ProviderConnectionError):
            self.provider.list_models()


# ---------------------------------------------------------------------------
# get_model_info() tests
# ---------------------------------------------------------------------------


class TestOllamaProviderGetModelInfo(unittest.TestCase):
    """Tests for OllamaProvider.get_model_info() delegation."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=OllamaClient)
        self.provider = OllamaProvider(client=self.mock_client)

    def test_delegates_to_show_model(self) -> None:
        """get_model_info() delegates to OllamaClient.show_model()."""
        self.mock_client.show_model.return_value = {
            "name": "qwen3:8b",
            "details": {"family": "qwen"},
            "capabilities": ["completion", "tools"],
        }
        result = self.provider.get_model_info("qwen3:8b")
        self.mock_client.show_model.assert_called_once_with("qwen3:8b")
        self.assertEqual(result["name"], "qwen3:8b")

    def test_returns_dict(self) -> None:
        """get_model_info() returns a dict."""
        self.mock_client.show_model.return_value = {"name": "qwen3:8b"}
        result = self.provider.get_model_info("qwen3:8b")
        self.assertIsInstance(result, dict)

    def test_result_has_name(self) -> None:
        """Result always includes a 'name' key."""
        self.mock_client.show_model.return_value = {"name": "qwen3:8b"}
        result = self.provider.get_model_info("qwen3:8b")
        self.assertIn("name", result)

    def test_adds_name_if_missing(self) -> None:
        """If show_model() omits 'name', it is added from the argument."""
        self.mock_client.show_model.return_value = {
            "details": {"family": "qwen"},
        }
        result = self.provider.get_model_info("qwen3:8b")
        self.assertEqual(result["name"], "qwen3:8b")

    def test_preserves_name_from_server(self) -> None:
        """If show_model() includes 'name', it is preserved."""
        self.mock_client.show_model.return_value = {
            "name": "qwen3:8b-custom",
        }
        result = self.provider.get_model_info("qwen3:8b")
        self.assertEqual(result["name"], "qwen3:8b-custom")

    def test_connection_error_propagates(self) -> None:
        """OllamaConnectionError propagates as ProviderConnectionError."""
        self.mock_client.show_model.side_effect = OllamaConnectionError(
            "offline"
        )
        with self.assertRaises(ProviderConnectionError):
            self.provider.get_model_info("qwen3:8b")

    def test_request_error_propagates(self) -> None:
        """OllamaRequestError propagates as ProviderRequestError."""
        self.mock_client.show_model.side_effect = OllamaRequestError(
            "model not found"
        )
        with self.assertRaises(ProviderRequestError):
            self.provider.get_model_info("nonexistent:latest")


# ---------------------------------------------------------------------------
# format_tools() tests
# ---------------------------------------------------------------------------


class TestOllamaProviderFormatTools(unittest.TestCase):
    """Tests for OllamaProvider.format_tools()."""

    def setUp(self) -> None:
        self.provider = OllamaProvider()

    def test_returns_list(self) -> None:
        """format_tools() returns a list."""
        result = self.provider.format_tools([])
        self.assertIsInstance(result, list)

    def test_empty_tools(self) -> None:
        """format_tools() with empty list returns empty list."""
        result = self.provider.format_tools([])
        self.assertEqual(result, [])

    def test_converts_single_tool(self) -> None:
        """format_tools() converts a single Tool to Ollama format."""
        tool = _FakeTool(
            name="bash",
            description="Execute a shell command.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        )
        result = self.provider.format_tools([tool])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "function")
        self.assertEqual(result[0]["function"]["name"], "bash")
        self.assertEqual(
            result[0]["function"]["description"],
            "Execute a shell command.",
        )
        self.assertIn("properties", result[0]["function"]["parameters"])

    def test_converts_multiple_tools(self) -> None:
        """format_tools() converts multiple Tools."""
        tools = [
            _FakeTool(
                name="bash",
                description="Run shell commands.",
                parameters={"type": "object", "properties": {}},
            ),
            _FakeTool(
                name="read",
                description="Read a file.",
                parameters={"type": "object", "properties": {}},
            ),
            _FakeTool(
                name="write",
                description="Write a file.",
                parameters={"type": "object", "properties": {}},
            ),
        ]
        result = self.provider.format_tools(tools)
        self.assertEqual(len(result), 3)
        names = [t["function"]["name"] for t in result]
        self.assertEqual(names, ["bash", "read", "write"])

    def test_format_matches_to_ollama_tool(self) -> None:
        """format_tools() output matches Tool.to_ollama_tool()."""
        tool = _FakeTool(
            name="edit",
            description="Edit a file.",
            parameters={"type": "object", "properties": {}},
        )
        result = self.provider.format_tools([tool])
        expected = tool.to_ollama_tool()
        self.assertEqual(result[0], expected)


# ---------------------------------------------------------------------------
# Interface conformance tests
# ---------------------------------------------------------------------------


class TestOllamaProviderConformance(unittest.TestCase):
    """Verify OllamaProvider satisfies the LLMProvider interface."""

    def test_is_subclass_of_llm_provider(self) -> None:
        """OllamaProvider is a subclass of LLMProvider."""
        self.assertTrue(issubclass(OllamaProvider, LLMProvider))

    def test_is_instance_of_llm_provider(self) -> None:
        """OllamaProvider instances pass isinstance check."""
        provider = OllamaProvider()
        self.assertIsInstance(provider, LLMProvider)

    def test_has_all_abstract_methods(self) -> None:
        """OllamaProvider implements all abstract methods from LLMProvider."""
        # If any abstract method is missing, instantiation would raise
        # TypeError.  This test verifies that does not happen.
        provider = OllamaProvider()
        self.assertIsNotNone(provider)

    def test_name_property_type(self) -> None:
        """name property returns a non-empty string."""
        provider = OllamaProvider()
        self.assertIsInstance(provider.name, str)
        self.assertTrue(len(provider.name) > 0)


# ---------------------------------------------------------------------------
# Exception hierarchy integration tests
# ---------------------------------------------------------------------------


class TestOllamaProviderExceptionHierarchy(unittest.TestCase):
    """Verify that Ollama-specific exceptions inherit from provider base."""

    def test_connection_error_is_provider_connection_error(self) -> None:
        """OllamaConnectionError is a ProviderConnectionError."""
        self.assertTrue(
            issubclass(OllamaConnectionError, ProviderConnectionError)
        )

    def test_request_error_is_provider_request_error(self) -> None:
        """OllamaRequestError is a ProviderRequestError."""
        self.assertTrue(
            issubclass(OllamaRequestError, ProviderRequestError)
        )

    def test_stream_error_is_provider_stream_error(self) -> None:
        """OllamaStreamError is a ProviderStreamError."""
        self.assertTrue(
            issubclass(OllamaStreamError, ProviderStreamError)
        )

    def test_catch_provider_connection_error(self) -> None:
        """OllamaConnectionError can be caught as ProviderConnectionError."""
        with self.assertRaises(ProviderConnectionError):
            raise OllamaConnectionError("test")

    def test_catch_provider_request_error(self) -> None:
        """OllamaRequestError can be caught as ProviderRequestError."""
        with self.assertRaises(ProviderRequestError):
            raise OllamaRequestError("test")

    def test_catch_provider_stream_error(self) -> None:
        """OllamaStreamError can be caught as ProviderStreamError."""
        with self.assertRaises(ProviderStreamError):
            raise OllamaStreamError("test")


if __name__ == "__main__":
    unittest.main()
