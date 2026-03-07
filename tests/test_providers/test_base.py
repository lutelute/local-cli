"""Tests for local_cli.providers.base module.

Verifies the LLMProvider ABC, the provider-agnostic exception hierarchy,
and conformance requirements for concrete provider implementations.
"""

import unittest
from abc import ABC
from typing import Any, Generator

from local_cli.providers.base import (
    LLMProvider,
    ProviderConnectionError,
    ProviderRequestError,
    ProviderStreamError,
)


# ---------------------------------------------------------------------------
# Concrete stub for testing the ABC contract
# ---------------------------------------------------------------------------


class _StubProvider(LLMProvider):
    """Minimal concrete provider for testing ABC requirements."""

    @property
    def name(self) -> str:
        return "stub"

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        return {
            "message": {
                "role": "assistant",
                "content": "stub response",
            }
        }

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        yield {"message": {"content": "chunk"}, "done": False}
        yield {"message": {"content": ""}, "done": True}

    def list_models(self) -> list[dict[str, Any]]:
        return [{"name": "stub-model"}]

    def get_model_info(self, model: str) -> dict[str, Any]:
        return {"name": model}

    def format_tools(self, tools: list) -> list[dict[str, Any]]:
        return [{"name": "stub_tool"}]


class _IncompleteProvider(LLMProvider):
    """Provider missing all abstract methods -- should not be instantiable."""

    pass


# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------


class TestExceptionHierarchy(unittest.TestCase):
    """Provider-agnostic exceptions are proper Exception subclasses."""

    def test_provider_connection_error_is_exception(self) -> None:
        self.assertTrue(issubclass(ProviderConnectionError, Exception))

    def test_provider_request_error_is_exception(self) -> None:
        self.assertTrue(issubclass(ProviderRequestError, Exception))

    def test_provider_stream_error_is_exception(self) -> None:
        self.assertTrue(issubclass(ProviderStreamError, Exception))

    def test_connection_error_can_carry_message(self) -> None:
        exc = ProviderConnectionError("cannot reach backend")
        self.assertEqual(str(exc), "cannot reach backend")

    def test_request_error_can_carry_message(self) -> None:
        exc = ProviderRequestError("bad request")
        self.assertEqual(str(exc), "bad request")

    def test_stream_error_can_carry_message(self) -> None:
        exc = ProviderStreamError("mid-stream failure")
        self.assertEqual(str(exc), "mid-stream failure")

    def test_connection_error_raiseable(self) -> None:
        with self.assertRaises(ProviderConnectionError):
            raise ProviderConnectionError("offline")

    def test_request_error_raiseable(self) -> None:
        with self.assertRaises(ProviderRequestError):
            raise ProviderRequestError("404")

    def test_stream_error_raiseable(self) -> None:
        with self.assertRaises(ProviderStreamError):
            raise ProviderStreamError("broken pipe")

    def test_exceptions_are_distinct_classes(self) -> None:
        """Each exception type is independent (not a subclass of another)."""
        self.assertFalse(issubclass(ProviderConnectionError, ProviderRequestError))
        self.assertFalse(issubclass(ProviderConnectionError, ProviderStreamError))
        self.assertFalse(issubclass(ProviderRequestError, ProviderStreamError))

    def test_catch_base_exception_catches_provider_exceptions(self) -> None:
        """All provider exceptions are catchable via bare ``except Exception``."""
        for exc_cls in (
            ProviderConnectionError,
            ProviderRequestError,
            ProviderStreamError,
        ):
            with self.assertRaises(Exception):
                raise exc_cls("test")


# ---------------------------------------------------------------------------
# LLMProvider ABC contract tests
# ---------------------------------------------------------------------------


class TestLLMProviderABC(unittest.TestCase):
    """LLMProvider is a proper ABC that cannot be instantiated directly."""

    def test_is_abstract_base_class(self) -> None:
        self.assertTrue(issubclass(LLMProvider, ABC))

    def test_cannot_instantiate_directly(self) -> None:
        with self.assertRaises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_incomplete_provider_cannot_instantiate(self) -> None:
        """A subclass that does not implement all abstract methods is
        rejected at instantiation time."""
        with self.assertRaises(TypeError):
            _IncompleteProvider()  # type: ignore[abstract]

    def test_required_abstract_methods(self) -> None:
        """LLMProvider defines exactly the expected set of abstract members."""
        expected = {"chat", "chat_stream", "list_models", "get_model_info",
                    "format_tools", "name"}
        # ABC stores abstract method names in __abstractmethods__
        self.assertEqual(LLMProvider.__abstractmethods__, expected)


# ---------------------------------------------------------------------------
# Conformance tests for concrete provider stubs
# ---------------------------------------------------------------------------


class TestStubProviderConformance(unittest.TestCase):
    """A fully-implemented concrete provider satisfies the ABC contract."""

    def setUp(self) -> None:
        self.provider = _StubProvider()

    def test_instantiation(self) -> None:
        """A complete implementation can be instantiated."""
        self.assertIsInstance(self.provider, LLMProvider)

    def test_isinstance_check(self) -> None:
        """Concrete providers pass isinstance check against LLMProvider."""
        self.assertIsInstance(self.provider, LLMProvider)

    def test_name_property(self) -> None:
        """name returns a non-empty string."""
        self.assertIsInstance(self.provider.name, str)
        self.assertTrue(len(self.provider.name) > 0)

    def test_chat_returns_dict(self) -> None:
        """chat() returns a dict with a 'message' key."""
        result = self.provider.chat(
            model="stub-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertIsInstance(result, dict)
        self.assertIn("message", result)

    def test_chat_message_has_role(self) -> None:
        """chat() response message includes 'role'."""
        result = self.provider.chat(
            model="stub-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertIn("role", result["message"])
        self.assertEqual(result["message"]["role"], "assistant")

    def test_chat_message_has_content(self) -> None:
        """chat() response message includes 'content'."""
        result = self.provider.chat(
            model="stub-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertIn("content", result["message"])

    def test_chat_accepts_tools(self) -> None:
        """chat() accepts an optional tools parameter."""
        result = self.provider.chat(
            model="stub-model",
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
        )
        self.assertIsInstance(result, dict)

    def test_chat_accepts_max_tokens(self) -> None:
        """chat() accepts an optional max_tokens parameter."""
        result = self.provider.chat(
            model="stub-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
        )
        self.assertIsInstance(result, dict)

    def test_chat_stream_returns_generator(self) -> None:
        """chat_stream() returns a generator."""
        stream = self.provider.chat_stream(
            model="stub-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertIsInstance(stream, Generator)

    def test_chat_stream_yields_dicts(self) -> None:
        """chat_stream() yields dicts with 'message' and 'done' keys."""
        stream = self.provider.chat_stream(
            model="stub-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        chunks = list(stream)
        self.assertTrue(len(chunks) > 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, dict)
            self.assertIn("message", chunk)
            self.assertIn("done", chunk)

    def test_chat_stream_last_chunk_is_done(self) -> None:
        """The final chunk from chat_stream() has done=True."""
        stream = self.provider.chat_stream(
            model="stub-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        chunks = list(stream)
        self.assertTrue(chunks[-1]["done"])

    def test_chat_stream_accepts_tools(self) -> None:
        """chat_stream() accepts an optional tools parameter."""
        stream = self.provider.chat_stream(
            model="stub-model",
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
        )
        chunks = list(stream)
        self.assertTrue(len(chunks) > 0)

    def test_chat_stream_accepts_max_tokens(self) -> None:
        """chat_stream() accepts an optional max_tokens parameter."""
        stream = self.provider.chat_stream(
            model="stub-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=512,
        )
        chunks = list(stream)
        self.assertTrue(len(chunks) > 0)

    def test_list_models_returns_list(self) -> None:
        """list_models() returns a list."""
        models = self.provider.list_models()
        self.assertIsInstance(models, list)

    def test_list_models_items_have_name(self) -> None:
        """Each model in list_models() has a 'name' key."""
        models = self.provider.list_models()
        for model in models:
            self.assertIn("name", model)

    def test_get_model_info_returns_dict(self) -> None:
        """get_model_info() returns a dict."""
        info = self.provider.get_model_info("stub-model")
        self.assertIsInstance(info, dict)

    def test_get_model_info_has_name(self) -> None:
        """get_model_info() result includes the model name."""
        info = self.provider.get_model_info("stub-model")
        self.assertIn("name", info)

    def test_format_tools_returns_list(self) -> None:
        """format_tools() returns a list."""
        result = self.provider.format_tools([])
        self.assertIsInstance(result, list)


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------


class TestPackageImports(unittest.TestCase):
    """Provider package exports the expected names."""

    def test_import_llm_provider_from_package(self) -> None:
        from local_cli.providers import LLMProvider as LP
        self.assertIs(LP, LLMProvider)

    def test_import_connection_error_from_package(self) -> None:
        from local_cli.providers import ProviderConnectionError as PCE
        self.assertIs(PCE, ProviderConnectionError)

    def test_import_request_error_from_package(self) -> None:
        from local_cli.providers import ProviderRequestError as PRE
        self.assertIs(PRE, ProviderRequestError)

    def test_import_stream_error_from_package(self) -> None:
        from local_cli.providers import ProviderStreamError as PSE
        self.assertIs(PSE, ProviderStreamError)


if __name__ == "__main__":
    unittest.main()
