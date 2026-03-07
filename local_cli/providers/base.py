"""Abstract base class for LLM providers.

Every provider inherits from :class:`LLMProvider` and implements the five
abstract members: :pyattr:`name`, :meth:`chat`, :meth:`chat_stream`,
:meth:`list_models`, :meth:`get_model_info`, and :meth:`format_tools`.

The module also defines a provider-agnostic exception hierarchy that the
agent loop catches instead of provider-specific exceptions.  Provider-specific
exception classes (e.g. ``OllamaStreamError``) should inherit from the
corresponding base exception so that ``except ProviderStreamError`` catches
both.
"""

from abc import ABC, abstractmethod
from typing import Any, Generator


# ---------------------------------------------------------------------------
# Provider-agnostic exceptions (agent.py catches these instead of
# OllamaStreamError / ClaudeStreamError directly)
# ---------------------------------------------------------------------------


class ProviderConnectionError(Exception):
    """Raised when a provider cannot connect to its backend."""


class ProviderRequestError(Exception):
    """Raised when a provider receives an error response."""


class ProviderStreamError(Exception):
    """Raised when an error occurs mid-stream."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract interface for LLM providers.

    Each concrete provider adapts a specific backend (Ollama, Claude API,
    etc.) to this common interface.  The agent loop, orchestrator, and CLI
    operate exclusively through this interface so that they remain
    provider-agnostic.

    All methods accept and return a **normalized** message format based on the
    Ollama / OpenAI convention (``role``, ``content``, ``tool_calls``).
    Provider adapters convert to and from their native format internally.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. ``'ollama'``, ``'claude'``)."""
        ...

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion.

        Args:
            model: Model identifier (e.g. ``'qwen3:8b'`` or
                ``'claude-sonnet-4-5'``).
            messages: Conversation history in normalized format.
            tools: Tool definitions in the provider's native format
                (as returned by :meth:`format_tools`), or ``None``.
            max_tokens: Maximum tokens to generate.  **Required** for
                the Claude provider (API mandate); optional for Ollama.
                Each provider should define a sensible default in its
                constructor (e.g. ``ClaudeProvider(default_max_tokens=4096)``).

        Returns:
            A normalized response dict::

                {
                    "message": {
                        "role": "assistant",
                        "content": "...",
                        "tool_calls": [          # optional
                            {
                                "function": {
                                    "name": "tool_name",
                                    "arguments": { ... }
                                },
                                "id": "toolu_..."  # provider-assigned; may be None for Ollama
                            }
                        ]
                    }
                }

            The ``id`` field is **critical** for Claude -- the provider must
            preserve it so that subsequent ``tool_result`` messages can
            reference the correct ``tool_use_id``.

        Raises:
            ProviderConnectionError: If the backend is unreachable.
            ProviderRequestError: If the backend returns an error.
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Streaming chat completion.

        Args:
            model: Model identifier.
            messages: Conversation history in normalized format.
            tools: Tool definitions in the provider's native format, or
                ``None``.
            max_tokens: Maximum tokens to generate (see :meth:`chat`
                for details).

        Yields:
            Normalized chunks::

                {
                    "message": {
                        "content": "delta text",
                        "tool_calls": [...]  # in final chunk if present
                    },
                    "done": bool
                }

        Raises:
            ProviderConnectionError: If the backend is unreachable.
            ProviderStreamError: If an error occurs mid-stream.
        """
        ...

    @abstractmethod
    def list_models(self) -> list[dict[str, Any]]:
        """List available models.

        Returns:
            A list of dicts, each with at least a ``"name"`` key.

        Raises:
            ProviderConnectionError: If the backend is unreachable.
            ProviderRequestError: If the backend returns an error.
        """
        ...

    @abstractmethod
    def get_model_info(self, model: str) -> dict[str, Any]:
        """Get detailed model information including capabilities.

        Args:
            model: Model identifier.

        Returns:
            A dict with model details.  The exact keys depend on the
            provider but should include ``"name"`` at minimum.

        Raises:
            ProviderConnectionError: If the backend is unreachable.
            ProviderRequestError: If the backend returns an error.
        """
        ...

    @abstractmethod
    def format_tools(self, tools: list) -> list[dict[str, Any]]:
        """Convert :class:`~local_cli.tools.base.Tool` instances to this
        provider's tool format.

        Args:
            tools: A list of :class:`~local_cli.tools.base.Tool` instances.

        Returns:
            A list of tool definition dicts in the format expected by the
            provider's chat API.
        """
        ...
