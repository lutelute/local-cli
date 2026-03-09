"""Ollama provider adapter for local-cli.

Implements the :class:`~local_cli.providers.base.LLMProvider` interface by
wrapping an :class:`~local_cli.ollama_client.OllamaClient` instance.  All
Ollama-specific details (NDJSON streaming, ``/api/chat`` payload format,
``/api/show`` for model info) are handled internally so that the agent loop
and orchestrator see only the normalized provider interface.

Since Ollama already uses the Ollama / OpenAI message convention
(``role``, ``content``, ``tool_calls``), no format conversion is needed
for messages -- they pass through unchanged.
"""

from typing import Any, Generator

from local_cli.ollama_client import OllamaClient
from local_cli.providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama server.

    Wraps :class:`~local_cli.ollama_client.OllamaClient` and exposes the
    :class:`~local_cli.providers.base.LLMProvider` interface.  Exceptions
    raised by the client (``OllamaConnectionError``, ``OllamaRequestError``,
    ``OllamaStreamError``) already inherit from the provider-agnostic base
    exceptions, so they propagate unchanged.

    Args:
        client: An existing :class:`OllamaClient` instance.  If ``None``,
            a new client is created with default settings.
        base_url: Base URL for the Ollama API.  Ignored when *client* is
            provided.  Defaults to ``http://localhost:11434``.
    """

    def __init__(
        self,
        client: OllamaClient | None = None,
        base_url: str = "http://localhost:11434",
    ) -> None:
        if client is not None:
            self._client = client
        else:
            self._client = OllamaClient(base_url=base_url)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Provider name: ``'ollama'``."""
        return "ollama"

    @property
    def client(self) -> OllamaClient:
        """The underlying :class:`OllamaClient` instance."""
        return self._client

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        think: bool | None = None,
        format: str | dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion via Ollama.

        Delegates to :meth:`OllamaClient.chat`.  The ``max_tokens``
        parameter is accepted for interface conformance but is not
        forwarded to Ollama (Ollama does not require it).

        Args:
            model: Ollama model name (e.g. ``'qwen3:8b'``).
            messages: Conversation history in normalized format.
            tools: Tool definitions in Ollama format, or ``None``.
            max_tokens: Ignored for Ollama (accepted for interface
                conformance).
            options: Ollama inference parameters (e.g.
                ``{"num_ctx": 8192, "temperature": 0.6}``).  Passed
                through to :meth:`OllamaClient.chat`.
            think: Enable thinking/reasoning mode (e.g. for Qwen3).
            format: Response format — ``"json"`` or a JSON schema dict.
            keep_alive: Duration to keep the model loaded in memory
                (e.g. ``"5m"`` or ``300``).

        Returns:
            Normalized response dict with ``message`` key containing
            ``role``, ``content``, and optionally ``tool_calls``.

        Raises:
            ProviderConnectionError: If Ollama is unreachable.
            ProviderRequestError: If Ollama returns an error.
        """
        return self._client.chat(
            model=model,
            messages=messages,
            tools=tools,
            options=options,
            think=think,
            format=format,
            keep_alive=keep_alive,
        )

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        think: bool | None = None,
        format: str | dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Streaming chat completion via Ollama.

        Delegates to :meth:`OllamaClient.chat_stream`.  Each yielded
        chunk already conforms to the normalized streaming format
        (``message.content`` for delta text, ``done`` flag).

        Args:
            model: Ollama model name.
            messages: Conversation history in normalized format.
            tools: Tool definitions in Ollama format, or ``None``.
            max_tokens: Ignored for Ollama (accepted for interface
                conformance).
            options: Ollama inference parameters (e.g.
                ``{"num_ctx": 8192, "temperature": 0.6}``).  Passed
                through to :meth:`OllamaClient.chat_stream`.
            think: Enable thinking/reasoning mode (e.g. for Qwen3).
            format: Response format — ``"json"`` or a JSON schema dict.
            keep_alive: Duration to keep the model loaded in memory
                (e.g. ``"5m"`` or ``300``).

        Yields:
            Normalized streaming chunks.

        Raises:
            ProviderConnectionError: If Ollama is unreachable.
            ProviderStreamError: If an error occurs mid-stream.
        """
        yield from self._client.chat_stream(
            model=model,
            messages=messages,
            tools=tools,
            options=options,
            think=think,
            format=format,
            keep_alive=keep_alive,
        )

    def list_models(self) -> list[dict[str, Any]]:
        """List models available on the Ollama server.

        Delegates to :meth:`OllamaClient.list_models`.

        Returns:
            A list of model info dicts, each with at least a ``"name"`` key.

        Raises:
            ProviderConnectionError: If Ollama is unreachable.
            ProviderRequestError: If Ollama returns an error.
        """
        return self._client.list_models()

    def get_model_info(self, model: str) -> dict[str, Any]:
        """Get detailed model information from Ollama.

        Delegates to :meth:`OllamaClient.show_model` and adds a ``"name"``
        key to the result for consistency with the provider interface.

        Args:
            model: Ollama model name.

        Returns:
            A dict with model details including ``"name"``.

        Raises:
            ProviderConnectionError: If Ollama is unreachable.
            ProviderRequestError: If Ollama returns an error.
        """
        info = self._client.show_model(model)
        # Ensure the response always includes a "name" key as required
        # by the LLMProvider interface.
        if "name" not in info:
            info["name"] = model
        return info

    def format_tools(self, tools: list) -> list[dict[str, Any]]:
        """Convert :class:`~local_cli.tools.base.Tool` instances to Ollama
        tool format.

        Calls :meth:`~local_cli.tools.base.Tool.to_ollama_tool` on each
        tool to produce the ``{"type": "function", "function": {...}}``
        format expected by the Ollama ``/api/chat`` endpoint.

        Args:
            tools: A list of :class:`~local_cli.tools.base.Tool` instances.

        Returns:
            A list of tool definition dicts in Ollama format.
        """
        return [tool.to_ollama_tool() for tool in tools]
