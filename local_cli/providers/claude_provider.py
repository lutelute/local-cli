"""Claude API provider for local-cli.

Implements the :class:`~local_cli.providers.base.LLMProvider` interface by
communicating directly with the Anthropic Claude Messages API using
``urllib.request`` from the standard library.  Supports both streaming
(SSE) and non-streaming chat completions, tool use with Claude's
content-block format, and automatic retry with exponential backoff for
rate-limit (HTTP 429) responses.

Authentication is handled via the ``ANTHROPIC_API_KEY`` environment
variable (or an explicit *api_key* constructor argument).  The API key
is **never** logged or written to configuration files.

All messages are accepted and returned in the normalized (Ollama-style)
format.  Conversion to and from Claude's native content-block format is
handled internally via :mod:`local_cli.providers.message_converter`.
"""

import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Generator

from local_cli.providers.base import (
    LLMProvider,
    ProviderConnectionError,
    ProviderRequestError,
    ProviderStreamError,
)
from local_cli.providers.message_converter import (
    claude_response_to_normalized,
    claude_stream_to_normalized,
    messages_to_claude,
    tools_to_claude,
)
from local_cli.providers.sse_parser import parse_sse_stream

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------

_BASE_URL = "https://api.anthropic.com"
_API_VERSION = "2023-06-01"

_DEFAULT_MAX_TOKENS = 4096

# Timeout for non-streaming requests (seconds).
_DEFAULT_TIMEOUT = 60

# Timeout for streaming requests (seconds).
_STREAM_TIMEOUT = 120

# Rate-limit retry configuration.
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds; doubles each attempt (1, 2, 4)

# Hardcoded list of available Claude models.
_CLAUDE_MODELS = [
    {
        "name": "claude-opus-4-6",
        "description": "Most capable Claude model for complex tasks.",
    },
    {
        "name": "claude-sonnet-4-5",
        "description": "Balanced performance and cost.",
    },
    {
        "name": "claude-haiku-4-5",
        "description": "Fast and cost-effective for simpler tasks.",
    },
]


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ClaudeConnectionError(ProviderConnectionError):
    """Raised when the client cannot connect to the Claude API.

    Inherits from :class:`~local_cli.providers.base.ProviderConnectionError`
    so that ``except ProviderConnectionError`` catches this as well.
    """


class ClaudeRequestError(ProviderRequestError):
    """Raised when the Claude API returns an error response.

    Inherits from :class:`~local_cli.providers.base.ProviderRequestError`
    so that ``except ProviderRequestError`` catches this as well.
    """


class ClaudeStreamError(ProviderStreamError):
    """Raised when an error is encountered mid-stream.

    Inherits from :class:`~local_cli.providers.base.ProviderStreamError`
    so that ``except ProviderStreamError`` catches this as well.
    """


class ClaudeRateLimitError(ClaudeRequestError):
    """Raised when the Claude API returns HTTP 429 (rate limited).

    This is a subclass of :class:`ClaudeRequestError` so callers that
    catch ``ProviderRequestError`` will also catch rate-limit errors.
    The retry logic in :class:`ClaudeProvider` handles these internally.
    """


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class ClaudeProvider(LLMProvider):
    """LLM provider backed by the Anthropic Claude Messages API.

    Communicates with ``POST /v1/messages`` using ``urllib.request``.
    Messages are converted between the normalized (Ollama-style) format
    and Claude's content-block format at the provider boundary.

    Args:
        api_key: Anthropic API key.  If ``None``, reads from the
            ``ANTHROPIC_API_KEY`` environment variable.
        default_max_tokens: Default maximum tokens for responses when
            ``max_tokens`` is not provided to :meth:`chat` or
            :meth:`chat_stream`.  Claude **requires** this parameter.
        base_url: Base URL for the Anthropic API.  Defaults to
            ``https://api.anthropic.com``.
        timeout: Timeout in seconds for non-streaming requests.
        stream_timeout: Timeout in seconds for streaming requests.

    Raises:
        ValueError: If no API key is available (neither argument nor
            environment variable).
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_max_tokens: int = _DEFAULT_MAX_TOKENS,
        base_url: str = _BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
        stream_timeout: int = _STREAM_TIMEOUT,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key is required. Set the ANTHROPIC_API_KEY "
                "environment variable or pass api_key to ClaudeProvider."
            )

        self._api_key: str = resolved_key
        self._default_max_tokens: int = default_max_tokens
        self._base_url: str = base_url.rstrip("/")
        self._timeout: int = timeout
        self._stream_timeout: int = stream_timeout

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Provider name: ``'claude'``."""
        return "claude"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Build the standard request headers for the Claude API.

        Returns:
            A dict with the required authentication and content-type
            headers.
        """
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }

    def _request(
        self,
        data: dict[str, Any],
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Send a non-streaming POST to ``/v1/messages``.

        Includes automatic retry with exponential backoff for HTTP 429
        (rate limit) responses.

        Args:
            data: JSON body for the request.
            timeout: Socket timeout in seconds.  Defaults to the
                instance-level timeout.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            ClaudeConnectionError: On connection failure or timeout.
            ClaudeRateLimitError: If rate-limited and all retries exhausted.
            ClaudeRequestError: On other HTTP error responses.
        """
        if timeout is None:
            timeout = self._timeout

        url = f"{self._base_url}/v1/messages"
        body = json.dumps(data).encode("utf-8")
        headers = self._headers()

        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            req = urllib.request.Request(
                url, data=body, headers=headers, method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw)

            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    last_exc = ClaudeRateLimitError(
                        f"Claude API rate limited (attempt "
                        f"{attempt + 1}/{_MAX_RETRIES})"
                    )
                    # Exponential backoff: 1s, 2s, 4s
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                    continue

                # Read error body for diagnostics.
                error_body = ""
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                raise ClaudeRequestError(
                    f"Claude API error (HTTP {exc.code}): {error_body}"
                ) from exc

            except urllib.error.URLError as exc:
                raise ClaudeConnectionError(
                    f"Failed to connect to Claude API at {url}: {exc}"
                ) from exc

            except socket.timeout as exc:
                raise ClaudeConnectionError(
                    f"Request to Claude API timed out ({timeout}s): {url}"
                ) from exc

            except json.JSONDecodeError as exc:
                raise ClaudeRequestError(
                    f"Invalid JSON response from Claude API: {exc}"
                ) from exc

        # All retries exhausted.
        if last_exc is not None:
            raise last_exc
        raise ClaudeRequestError("Request failed after retries")  # pragma: no cover

    def _stream_request(
        self,
        data: dict[str, Any],
        timeout: int | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Send a streaming POST to ``/v1/messages`` and yield SSE events.

        Includes automatic retry with exponential backoff for HTTP 429
        (rate limit) responses on the initial connection.

        Args:
            data: JSON body for the request (must include
                ``"stream": true``).
            timeout: Socket timeout in seconds.  Defaults to the
                instance-level stream timeout.

        Yields:
            Normalized stream chunks.

        Raises:
            ClaudeConnectionError: On connection failure or timeout.
            ClaudeRateLimitError: If rate-limited and all retries exhausted.
            ClaudeRequestError: On other HTTP error responses.
            ClaudeStreamError: If the server sends an error event mid-stream.
        """
        if timeout is None:
            timeout = self._stream_timeout

        url = f"{self._base_url}/v1/messages"
        body = json.dumps(data).encode("utf-8")
        headers = self._headers()

        resp = None
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            req = urllib.request.Request(
                url, data=body, headers=headers, method="POST",
            )

            try:
                resp = urllib.request.urlopen(req, timeout=timeout)
                break  # Connection successful.

            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    last_exc = ClaudeRateLimitError(
                        f"Claude API rate limited (attempt "
                        f"{attempt + 1}/{_MAX_RETRIES})"
                    )
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                    continue

                error_body = ""
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                raise ClaudeRequestError(
                    f"Claude API error (HTTP {exc.code}): {error_body}"
                ) from exc

            except urllib.error.URLError as exc:
                raise ClaudeConnectionError(
                    f"Failed to connect to Claude API at {url}: {exc}"
                ) from exc

            except socket.timeout as exc:
                raise ClaudeConnectionError(
                    f"Request to Claude API timed out ({timeout}s): {url}"
                ) from exc

        if resp is None:
            if last_exc is not None:
                raise last_exc
            raise ClaudeRequestError("Stream request failed after retries")  # pragma: no cover

        # Parse the SSE stream and convert events to normalized chunks.
        # Accumulate tool input JSON deltas across events.
        tool_calls: list[dict[str, Any]] = []
        current_tool: dict[str, Any] | None = None
        json_delta_parts: list[str] = []

        try:
            for sse_event in parse_sse_stream(resp):
                event_type = sse_event.event_type
                event_data = sse_event.data

                # Handle error events.
                if event_type == "error":
                    error_msg = event_data.get("error", {})
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get("message", str(error_msg))
                    raise ClaudeStreamError(
                        f"Claude API stream error: {error_msg}"
                    )

                # Track tool use blocks via content_block_start.
                if event_type == "content_block_start":
                    block = event_data.get("content_block", {})
                    if block.get("type") == "tool_use":
                        current_tool = {
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                        }
                        json_delta_parts.clear()
                    continue

                # Accumulate input_json_delta for tool use.
                if event_type == "content_block_delta":
                    delta = event_data.get("delta", {})
                    if delta.get("type") == "input_json_delta":
                        json_delta_parts.append(delta.get("partial_json", ""))
                        continue

                # Finalize tool block on content_block_stop.
                if event_type == "content_block_stop":
                    if current_tool is not None:
                        raw_json = "".join(json_delta_parts)
                        try:
                            arguments = json.loads(raw_json) if raw_json else {}
                        except json.JSONDecodeError:
                            arguments = {}
                        tool_calls.append({
                            "function": {
                                "name": current_tool["name"],
                                "arguments": arguments,
                            },
                            "id": current_tool["id"],
                        })
                        current_tool = None
                        json_delta_parts.clear()
                    continue

                # Convert the event to a normalized chunk.
                chunk = claude_stream_to_normalized(event_type, event_data)
                if chunk is not None:
                    # On message_stop or done, attach accumulated tool calls.
                    if chunk.get("done") and tool_calls:
                        chunk["message"]["tool_calls"] = list(tool_calls)
                    yield chunk

        finally:
            resp.close()

    def _build_request_body(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build the JSON request body for the Claude Messages API.

        Converts normalized messages to Claude's content-block format,
        extracts the system prompt, and assembles the full request body.

        Args:
            model: Claude model identifier.
            messages: Conversation history in normalized format.
            tools: Tool definitions in Claude format, or ``None``.
            max_tokens: Maximum tokens to generate.
            stream: Whether to enable SSE streaming.

        Returns:
            A dict ready to be JSON-serialized and sent to the API.
        """
        system_text, claude_messages = messages_to_claude(messages)
        effective_max_tokens = max_tokens or self._default_max_tokens

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": effective_max_tokens,
            "messages": claude_messages,
        }

        if system_text is not None:
            body["system"] = system_text

        if tools:
            body["tools"] = tools

        if stream:
            body["stream"] = True

        return body

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion via the Claude Messages API.

        Converts normalized messages to Claude format, sends the request,
        and converts the response back to normalized format.

        Args:
            model: Claude model identifier (e.g. ``'claude-sonnet-4-5'``).
            messages: Conversation history in normalized format.
            tools: Tool definitions in Claude format (as returned by
                :meth:`format_tools`), or ``None``.
            max_tokens: Maximum tokens to generate.  Defaults to the
                provider's ``default_max_tokens``.

        Returns:
            Normalized response dict with ``message`` key containing
            ``role``, ``content``, and optionally ``tool_calls``.

        Raises:
            ClaudeConnectionError: If the Claude API is unreachable.
            ClaudeRequestError: If the API returns an error.
            ClaudeRateLimitError: If rate-limited after all retries.
        """
        body = self._build_request_body(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            stream=False,
        )

        response = self._request(body)
        return claude_response_to_normalized(response)

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Streaming chat completion via the Claude Messages API.

        Streams SSE events from the Claude API, converts text deltas to
        normalized chunks, and accumulates tool use input across events.
        Tool calls are attached to the final (``done: True``) chunk.

        Args:
            model: Claude model identifier.
            messages: Conversation history in normalized format.
            tools: Tool definitions in Claude format, or ``None``.
            max_tokens: Maximum tokens to generate.  Defaults to the
                provider's ``default_max_tokens``.

        Yields:
            Normalized streaming chunks.

        Raises:
            ClaudeConnectionError: If the Claude API is unreachable.
            ClaudeStreamError: If an error occurs mid-stream.
            ClaudeRateLimitError: If rate-limited after all retries.
        """
        body = self._build_request_body(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            stream=True,
        )

        yield from self._stream_request(body)

    def list_models(self) -> list[dict[str, Any]]:
        """List available Claude models.

        Returns a hardcoded list of Claude model families since the
        Anthropic API does not provide a model listing endpoint.

        Returns:
            A list of model info dicts, each with ``"name"`` and
            ``"description"`` keys.
        """
        return list(_CLAUDE_MODELS)

    def get_model_info(self, model: str) -> dict[str, Any]:
        """Get basic information about a Claude model.

        Returns a constructed info dict since the Anthropic API does not
        provide a dedicated model info endpoint.

        Args:
            model: Claude model identifier.

        Returns:
            A dict with model details including ``"name"``,
            ``"provider"``, and ``"capabilities"``.
        """
        return {
            "name": model,
            "provider": "claude",
            "capabilities": ["completion", "tools", "vision"],
        }

    def format_tools(self, tools: list) -> list[dict[str, Any]]:
        """Convert :class:`~local_cli.tools.base.Tool` instances to Claude
        tool format.

        Calls :meth:`~local_cli.tools.base.Tool.to_claude_tool` on each
        tool to produce the ``{"name": ..., "description": ...,
        "input_schema": ...}`` format expected by the Claude Messages API.

        Args:
            tools: A list of :class:`~local_cli.tools.base.Tool` instances.

        Returns:
            A list of tool definition dicts in Claude format.
        """
        return [tool.to_claude_tool() for tool in tools]
