"""Ollama HTTP client for local-cli.

Provides a zero-dependency client for communicating with the Ollama REST API
using only ``urllib.request`` from the standard library.  Supports streaming
chat completions (NDJSON), embeddings, model management, and version queries.
"""

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Generator

from local_cli.providers.base import (
    ProviderConnectionError,
    ProviderRequestError,
    ProviderStreamError,
)
from local_cli.security import validate_model_name, validate_ollama_host

# ---------------------------------------------------------------------------
# Default timeouts (seconds)
# ---------------------------------------------------------------------------

# Simple API calls (version, tags, etc.).
_DEFAULT_TIMEOUT = 30

# Streaming chat completions (model may take time to start generating).
_STREAM_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class OllamaConnectionError(ProviderConnectionError):
    """Raised when the client cannot connect to Ollama.

    Inherits from :class:`~local_cli.providers.base.ProviderConnectionError`
    so that ``except ProviderConnectionError`` catches this as well.
    """


class OllamaRequestError(ProviderRequestError):
    """Raised when the Ollama API returns an error response.

    Inherits from :class:`~local_cli.providers.base.ProviderRequestError`
    so that ``except ProviderRequestError`` catches this as well.
    """


class OllamaStreamError(ProviderStreamError):
    """Raised when an error is encountered mid-stream.

    Inherits from :class:`~local_cli.providers.base.ProviderStreamError`
    so that ``except ProviderStreamError`` catches this as well.
    """


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OllamaClient:
    """HTTP client for the Ollama REST API.

    Uses ``urllib.request`` for all HTTP communication.  Each request opens a
    new TCP connection (no connection pooling).

    Args:
        base_url: Base URL for the Ollama API.  Must point to localhost.
            Defaults to ``http://localhost:11434``.

    Raises:
        ValueError: If *base_url* does not point to a localhost address.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        base_url = base_url.rstrip("/")

        if not validate_ollama_host(base_url):
            raise ValueError(
                f"Ollama host must be a localhost address, got: {base_url}"
            )

        self.base_url: str = base_url

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        """Send an HTTP request to the Ollama API and return parsed JSON.

        Args:
            method: HTTP method (``GET``, ``POST``, etc.).
            path: API path (e.g. ``/api/version``).
            data: Optional JSON body for the request.
            timeout: Socket timeout in seconds.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            OllamaConnectionError: On connection failure or timeout.
            OllamaRequestError: On HTTP error responses.
        """
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode("utf-8") if data else None
        headers: dict[str, str] = {}
        if body is not None:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            url, data=body, headers=headers, method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            raise OllamaRequestError(
                f"Ollama API error {exc.code} at {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OllamaConnectionError(
                f"Failed to connect to Ollama at {url}: {exc}"
            ) from exc
        except socket.timeout as exc:
            raise OllamaConnectionError(
                f"Request to Ollama timed out ({timeout}s): {url}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise OllamaRequestError(
                f"Invalid JSON response from Ollama: {exc}"
            ) from exc

    def _stream_request(
        self,
        path: str,
        data: dict[str, Any],
        timeout: int = _STREAM_TIMEOUT,
    ) -> Generator[dict[str, Any], None, None]:
        """Send a POST request and yield parsed NDJSON lines.

        Each line of the response is parsed as a separate JSON object and
        yielded.  Mid-stream ``{"error": ...}`` objects raise
        :class:`OllamaStreamError`.

        Args:
            path: API path (e.g. ``/api/chat``).
            data: JSON body for the POST request.
            timeout: Socket timeout in seconds.

        Yields:
            Parsed JSON chunks from the NDJSON response stream.

        Raises:
            OllamaConnectionError: On connection failure or timeout.
            OllamaStreamError: If the server sends an error object mid-stream.
        """
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            raise OllamaRequestError(
                f"Ollama API error {exc.code} at {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OllamaConnectionError(
                f"Failed to connect to Ollama at {url}: {exc}"
            ) from exc
        except socket.timeout as exc:
            raise OllamaConnectionError(
                f"Request to Ollama timed out ({timeout}s): {url}"
            ) from exc

        try:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk:
                    raise OllamaStreamError(
                        f"Ollama error: {chunk['error']}"
                    )
                yield chunk
        finally:
            resp.close()

    def _request_no_content(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        """Send an HTTP request that returns no JSON body (HTTP 200 OK).

        Used by endpoints like ``DELETE /api/delete`` and ``POST /api/copy``
        that return an empty or status-only response.

        Args:
            method: HTTP method (``DELETE``, ``POST``, etc.).
            path: API path (e.g. ``/api/delete``).
            data: Optional JSON body for the request.
            timeout: Socket timeout in seconds.

        Raises:
            OllamaConnectionError: On connection failure or timeout.
            OllamaRequestError: On HTTP error responses.
        """
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode("utf-8") if data else None
        headers: dict[str, str] = {}
        if body is not None:
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            url, data=body, headers=headers, method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()  # consume body without parsing
        except urllib.error.URLError as exc:
            raise OllamaConnectionError(
                f"Failed to connect to Ollama at {url}: {exc}"
            ) from exc
        except socket.timeout as exc:
            raise OllamaConnectionError(
                f"Request to Ollama timed out ({timeout}s): {url}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_version(self) -> dict[str, Any]:
        """Get the Ollama server version.

        Returns:
            Version information dict (e.g. ``{"version": "0.5.1"}``).

        Raises:
            OllamaConnectionError: On connection failure.
            OllamaRequestError: On unexpected response.
        """
        return self._request("GET", "/api/version")

    def list_models(self) -> list[dict[str, Any]]:
        """List models available on the Ollama server.

        Returns:
            A list of model info dicts from the ``models`` key of the
            ``/api/tags`` response.

        Raises:
            OllamaConnectionError: On connection failure.
            OllamaRequestError: On unexpected response.
        """
        result = self._request("GET", "/api/tags")
        return result.get("models", [])

    def _validate_model(self, model: str) -> None:
        """Validate a model name before sending to the API.

        Args:
            model: Model name to validate.

        Raises:
            ValueError: If the model name is invalid.
        """
        if not validate_model_name(model):
            raise ValueError(f"Invalid model name: {model!r}")

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        think: bool | None = None,
        format: str | dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream a chat completion from Ollama.

        Sends a ``POST /api/chat`` request with ``stream: true`` and yields
        each NDJSON chunk as a parsed dictionary.  Token content appears in
        ``chunk["message"]["content"]`` for most chunks.  Tool calls
        typically appear in the final chunk (``done: true``), but callers
        should accumulate them across chunks for robustness.

        Args:
            model: Ollama model name (e.g. ``qwen3:8b``).
            messages: Conversation messages list.
            tools: Optional list of tool definitions in Ollama format.
            options: Optional dict of inference parameters (e.g.
                ``{"num_ctx": 8192, "temperature": 0.6}``).  Passed as
                the nested ``options`` key in the Ollama request body.
                When *None*, a default of ``{"num_ctx": 8192}`` is used.
            think: Optional flag to enable thinking mode (e.g. for Qwen3).
                Sent as a top-level ``think`` key in the request body.
            format: Optional response format specification (e.g. ``"json"``
                or a JSON schema dict).  Sent as a top-level ``format`` key.
            keep_alive: Optional duration to keep the model loaded (e.g.
                ``"5m"`` or ``300``).  Sent as a top-level ``keep_alive`` key.

        Yields:
            Parsed JSON chunks from the streaming response.

        Raises:
            OllamaConnectionError: On connection failure or timeout.
            OllamaStreamError: If an error is received mid-stream.
            ValueError: If the model name is invalid.
        """
        self._validate_model(model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        # Inference options: use caller-provided options or sensible defaults.
        payload["options"] = options if options is not None else {"num_ctx": 8192}
        if think is not None:
            payload["think"] = think
        if format is not None:
            payload["format"] = format
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        yield from self._stream_request(
            "/api/chat", payload, timeout=_STREAM_TIMEOUT
        )

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        think: bool | None = None,
        format: str | dict[str, Any] | None = None,
        keep_alive: str | int | None = None,
    ) -> dict[str, Any]:
        """Send a non-streaming chat completion request.

        Args:
            model: Ollama model name.
            messages: Conversation messages list.
            tools: Optional list of tool definitions in Ollama format.
            options: Optional dict of inference parameters (e.g.
                ``{"num_ctx": 8192, "temperature": 0.6}``).  Passed as
                the nested ``options`` key in the Ollama request body.
                When *None*, a default of ``{"num_ctx": 8192}`` is used.
            think: Optional flag to enable thinking mode (e.g. for Qwen3).
                Sent as a top-level ``think`` key in the request body.
            format: Optional response format specification (e.g. ``"json"``
                or a JSON schema dict).  Sent as a top-level ``format`` key.
            keep_alive: Optional duration to keep the model loaded (e.g.
                ``"5m"`` or ``300``).  Sent as a top-level ``keep_alive`` key.

        Returns:
            The full response dict from Ollama.

        Raises:
            OllamaConnectionError: On connection failure or timeout.
            OllamaRequestError: On error response.
            ValueError: If the model name is invalid.
        """
        self._validate_model(model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        # Inference options: use caller-provided options or sensible defaults.
        payload["options"] = options if options is not None else {"num_ctx": 8192}
        if think is not None:
            payload["think"] = think
        if format is not None:
            payload["format"] = format
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        return self._request("POST", "/api/chat", data=payload, timeout=_STREAM_TIMEOUT)

    def generate_stream(
        self,
        model: str,
        prompt: str,
        **kwargs: Any,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream a raw text generation from Ollama.

        Sends a ``POST /api/generate`` request with ``stream: true`` and
        yields each NDJSON chunk as a parsed dictionary.  Token content
        appears in ``chunk["response"]`` for most chunks.

        This is useful for single-shot generation without chat context
        (e.g. ideation mode).

        Args:
            model: Ollama model name (e.g. ``qwen3:8b``).
            prompt: The prompt string for generation.
            **kwargs: Additional parameters forwarded to the Ollama API
                (e.g. ``think=True``, ``keep_alive="5m"``).

        Yields:
            Parsed JSON chunks from the streaming response.

        Raises:
            OllamaConnectionError: On connection failure or timeout.
            OllamaStreamError: If an error is received mid-stream.
            ValueError: If the model name is invalid.
        """
        self._validate_model(model)
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            **kwargs,
        }
        yield from self._stream_request(
            "/api/generate", payload, timeout=_STREAM_TIMEOUT
        )

    def embed(
        self,
        model: str,
        input: str | list[str],
    ) -> list[list[float]]:
        """Generate embeddings using Ollama.

        Args:
            model: Embedding model name (e.g. ``all-minilm``).
            input: A single string or list of strings to embed.

        Returns:
            A list of embedding vectors (each a list of floats).

        Raises:
            OllamaConnectionError: On connection failure or timeout.
            OllamaRequestError: On error response.
            ValueError: If the model name is invalid.
        """
        self._validate_model(model)
        payload: dict[str, Any] = {
            "model": model,
            "input": input,
        }
        result = self._request("POST", "/api/embed", data=payload)
        return result.get("embeddings", [])

    def pull_model(
        self,
        model: str,
    ) -> Generator[dict[str, Any], None, None]:
        """Pull (download) a model from the Ollama registry.

        Streams progress updates as NDJSON chunks.  Each chunk typically
        contains ``status`` and may include ``completed`` / ``total`` byte
        counts for download progress.

        Args:
            model: Model name to pull (e.g. ``qwen3:8b``).

        Yields:
            Progress update dicts from the streaming response.

        Raises:
            OllamaConnectionError: On connection failure.
            OllamaStreamError: If an error is received mid-stream.
            ValueError: If the model name is invalid.
        """
        self._validate_model(model)
        payload: dict[str, Any] = {
            "model": model,
            "stream": True,
        }
        # Model pulls can take a very long time; use a generous timeout
        # for the initial connection but rely on the streaming keep-alive
        # for ongoing progress.
        yield from self._stream_request(
            "/api/pull", payload, timeout=_STREAM_TIMEOUT
        )

    def show_model(self, model: str) -> dict[str, Any]:
        """Show detailed information about a model.

        Calls ``POST /api/show`` to retrieve model metadata including
        template, parameters, license, and capabilities.

        Args:
            model: Model name to inspect (e.g. ``qwen3:8b``).

        Returns:
            A dict with model details.  Typically includes keys such as
            ``modelfile``, ``parameters``, ``template``, ``details``,
            and ``capabilities`` (if the model advertises them).

        Raises:
            OllamaConnectionError: On connection failure.
            OllamaRequestError: On error response.
            ValueError: If the model name is invalid.
        """
        self._validate_model(model)
        payload: dict[str, Any] = {"model": model}
        return self._request("POST", "/api/show", data=payload)

    def list_running_models(self) -> list[dict[str, Any]]:
        """List models currently loaded in VRAM.

        Calls ``GET /api/ps`` to retrieve information about models that
        are currently loaded and ready for inference.

        Returns:
            A list of running model info dicts from the ``models`` key.

        Raises:
            OllamaConnectionError: On connection failure.
            OllamaRequestError: On error response.
        """
        result = self._request("GET", "/api/ps")
        return result.get("models", [])

    def delete_model(self, model: str) -> None:
        """Delete a model from Ollama.

        Calls ``DELETE /api/delete`` with a JSON body.  The API returns
        HTTP 200 with no JSON body on success.

        Args:
            model: Model name to delete (e.g. ``phi4-mini``).

        Raises:
            OllamaConnectionError: On connection failure.
            OllamaRequestError: On error response (e.g. model not found).
            ValueError: If the model name is invalid.
        """
        self._validate_model(model)
        payload: dict[str, Any] = {"model": model}
        self._request_no_content("DELETE", "/api/delete", data=payload)

    def copy_model(self, source: str, destination: str) -> None:
        """Copy a model to a new name.

        Calls ``POST /api/copy`` with source and destination names.
        The API returns HTTP 200 with no JSON body on success.

        Args:
            source: Source model name.
            destination: Destination model name.

        Raises:
            OllamaConnectionError: On connection failure.
            OllamaRequestError: On error response (e.g. source not found).
            ValueError: If either model name is invalid.
        """
        self._validate_model(source)
        self._validate_model(destination)
        payload: dict[str, Any] = {
            "source": source,
            "destination": destination,
        }
        self._request_no_content("POST", "/api/copy", data=payload)

    def create_model(
        self,
        name: str,
        from_model: str | None = None,
        modelfile: str | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Create a new model or variant.

        Calls ``POST /api/create`` with streaming progress updates.  At
        least one of *from_model* or *modelfile* must be provided.

        Args:
            name: Name for the new model.
            from_model: Base model to derive from (Ollama API ``from``
                parameter).
            modelfile: Full Modelfile content as a string.

        Yields:
            Progress update dicts from the streaming response.

        Raises:
            OllamaConnectionError: On connection failure.
            OllamaStreamError: If an error is received mid-stream.
            ValueError: If the model name is invalid, or if neither
                *from_model* nor *modelfile* is provided.
        """
        self._validate_model(name)
        if from_model is not None:
            self._validate_model(from_model)

        if from_model is None and modelfile is None:
            raise ValueError(
                "At least one of 'from_model' or 'modelfile' must be provided"
            )

        payload: dict[str, Any] = {
            "model": name,
            "stream": True,
        }
        if from_model is not None:
            payload["from"] = from_model
        if modelfile is not None:
            payload["modelfile"] = modelfile

        yield from self._stream_request(
            "/api/create", payload, timeout=_STREAM_TIMEOUT
        )
