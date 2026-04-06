"""llama-server (llama.cpp) provider adapter for local-cli.

Implements the :class:`~local_cli.providers.base.LLMProvider` interface by
communicating with a remote llama-server instance via its OpenAI-compatible
``/v1/chat/completions`` endpoint.  This enables GPU-accelerated inference
on a remote machine (e.g. a server with an RTX 4090 running Bonsai 8B)
while using local-cli on any CPU-only machine.

Unlike the Ollama provider, this provider:
- Does NOT require localhost (designed for remote access via Tailscale etc.)
- Uses the OpenAI chat completions API format (SSE streaming)
- Normalizes responses to the same format used by the Ollama provider
"""

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Generator

from local_cli.providers.base import (
    LLMProvider,
    ProviderConnectionError,
    ProviderRequestError,
    ProviderStreamError,
)


# ---------------------------------------------------------------------------
# Default timeouts (seconds)
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30
_STREAM_TIMEOUT = 600


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class LlamaServerConnectionError(ProviderConnectionError):
    """Raised when the client cannot connect to llama-server."""


class LlamaServerRequestError(ProviderRequestError):
    """Raised when llama-server returns an error response."""


class LlamaServerStreamError(ProviderStreamError):
    """Raised when an error occurs mid-stream."""


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class LlamaServerProvider(LLMProvider):
    """LLM provider backed by a remote llama-server instance.

    Communicates via the OpenAI-compatible ``/v1/chat/completions``
    endpoint that llama-server exposes.

    Args:
        base_url: Base URL for the llama-server API
            (e.g. ``http://100.126.144.110:8090``).
    """

    def __init__(self, base_url: str = "http://localhost:8090") -> None:
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "llama-server"

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
        url = f"{self._base_url}{path}"
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
            raise LlamaServerRequestError(
                f"llama-server API error {exc.code} at {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LlamaServerConnectionError(
                f"Failed to connect to llama-server at {url}: {exc}"
            ) from exc
        except socket.timeout as exc:
            raise LlamaServerConnectionError(
                f"Request to llama-server timed out ({timeout}s): {url}"
            ) from exc

    def _build_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert normalized messages to OpenAI format.

        Handles tool_calls and tool results which use slightly different
        field names between Ollama and OpenAI formats.
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "user")

            # Tool result messages
            if role == "tool":
                converted.append({
                    "role": "tool",
                    "content": str(msg.get("content", "")),
                    "tool_call_id": msg.get("tool_call_id", ""),
                })
                continue

            out: dict[str, Any] = {
                "role": role,
                "content": msg.get("content", ""),
            }

            # Assistant messages with tool_calls
            tool_calls = msg.get("tool_calls")
            if tool_calls and role == "assistant":
                openai_calls = []
                for i, tc in enumerate(tool_calls):
                    func = tc.get("function", {})
                    args = func.get("arguments", {})
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    openai_calls.append({
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": args,
                        },
                    })
                out["tool_calls"] = openai_calls

            converted.append(out)
        return converted

    def _normalize_response(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert OpenAI chat completion response to normalized format."""
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})

        result: dict[str, Any] = {
            "role": msg.get("role", "assistant"),
            "content": msg.get("content", ""),
        }

        # Convert OpenAI tool_calls to normalized format
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            normalized_calls = []
            for tc in tool_calls:
                func = tc.get("function", {})
                args = func.get("arguments", "")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                normalized_calls.append({
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": args,
                    },
                    "id": tc.get("id"),
                })
            result["tool_calls"] = normalized_calls

        return {"message": result}

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._build_messages(messages),
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        else:
            payload["max_tokens"] = 4096
        if tools:
            payload["tools"] = tools

        data = self._request(
            "POST", "/v1/chat/completions", data=payload,
            timeout=_STREAM_TIMEOUT,
        )
        return self._normalize_response(data)

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Generator[dict[str, Any], None, None]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._build_messages(messages),
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        else:
            payload["max_tokens"] = 4096
        if tools:
            payload["tools"] = tools

        url = f"{self._base_url}/v1/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req, timeout=_STREAM_TIMEOUT)
        except urllib.error.HTTPError as exc:
            raise LlamaServerRequestError(
                f"llama-server API error {exc.code}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LlamaServerConnectionError(
                f"Failed to connect to llama-server at {url}: {exc}"
            ) from exc
        except socket.timeout as exc:
            raise LlamaServerConnectionError(
                f"Request timed out ({_STREAM_TIMEOUT}s): {url}"
            ) from exc

        # Accumulate tool_calls across chunks
        accumulated_tool_calls: list[dict[str, Any]] = []
        tool_call_buffers: dict[int, dict[str, Any]] = {}

        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    # Final chunk — emit with tool_calls if any
                    if accumulated_tool_calls:
                        yield {
                            "message": {
                                "content": "",
                                "tool_calls": accumulated_tool_calls,
                            },
                            "done": True,
                        }
                    else:
                        yield {"message": {"content": ""}, "done": True}
                    return

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                finish = choice.get("finish_reason")

                # Accumulate tool_call deltas
                tc_deltas = delta.get("tool_calls")
                if tc_deltas:
                    for tcd in tc_deltas:
                        idx = tcd.get("index", 0)
                        if idx not in tool_call_buffers:
                            tool_call_buffers[idx] = {
                                "id": tcd.get("id", f"call_{idx}"),
                                "function": {"name": "", "arguments": ""},
                            }
                        buf = tool_call_buffers[idx]
                        func_delta = tcd.get("function", {})
                        if "name" in func_delta:
                            buf["function"]["name"] += func_delta["name"]
                        if "arguments" in func_delta:
                            buf["function"]["arguments"] += func_delta["arguments"]

                # Emit content delta
                content = delta.get("content")
                if content:
                    yield {
                        "message": {"content": content},
                        "done": False,
                    }

                if finish == "tool_calls":
                    # Finalize tool_calls
                    for idx in sorted(tool_call_buffers.keys()):
                        buf = tool_call_buffers[idx]
                        args = buf["function"]["arguments"]
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                        accumulated_tool_calls.append({
                            "function": {
                                "name": buf["function"]["name"],
                                "arguments": args,
                            },
                            "id": buf["id"],
                        })

                if finish is not None and finish != "tool_calls":
                    yield {"message": {"content": ""}, "done": True}
                    return
        finally:
            resp.close()

    def list_models(self) -> list[dict[str, Any]]:
        result = self._request("GET", "/v1/models")
        models = result.get("data", [])
        return [{"name": m.get("id", "unknown")} for m in models]

    def get_model_info(self, model: str) -> dict[str, Any]:
        models = self.list_models()
        for m in models:
            if m.get("name") == model:
                return m
        return {"name": model}

    def format_tools(self, tools: list) -> list[dict[str, Any]]:
        """Convert Tool instances to OpenAI function-calling format."""
        result = []
        for tool in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return result
