"""Stdin/stdout JSON-line server for desktop GUI integration.

Reads newline-delimited JSON requests from stdin and writes
newline-delimited JSON responses to stdout. This avoids any
network dependency and lets Electron spawn the Python process
directly.

Protocol
--------
Request (one JSON object per line on stdin)::

    {"id": 1, "type": "chat", "content": "explain this code"}
    {"id": 2, "type": "command", "command": "/status"}
    {"id": 3, "type": "models"}

Response (newline-delimited JSON on stdout)::

    {"id": 1, "type": "stream", "content": "The"}
    {"id": 1, "type": "stream", "content": " code"}
    {"id": 1, "type": "tool_call", "name": "read", "args": {"file_path": "x"}}
    {"id": 1, "type": "tool_result", "name": "read", "output": "..."}
    {"id": 1, "type": "done"}
    {"id": 2, "type": "status", "data": {...}}
    {"id": 3, "type": "models", "data": [...]}
"""

import json
import sys
import threading
from typing import Any

from local_cli.config import Config
from local_cli.model_catalog import get_merged_catalog, update_catalog
from local_cli.model_search import search_models
from local_cli.ollama_client import OllamaClient, OllamaConnectionError
from local_cli.security import validate_model_name
from local_cli.tools import get_default_tools
from local_cli.tools.base import Tool


def _send(obj: dict[str, Any]) -> None:
    """Write a JSON line to stdout (thread-safe)."""
    line = json.dumps(obj, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _build_system_prompt(tools: list[Tool]) -> str:
    tool_lines = [f"- {t.name}: {t.description}" for t in tools]
    return (
        "You are a helpful AI coding assistant running locally via Ollama. "
        "You have access to the following tools:\n\n"
        + "\n".join(tool_lines)
        + "\n\nUse these tools to help the user with their coding tasks. "
        "Be concise and accurate."
    )


class JsonLineServer:
    """Stdin/stdout server for desktop integration."""

    def __init__(self) -> None:
        self._config = Config()
        try:
            self._client = OllamaClient(base_url=self._config.ollama_host)
        except ValueError:
            self._client = OllamaClient()

        self._tools = get_default_tools()
        self._system_prompt = _build_system_prompt(self._tools)
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
        ]
        self._tool_defs = [t.to_ollama_tool() for t in self._tools]
        self._tool_map = {t.name: t for t in self._tools}

    def run(self) -> None:
        """Main loop: read stdin lines, dispatch, write responses."""
        # Send ready signal.
        tool_names = [t.name for t in self._tools]
        _send({
            "type": "ready",
            "model": self._config.model,
            "tools": tool_names,
            "provider": getattr(self._config, "provider", "ollama"),
        })

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                _send({"type": "error", "message": "Invalid JSON"})
                continue

            req_id = req.get("id", 0)
            req_type = req.get("type", "")

            try:
                if req_type == "chat":
                    self._handle_chat(req_id, req.get("content", ""))
                elif req_type == "command":
                    self._handle_command(req_id, req.get("command", ""))
                elif req_type == "models":
                    self._handle_models(req_id)
                elif req_type == "catalog":
                    self._handle_catalog(req_id)
                elif req_type == "pull_model":
                    self._handle_pull_model(req_id, req.get("model", ""))
                elif req_type == "delete_model":
                    self._handle_delete_model(req_id, req.get("model", ""))
                elif req_type == "update_catalog":
                    self._handle_update_catalog(req_id)
                elif req_type == "search_models":
                    self._handle_search_models(
                        req_id,
                        req.get("query", ""),
                        req.get("sort", "popular"),
                        req.get("capability", ""),
                    )
                elif req_type == "status":
                    self._handle_status(req_id)
                elif req_type == "switch_model":
                    self._handle_switch_model(req_id, req.get("model", ""))
                elif req_type == "clear":
                    self._handle_clear(req_id)
                else:
                    _send({"id": req_id, "type": "error", "message": f"Unknown type: {req_type}"})
            except Exception as exc:
                _send({"id": req_id, "type": "error", "message": str(exc)})

    def _handle_chat(self, req_id: int, content: str) -> None:
        if not content.strip():
            _send({"id": req_id, "type": "error", "message": "Empty message"})
            return

        self._messages.append({"role": "user", "content": content})

        max_iterations = 15
        for _ in range(max_iterations):
            # Stream response from LLM.
            full_content = ""
            tool_calls = []

            try:
                for chunk in self._client.chat_stream(
                    model=self._config.model,
                    messages=self._messages,
                    tools=self._tool_defs,
                ):
                    msg = chunk.get("message", {})
                    delta = msg.get("content", "")
                    if delta:
                        full_content += delta
                        _send({"id": req_id, "type": "stream", "content": delta})

                    if msg.get("tool_calls"):
                        tool_calls = msg["tool_calls"]

                    if chunk.get("done"):
                        break

            except OllamaConnectionError as exc:
                _send({"id": req_id, "type": "error", "message": f"Ollama connection error: {exc}"})
                return
            except Exception as exc:
                _send({"id": req_id, "type": "error", "message": str(exc)})
                return

            # Append assistant message to history.
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": full_content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            self._messages.append(assistant_msg)

            # If no tool calls, we're done.
            if not tool_calls:
                _send({"id": req_id, "type": "done"})
                return

            # Execute tool calls.
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                tool_args = func.get("arguments", {})

                _send({
                    "id": req_id,
                    "type": "tool_call",
                    "name": tool_name,
                    "args": tool_args,
                })

                tool = self._tool_map.get(tool_name)
                if tool is None:
                    result = f"Error: unknown tool '{tool_name}'"
                else:
                    try:
                        result = tool.execute(**tool_args)
                    except Exception as exc:
                        result = f"Error: {exc}"

                _send({
                    "id": req_id,
                    "type": "tool_result",
                    "name": tool_name,
                    "output": result if len(result) <= 10000 else result[:10000] + "\n...(truncated)",
                })

                self._messages.append({
                    "role": "tool",
                    "content": result,
                })

            # Loop back to let LLM process tool results.

        _send({"id": req_id, "type": "done"})

    def _handle_command(self, req_id: int, command: str) -> None:
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""

        if cmd == "/clear":
            self._handle_clear(req_id)
        elif cmd == "/model":
            if len(parts) > 1:
                self._handle_switch_model(req_id, parts[1].strip())
            else:
                _send({"id": req_id, "type": "status", "data": {"model": self._config.model}})
        elif cmd == "/status":
            self._handle_status(req_id)
        elif cmd == "/models":
            self._handle_models(req_id)
        else:
            _send({"id": req_id, "type": "error", "message": f"Unknown command: {command}"})

    def _handle_models(self, req_id: int) -> None:
        try:
            models = self._client.list_models()
            model_list = []
            for m in models:
                name = m.get("name", "")
                size = m.get("size", 0)
                model_list.append({"name": name, "size": size})
            _send({"id": req_id, "type": "models", "data": model_list})
        except OllamaConnectionError:
            _send({"id": req_id, "type": "error", "message": "Cannot connect to Ollama"})

    def _handle_status(self, req_id: int) -> None:
        user_msgs = sum(1 for m in self._messages if m.get("role") == "user")
        connected = False
        version = ""
        try:
            info = self._client.get_version()
            connected = True
            version = info.get("version", "")
        except OllamaConnectionError:
            pass

        _send({
            "id": req_id,
            "type": "status",
            "data": {
                "model": self._config.model,
                "provider": getattr(self._config, "provider", "ollama"),
                "messages": user_msgs,
                "connected": connected,
                "ollama_version": version,
            },
        })

    def _handle_switch_model(self, req_id: int, model: str) -> None:
        if not model:
            _send({"id": req_id, "type": "error", "message": "No model specified"})
            return

        if not validate_model_name(model):
            _send({"id": req_id, "type": "error", "message": f"Invalid model name: {model}"})
            return

        self._config.model = model
        _send({"id": req_id, "type": "model_changed", "model": model})

    def _handle_catalog(self, req_id: int) -> None:
        """Return merged model catalog (built-in + cache) + installed status."""
        catalog, categories = get_merged_catalog()

        # Get installed models to mark which are available.
        installed_names: set[str] = set()
        try:
            models = self._client.list_models()
            for m in models:
                name = m.get("name", "")
                installed_names.add(name)
                base = name.split(":")[0]
                if base:
                    installed_names.add(base)
        except OllamaConnectionError:
            pass

        # Mark installed status on catalog entries.
        for entry in catalog:
            entry["installed"] = (
                entry["name"] in installed_names
                or entry["name"].split(":")[0] in installed_names
            )

        # Include locally installed models not in the catalog.
        catalog_names = {e["name"] for e in catalog}
        try:
            models = self._client.list_models()
            for m in models:
                name = m.get("name", "")
                if name and name not in catalog_names:
                    catalog.append({
                        "name": name,
                        "display": name,
                        "category": "Installed",
                        "params": "",
                        "size_gb": round(m.get("size", 0) / (1024**3), 1),
                        "description": "Locally installed model.",
                        "tags": [],
                        "installed": True,
                    })
        except OllamaConnectionError:
            pass

        if any(e.get("category") == "Installed" for e in catalog):
            categories = ["Installed"] + [c for c in categories if c != "Installed"]

        _send({
            "id": req_id,
            "type": "catalog",
            "data": {"categories": categories, "models": catalog},
        })

    def _handle_update_catalog(self, req_id: int) -> None:
        """Fetch latest models from ollama.com and update local cache."""
        _send({"id": req_id, "type": "catalog_updating"})

        try:
            result = update_catalog()
            _send({
                "id": req_id,
                "type": "catalog_updated",
                "data": result,
            })
        except Exception as exc:
            _send({"id": req_id, "type": "error", "message": f"Catalog update failed: {exc}"})

    def _handle_pull_model(self, req_id: int, model: str) -> None:
        """Pull/download a model with streaming progress."""
        if not model:
            _send({"id": req_id, "type": "error", "message": "No model specified"})
            return
        if not validate_model_name(model):
            _send({"id": req_id, "type": "error", "message": f"Invalid model name: {model}"})
            return

        _send({"id": req_id, "type": "pull_start", "model": model})

        try:
            for chunk in self._client.pull_model(model):
                status = chunk.get("status", "")
                completed = chunk.get("completed")
                total = chunk.get("total")
                _send({
                    "id": req_id,
                    "type": "pull_progress",
                    "model": model,
                    "status": status,
                    "completed": completed,
                    "total": total,
                })
        except OllamaConnectionError as exc:
            _send({"id": req_id, "type": "error", "message": f"Connection error: {exc}"})
            return
        except Exception as exc:
            _send({"id": req_id, "type": "error", "message": f"Pull failed: {exc}"})
            return

        _send({"id": req_id, "type": "pull_done", "model": model})

    def _handle_delete_model(self, req_id: int, model: str) -> None:
        """Delete an installed model."""
        if not model:
            _send({"id": req_id, "type": "error", "message": "No model specified"})
            return
        if not validate_model_name(model):
            _send({"id": req_id, "type": "error", "message": f"Invalid model name: {model}"})
            return

        try:
            self._client.delete_model(model)
            _send({"id": req_id, "type": "delete_done", "model": model})
        except OllamaConnectionError as exc:
            _send({"id": req_id, "type": "error", "message": f"Connection error: {exc}"})
        except Exception as exc:
            _send({"id": req_id, "type": "error", "message": f"Delete failed: {exc}"})

    def _handle_search_models(
        self, req_id: int, query: str, sort: str, capability: str,
    ) -> None:
        """Search Ollama library for models."""
        try:
            results = search_models(query=query, sort=sort, capability=capability)

            # Mark installed status.
            installed_names: set[str] = set()
            try:
                models = self._client.list_models()
                for m in models:
                    name = m.get("name", "")
                    installed_names.add(name)
                    installed_names.add(name.split(":")[0])
            except OllamaConnectionError:
                pass

            for r in results:
                r["installed"] = (
                    r["name"] in installed_names
                    or r["name"].split(":")[0] in installed_names
                )

            _send({"id": req_id, "type": "search_results", "data": results})
        except Exception as exc:
            _send({"id": req_id, "type": "error", "message": f"Search failed: {exc}"})

    def _handle_clear(self, req_id: int) -> None:
        self._messages.clear()
        self._messages.append({"role": "system", "content": self._system_prompt})
        _send({"id": req_id, "type": "cleared"})


def run_server() -> None:
    """Entry point for the JSON-line server."""
    server = JsonLineServer()
    server.run()
