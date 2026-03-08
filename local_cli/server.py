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
import os
import sys
import threading
from typing import Any

from local_cli.config import Config
from local_cli.model_catalog import get_merged_catalog, update_catalog
from local_cli.model_search import search_models
from local_cli.ollama_client import OllamaClient, OllamaConnectionError
from local_cli.providers import LLMProvider, ProviderConnectionError, ProviderRequestError, ProviderStreamError
from local_cli.providers.claude_provider import ClaudeProvider
from local_cli.providers.ollama_provider import OllamaProvider
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
    cwd = os.getcwd()
    return (
        "You are a coding agent — an autonomous AI assistant that completes tasks by "
        "using tools. You operate in an agent loop: think about what to do, use a tool, "
        "observe the result, then decide the next step. Continue until the task is fully done.\n\n"
        f"WORKING DIRECTORY: {cwd}\n"
        "All file paths should be relative to or within this directory unless the user "
        "specifies an absolute path.\n\n"
        "AVAILABLE TOOLS:\n"
        + "\n".join(tool_lines)
        + "\n\nRULES:\n"
        "1. ALWAYS use tools to interact with the filesystem. Never guess file contents.\n"
        "2. Before editing a file, ALWAYS read it first to understand its current state.\n"
        "3. Use glob/grep to find files before reading them.\n"
        "4. When asked to write or modify code, actually do it using write/edit tools. "
        "Do NOT just show code in your response.\n"
        "5. After making changes, verify them (read the file back, run tests if applicable).\n"
        "6. Use bash to run commands (tests, builds, git, etc.) when needed.\n"
        "7. If a task requires multiple steps, execute them one by one. Do not stop halfway.\n"
        "8. Be concise in your explanations. Let tool outputs speak for themselves.\n"
        "9. If you encounter an error, try to fix it rather than just reporting it.\n"
        "10. When creating new files, use the write tool. When modifying existing files, "
        "prefer the edit tool for precise changes.\n"
    )


class JsonLineServer:
    """Stdin/stdout server for desktop integration."""

    def __init__(self) -> None:
        self._config = Config()
        try:
            self._client = OllamaClient(base_url=self._config.ollama_host)
        except ValueError:
            self._client = OllamaClient()

        # Wrap the client in a provider for normalized chat operations.
        # Keep self._client for Ollama-specific ops (pull, delete, catalog, search).
        self._provider: LLMProvider = OllamaProvider(client=self._client)

        self._tools = get_default_tools()
        self._system_prompt = _build_system_prompt(self._tools)
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
        ]
        self._tool_defs = self._provider.format_tools(self._tools)
        self._tool_map = {t.name: t for t in self._tools}
        self._stop_flag = threading.Event()

    def run(self) -> None:
        """Main loop: read stdin lines, dispatch, write responses."""
        # Send ready signal.
        tool_names = [t.name for t in self._tools]
        has_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
        _send({
            "type": "ready",
            "model": self._config.model,
            "tools": tool_names,
            "provider": getattr(self._config, "provider", "ollama"),
            "has_claude": has_claude,
        })

        # Background auto-update check.
        def _bg_update_check() -> None:
            try:
                from local_cli.updater import check_for_updates
                has_updates, message = check_for_updates()
                if has_updates:
                    _send({"type": "update_available", "message": message})
            except Exception:
                pass

        update_thread = threading.Thread(target=_bg_update_check, daemon=True)
        update_thread.start()

        self._chat_thread: threading.Thread | None = None

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

            # Stop can arrive while chat is running in a thread.
            if req_type == "stop":
                self._handle_stop(req_id)
                continue

            # Wait for any running chat to finish before processing
            # non-stop requests (except stop which is handled above).
            if self._chat_thread and self._chat_thread.is_alive():
                self._chat_thread.join()

            try:
                if req_type == "chat":
                    # Run chat in a thread so stdin can still read stop.
                    self._stop_flag.clear()
                    self._chat_thread = threading.Thread(
                        target=self._handle_chat,
                        args=(req_id, req.get("content", "")),
                        daemon=True,
                    )
                    self._chat_thread.start()
                elif req_type == "command":
                    self._handle_command(req_id, req.get("command", ""))
                elif req_type == "models":
                    self._handle_models(req_id)
                elif req_type == "catalog":
                    self._handle_catalog(req_id)
                elif req_type == "pull_model":
                    # Run pull in a background thread so it doesn't block
                    # the stdin loop (allows model switch, catalog, etc.).
                    threading.Thread(
                        target=self._handle_pull_model,
                        args=(req_id, req.get("model", "")),
                        daemon=True,
                    ).start()
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
                elif req_type == "switch_provider":
                    self._handle_switch_provider(req_id, req.get("provider", ""))
                elif req_type == "check_update":
                    self._handle_check_update(req_id)
                elif req_type == "do_update":
                    self._handle_do_update(req_id)
                elif req_type == "set_api_key":
                    self._handle_set_api_key(req_id, req.get("api_key", ""))
                elif req_type == "claude_logout":
                    self._handle_claude_logout(req_id)
                elif req_type == "recommend":
                    self._handle_recommend(req_id)
                elif req_type == "set_cwd":
                    self._handle_set_cwd(req_id, req.get("path", ""))
                else:
                    _send({"id": req_id, "type": "error", "message": f"Unknown type: {req_type}"})
            except Exception as exc:
                _send({"id": req_id, "type": "error", "message": str(exc)})

    def _handle_stop(self, req_id: int) -> None:
        """Signal the current chat to stop."""
        self._stop_flag.set()
        _send({"id": req_id, "type": "stopped"})

    def _handle_chat(self, req_id: int, content: str) -> None:
        if not content.strip():
            _send({"id": req_id, "type": "error", "message": "Empty message"})
            return

        self._stop_flag.clear()
        self._messages.append({"role": "user", "content": content})

        max_iterations = 15
        for _ in range(max_iterations):
            if self._stop_flag.is_set():
                _send({"id": req_id, "type": "done"})
                return

            # Stream response from LLM.
            full_content = ""
            tool_calls = []

            try:
                for chunk in self._provider.chat_stream(
                    model=self._config.model,
                    messages=self._messages,
                    tools=self._tool_defs,
                ):
                    if self._stop_flag.is_set():
                        break

                    msg = chunk.get("message", {})
                    delta = msg.get("content", "")
                    if delta:
                        full_content += delta
                        _send({"id": req_id, "type": "stream", "content": delta})

                    if msg.get("tool_calls"):
                        tool_calls = msg["tool_calls"]

                    if chunk.get("done"):
                        break

            except ProviderRequestError as exc:
                _send({"id": req_id, "type": "error", "message": f"API error: {exc}"})
                return
            except ProviderConnectionError as exc:
                _send({"id": req_id, "type": "error", "message": f"Connection error: {exc}"})
                return
            except ProviderStreamError as exc:
                _send({"id": req_id, "type": "error", "message": f"Stream error: {exc}"})
                return
            except Exception as exc:
                _send({"id": req_id, "type": "error", "message": str(exc)})
                return

            if self._stop_flag.is_set():
                # Save whatever was generated so far.
                if full_content:
                    self._messages.append({"role": "assistant", "content": full_content})
                _send({"id": req_id, "type": "done"})
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
                if self._stop_flag.is_set():
                    _send({"id": req_id, "type": "done"})
                    return

                func = tc.get("function", {})
                tool_name = func.get("name", "")
                tool_args = func.get("arguments", {})
                # Ollama sometimes returns arguments as a JSON string.
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except (json.JSONDecodeError, ValueError):
                        tool_args = {}
                tool_call_id = tc.get("id")

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

                # Include tool_name and tool_call_id in the tool result
                # message (critical for Claude compatibility).
                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": result,
                }
                if tool_call_id:
                    tool_msg["tool_call_id"] = tool_call_id
                self._messages.append(tool_msg)

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
                "provider": self._provider.name,
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

        # Verify the model is installed on Ollama (when using Ollama provider).
        if self._provider.name == "ollama":
            try:
                models = self._client.list_models()
                installed = {m.get("name", "") for m in models}
                # Also check base name (e.g. "qwen3" matches "qwen3:8b").
                installed_bases = {n.split(":")[0] for n in installed}
                if model not in installed and model not in installed_bases:
                    _send({"id": req_id, "type": "error",
                           "message": f"Model '{model}' is not installed. Pull it first."})
                    return
            except OllamaConnectionError:
                pass  # Can't verify, proceed anyway.

        self._config.model = model
        # Reset conversation to avoid sending incompatible tool_calls
        # from the previous model.
        self._messages.clear()
        self._messages.append({"role": "system", "content": self._system_prompt})
        _send({"id": req_id, "type": "model_changed", "model": model})
        _send({"id": req_id, "type": "cleared"})

    def _handle_switch_provider(self, req_id: int, provider: str) -> None:
        """Switch the active LLM provider (ollama or claude)."""
        if provider not in ("ollama", "claude"):
            _send({
                "id": req_id,
                "type": "error",
                "message": f"Unknown provider: {provider}. Must be 'ollama' or 'claude'.",
            })
            return

        if provider == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                _send({
                    "id": req_id,
                    "type": "error",
                    "message": "ANTHROPIC_API_KEY environment variable is not set.",
                })
                return
            self._provider = ClaudeProvider(api_key=api_key)
        else:
            self._provider = OllamaProvider(client=self._client)

        self._tool_defs = self._provider.format_tools(self._tools)
        self._messages.clear()
        self._messages.append({"role": "system", "content": self._system_prompt})

        _send({"id": req_id, "type": "provider_changed", "provider": provider})

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
                    details = m.get("details", {})
                    params = details.get("parameter_size", "")
                    family = details.get("family", "")
                    quant = details.get("quantization_level", "")
                    desc_parts = []
                    if family:
                        desc_parts.append(family)
                    if quant:
                        desc_parts.append(quant)
                    desc = " · ".join(desc_parts) if desc_parts else "Locally installed model."
                    catalog.append({
                        "name": name,
                        "display": name,
                        "category": "Installed",
                        "params": params,
                        "size_gb": round(m.get("size", 0) / (1024**3), 1),
                        "description": desc,
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

    def _handle_check_update(self, req_id: int) -> None:
        """Check if updates are available."""
        from local_cli.updater import check_for_updates

        has_updates, message = check_for_updates()
        _send({
            "id": req_id,
            "type": "update_status",
            "has_updates": has_updates,
            "message": message,
        })

    def _handle_do_update(self, req_id: int) -> None:
        """Perform the update (git pull)."""
        from local_cli.updater import perform_update

        _send({"id": req_id, "type": "updating"})
        success, message = perform_update()
        _send({
            "id": req_id,
            "type": "update_done",
            "success": success,
            "message": message,
        })

    def _handle_set_api_key(self, req_id: int, api_key: str) -> None:
        """Set the Anthropic API key at runtime (from desktop login)."""
        if not api_key:
            _send({"id": req_id, "type": "error", "message": "No API key provided."})
            return
        os.environ["ANTHROPIC_API_KEY"] = api_key
        _send({"id": req_id, "type": "api_key_set", "has_claude": True})

    def _handle_claude_logout(self, req_id: int) -> None:
        """Clear the stored API key and revert to Ollama if active."""
        os.environ.pop("ANTHROPIC_API_KEY", None)
        if self._provider.name == "claude":
            self._provider = OllamaProvider(client=self._client)
            self._tool_defs = self._provider.format_tools(self._tools)
            self._messages.clear()
            self._messages.append({"role": "system", "content": self._system_prompt})
            _send({"id": req_id, "type": "provider_changed", "provider": "ollama"})
        _send({"id": req_id, "type": "api_key_set", "has_claude": False})

    def _handle_recommend(self, req_id: int) -> None:
        """Return model recommendations based on system specs."""
        from local_cli.system_info import get_system_info, recommend_models

        info = get_system_info()
        recommendations = recommend_models(ram_gb=info.get("ram_gb"))

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

        for rec in recommendations:
            rec["installed"] = (
                rec["name"] in installed_names
                or rec["name"].split(":")[0] in installed_names
            )

        _send({
            "id": req_id,
            "type": "recommend",
            "system": info,
            "models": recommendations,
        })

    def _handle_set_cwd(self, req_id: int, path: str) -> None:
        """Change the working directory and update the system prompt."""
        if not path:
            _send({"id": req_id, "type": "error", "message": "No path specified"})
            return
        try:
            os.chdir(path)
            # Rebuild system prompt with new cwd.
            self._system_prompt = _build_system_prompt(self._tools)
            # Update system message in conversation history.
            if self._messages and self._messages[0].get("role") == "system":
                self._messages[0] = {"role": "system", "content": self._system_prompt}
            _send({"id": req_id, "type": "cwd_changed", "path": path})
        except OSError as exc:
            _send({"id": req_id, "type": "error", "message": f"Cannot change directory: {exc}"})

    def _handle_clear(self, req_id: int) -> None:
        self._messages.clear()
        self._messages.append({"role": "system", "content": self._system_prompt})
        _send({"id": req_id, "type": "cleared"})


def run_server() -> None:
    """Entry point for the JSON-line server."""
    server = JsonLineServer()
    server.run()
