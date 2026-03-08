"""Argument parsing and interactive REPL for local-cli.

Provides :func:`build_parser` for CLI argument parsing using ``argparse``,
and :func:`run_repl` for the interactive read-eval-print loop.
"""

import argparse
import readline  # noqa: F401 — imported for side-effect (line editing/history)
import sys

from local_cli import __version__
from local_cli.agent import agent_loop
from local_cli.config import Config
from local_cli.git_ops import GitError, GitNotInstalledError, GitOps
from local_cli.ollama_client import OllamaClient, OllamaConnectionError
from local_cli.session import SessionManager
from local_cli.tools.base import Tool

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _build_system_prompt(tools: list[Tool]) -> str:
    """Build the system prompt including tool descriptions.

    Generates a system prompt that describes the assistant's capabilities
    and lists all available tools with their descriptions so the LLM
    knows what it can use.

    Args:
        tools: The list of tool instances available to the agent.

    Returns:
        The full system prompt string.
    """
    import os

    tool_lines = []
    for tool in tools:
        tool_lines.append(f"- {tool.name}: {tool.description}")

    tool_section = "\n".join(tool_lines)
    cwd = os.getcwd()

    return (
        "You are a coding agent — an autonomous AI assistant that completes tasks by "
        "using tools. You operate in an agent loop: think about what to do, use a tool, "
        "observe the result, then decide the next step. Continue until the task is fully done.\n\n"
        f"WORKING DIRECTORY: {cwd}\n"
        "All file paths should be relative to or within this directory unless the user "
        "specifies an absolute path.\n\n"
        "AVAILABLE TOOLS:\n"
        f"{tool_section}\n\n"
        "RULES:\n"
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


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

_SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show this help message.",
    "/exit": "Exit the REPL.",
    "/quit": "Exit the REPL (alias for /exit).",
    "/clear": "Clear conversation history.",
    "/model <name>": "Switch to a different model.",
    "/status": "Show current model, message count, connection status.",
    "/save": "Save the current session.",
    "/models": "Open interactive model selector.",
    "/checkpoint": "Create a git checkpoint (tagged commit).",
    "/rollback [tag]": "Roll back to a checkpoint (latest if no tag given).",
    "/install <model>": "Pull/install a model from Ollama registry.",
    "/uninstall <model>": "Delete a model from Ollama.",
    "/info <model>": "Show model details and capabilities.",
    "/running": "List models currently loaded in VRAM.",
    "/provider [name]": "Switch or show the active LLM provider.",
    "/brain [model]": "Set or show the orchestrator brain model.",
    "/registry": "Show current model-to-task routing registry.",
    "/update": "Check for updates and pull the latest version.",
}


class _ReplContext:
    """Mutable state shared between the REPL loop and slash command handler.

    Attributes:
        config: Application configuration.
        client: The Ollama client instance.
        tools: Available tool instances.
        messages: Conversation message history (mutated in place).
        session_manager: Session persistence manager.
        system_prompt: The system prompt string used to reset on /clear.
        rag_engine: Optional RAG engine for context augmentation.
        rag_topk: Number of RAG results per query.
        git_ops: GitOps instance for checkpoint/rollback commands.
        orchestrator: Optional orchestrator for provider/brain management.
        model_manager: Optional model manager for install/delete operations.
    """

    __slots__ = (
        "config",
        "client",
        "tools",
        "messages",
        "session_manager",
        "system_prompt",
        "rag_engine",
        "rag_topk",
        "git_ops",
        "orchestrator",
        "model_manager",
    )

    def __init__(
        self,
        config: Config,
        client: OllamaClient,
        tools: list[Tool],
        messages: list[dict],
        session_manager: SessionManager,
        system_prompt: str,
        rag_engine: object | None = None,
        rag_topk: int = 5,
        orchestrator: object | None = None,
        model_manager: object | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.tools = tools
        self.messages = messages
        self.session_manager = session_manager
        self.system_prompt = system_prompt
        self.rag_engine = rag_engine
        self.rag_topk = rag_topk
        self.git_ops = GitOps()
        self.orchestrator = orchestrator
        self.model_manager = model_manager


def _handle_slash_command(command: str, ctx: _ReplContext) -> bool:
    """Handle a slash command.

    Args:
        command: The raw user input starting with ``/``.
        ctx: The REPL context containing shared state.

    Returns:
        True if the REPL should continue, False if it should exit.
    """
    stripped = command.strip()
    parts = stripped.split(maxsplit=1)
    cmd = parts[0].lower()

    # -- /exit, /quit -------------------------------------------------------
    if cmd in ("/exit", "/quit"):
        print("Goodbye!")
        return False

    # -- /help --------------------------------------------------------------
    if cmd == "/help":
        print("\nAvailable commands:")
        for name, description in _SLASH_COMMANDS.items():
            print(f"  {name:<20} {description}")
        print()
        return True

    # -- /clear -------------------------------------------------------------
    if cmd == "/clear":
        ctx.messages.clear()
        ctx.messages.append({"role": "system", "content": ctx.system_prompt})
        print("Conversation history cleared.")
        return True

    # -- /model <name> ------------------------------------------------------
    if cmd == "/model":
        if len(parts) < 2 or not parts[1].strip():
            print("Usage: /model <name>")
            return True

        new_model = parts[1].strip()

        # Validate the model exists on the Ollama server.
        try:
            models = ctx.client.list_models()
            model_names = [m.get("name", "") for m in models]
            model_found = any(
                new_model == name or new_model == name.split(":")[0]
                for name in model_names
            )
            if not model_found:
                print(f"Model '{new_model}' not found on Ollama server.")
                if model_names:
                    print(f"Available models: {', '.join(model_names)}")
                return True
        except OllamaConnectionError:
            print("Warning: could not connect to Ollama to validate model.")
            print(f"Switching to '{new_model}' anyway.")

        ctx.config.model = new_model
        print(f"Switched to model: {new_model}")
        return True

    # -- /status ------------------------------------------------------------
    if cmd == "/status":
        # Count user messages (exclude system and tool messages).
        user_msg_count = sum(
            1 for m in ctx.messages if m.get("role") == "user"
        )
        print(f"\nModel: {ctx.config.model}")
        print(f"Messages: {user_msg_count}")

        # Check Ollama connection status.
        try:
            version_info = ctx.client.get_version()
            version = version_info.get("version", "unknown")
            print(f"Ollama: connected (v{version})")
        except OllamaConnectionError:
            print("Ollama: disconnected")

        print()
        return True

    # -- /models ------------------------------------------------------------
    if cmd == "/models":
        try:
            from local_cli.model_selector import select_model_interactive

            result = select_model_interactive(ctx.client, ctx.config.model)
            if result is not None:
                ctx.config.model = result
                print(f"Switched to model: {result}")
        except Exception as exc:
            print(f"Model selection failed: {exc}")
        return True

    # -- /save --------------------------------------------------------------
    if cmd == "/save":
        try:
            session_id = ctx.session_manager.save_session(ctx.messages)
            print(f"Session saved: {session_id}")
        except OSError as exc:
            print(f"Failed to save session: {exc}")
        return True

    # -- /checkpoint --------------------------------------------------------
    if cmd == "/checkpoint":
        # Optional message from the rest of the input.
        checkpoint_msg = parts[1].strip() if len(parts) > 1 else ""
        try:
            if not ctx.git_ops.is_git_repo():
                print("Not a git repository. Cannot create checkpoint.")
                return True
            tag = ctx.git_ops.create_checkpoint(checkpoint_msg)
            print(f"Checkpoint created: {tag}")
        except GitNotInstalledError:
            print("git is not installed. Cannot create checkpoint.")
        except GitError as exc:
            print(f"Checkpoint failed: {exc}")
        return True

    # -- /rollback [tag] ----------------------------------------------------
    if cmd == "/rollback":
        try:
            if not ctx.git_ops.is_git_repo():
                print("Not a git repository. Cannot rollback.")
                return True

            # Determine which tag to roll back to.
            if len(parts) > 1 and parts[1].strip():
                target_tag = parts[1].strip()
            else:
                # Use the most recent checkpoint.
                checkpoints = ctx.git_ops.list_checkpoints()
                if not checkpoints:
                    print("No checkpoints found. Use /checkpoint first.")
                    return True
                target_tag = checkpoints[0]

            ctx.git_ops.rollback_to_checkpoint(target_tag)
            print(f"Rolled back to checkpoint: {target_tag}")
        except GitNotInstalledError:
            print("git is not installed. Cannot rollback.")
        except GitError as exc:
            print(f"Rollback failed: {exc}")
        return True

    # -- /install <model> ---------------------------------------------------
    if cmd == "/install":
        if len(parts) < 2 or not parts[1].strip():
            print("Usage: /install <model>")
            return True

        if ctx.model_manager is None:
            print("Model management not available.")
            return True

        model_name = parts[1].strip()

        def _print_progress(
            status: str, completed: int | None, total: int | None
        ) -> None:
            if completed is not None and total is not None and total > 0:
                pct = completed * 100 // total
                print(f"\r  {status}: {pct}%", end="", flush=True)
            else:
                print(f"\r  {status}", end="", flush=True)

        try:
            print(f"Installing {model_name}...")
            ctx.model_manager.install_model(
                model_name, progress_callback=_print_progress
            )
            print(f"\nModel '{model_name}' installed successfully.")
        except ValueError as exc:
            print(f"Invalid model name: {exc}")
        except Exception as exc:
            print(f"\nInstallation failed: {exc}")
        return True

    # -- /uninstall <model> -------------------------------------------------
    if cmd == "/uninstall":
        if len(parts) < 2 or not parts[1].strip():
            print("Usage: /uninstall <model>")
            return True

        if ctx.model_manager is None:
            print("Model management not available.")
            return True

        model_name = parts[1].strip()
        try:
            ctx.model_manager.delete_model(model_name)
            print(f"Model '{model_name}' deleted.")
        except ValueError as exc:
            print(f"Invalid model name: {exc}")
        except Exception as exc:
            print(f"Deletion failed: {exc}")
        return True

    # -- /info <model> ------------------------------------------------------
    if cmd == "/info":
        if len(parts) < 2 or not parts[1].strip():
            print("Usage: /info <model>")
            return True

        if ctx.model_manager is None:
            print("Model management not available.")
            return True

        model_name = parts[1].strip()
        try:
            info = ctx.model_manager.get_model_info(model_name)
            print(f"\nModel: {model_name}")
            details = info.get("details", {})
            if isinstance(details, dict):
                for key, value in details.items():
                    print(f"  {key}: {value}")
            capabilities = info.get("capabilities")
            if capabilities:
                print(f"  capabilities: {', '.join(capabilities)}")
            license_text = info.get("license")
            if license_text:
                # Show only the first line of the license.
                first_line = license_text.strip().split("\n")[0]
                print(f"  license: {first_line}")
            print()
        except ValueError as exc:
            print(f"Invalid model name: {exc}")
        except Exception as exc:
            print(f"Failed to get model info: {exc}")
        return True

    # -- /running -----------------------------------------------------------
    if cmd == "/running":
        if ctx.model_manager is None:
            print("Model management not available.")
            return True

        try:
            running = ctx.model_manager.list_running()
            if not running:
                print("No models currently loaded in VRAM.")
            else:
                print(f"\nModels loaded in VRAM ({len(running)}):")
                for model_info in running:
                    name = model_info.get("name", "unknown")
                    size = model_info.get("size", 0)
                    size_gb = size / (1024 ** 3) if size else 0
                    print(f"  {name} ({size_gb:.1f} GB)")
                print()
        except Exception as exc:
            print(f"Failed to list running models: {exc}")
        return True

    # -- /provider [name] ---------------------------------------------------
    if cmd == "/provider":
        if ctx.orchestrator is None:
            print("Provider management not available.")
            return True

        if len(parts) < 2 or not parts[1].strip():
            # Show current provider.
            current = ctx.orchestrator.get_active_provider_name()
            print(f"Active provider: {current}")
            return True

        new_provider = parts[1].strip().lower()
        try:
            ctx.orchestrator.switch_provider(new_provider)
            print(f"Switched to provider: {new_provider}")
        except ValueError as exc:
            print(f"Failed to switch provider: {exc}")
        return True

    # -- /brain [model] -----------------------------------------------------
    if cmd == "/brain":
        if ctx.orchestrator is None:
            print("Orchestrator not available.")
            return True

        if len(parts) < 2 or not parts[1].strip():
            # Show current brain model.
            brain = ctx.orchestrator.get_brain_model()
            print(f"Brain model: {brain}")
            return True

        new_brain = parts[1].strip()
        try:
            ctx.orchestrator.set_brain_model(new_brain)
            print(f"Brain model set to: {new_brain}")
        except ValueError as exc:
            print(f"Invalid brain model: {exc}")
        return True

    # -- /registry ----------------------------------------------------------
    if cmd == "/registry":
        if ctx.orchestrator is None:
            print("Orchestrator not available.")
            return True

        registry = ctx.orchestrator.registry
        if registry is None:
            print("No model registry configured.")
            return True

        routes = registry.list_routes()
        if not routes:
            print("Model registry is empty (using defaults).")
            default_provider, default_model = registry.get_default()
            print(f"Default: {default_provider}/{default_model}")
        else:
            print(f"\nModel Registry:")
            default_provider, default_model = registry.get_default()
            print(f"  Default: {default_provider}/{default_model}")
            for task_type, entries in routes.items():
                print(f"  {task_type}:")
                for entry in entries:
                    provider = entry.get("provider", "?")
                    model = entry.get("model", "?")
                    priority = entry.get("priority", "?")
                    print(f"    [{priority}] {provider}/{model}")
            print()
        return True

    # -- /update ------------------------------------------------------------
    if cmd == "/update":
        from local_cli.updater import check_for_updates, perform_update

        print("Checking for updates...")
        has_updates, check_msg = check_for_updates()
        if not has_updates:
            print(check_msg)
            return True

        print(check_msg)
        print("Updating...")
        success, update_msg = perform_update()
        print(update_msg)
        return True

    # -- Unknown command ----------------------------------------------------
    print(f"Unknown command: {stripped}")
    print("Type /help for a list of commands.")
    return True


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        An :class:`argparse.ArgumentParser` configured with all
        supported flags.
    """
    parser = argparse.ArgumentParser(
        prog="local-cli",
        description="Local-first AI coding agent powered by Ollama.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model to use (default: qwen3:8b).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=None,
        help="Enable debug output.",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        default=None,
        help="Enable RAG (retrieval-augmented generation) engine.",
    )
    parser.add_argument(
        "--rag-path",
        type=str,
        default=None,
        help="Directory to index for RAG (default: current directory).",
    )
    parser.add_argument(
        "--rag-topk",
        type=int,
        default=None,
        help="Number of RAG results per query (default: 5).",
    )
    parser.add_argument(
        "--rag-model",
        type=str,
        default=None,
        help="Embedding model for RAG (default: all-minilm).",
    )
    parser.add_argument(
        "--select-model",
        action="store_true",
        default=None,
        help="Interactively select a model from available Ollama models at startup.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Set the LLM provider (ollama or claude).",
    )
    parser.add_argument(
        "--brain-model",
        type=str,
        default=None,
        help="Set the orchestrator brain model.",
    )
    parser.add_argument(
        "--registry-file",
        type=str,
        default=None,
        help="Path to model registry JSON file.",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        default=False,
        help="Run in JSON-line server mode (for desktop GUI).",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        default=False,
        help="Check for updates and pull the latest version.",
    )
    return parser


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


def run_repl(
    config: Config,
    client: OllamaClient,
    tools: list[Tool],
    rag_engine: object | None = None,
    rag_topk: int = 5,
    orchestrator: object | None = None,
    model_manager: object | None = None,
) -> None:
    """Run the interactive REPL loop.

    Reads user input line-by-line, detects slash commands, and forwards
    natural-language prompts to :func:`agent_loop` for LLM processing.

    Uses ``readline`` for line editing and input history (automatically
    available via the import at module level).

    Args:
        config: Application configuration.
        client: An :class:`OllamaClient` instance.
        tools: A list of :class:`Tool` instances available to the agent.
        rag_engine: Optional :class:`RAGEngine` for context augmentation.
        rag_topk: Number of RAG results per query.
        orchestrator: Optional :class:`Orchestrator` for provider/brain
            management and task routing.
        model_manager: Optional :class:`ModelManager` for model
            install/delete operations.
    """
    # Print welcome banner.
    tool_names = ", ".join(t.name for t in tools)
    print(f"local-cli v{__version__} | model: {config.model}")
    print(f"Tools: {tool_names}")
    if rag_engine is not None:
        print("RAG: enabled")
    if orchestrator is not None:
        print(f"Provider: {orchestrator.get_active_provider_name()}")
    print("Type /help for commands, /exit to quit.\n")

    # Build system prompt with tool descriptions.
    system_prompt = _build_system_prompt(tools)

    # Conversation history (persists across the session).
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
    ]

    # Session manager for /save command.
    session_manager = SessionManager(config.state_dir)

    # Build the REPL context for slash commands.
    ctx = _ReplContext(
        config=config,
        client=client,
        tools=tools,
        messages=messages,
        session_manager=session_manager,
        system_prompt=system_prompt,
        rag_engine=rag_engine,
        rag_topk=rag_topk,
        orchestrator=orchestrator,
        model_manager=model_manager,
    )

    while True:
        # Read user input.
        try:
            user_input = input("You> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        # Skip empty input.
        stripped = user_input.strip()
        if not stripped:
            continue

        # Handle slash commands.
        if stripped.startswith("/"):
            should_continue = _handle_slash_command(stripped, ctx)
            if not should_continue:
                break
            continue

        # Augment prompt with RAG context if available.
        prompt_content = stripped
        if ctx.rag_engine is not None:
            try:
                prompt_content = ctx.rag_engine.augment_prompt(
                    stripped, top_k=ctx.rag_topk,
                )
            except Exception:
                # RAG failure is non-fatal; fall back to the raw prompt.
                pass

        # Build user message and add to history.
        messages.append({"role": "user", "content": prompt_content})

        # Run the agent loop (streams response to stdout).
        try:
            agent_loop(
                client=client,
                model=config.model,
                tools=tools,
                messages=messages,
                debug=config.debug,
            )
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as exc:
            sys.stderr.write(f"Error: {exc}\n")
