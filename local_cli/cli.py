"""Argument parsing and interactive REPL for local-cli.

Provides :func:`build_parser` for CLI argument parsing using ``argparse``,
and :func:`run_repl` for the interactive read-eval-print loop.
"""

import argparse
import readline  # noqa: F401 — imported for side-effect (line editing/history)
import sys

from local_cli import __version__
from local_cli.agent import (
    _COMPACT_MESSAGE_THRESHOLD,
    _COMPACT_TOKEN_THRESHOLD,
    _estimate_tokens,
    _is_complex_request,
    _needs_compaction,
    agent_loop,
    build_plan_context,
    ideation_loop,
)
from local_cli.clipboard import (
    ClipboardError,
    ClipboardUnavailableError,
    copy_to_clipboard,
)
from local_cli.config import Config
from local_cli.git_ops import GitError, GitNotInstalledError, GitOps
from local_cli.ideation import IdeationEngine
from local_cli.knowledge import KnowledgeError, KnowledgeNotFoundError, KnowledgeStore
from local_cli.model_presets import SUPPORTS_THINKING, get_model_family, get_model_preset
from local_cli.ollama_client import OllamaClient, OllamaConnectionError
from local_cli.plan_manager import PlanError, PlanManager, PlanNotFoundError
from local_cli.session import SessionManager
from local_cli.skills import SkillsLoader
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
        "THINKING PROCESS:\n"
        "Before taking action, think through these steps:\n"
        "1. What is the goal? Restate the task in your own words.\n"
        "2. What information do I need? Identify files, context, or state to gather.\n"
        "3. What tool should I use? Pick the most appropriate tool for this step.\n"
        "4. What could go wrong? Anticipate errors and plan fallbacks.\n"
        "Work step by step. Do not try to do everything in one tool call.\n\n"
        "TOOL USAGE PATTERNS:\n"
        "- Find then read: Use glob to locate files, then read the matches.\n"
        "- Read then edit: Always read a file before editing it.\n"
        "- Search then act: Use grep to find relevant code, then read surrounding context.\n"
        "- Edit then verify: After editing, read the file back or run tests with bash.\n"
        "- Write then test: After writing new code, run it with bash to check for errors.\n\n"
        "ERROR RECOVERY:\n"
        "If a tool returns an error, do NOT give up. Instead:\n"
        "1. Read the error message carefully — it usually tells you what went wrong.\n"
        "2. Adjust your approach (fix the path, correct the syntax, try a different tool).\n"
        "3. Retry. If it fails again, try an alternative strategy.\n\n"
        "OUTPUT FORMAT:\n"
        "- Be concise. Show what you did and the result.\n"
        "- Don't repeat file contents unless the user asks.\n"
        "- Let tool outputs speak for themselves.\n"
        "- Summarize changes at the end of multi-step tasks.\n\n"
        "RULES:\n"
        "1. ALWAYS use tools to interact with the filesystem. Never guess file contents.\n"
        "2. Before editing a file, ALWAYS read it first to understand its current state.\n"
        "3. Use glob/grep to find files before reading them.\n"
        "4. When asked to write or modify code, actually do it using write/edit tools. "
        "Do NOT just show code in your response.\n"
        "5. After making changes, verify them (read the file back, run tests if applicable).\n"
        "6. Use bash to run commands (tests, builds, git, etc.) when needed.\n"
        "7. If a task requires multiple steps, execute them one by one. Do not stop halfway.\n"
        "8. If you encounter an error, try to fix it rather than just reporting it.\n"
        "9. When creating new files, use the write tool. When modifying existing files, "
        "prefer the edit tool for precise changes.\n"
        "10. When the user asks about the system, environment, files, or anything that "
        "can be answered by running a command or reading a file, ALWAYS use a tool "
        "(bash, read, glob, grep) to get the real answer. NEVER guess or say "
        "'I cannot access your system'. You ARE running on their system.\n"
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
    "/undo": "Undo the most recent file modifications (git checkout).",
    "/diff": "Show uncommitted changes in the working tree.",
    "/context": "Show context window usage (messages, tokens, compaction).",
    "/copy": "Copy last assistant response to clipboard.",
    "/usage": "Show per-message token usage and session totals.",
    "/agents": "List background sub-agent status.",
    "/plan": "Show, create, or update plans.",
    "/ideate": "Enter ideation (brainstorming) mode.",
    "/knowledge": "Save, load, or list knowledge items.",
    "/skills": "List or show discovered skills.",
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
        token_tracker: Optional token usage tracker for /usage command.
        tool_cache: Optional tool result cache for read/glob/grep caching.
        sub_agent_runner: Optional SubAgentRunner for background agent
            status queries.
        plan_manager: Optional plan manager for plan CRUD operations.
        knowledge_store: Optional knowledge store for persistent knowledge.
        skills_loader: Optional skills loader for auto-discovered skills.
        ideation_engine: Optional ideation engine for brainstorming mode.
        active_plan_id: ID of the currently active plan (or None).
        current_mode: Current REPL mode ('agent' or 'ideate').
        ideation_messages: Separate message history for ideation mode.
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
        "token_tracker",
        "tool_cache",
        "sub_agent_runner",
        "plan_manager",
        "knowledge_store",
        "skills_loader",
        "ideation_engine",
        "active_plan_id",
        "current_mode",
        "ideation_messages",
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
        token_tracker: object | None = None,
        tool_cache: object | None = None,
        sub_agent_runner: object | None = None,
        plan_manager: PlanManager | None = None,
        knowledge_store: KnowledgeStore | None = None,
        skills_loader: SkillsLoader | None = None,
        ideation_engine: IdeationEngine | None = None,
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
        self.token_tracker = token_tracker
        self.tool_cache = tool_cache
        self.sub_agent_runner = sub_agent_runner
        self.plan_manager = plan_manager
        self.knowledge_store = knowledge_store
        self.skills_loader = skills_loader
        self.ideation_engine = ideation_engine
        self.active_plan_id: str | None = None
        self.current_mode: str = "agent"
        self.ideation_messages: list[dict] = []


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
        print(f"Mode: {ctx.current_mode}")

        # Show active plan if any.
        if ctx.active_plan_id is not None:
            print(f"Active plan: {ctx.active_plan_id}")

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

    # -- /undo --------------------------------------------------------------
    if cmd == "/undo":
        try:
            if not ctx.git_ops.is_git_repo():
                print("Not a git repository. Cannot undo.")
                return True
            result = ctx.git_ops.undo_last_change()
            print(result)
        except GitNotInstalledError:
            print("git is not installed. Cannot undo.")
        except GitError as exc:
            print(f"Undo failed: {exc}")
        return True

    # -- /diff --------------------------------------------------------------
    if cmd == "/diff":
        try:
            if not ctx.git_ops.is_git_repo():
                print("Not a git repository. Cannot show diff.")
                return True
            result = ctx.git_ops.diff_working_tree()
            print(result)
        except GitNotInstalledError:
            print("git is not installed. Cannot show diff.")
        except GitError as exc:
            print(f"Diff failed: {exc}")
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

    # -- /context -----------------------------------------------------------
    if cmd == "/context":
        msg_count = len(ctx.messages)
        est_tokens = _estimate_tokens(ctx.messages)
        token_limit = _COMPACT_TOKEN_THRESHOLD
        compaction_triggered = _needs_compaction(ctx.messages)
        compaction_status = "triggered" if compaction_triggered else "not triggered"
        print(
            f"Messages: {msg_count} | "
            f"Est. tokens: ~{est_tokens} / {token_limit} | "
            f"Compaction: {compaction_status}"
        )
        return True

    # -- /copy --------------------------------------------------------------
    if cmd == "/copy":
        # Find the last assistant message in the conversation history.
        last_assistant = None
        for msg in reversed(ctx.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if content:
                    last_assistant = content
                    break

        if last_assistant is None:
            print("Nothing to copy.")
            return True

        try:
            copy_to_clipboard(last_assistant)
            print("Copied to clipboard.")
        except ClipboardUnavailableError:
            print("Clipboard not available.")
        except ClipboardError as exc:
            print(f"Copy failed: {exc}")
        return True

    # -- /usage -------------------------------------------------------------
    if cmd == "/usage":
        if ctx.token_tracker is None:
            print("Token tracking not available.")
            return True

        print(ctx.token_tracker.format_table())
        return True

    # -- /agents ------------------------------------------------------------
    if cmd == "/agents":
        if ctx.sub_agent_runner is None:
            print("Sub-agent support not available.")
            return True

        agents = ctx.sub_agent_runner.list_background_agents()
        if not agents:
            print("No background agents.")
        else:
            print(f"\nBackground agents ({len(agents)}):")
            for info in agents:
                agent_id = info.get("agent_id", "?")
                status = info.get("status", "?")
                print(f"  {agent_id}  {status}")
            print()
        return True

    # -- /plan [subcommand] -------------------------------------------------
    if cmd == "/plan":
        return _handle_plan_command(parts, ctx)

    # -- /ideate [subcommand] -----------------------------------------------
    if cmd == "/ideate":
        return _handle_ideate_command(parts, ctx)

    # -- /knowledge [subcommand] --------------------------------------------
    if cmd == "/knowledge":
        return _handle_knowledge_command(parts, ctx)

    # -- /skills [subcommand] -----------------------------------------------
    if cmd == "/skills":
        return _handle_skills_command(parts, ctx)

    # -- Unknown command ----------------------------------------------------
    print(f"Unknown command: {stripped}")
    print("Type /help for a list of commands.")
    return True


# ---------------------------------------------------------------------------
# Plan command handler
# ---------------------------------------------------------------------------


def _handle_plan_command(parts: list[str], ctx: _ReplContext) -> bool:
    """Handle /plan slash command and its subcommands.

    Subcommands:
        - ``/plan`` or ``/plan list`` — list all plans.
        - ``/plan create <title>`` — create a new plan.
        - ``/plan show <id>`` — show a plan's details.
        - ``/plan activate <id>`` — set a plan as the active plan.
        - ``/plan update <id> <step> done|undone`` — mark a step.
        - ``/plan review <id>`` — request an LLM review of a plan.
        - ``/plan abandon <id>`` — abandon a plan.

    Args:
        parts: The split command parts (``["/plan", ...]``).
        ctx: The REPL context containing shared state.

    Returns:
        True to continue the REPL.
    """
    if ctx.plan_manager is None:
        print("Plan management not available.")
        return True

    # No subcommand or "list" → list plans.
    if len(parts) < 2 or not parts[1].strip():
        return _plan_list(ctx)

    sub_parts = parts[1].strip().split(maxsplit=1)
    subcmd = sub_parts[0].lower()
    sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""

    if subcmd == "list":
        return _plan_list(ctx)

    if subcmd == "create":
        if not sub_arg:
            print("Usage: /plan create <title>")
            return True
        try:
            plan = ctx.plan_manager.create_plan(
                title=sub_arg,
                model=ctx.config.model,
            )
            print(f"Plan {plan.plan_id} created: {plan.title}")
        except PlanError as exc:
            print(f"Failed to create plan: {exc}")
        return True

    if subcmd == "show":
        if not sub_arg:
            print("Usage: /plan show <id>")
            return True
        try:
            plan = ctx.plan_manager.show_plan(sub_arg)
            _print_plan(plan)
        except PlanNotFoundError:
            print(f"Plan '{sub_arg}' not found.")
        except PlanError as exc:
            print(f"Failed to show plan: {exc}")
        return True

    if subcmd == "activate":
        if not sub_arg:
            print("Usage: /plan activate <id>")
            return True
        try:
            plan = ctx.plan_manager.activate_plan(sub_arg)
            ctx.active_plan_id = plan.plan_id
            print(f"Plan {plan.plan_id} activated: {plan.title}")
        except PlanNotFoundError:
            print(f"Plan '{sub_arg}' not found.")
        except PlanError as exc:
            print(f"Failed to activate plan: {exc}")
        return True

    if subcmd == "update":
        # Expected format: /plan update <id> <step> done|undone
        update_parts = sub_arg.split(maxsplit=2)
        if len(update_parts) < 3:
            print("Usage: /plan update <id> <step> done|undone")
            return True
        plan_id = update_parts[0]
        try:
            step_num = int(update_parts[1])
        except ValueError:
            print("Step number must be an integer.")
            return True
        done_str = update_parts[2].lower()
        if done_str not in ("done", "undone"):
            print("Status must be 'done' or 'undone'.")
            return True
        done = done_str == "done"
        try:
            plan = ctx.plan_manager.update_step(plan_id, step_num, done)
            mark = "done" if done else "undone"
            print(f"Step {step_num} marked as {mark}.")
            if plan.status == "complete":
                print(f"Plan {plan.plan_id} is now complete!")
        except PlanNotFoundError:
            print(f"Plan '{plan_id}' not found.")
        except PlanError as exc:
            print(f"Failed to update step: {exc}")
        return True

    if subcmd == "review":
        if not sub_arg:
            print("Usage: /plan review <id>")
            return True
        try:
            content = ctx.plan_manager.get_plan_content(sub_arg)
        except PlanNotFoundError:
            print(f"Plan '{sub_arg}' not found.")
            return True
        except PlanError as exc:
            print(f"Failed to read plan: {exc}")
            return True

        # Send plan content to LLM for review via ideation-style loop.
        review_prompt = (
            "Please review and critique the following plan. "
            "Identify risks, suggest improvements, and assess feasibility.\n\n"
            f"{content}"
        )
        review_messages: list[dict] = [
            {"role": "system", "content": ctx.system_prompt},
            {"role": "user", "content": review_prompt},
        ]
        try:
            ideation_loop(
                client=ctx.client,
                model=ctx.config.model,
                messages=review_messages,
                think=True,
            )
        except KeyboardInterrupt:
            print("\nReview interrupted.")
        return True

    if subcmd == "abandon":
        if not sub_arg:
            print("Usage: /plan abandon <id>")
            return True
        try:
            plan = ctx.plan_manager.abandon_plan(sub_arg)
            print(f"Plan {plan.plan_id} abandoned.")
            if ctx.active_plan_id == plan.plan_id:
                ctx.active_plan_id = None
        except PlanNotFoundError:
            print(f"Plan '{sub_arg}' not found.")
        except PlanError as exc:
            print(f"Failed to abandon plan: {exc}")
        return True

    print(f"Unknown plan subcommand: {subcmd}")
    print("Usage: /plan [list|create|show|activate|update|review|abandon]")
    return True


def _plan_list(ctx: _ReplContext) -> bool:
    """List all plans with their status.

    Args:
        ctx: The REPL context.

    Returns:
        True to continue the REPL.
    """
    try:
        plans = ctx.plan_manager.list_plans()
    except PlanError as exc:
        print(f"Failed to list plans: {exc}")
        return True

    if not plans:
        print("No plans found.")
        return True

    print("\nPlans:")
    for plan in plans:
        active_marker = " *" if plan.plan_id == ctx.active_plan_id else ""
        done_count = sum(1 for done, _ in plan.steps if done)
        total = len(plan.steps)
        progress = f"[{done_count}/{total}]" if total > 0 else ""
        print(
            f"  {plan.plan_id}  {plan.status:<10}  "
            f"{plan.title}  {progress}{active_marker}"
        )
    print()
    return True


def _print_plan(plan: "object") -> None:
    """Pretty-print a plan to stdout.

    Args:
        plan: A :class:`Plan` instance.
    """
    print(f"\n# Plan {plan.plan_id}: {plan.title}")
    print(f"  Status:  {plan.status}")
    print(f"  Created: {plan.created}")
    if plan.model:
        print(f"  Model:   {plan.model}")
    if plan.description:
        print(f"\n  {plan.description}")
    if plan.steps:
        print("\n  Steps:")
        for i, (done, text) in enumerate(plan.steps, 1):
            checkbox = "[x]" if done else "[ ]"
            print(f"    {i}. {checkbox} {text}")
    if plan.notes:
        print(f"\n  Notes: {plan.notes}")
    print()


# ---------------------------------------------------------------------------
# Ideate command handler
# ---------------------------------------------------------------------------


def _handle_ideate_command(parts: list[str], ctx: _ReplContext) -> bool:
    """Handle /ideate slash command and its subcommands.

    Subcommands:
        - ``/ideate`` — enter ideation (brainstorming) mode.
        - ``/ideate exit`` — return to normal agent mode.
        - ``/ideate clear`` — clear ideation history.
        - ``/ideate once <prompt>`` — single-shot ideation via /api/generate.

    Args:
        parts: The split command parts (``["/ideate", ...]``).
        ctx: The REPL context containing shared state.

    Returns:
        True to continue the REPL.
    """
    if ctx.ideation_engine is None:
        print("Ideation engine not available.")
        return True

    # No subcommand → enter ideation mode.
    if len(parts) < 2 or not parts[1].strip():
        ctx.current_mode = "ideate"
        if not ctx.ideation_engine.has_session:
            ctx.ideation_engine.start_session()
        print("Entered ideation mode. Type /ideate exit to return.")
        return True

    sub_parts = parts[1].strip().split(maxsplit=1)
    subcmd = sub_parts[0].lower()
    sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""

    if subcmd == "exit":
        ctx.current_mode = "agent"
        print("Returned to agent mode.")
        return True

    if subcmd == "clear":
        ctx.ideation_engine.clear_history()
        print("Ideation history cleared.")
        return True

    if subcmd == "once":
        if not sub_arg:
            print("Usage: /ideate once <prompt>")
            return True
        try:
            ctx.ideation_engine.single_shot(
                prompt=sub_arg,
                model=ctx.config.model,
            )
        except Exception as exc:
            print(f"Ideation failed: {exc}")
        return True

    print(f"Unknown ideate subcommand: {subcmd}")
    print("Usage: /ideate [exit|clear|once <prompt>]")
    return True


# ---------------------------------------------------------------------------
# Knowledge command handler
# ---------------------------------------------------------------------------


def _handle_knowledge_command(parts: list[str], ctx: _ReplContext) -> bool:
    """Handle /knowledge slash command and its subcommands.

    Subcommands:
        - ``/knowledge`` or ``/knowledge list`` — list all knowledge items.
        - ``/knowledge save <name>`` — save a knowledge item.
        - ``/knowledge load <name>`` — load a knowledge item into context.
        - ``/knowledge delete <name>`` — delete a knowledge item.

    Args:
        parts: The split command parts (``["/knowledge", ...]``).
        ctx: The REPL context containing shared state.

    Returns:
        True to continue the REPL.
    """
    if ctx.knowledge_store is None:
        print("Knowledge store not available.")
        return True

    # No subcommand or "list" → list items.
    if len(parts) < 2 or not parts[1].strip():
        return _knowledge_list(ctx)

    sub_parts = parts[1].strip().split(maxsplit=1)
    subcmd = sub_parts[0].lower()
    sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""

    if subcmd == "list":
        return _knowledge_list(ctx)

    if subcmd == "save":
        if not sub_arg:
            print("Usage: /knowledge save <name>")
            return True
        # Save with description from last assistant message if available.
        description = ""
        content = ""
        for msg in reversed(ctx.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                description = content[:100] if content else ""
                break
        try:
            ctx.knowledge_store.save_item(
                name=sub_arg,
                description=description,
                content=content,
            )
            print(f"Knowledge item '{sub_arg}' saved.")
        except KnowledgeError as exc:
            print(f"Failed to save knowledge: {exc}")
        return True

    if subcmd == "load":
        if not sub_arg:
            print("Usage: /knowledge load <name>")
            return True
        try:
            item = ctx.knowledge_store.load_item(sub_arg)
            # Inject knowledge content into the conversation as a system message.
            artifacts = item.get("artifacts_content", {})
            content_parts = []
            for artifact_name, artifact_content in artifacts.items():
                content_parts.append(
                    f"--- {artifact_name} ---\n{artifact_content}"
                )
            if content_parts:
                knowledge_content = "\n\n".join(content_parts)
                ctx.messages.append({
                    "role": "system",
                    "content": (
                        f"Knowledge item '{sub_arg}' loaded:\n\n"
                        f"{knowledge_content}"
                    ),
                })
            print(f"Knowledge item '{sub_arg}' loaded into context.")
        except KnowledgeNotFoundError:
            print(f"Knowledge item '{sub_arg}' not found.")
        except KnowledgeError as exc:
            print(f"Failed to load knowledge: {exc}")
        return True

    if subcmd == "delete":
        if not sub_arg:
            print("Usage: /knowledge delete <name>")
            return True
        try:
            ctx.knowledge_store.delete_item(sub_arg)
            print(f"Knowledge item '{sub_arg}' deleted.")
        except KnowledgeNotFoundError:
            print(f"Knowledge item '{sub_arg}' not found.")
        except KnowledgeError as exc:
            print(f"Failed to delete knowledge: {exc}")
        return True

    print(f"Unknown knowledge subcommand: {subcmd}")
    print("Usage: /knowledge [list|save|load|delete] <name>")
    return True


def _knowledge_list(ctx: _ReplContext) -> bool:
    """List all knowledge items.

    Args:
        ctx: The REPL context.

    Returns:
        True to continue the REPL.
    """
    try:
        items = ctx.knowledge_store.list_items()
    except KnowledgeError as exc:
        print(f"Failed to list knowledge: {exc}")
        return True

    if not items:
        print("No knowledge items found.")
        return True

    print("\nKnowledge items:")
    for item in items:
        name = item.get("name", "?")
        desc = item.get("description", "")
        tags = item.get("tags", [])
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        print(f"  {name}: {desc[:60]}{tag_str}")
    print()
    return True


# ---------------------------------------------------------------------------
# Skills command handler
# ---------------------------------------------------------------------------


def _handle_skills_command(parts: list[str], ctx: _ReplContext) -> bool:
    """Handle /skills slash command and its subcommands.

    Subcommands:
        - ``/skills`` or ``/skills list`` — list all discovered skills.
        - ``/skills show <name>`` — show a skill's content.

    Args:
        parts: The split command parts (``["/skills", ...]``).
        ctx: The REPL context containing shared state.

    Returns:
        True to continue the REPL.
    """
    if ctx.skills_loader is None:
        print("Skills system not available.")
        return True

    # No subcommand or "list" → list skills.
    if len(parts) < 2 or not parts[1].strip():
        return _skills_list(ctx)

    sub_parts = parts[1].strip().split(maxsplit=1)
    subcmd = sub_parts[0].lower()
    sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""

    if subcmd == "list":
        return _skills_list(ctx)

    if subcmd == "show":
        if not sub_arg:
            print("Usage: /skills show <name>")
            return True
        try:
            content = ctx.skills_loader.get_skill_content(sub_arg)
            print(f"\n{content}\n")
        except Exception:
            print(f"Skill '{sub_arg}' not found.")
        return True

    print(f"Unknown skills subcommand: {subcmd}")
    print("Usage: /skills [list|show <name>]")
    return True


def _skills_list(ctx: _ReplContext) -> bool:
    """List all discovered skills.

    Args:
        ctx: The REPL context.

    Returns:
        True to continue the REPL.
    """
    skills = ctx.skills_loader.list_skills()
    if not skills:
        print("No skills discovered.")
        return True

    print("\nSkills:")
    for skill in skills:
        triggers = ", ".join(skill.triggers) if skill.triggers else ""
        print(f"  {skill.name}: {skill.description}")
        if triggers:
            print(f"    triggers: {triggers}")
    print()
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
        "--num-ctx",
        type=int,
        default=None,
        help="Context window size in tokens (default: model-specific).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: model-specific).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling threshold (default: model-specific).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling limit (default: model-specific).",
    )
    parser.add_argument(
        "--think-mode",
        action="store_true",
        default=None,
        help="Enable extended thinking mode for supported models.",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        default=False,
        help="Run in JSON-line server mode (for desktop GUI).",
    )
    parser.add_argument(
        "--web-monitor",
        action="store_true",
        default=False,
        help="Run web-based agent monitor (browser dashboard with SSE streaming).",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=7070,
        help="Port for --web-monitor (default: 7070).",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        default=False,
        help="Check for updates and pull the latest version.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        default=False,
        help="Start in plan mode with plan commands available.",
    )
    parser.add_argument(
        "--ideate",
        action="store_true",
        default=False,
        help="Start directly in ideation (brainstorming) mode.",
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
    sub_agent_runner: object | None = None,
    plan_manager: PlanManager | None = None,
    knowledge_store: KnowledgeStore | None = None,
    skills_loader: SkillsLoader | None = None,
    ideation_engine: IdeationEngine | None = None,
    initial_mode: str = "agent",
) -> None:
    """Run the interactive REPL loop.

    Reads user input line-by-line, detects slash commands, and forwards
    natural-language prompts to :func:`agent_loop` for LLM processing.
    Supports multiple modes: ``agent`` (default tool-using mode) and
    ``ideate`` (tool-free brainstorming mode).

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
        sub_agent_runner: Optional :class:`SubAgentRunner` for background
            agent status queries via the ``/agents`` command.
        plan_manager: Optional :class:`PlanManager` for plan CRUD.
        knowledge_store: Optional :class:`KnowledgeStore` for persistent
            knowledge items.
        skills_loader: Optional :class:`SkillsLoader` for skill
            auto-discovery and contextual injection.
        ideation_engine: Optional :class:`IdeationEngine` for tool-free
            brainstorming mode.
        initial_mode: Starting REPL mode (``"agent"`` or ``"ideate"``).
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
        sub_agent_runner=sub_agent_runner,
        plan_manager=plan_manager,
        knowledge_store=knowledge_store,
        skills_loader=skills_loader,
        ideation_engine=ideation_engine,
    )

    # Set initial mode.
    ctx.current_mode = initial_mode
    if initial_mode == "ideate" and ideation_engine is not None:
        if not ideation_engine.has_session:
            ideation_engine.start_session()
        print("Starting in ideation mode. Type /ideate exit to return.\n")

    while True:
        # Read user input with mode-aware prompt.
        prompt_label = "Ideate> " if ctx.current_mode == "ideate" else "You> "
        try:
            user_input = input(prompt_label)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        # Skip empty input.
        stripped = user_input.strip()
        if not stripped:
            continue

        # Handle slash commands (available in all modes).
        if stripped.startswith("/"):
            should_continue = _handle_slash_command(stripped, ctx)
            if not should_continue:
                break
            continue

        # -- Ideation mode --------------------------------------------------
        if ctx.current_mode == "ideate":
            if ctx.ideation_engine is not None:
                try:
                    ctx.ideation_engine.chat_turn(
                        user_input=stripped,
                        model=ctx.config.model,
                    )
                except KeyboardInterrupt:
                    print("\nInterrupted.")
                except Exception as exc:
                    sys.stderr.write(f"Ideation error: {exc}\n")
            else:
                print("Ideation engine not available. Use /ideate exit.")
            continue

        # -- Agent mode -----------------------------------------------------

        # Inject skills context if skills match the user input.
        if ctx.skills_loader is not None:
            matching_skills = ctx.skills_loader.get_matching_skills(stripped)
            for skill in matching_skills:
                messages.append({
                    "role": "system",
                    "content": (
                        f"--- SKILL: {skill.name} ---\n"
                        f"{skill.content}\n"
                        f"--- END SKILL ---"
                    ),
                })

        # Fast-mode heuristic: suggest plan for complex requests.
        if (
            ctx.plan_manager is not None
            and ctx.active_plan_id is None
            and _is_complex_request(stripped)
        ):
            print(
                "This looks complex. Consider creating a plan first "
                "with /plan create <title>."
            )

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

        # Inject active plan context if a plan is active.
        if ctx.active_plan_id is not None and ctx.plan_manager is not None:
            try:
                plan_content = ctx.plan_manager.get_plan_content(
                    ctx.active_plan_id,
                )
                plan_msg = build_plan_context(plan_content)
                messages.append(plan_msg)
            except PlanError:
                # Plan read failure is non-fatal; skip injection.
                pass

        # Build user message and add to history.
        messages.append({"role": "user", "content": prompt_content})

        # Build merged inference options: defaults < presets < user config.
        default_options: dict = {"num_ctx": 8192}
        preset_options = get_model_preset(config.model)
        user_options: dict = {"num_ctx": config.num_ctx}
        if config.temperature is not None:
            user_options["temperature"] = config.temperature
        if config.top_p is not None:
            user_options["top_p"] = config.top_p
        if config.top_k is not None:
            user_options["top_k"] = config.top_k
        inference_options = {**default_options, **preset_options, **user_options}

        # Determine think mode for models that support it.
        family = get_model_family(config.model)
        think: bool | None = None
        if family in SUPPORTS_THINKING:
            think = True if config.think_mode else False

        # Run the agent loop (streams response to stdout).
        try:
            agent_loop(
                client=client,
                model=config.model,
                tools=tools,
                messages=messages,
                debug=config.debug,
                options=inference_options,
                think=think,
            )
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as exc:
            sys.stderr.write(f"Error: {exc}\n")
