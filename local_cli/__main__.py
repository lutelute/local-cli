"""Entry point for python -m local_cli.

The startup sequence is decomposed into small, individually testable
steps; :func:`main` is a thin orchestration of them (it was previously a
single 300-line function with no test coverage).
"""

import sys
import threading

from local_cli.cli import build_parser, run_repl
from local_cli.config import Config
from local_cli.health_check import (
    STATUS_OK,
    format_health_check,
    run_health_check,
)
from local_cli.ideation import IdeationEngine
from local_cli.knowledge import KnowledgeStore
from local_cli.model_manager import ModelManager
from local_cli.model_registry import ModelRegistry
from local_cli.model_selector import select_model_interactive
from local_cli.ollama_client import OllamaClient
from local_cli.orchestrator import Orchestrator
from local_cli.plan_manager import PlanManager
from local_cli.rag import RAGEngine
from local_cli.security import validate_model_name
from local_cli.skills import SkillsLoader
from local_cli.tools import get_default_tools, get_sub_agent_tools
from local_cli.tools.base import Tool


def _cli_confirm(command: str) -> bool:
    """Ask the user to approve a risky shell command before it runs.

    Used by the interactive REPL when confirmation is enabled (the user did
    not pass --yes).  Reads a yes/no answer from stdin; EOF or Ctrl-C is
    treated as a decline.

    Args:
        command: The risky shell command awaiting approval.

    Returns:
        True to run the command, False to decline.
    """
    sys.stderr.write(
        f"\n  ⚠  About to run a risky command:\n    {command}\n"
        "  Proceed? [y/N] "
    )
    sys.stderr.flush()
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        sys.stderr.write("\n")
        return False
    return answer in ("y", "yes")


# ---------------------------------------------------------------------------
# Startup steps (each one small and testable; main() just sequences them)
# ---------------------------------------------------------------------------


def _apply_arg_aliases(args: object, config: Config) -> None:
    """Map CLI args whose names differ from their config keys.

    argparse converts ``--brain-model`` to ``brain_model`` and
    ``--registry-file`` to ``registry_file``, but config uses
    ``orchestrator_model`` and ``model_registry_file`` respectively.

    Args:
        args: The parsed argparse namespace.
        config: The configuration to update in place.
    """
    brain_model_arg = getattr(args, "brain_model", None)
    if brain_model_arg is not None:
        config.orchestrator_model = brain_model_arg

    registry_file_arg = getattr(args, "registry_file", None)
    if registry_file_arg is not None:
        config.model_registry_file = registry_file_arg


def _dispatch_alternate_modes(args: object, config: Config) -> bool:
    """Run a non-REPL mode (server / web monitor / bench / update).

    These modes skip the heavy REPL initialization entirely, so their
    modules are imported lazily.

    Args:
        args: The parsed argparse namespace.
        config: The application configuration.

    Returns:
        ``True`` when an alternate mode ran (the caller should return),
        ``False`` to continue with the interactive REPL startup.
    """
    if getattr(args, "server", False):
        from local_cli.server import run_server
        run_server()
        return True

    if getattr(args, "web_monitor", False):
        from local_cli.web_monitor import run_web_monitor
        run_web_monitor(config=config, port=getattr(args, "web_port", 7070))
        return True

    if getattr(args, "bench", False):
        from local_cli.bench import run_bench
        from local_cli.providers import get_provider

        if config.provider == "llama-server":
            prov = get_provider("llama-server", base_url=config.llama_server_url)
        elif config.provider == "claude":
            prov = get_provider("claude")
        else:
            prov = get_provider("ollama", base_url=config.ollama_host)
        run_bench(prov, config.model)
        return True

    if getattr(args, "update", False):
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

    return False


def _start_update_check() -> tuple[threading.Thread, dict]:
    """Start the background auto-update check.

    Returns:
        ``(thread, result)`` where *result* is a shared dict that gains
        a ``"message"`` key when an update is available.
    """
    result: dict = {}

    def _bg_update_check() -> None:
        try:
            from local_cli.updater import check_for_updates
            has_updates, message = check_for_updates()
            if has_updates:
                result["message"] = message
        except Exception:
            pass

    thread = threading.Thread(target=_bg_update_check, daemon=True)
    thread.start()
    return thread, result


def _show_update_notice(thread: threading.Thread, result: dict) -> None:
    """Print the update notice if the background check found one.

    Args:
        thread: The background check thread (joined briefly).
        result: The shared result dict from :func:`_start_update_check`.
    """
    thread.join(timeout=0.5)
    if result.get("message") is not None:
        sys.stderr.write(
            "\n  Update available! Run /update or local-cli --update "
            "to update.\n\n"
        )


def _build_client(config: Config) -> OllamaClient:
    """Validate the model name, build the Ollama client, run health check.

    Exits the process with status 1 on an invalid model name or an
    invalid Ollama host.

    Args:
        config: The application configuration.

    Returns:
        A connected :class:`OllamaClient`.
    """
    if not validate_model_name(config.model):
        sys.stderr.write(f"Error: invalid model name: {config.model!r}\n")
        sys.exit(1)

    try:
        client = OllamaClient(base_url=config.ollama_host)
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)

    health_results = run_health_check(client, config.model)
    sys.stderr.write(format_health_check(health_results) + "\n")
    return client


def _maybe_select_model(args: object, config: Config, client: OllamaClient) -> None:
    """Run the interactive model selector when --select-model was passed.

    Exits with status 1 if the selected name fails validation.

    Args:
        args: The parsed argparse namespace.
        config: The configuration (``model`` is updated in place).
        client: The Ollama client used to list installed models.
    """
    if not getattr(args, "select_model", False):
        return
    selected = select_model_interactive(client, config.model)
    if selected is not None:
        if not validate_model_name(selected):
            sys.stderr.write(f"Error: invalid model name: {selected!r}\n")
            sys.exit(1)
        config.model = selected


def _build_tools(config: Config) -> list[Tool]:
    """Build the REPL tool set, wiring risky-command confirmation.

    Unless the user opted into auto-approval (--yes), the bash tool is
    replaced with one that asks for confirmation on risky commands.
    Sub-agents and the GUI server keep the unconfirmed bash tool (no
    stdin to prompt on).

    Args:
        config: The application configuration.

    Returns:
        The tool instances for the interactive REPL.
    """
    tools = get_default_tools()
    if not config.auto_approve:
        from local_cli.tools.bash_tool import BashTool
        for i, t in enumerate(tools):
            if t.name == "bash":
                tools[i] = BashTool(confirm=_cli_confirm)
                break
    return tools


def _load_registry(config: Config) -> ModelRegistry | None:
    """Load the task-to-model registry file when configured.

    Args:
        config: The application configuration.

    Returns:
        The loaded registry, or ``None`` when unconfigured or on error
        (a warning is printed).
    """
    if not config.model_registry_file:
        return None
    try:
        registry = ModelRegistry.load(config.model_registry_file)
        if config.debug:
            sys.stderr.write(
                f"[debug] Loaded model registry from "
                f"{config.model_registry_file}\n"
            )
        return registry
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"Warning: Failed to load model registry: {exc}\n")
        return None


def _init_orchestrator(
    config: Config,
    registry: ModelRegistry | None,
) -> Orchestrator:
    """Create the orchestrator and initialize its active provider.

    A non-Ollama provider that fails to initialize (e.g. Claude without
    an API key) falls back to Ollama with a warning; if no provider can
    be initialized the process exits with status 1.

    Args:
        config: The application configuration.
        registry: Optional task-to-model registry.

    Returns:
        The orchestrator with a working active provider.
    """
    orchestrator = Orchestrator(config, registry=registry)
    try:
        orchestrator.get_active_provider()
        if config.debug:
            sys.stderr.write(
                f"[debug] Active provider: "
                f"{orchestrator.get_active_provider_name()}\n"
            )
    except ValueError as exc:
        if config.provider != "ollama":
            sys.stderr.write(
                f"Warning: Failed to initialize provider "
                f"{config.provider!r}: {exc}\n"
                f"Falling back to Ollama provider.\n"
            )
            try:
                orchestrator.switch_provider("ollama")
            except ValueError:
                sys.stderr.write("Error: Cannot initialize any provider.\n")
                sys.exit(1)
        else:
            sys.stderr.write(f"Error: Failed to initialize provider: {exc}\n")
            sys.exit(1)
    return orchestrator


def _attach_agent_tool(
    tools: list[Tool],
    orchestrator: Orchestrator,
    config: Config,
) -> object | None:
    """Create the SubAgentRunner and append the agent tool to *tools*.

    Failure is non-fatal: the REPL simply runs without sub-agent
    support (a debug note is printed).

    Args:
        tools: The REPL tool list (mutated in place).
        orchestrator: Source of the provider handed to sub-agents.
        config: The application configuration.

    Returns:
        The :class:`SubAgentRunner`, or ``None`` when unavailable.
    """
    try:
        from local_cli.sub_agent import SubAgentRunner
        from local_cli.tools.agent_tool import AgentTool

        sub_agent_runner = SubAgentRunner()
        provider = orchestrator.get_active_provider()
        agent_tool = AgentTool(
            runner=sub_agent_runner,
            provider=provider,
            model=config.model,
            sub_agent_tools=get_sub_agent_tools(),
        )
        tools.append(agent_tool)
        if config.debug:
            sys.stderr.write("[debug] AgentTool enabled (sub-agent support)\n")
        return sub_agent_runner
    except Exception as exc:
        if config.debug:
            sys.stderr.write(f"[debug] AgentTool not available: {exc}\n")
        return None


def _init_rag(args: object, client: OllamaClient) -> tuple[RAGEngine | None, int]:
    """Initialize the RAG engine when --rag was passed.

    Args:
        args: The parsed argparse namespace.
        client: The Ollama client used for embeddings.

    Returns:
        ``(rag_engine, rag_topk)``; the engine is ``None`` when RAG is
        disabled or its initialization failed (a warning is printed).
    """
    if not (getattr(args, "rag", False) or False):
        return None, 5

    rag_path = getattr(args, "rag_path", None) or "."
    rag_model = getattr(args, "rag_model", None) or "all-minilm"
    rag_topk = getattr(args, "rag_topk", None) or 5

    try:
        rag_engine = RAGEngine(client=client, embedding_model=rag_model)
        sys.stderr.write(f"RAG: indexing {rag_path}...\n")
        stats = rag_engine.index_directory(rag_path, embedding_model=rag_model)
        sys.stderr.write(
            f"RAG: indexed {stats['files_indexed']} files "
            f"({stats['chunks_indexed']} chunks), "
            f"{stats['files_unchanged']} unchanged, "
            f"{stats['files_skipped']} skipped\n"
        )
        return rag_engine, rag_topk
    except Exception as exc:
        sys.stderr.write(f"Warning: RAG initialization failed: {exc}\n")
        return None, rag_topk


def _init_optional_components(
    config: Config,
    client: OllamaClient,
) -> tuple[
    PlanManager | None,
    KnowledgeStore | None,
    SkillsLoader | None,
    IdeationEngine | None,
]:
    """Initialize the optional REPL components (plan/knowledge/skills/ideation).

    Each component fails independently with a warning; the REPL runs
    with whatever subset initialized successfully.

    Args:
        config: The application configuration.
        client: The Ollama client (used by the ideation engine).

    Returns:
        ``(plan_manager, knowledge_store, skills_loader, ideation_engine)``,
        any of which may be ``None``.
    """
    plan_manager: PlanManager | None = None
    try:
        plan_manager = PlanManager(plans_dir=config.plan_dir)
        if config.debug:
            sys.stderr.write(
                f"[debug] Plan manager initialized: {config.plan_dir}\n"
            )
    except Exception as exc:
        sys.stderr.write(f"Warning: Plan manager initialization failed: {exc}\n")

    knowledge_store: KnowledgeStore | None = None
    try:
        knowledge_store = KnowledgeStore(knowledge_dir=config.knowledge_dir)
        if config.debug:
            sys.stderr.write(
                f"[debug] Knowledge store initialized: {config.knowledge_dir}\n"
            )
    except Exception as exc:
        sys.stderr.write(
            f"Warning: Knowledge store initialization failed: {exc}\n"
        )

    skills_loader: SkillsLoader | None = None
    try:
        skills_loader = SkillsLoader(skills_dir=config.skills_dir)
        discovered = skills_loader.discover_skills()
        if config.debug and discovered:
            skill_names = ", ".join(s.name for s in discovered)
            sys.stderr.write(
                f"[debug] Discovered {len(discovered)} skills: {skill_names}\n"
            )
    except Exception as exc:
        sys.stderr.write(
            f"Warning: Skills loader initialization failed: {exc}\n"
        )

    ideation_engine: IdeationEngine | None = None
    try:
        ideation_engine = IdeationEngine(client=client)
        if config.debug:
            sys.stderr.write("[debug] Ideation engine initialized\n")
    except Exception as exc:
        sys.stderr.write(
            f"Warning: Ideation engine initialization failed: {exc}\n"
        )

    return plan_manager, knowledge_store, skills_loader, ideation_engine


def _resolve_initial_mode(args: object, config: Config) -> str:
    """Determine the starting REPL mode from CLI flags and config.

    Args:
        args: The parsed argparse namespace.
        config: The application configuration.

    Returns:
        ``"agent"`` or ``"ideate"``.
    """
    initial_mode = config.default_mode
    if getattr(args, "plan", False):
        initial_mode = "agent"  # Plan mode uses agent mode with plan context.
    elif getattr(args, "ideate", False):
        initial_mode = "ideate"
    return initial_mode


def main() -> None:
    """Run the local-cli application."""
    parser = build_parser()
    args = parser.parse_args()

    config = Config(cli_args=args)
    _apply_arg_aliases(args, config)

    # Non-REPL modes (server / web monitor / bench / update) return early.
    if _dispatch_alternate_modes(args, config):
        return

    update_thread, update_result = _start_update_check()

    client = _build_client(config)
    _maybe_select_model(args, config, client)

    tools = _build_tools(config)
    registry = _load_registry(config)
    model_manager = ModelManager(client)
    orchestrator = _init_orchestrator(config, registry)
    sub_agent_runner = _attach_agent_tool(tools, orchestrator, config)

    rag_engine, rag_topk = _init_rag(args, client)
    plan_manager, knowledge_store, skills_loader, ideation_engine = (
        _init_optional_components(config, client)
    )
    initial_mode = _resolve_initial_mode(args, config)

    _show_update_notice(update_thread, update_result)

    run_repl(
        config,
        client,
        tools,
        rag_engine=rag_engine,
        rag_topk=rag_topk,
        orchestrator=orchestrator,
        model_manager=model_manager,
        sub_agent_runner=sub_agent_runner,
        plan_manager=plan_manager,
        knowledge_store=knowledge_store,
        skills_loader=skills_loader,
        ideation_engine=ideation_engine,
        initial_mode=initial_mode,
    )


if __name__ == "__main__":
    main()
