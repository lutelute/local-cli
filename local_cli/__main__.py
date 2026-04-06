"""Entry point for python -m local_cli."""

import sys

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
from local_cli.ollama_client import OllamaClient, OllamaConnectionError
from local_cli.orchestrator import Orchestrator
from local_cli.plan_manager import PlanManager
from local_cli.rag import RAGEngine
from local_cli.security import validate_model_name
from local_cli.skills import SkillsLoader
from local_cli.tools import get_default_tools, get_sub_agent_tools


def main() -> None:
    """Run the local-cli application."""
    # 1. Parse CLI arguments.
    parser = build_parser()
    args = parser.parse_args()

    # 2. Build configuration (CLI args > env vars > config file > defaults).
    config = Config(cli_args=args)

    # 2b. Map CLI args that don't match config key names.
    # argparse converts --brain-model to brain_model and --registry-file
    # to registry_file, but config uses orchestrator_model and
    # model_registry_file respectively.
    brain_model_arg = getattr(args, "brain_model", None)
    if brain_model_arg is not None:
        config.orchestrator_model = brain_model_arg

    registry_file_arg = getattr(args, "registry_file", None)
    if registry_file_arg is not None:
        config.model_registry_file = registry_file_arg

    # 2c. Server mode — skip all heavy initialization, go straight to server.
    if getattr(args, "server", False):
        from local_cli.server import run_server
        run_server()
        return

    # 2c2. Web monitor mode — browser dashboard with SSE streaming.
    if getattr(args, "web_monitor", False):
        from local_cli.web_monitor import run_web_monitor
        run_web_monitor(config=config, port=getattr(args, "web_port", 7070))
        return

    # 2d. Handle --update flag (explicit update).
    if getattr(args, "update", False):
        from local_cli.updater import check_for_updates, perform_update

        print("Checking for updates...")
        has_updates, check_msg = check_for_updates()
        if not has_updates:
            print(check_msg)
            return

        print(check_msg)
        print("Updating...")
        success, update_msg = perform_update()
        print(update_msg)
        return

    # 2d. Background auto-update check on startup.
    _auto_update_check_result: dict | None = None

    def _bg_update_check() -> None:
        nonlocal _auto_update_check_result
        try:
            from local_cli.updater import check_for_updates
            has_updates, message = check_for_updates()
            if has_updates:
                _auto_update_check_result = {"message": message}
        except Exception:
            pass

    import threading
    _update_thread = threading.Thread(target=_bg_update_check, daemon=True)
    _update_thread.start()

    # 3. Validate model name.
    if not validate_model_name(config.model):
        sys.stderr.write(
            f"Error: invalid model name: {config.model!r}\n"
        )
        sys.exit(1)

    # 4. Create Ollama client (needed for ModelManager and Ollama provider).
    try:
        client = OllamaClient(base_url=config.ollama_host)
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)

    # 5. Run startup health check (non-blocking).
    health_results = run_health_check(client, config.model)
    sys.stderr.write(format_health_check(health_results) + "\n")

    # Derive ollama_available from health check results.
    ollama_result = health_results[0] if health_results else None
    ollama_available = ollama_result is not None and ollama_result.status == STATUS_OK

    # 5b. Optionally run interactive model selector (--select-model).
    if args.select_model:
        selected = select_model_interactive(client, config.model)
        if selected is not None:
            if not validate_model_name(selected):
                sys.stderr.write(
                    f"Error: invalid model name: {selected!r}\n"
                )
                sys.exit(1)
            config.model = selected

    # 6. Get default tools.
    tools = get_default_tools()

    # 7. Load model registry (if configured).
    registry: ModelRegistry | None = None
    if config.model_registry_file:
        try:
            registry = ModelRegistry.load(config.model_registry_file)
            if config.debug:
                sys.stderr.write(
                    f"[debug] Loaded model registry from "
                    f"{config.model_registry_file}\n"
                )
        except (FileNotFoundError, ValueError) as exc:
            sys.stderr.write(
                f"Warning: Failed to load model registry: {exc}\n"
            )

    # 8. Create model manager.
    model_manager = ModelManager(client)

    # 9. Create orchestrator.
    orchestrator = Orchestrator(config, registry=registry)

    # 10. Initialize the active provider.
    try:
        orchestrator.get_active_provider()
        if config.debug:
            sys.stderr.write(
                f"[debug] Active provider: "
                f"{orchestrator.get_active_provider_name()}\n"
            )
    except ValueError as exc:
        # Provider initialization failed (e.g. Claude without API key).
        if config.provider != "ollama":
            sys.stderr.write(
                f"Warning: Failed to initialize provider "
                f"{config.provider!r}: {exc}\n"
                f"Falling back to Ollama provider.\n"
            )
            try:
                orchestrator.switch_provider("ollama")
            except ValueError:
                sys.stderr.write(
                    "Error: Cannot initialize any provider.\n"
                )
                sys.exit(1)
        else:
            sys.stderr.write(f"Error: Failed to initialize provider: {exc}\n")
            sys.exit(1)

    # 10b. Create SubAgentRunner and AgentTool, append to tools list.
    sub_agent_runner = None
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
    except Exception as exc:
        if config.debug:
            sys.stderr.write(
                f"[debug] AgentTool not available: {exc}\n"
            )

    # 11. Optionally initialize RAG engine.
    rag_engine: RAGEngine | None = None
    rag_enabled = getattr(args, "rag", False) or False
    if rag_enabled:
        rag_path = getattr(args, "rag_path", None) or "."
        rag_model = getattr(args, "rag_model", None) or "all-minilm"
        rag_topk = getattr(args, "rag_topk", None) or 5

        try:
            rag_engine = RAGEngine(
                client=client,
                embedding_model=rag_model,
            )
            sys.stderr.write(f"RAG: indexing {rag_path}...\n")
            stats = rag_engine.index_directory(rag_path, embedding_model=rag_model)
            sys.stderr.write(
                f"RAG: indexed {stats['files_indexed']} files "
                f"({stats['chunks_indexed']} chunks), "
                f"{stats['files_unchanged']} unchanged, "
                f"{stats['files_skipped']} skipped\n"
            )
        except Exception as exc:
            sys.stderr.write(f"Warning: RAG initialization failed: {exc}\n")
            rag_engine = None
    else:
        rag_topk = 5

    # 12. Initialize plan manager.
    plan_manager: PlanManager | None = None
    try:
        plan_manager = PlanManager(plans_dir=config.plan_dir)
        if config.debug:
            sys.stderr.write(
                f"[debug] Plan manager initialized: {config.plan_dir}\n"
            )
    except Exception as exc:
        sys.stderr.write(f"Warning: Plan manager initialization failed: {exc}\n")

    # 13. Initialize knowledge store.
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

    # 14. Initialize skills loader and discover skills.
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

    # 15. Initialize ideation engine.
    ideation_engine: IdeationEngine | None = None
    try:
        ideation_engine = IdeationEngine(client=client)
        if config.debug:
            sys.stderr.write("[debug] Ideation engine initialized\n")
    except Exception as exc:
        sys.stderr.write(
            f"Warning: Ideation engine initialization failed: {exc}\n"
        )

    # 16. Determine initial REPL mode from CLI flags.
    initial_mode = config.default_mode
    if getattr(args, "plan", False):
        initial_mode = "agent"  # Plan mode uses agent mode with plan context.
    elif getattr(args, "ideate", False):
        initial_mode = "ideate"

    # 17. Show update notice if available (non-blocking check finished).
    _update_thread.join(timeout=0.5)  # Wait briefly for result.
    if _auto_update_check_result is not None:
        sys.stderr.write(
            f"\n  Update available! Run /update or local-cli --update to update.\n\n"
        )

    # 18. Start the interactive REPL.
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
