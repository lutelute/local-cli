"""Entry point for python -m local_cli."""

import sys

from local_cli.cli import build_parser, run_repl
from local_cli.config import Config
from local_cli.model_manager import ModelManager
from local_cli.model_registry import ModelRegistry
from local_cli.model_selector import select_model_interactive
from local_cli.ollama_client import OllamaClient, OllamaConnectionError
from local_cli.orchestrator import Orchestrator
from local_cli.rag import RAGEngine
from local_cli.security import validate_model_name
from local_cli.tools import get_default_tools


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

    # 2c. Handle --update flag.
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

    # 5. Optionally check Ollama connectivity and model availability.
    ollama_available = True
    try:
        version_info = client.get_version()
        if config.debug:
            sys.stderr.write(
                f"[debug] Ollama version: {version_info.get('version', 'unknown')}\n"
            )

        # Check if the requested model is available on Ollama.
        # Only relevant when using the Ollama provider.
        if config.provider == "ollama":
            models = client.list_models()
            model_names = [m.get("name", "") for m in models]
            # Ollama model names may include a tag suffix (e.g. ":latest").
            # Match if the configured model equals the full name or the
            # base name without tag.
            model_found = any(
                config.model == name or config.model == name.split(":")[0]
                for name in model_names
            )
            if not model_found and model_names:
                sys.stderr.write(
                    f"Warning: model '{config.model}' not found on Ollama server.\n"
                    f"Available models: {', '.join(model_names)}\n"
                )

    except OllamaConnectionError:
        ollama_available = False
        sys.stderr.write(
            "Warning: could not connect to Ollama. "
            "Make sure Ollama is running.\n"
        )

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

    # 12. Server mode (for desktop GUI).
    if getattr(args, "server", False):
        from local_cli.server import run_server
        run_server()
        return

    # 13. Start the interactive REPL.
    run_repl(
        config,
        client,
        tools,
        rag_engine=rag_engine,
        rag_topk=rag_topk,
        orchestrator=orchestrator,
        model_manager=model_manager,
    )


if __name__ == "__main__":
    main()
