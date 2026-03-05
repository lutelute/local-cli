"""Entry point for python -m local_cli."""

import sys

from local_cli.cli import build_parser, run_repl
from local_cli.config import Config
from local_cli.ollama_client import OllamaClient, OllamaConnectionError
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

    # 3. Validate model name.
    if not validate_model_name(config.model):
        sys.stderr.write(
            f"Error: invalid model name: {config.model!r}\n"
        )
        sys.exit(1)

    # 4. Create Ollama client.
    try:
        client = OllamaClient(base_url=config.ollama_host)
    except ValueError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.exit(1)

    # 5. Optionally check Ollama connectivity and model availability.
    try:
        version_info = client.get_version()
        if config.debug:
            sys.stderr.write(
                f"[debug] Ollama version: {version_info.get('version', 'unknown')}\n"
            )

        # Check if the requested model is available.
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
        sys.stderr.write(
            "Warning: could not connect to Ollama. "
            "Make sure Ollama is running.\n"
        )

    # 6. Get default tools.
    tools = get_default_tools()

    # 7. Optionally initialize RAG engine.
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

    # 8. Start the interactive REPL.
    run_repl(config, client, tools, rag_engine=rag_engine, rag_topk=rag_topk)


if __name__ == "__main__":
    main()
