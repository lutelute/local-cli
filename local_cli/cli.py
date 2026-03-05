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
from local_cli.ollama_client import OllamaClient
from local_cli.tools.base import Tool

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful AI coding assistant running locally via Ollama. "
    "You have access to tools for reading files, writing files, editing "
    "files, and running shell commands. Use these tools to help the user "
    "with their coding tasks. Be concise and accurate."
)

# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

_SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show this help message.",
    "/exit": "Exit the REPL.",
    "/quit": "Exit the REPL (alias for /exit).",
}


def _handle_slash_command(command: str) -> bool:
    """Handle a slash command.

    Args:
        command: The raw user input starting with ``/``.

    Returns:
        True if the REPL should continue, False if it should exit.
    """
    cmd = command.strip().lower()

    if cmd in ("/exit", "/quit"):
        print("Goodbye!")
        return False

    if cmd == "/help":
        print("\nAvailable commands:")
        for name, description in _SLASH_COMMANDS.items():
            print(f"  {name:<12} {description}")
        print()
        return True

    print(f"Unknown command: {command.strip()}")
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
    return parser


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


def run_repl(
    config: Config,
    client: OllamaClient,
    tools: list[Tool],
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
    """
    # Print welcome banner.
    tool_names = ", ".join(t.name for t in tools)
    print(f"local-cli v{__version__} | model: {config.model}")
    print(f"Tools: {tool_names}")
    print("Type /help for commands, /exit to quit.\n")

    # Conversation history (persists across the session).
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

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
            should_continue = _handle_slash_command(stripped)
            if not should_continue:
                break
            continue

        # Build user message and add to history.
        messages.append({"role": "user", "content": stripped})

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
