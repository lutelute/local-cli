"""Local CLI - Local-first AI coding agent powered by Ollama.

Import this module to launch the interactive REPL (antigravity-style)::

    >>> import local_cli
"""

__version__ = "0.8.0"


def _launch() -> None:
    """Launch the interactive REPL when imported directly."""
    import sys

    # Only auto-launch in interactive mode (not during import by tests/tools).
    if not hasattr(sys, "ps1") and not sys.flags.interactive:
        return
    from local_cli.__main__ import main
    main()


_launch()
