"""Interactive model selector for local-cli.

Provides an interactive terminal interface for selecting from locally available
Ollama models.  Includes helper functions for formatting model metadata and a
simple numbered-list fallback selector for non-TTY environments.
"""

import sys
from typing import Any


# ---------------------------------------------------------------------------
# Display data type
# ---------------------------------------------------------------------------

# Keys used in the model display data dictionary.
_KEY_NAME = "name"
_KEY_SIZE = "size"
_KEY_PARAMETER_SIZE = "parameter_size"
_KEY_QUANTIZATION_LEVEL = "quantization_level"
_KEY_FAMILY = "family"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _format_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Converts raw byte counts into ``GB``, ``MB``, or ``KB`` suffixed strings
    with one decimal place.

    Args:
        size_bytes: Size in bytes.  Must be non-negative.

    Returns:
        Human-readable size string (e.g. ``"5.2 GB"``, ``"450.0 MB"``).
    """
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.1f} GB"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    return f"{size_bytes / 1_000:.1f} KB"


def _build_model_display_data(model: dict[str, Any]) -> dict[str, str]:
    """Extract display metadata from an Ollama model info dictionary.

    Safely extracts the model name, formatted size, parameter size,
    quantization level, and family from the dictionary returned by
    ``OllamaClient.list_models()``.  Missing fields default to ``"N/A"``.

    Args:
        model: A single model info dict from the Ollama ``/api/tags``
            response.  Expected keys: ``name``, ``size``, ``details``
            (which may contain ``parameter_size``, ``quantization_level``,
            ``family``).

    Returns:
        A dictionary with string values for display:
        ``name``, ``size``, ``parameter_size``, ``quantization_level``,
        ``family``.
    """
    details: dict[str, Any] = model.get("details") or {}

    raw_size = model.get("size")
    if isinstance(raw_size, (int, float)) and raw_size > 0:
        size_str = _format_size(int(raw_size))
    else:
        size_str = "N/A"

    return {
        _KEY_NAME: model.get("name", "N/A"),
        _KEY_SIZE: size_str,
        _KEY_PARAMETER_SIZE: details.get("parameter_size") or "N/A",
        _KEY_QUANTIZATION_LEVEL: details.get("quantization_level") or "N/A",
        _KEY_FAMILY: details.get("family") or "N/A",
    }


# ---------------------------------------------------------------------------
# Simple fallback selector
# ---------------------------------------------------------------------------


def _select_model_simple(
    models: list[dict[str, Any]],
    current_model: str = "",
) -> str | None:
    """Display a numbered list of models and prompt the user to select one.

    This is a simple fallback selector for non-TTY environments or when
    the curses TUI is unavailable.  Models are displayed with index numbers
    and metadata.  The user types a number to select or ``q`` to cancel.

    Args:
        models: List of model info dicts from ``OllamaClient.list_models()``.
        current_model: Name of the currently active model (marked with
            ``*`` in the listing).  Defaults to ``""``.

    Returns:
        The selected model name string, or ``None`` if the user cancels
        or the model list is empty.
    """
    if not models:
        sys.stderr.write(
            "No models found. Pull a model with: ollama pull <model>\n"
        )
        return None

    # Build display data for all models.
    display_data = [_build_model_display_data(m) for m in models]

    # Print header.
    sys.stdout.write("\nAvailable models:\n")
    sys.stdout.write("-" * 60 + "\n")

    # Print each model with index and metadata.
    for idx, data in enumerate(display_data, start=1):
        marker = "*" if data[_KEY_NAME] == current_model else " "
        sys.stdout.write(
            f"  {marker} {idx:>2}) {data[_KEY_NAME]}"
            f"  ({data[_KEY_SIZE]}"
            f", {data[_KEY_PARAMETER_SIZE]}"
            f", {data[_KEY_QUANTIZATION_LEVEL]})\n"
        )

    sys.stdout.write("-" * 60 + "\n")
    sys.stdout.write("Enter number to select, or 'q' to cancel: ")
    sys.stdout.flush()

    # Prompt loop.
    while True:
        try:
            choice = input().strip()
        except (KeyboardInterrupt, EOFError):
            sys.stdout.write("\n")
            return None

        if not choice or choice.lower() == "q":
            return None

        try:
            index = int(choice)
        except ValueError:
            sys.stderr.write(
                f"Invalid input: {choice!r}. Enter a number or 'q'.\n"
            )
            sys.stdout.write("Enter number to select, or 'q' to cancel: ")
            sys.stdout.flush()
            continue

        if index < 1 or index > len(models):
            sys.stderr.write(
                f"Out of range. Enter 1-{len(models)} or 'q'.\n"
            )
            sys.stdout.write("Enter number to select, or 'q' to cancel: ")
            sys.stdout.flush()
            continue

        return display_data[index - 1][_KEY_NAME]
