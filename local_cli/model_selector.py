"""Interactive model selector for local-cli.

Provides an interactive terminal interface for selecting from locally available
Ollama models.  Includes a curses-based TUI with two-panel layout, helper
functions for formatting model metadata, and a simple numbered-list fallback
selector for non-TTY environments.
"""

import sys
from typing import Any

from local_cli.ollama_client import OllamaClient, OllamaConnectionError

# Curses is optional; unavailable on some platforms (e.g. Windows without
# windows-curses).  We track availability and fall back gracefully.
try:
    import curses

    _CURSES_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CURSES_AVAILABLE = False


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


# ---------------------------------------------------------------------------
# Curses TUI selector
# ---------------------------------------------------------------------------

# Minimum terminal dimensions for the curses TUI.
_MIN_COLS = 60
_MIN_LINES = 10


def _draw_curses_ui(
    stdscr: Any,
    display_data: list[dict[str, str]],
    selected: int,
    current_model: str,
) -> None:
    """Draw the two-panel curses TUI layout.

    Renders the model list on the left and details for the highlighted
    model on the right.  The currently active model is marked with ``*``.

    Args:
        stdscr: The curses standard screen window.
        display_data: Pre-built display data for each model.
        selected: Zero-based index of the currently highlighted model.
        current_model: Name of the currently active model.
    """
    stdscr.clear()
    max_y, max_x = stdscr.getmaxyx()

    # -- Title bar ----------------------------------------------------------
    title = " Model Selector - local-cli "
    title_padded = title.center(max_x - 1)
    stdscr.addnstr(0, 0, title_padded, max_x - 1, curses.A_REVERSE)

    # -- Panel dimensions ---------------------------------------------------
    # Left panel: model list.  Right panel: model details.
    divider_col = max_x // 2
    content_top = 2
    content_bottom = max_y - 2  # Reserve last line for help bar.
    visible_rows = content_bottom - content_top

    # -- Left panel: model list ---------------------------------------------
    for idx, data in enumerate(display_data):
        row = content_top + idx
        if row >= content_bottom:
            break

        marker = "*" if data[_KEY_NAME] == current_model else " "
        label = f" {marker} {data[_KEY_NAME]}"

        # Truncate to fit the left panel.
        label = label[: divider_col - 1]

        if idx == selected:
            stdscr.addnstr(row, 0, label.ljust(divider_col - 1),
                           divider_col - 1, curses.A_REVERSE)
        else:
            stdscr.addnstr(row, 0, label, divider_col - 1)

    # -- Divider line -------------------------------------------------------
    for row in range(content_top, content_bottom):
        if divider_col < max_x:
            try:
                stdscr.addch(row, divider_col, curses.ACS_VLINE)
            except curses.error:
                pass

    # -- Right panel: model details -----------------------------------------
    if 0 <= selected < len(display_data):
        detail = display_data[selected]
        detail_col = divider_col + 2
        detail_width = max_x - detail_col - 1
        if detail_width > 0:
            lines = [
                f"Name:   {detail[_KEY_NAME]}",
                f"Size:   {detail[_KEY_SIZE]}",
                f"Params: {detail[_KEY_PARAMETER_SIZE]}",
                f"Quant:  {detail[_KEY_QUANTIZATION_LEVEL]}",
                f"Family: {detail[_KEY_FAMILY]}",
            ]
            for i, line in enumerate(lines):
                row = content_top + i
                if row >= content_bottom:
                    break
                stdscr.addnstr(row, detail_col, line, detail_width)

    # -- Help bar -----------------------------------------------------------
    help_text = " [Up/Down] Navigate  [Enter] Select  [q/Esc] Cancel "
    help_padded = help_text.center(max_x - 1)
    stdscr.addnstr(max_y - 1, 0, help_padded, max_x - 1, curses.A_REVERSE)

    stdscr.refresh()


def _curses_main(
    stdscr: Any,
    models: list[dict[str, Any]],
    current_model: str,
) -> str | None:
    """Curses main loop for the interactive model selector.

    This function is passed to ``curses.wrapper()`` to ensure proper
    terminal setup and cleanup.

    Args:
        stdscr: The curses standard screen window.
        models: List of model info dicts from ``OllamaClient.list_models()``.
        current_model: Name of the currently active model.

    Returns:
        The selected model name, or ``None`` if cancelled.

    Raises:
        curses.error: If the terminal is too small.
    """
    max_y, max_x = stdscr.getmaxyx()
    if max_y < _MIN_LINES or max_x < _MIN_COLS:
        raise curses.error(
            f"Terminal too small ({max_x}x{max_y}). "
            f"Need at least {_MIN_COLS}x{_MIN_LINES}."
        )

    # Hide the cursor.
    try:
        curses.curs_set(0)
    except curses.error:
        pass

    # Build display data once.
    display_data = [_build_model_display_data(m) for m in models]

    # Start with the current model highlighted if it exists in the list.
    selected = 0
    for idx, data in enumerate(display_data):
        if data[_KEY_NAME] == current_model:
            selected = idx
            break

    while True:
        _draw_curses_ui(stdscr, display_data, selected, current_model)

        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            return None

        # Navigation.
        if key == curses.KEY_UP:
            if selected > 0:
                selected -= 1
        elif key == curses.KEY_DOWN:
            if selected < len(display_data) - 1:
                selected += 1

        # Confirm selection.
        elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            return display_data[selected][_KEY_NAME]

        # Cancel.
        elif key in (27, ord("q"), ord("Q")):  # 27 = Escape
            return None


def _select_model_curses(
    models: list[dict[str, Any]],
    current_model: str = "",
) -> str | None:
    """Display a curses-based interactive model selector.

    Uses ``curses.wrapper()`` to ensure proper terminal setup and cleanup.
    The TUI has a two-panel layout with the model list on the left and
    details on the right.

    Args:
        models: List of model info dicts from ``OllamaClient.list_models()``.
        current_model: Name of the currently active model (highlighted
            and marked with ``*``).  Defaults to ``""``.

    Returns:
        The selected model name string, or ``None`` if the user cancels.

    Raises:
        curses.error: If the terminal is too small for the TUI.
    """
    return curses.wrapper(_curses_main, models, current_model)


# ---------------------------------------------------------------------------
# Public API — main routing function
# ---------------------------------------------------------------------------


def select_model_interactive(
    client: OllamaClient,
    current_model: str = "",
) -> str | None:
    """Open the interactive model selector and return the chosen model.

    Fetches available models from the Ollama server, then routes to the
    curses TUI selector (if the terminal supports it) or the simple
    numbered-list fallback.

    Args:
        client: An :class:`OllamaClient` instance for fetching models.
        current_model: Name of the currently active model.  Shown as
            highlighted/marked in the selector.  Defaults to ``""``.

    Returns:
        The selected model name string, or ``None`` if the user cancels,
        no models are available, or the Ollama server is unreachable.
    """
    # Fetch models from Ollama.
    try:
        models = client.list_models()
    except OllamaConnectionError:
        sys.stderr.write(
            "Error: Could not connect to Ollama. "
            "Is the server running? (ollama serve)\n"
        )
        return None

    # Handle empty model list.
    if not models:
        sys.stderr.write(
            "No models found. Pull a model with: ollama pull <model>\n"
        )
        return None

    # Route to the appropriate selector.
    use_curses = (
        _CURSES_AVAILABLE
        and sys.stdin.isatty()
    )

    if use_curses:
        try:
            return _select_model_curses(models, current_model)
        except curses.error:
            # Terminal too small or other curses issue; fall back.
            return _select_model_simple(models, current_model)

    return _select_model_simple(models, current_model)
