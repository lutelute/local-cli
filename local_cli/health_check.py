"""Startup health check module for local-cli.

Validates the runtime environment before launching the interactive REPL:

- **Ollama connectivity** — HTTP GET ``/api/version`` to confirm the server
  is reachable.
- **Model availability** — HTTP GET ``/api/tags`` to verify the requested
  model is present on the Ollama server.
- **Disk space** — :func:`shutil.disk_usage` to ensure at least 1 GB of free
  space is available.

The health check is **non-blocking**: it prints color-coded status lines
(green ✓, yellow ⚠, red ✗) but never prevents startup.  Results are returned
as structured :class:`CheckResult` objects for programmatic inspection.
"""

import shutil
from typing import Any

from local_cli.ollama_client import OllamaClient, OllamaConnectionError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum free disk space in bytes (1 GB).
_MIN_FREE_DISK_BYTES: int = 1_073_741_824  # 1 * 1024 * 1024 * 1024

# Default disk path to check (root filesystem).
_DEFAULT_DISK_PATH: str = "/"

# ---------------------------------------------------------------------------
# Status codes
# ---------------------------------------------------------------------------

STATUS_OK: str = "ok"
STATUS_WARNING: str = "warning"
STATUS_ERROR: str = "error"

# ---------------------------------------------------------------------------
# ANSI color codes
# ---------------------------------------------------------------------------

_GREEN: str = "\033[32m"
_YELLOW: str = "\033[33m"
_RED: str = "\033[31m"
_RESET: str = "\033[0m"

_STATUS_COLORS: dict[str, str] = {
    STATUS_OK: _GREEN,
    STATUS_WARNING: _YELLOW,
    STATUS_ERROR: _RED,
}

_STATUS_SYMBOLS: dict[str, str] = {
    STATUS_OK: "✓",
    STATUS_WARNING: "⚠",
    STATUS_ERROR: "✗",
}


# ---------------------------------------------------------------------------
# Check result
# ---------------------------------------------------------------------------


class CheckResult:
    """Structured result from a single health check.

    Attributes:
        name: Short human-readable name of the check (e.g. ``"Ollama"``).
        status: One of :data:`STATUS_OK`, :data:`STATUS_WARNING`, or
            :data:`STATUS_ERROR`.
        message: Descriptive message explaining the status.
        details: Optional dictionary with additional data (e.g. version info,
            available models, disk usage bytes).
    """

    __slots__ = ("name", "status", "message", "details")

    def __init__(
        self,
        name: str,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.name: str = name
        self.status: str = status
        self.message: str = message
        self.details: dict[str, Any] | None = details

    def __repr__(self) -> str:
        return (
            f"CheckResult(name={self.name!r}, status={self.status!r}, "
            f"message={self.message!r})"
        )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_ollama_connectivity(client: OllamaClient) -> CheckResult:
    """Check whether the Ollama server is reachable.

    Sends ``GET /api/version`` via the provided :class:`OllamaClient`.

    Args:
        client: An :class:`OllamaClient` instance configured with the
            target Ollama host.

    Returns:
        A :class:`CheckResult` with status ``ok`` (server reachable) or
        ``error`` (connection failed).
    """
    try:
        version_info = client.get_version()
        version = version_info.get("version", "unknown")
        return CheckResult(
            name="Ollama",
            status=STATUS_OK,
            message=f"Connected (v{version})",
            details={"version": version},
        )
    except OllamaConnectionError as exc:
        return CheckResult(
            name="Ollama",
            status=STATUS_ERROR,
            message=f"Cannot connect to Ollama: {exc}",
            details={"error": str(exc)},
        )
    except Exception as exc:
        return CheckResult(
            name="Ollama",
            status=STATUS_ERROR,
            message=f"Unexpected error checking Ollama: {exc}",
            details={"error": str(exc)},
        )


def check_model_availability(
    client: OllamaClient,
    model: str,
) -> CheckResult:
    """Check whether a specific model is available on the Ollama server.

    Sends ``GET /api/tags`` via the provided :class:`OllamaClient` and
    searches the response for the requested model name.

    Args:
        client: An :class:`OllamaClient` instance.
        model: The model name to look for (e.g. ``"qwen3:8b"``).

    Returns:
        A :class:`CheckResult` with status ``ok`` (model found),
        ``warning`` (model not found but server reachable), or
        ``error`` (connection failed).
    """
    try:
        models = client.list_models()
        model_names = [m.get("name", "") for m in models]

        # Ollama model names may include a tag suffix (e.g. ":latest").
        # Match if the configured model equals the full name or the
        # base name without tag.
        model_found = any(
            model == name or model == name.split(":")[0]
            for name in model_names
        )

        if model_found:
            return CheckResult(
                name="Model",
                status=STATUS_OK,
                message=f"Model '{model}' available",
                details={"model": model, "available_models": model_names},
            )
        else:
            available = ", ".join(model_names) if model_names else "none"
            return CheckResult(
                name="Model",
                status=STATUS_WARNING,
                message=(
                    f"Model '{model}' not found. "
                    f"Available: {available}"
                ),
                details={"model": model, "available_models": model_names},
            )
    except OllamaConnectionError as exc:
        return CheckResult(
            name="Model",
            status=STATUS_ERROR,
            message=f"Cannot check models: {exc}",
            details={"error": str(exc)},
        )
    except Exception as exc:
        return CheckResult(
            name="Model",
            status=STATUS_ERROR,
            message=f"Unexpected error checking models: {exc}",
            details={"error": str(exc)},
        )


def check_disk_space(
    path: str = _DEFAULT_DISK_PATH,
    min_free_bytes: int = _MIN_FREE_DISK_BYTES,
) -> CheckResult:
    """Check whether sufficient disk space is available.

    Uses :func:`shutil.disk_usage` to query the filesystem containing
    *path*.

    Args:
        path: Filesystem path to check (defaults to ``/``).
        min_free_bytes: Minimum required free space in bytes.
            Defaults to 1 GB.

    Returns:
        A :class:`CheckResult` with status ``ok`` (enough space),
        ``warning`` (below threshold), or ``error`` (check failed).
    """
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)

        if usage.free >= min_free_bytes:
            return CheckResult(
                name="Disk space",
                status=STATUS_OK,
                message=f"{free_gb:.1f} GB free",
                details={
                    "free_bytes": usage.free,
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                },
            )
        else:
            return CheckResult(
                name="Disk space",
                status=STATUS_WARNING,
                message=(
                    f"Low disk space: {free_gb:.1f} GB free "
                    f"(recommended: ≥{min_free_bytes / (1024 ** 3):.0f} GB)"
                ),
                details={
                    "free_bytes": usage.free,
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                },
            )
    except OSError as exc:
        return CheckResult(
            name="Disk space",
            status=STATUS_ERROR,
            message=f"Cannot check disk space: {exc}",
            details={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Aggregated health check
# ---------------------------------------------------------------------------


def run_health_check(
    client: OllamaClient,
    model: str,
    disk_path: str = _DEFAULT_DISK_PATH,
) -> list[CheckResult]:
    """Run all startup health checks and return structured results.

    Executes the following checks in order:

    1. Ollama connectivity
    2. Model availability (skipped if Ollama is unreachable)
    3. Disk space

    Args:
        client: An :class:`OllamaClient` instance.
        model: The model name to check availability for.
        disk_path: Filesystem path for disk space check.

    Returns:
        A list of :class:`CheckResult` objects, one per check.
    """
    results: list[CheckResult] = []

    # 1. Ollama connectivity.
    ollama_result = check_ollama_connectivity(client)
    results.append(ollama_result)

    # 2. Model availability — only if Ollama is reachable.
    if ollama_result.status == STATUS_OK:
        model_result = check_model_availability(client, model)
        results.append(model_result)
    else:
        results.append(CheckResult(
            name="Model",
            status=STATUS_WARNING,
            message=f"Skipped (Ollama not reachable)",
        ))

    # 3. Disk space.
    disk_result = check_disk_space(path=disk_path)
    results.append(disk_result)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_check_result(result: CheckResult, color: bool = True) -> str:
    """Format a single check result as a color-coded status line.

    Args:
        result: The :class:`CheckResult` to format.
        color: Whether to include ANSI color codes.  Set to ``False``
            for plain-text output (e.g. logging).

    Returns:
        A formatted string like ``"  ✓ Ollama: Connected (v0.5.1)"``.
    """
    symbol = _STATUS_SYMBOLS.get(result.status, "?")

    if color:
        clr = _STATUS_COLORS.get(result.status, "")
        return f"  {clr}{symbol}{_RESET} {result.name}: {result.message}"
    else:
        return f"  {symbol} {result.name}: {result.message}"


def format_health_check(
    results: list[CheckResult],
    color: bool = True,
) -> str:
    """Format all health check results as a multi-line status block.

    Args:
        results: List of :class:`CheckResult` objects from
            :func:`run_health_check`.
        color: Whether to include ANSI color codes.

    Returns:
        A multi-line string suitable for printing to the terminal.
    """
    lines: list[str] = []
    for result in results:
        lines.append(format_check_result(result, color=color))
    return "\n".join(lines)
