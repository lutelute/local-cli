"""Bash tool for executing shell commands.

Runs shell commands via ``subprocess.run()``, captures stdout and stderr,
and integrates with the security module to block dangerous commands and
sanitize the subprocess environment.
"""

import subprocess
from typing import Callable

from local_cli.security import (
    get_sanitized_env,
    is_command_dangerous,
    is_command_risky,
)
from local_cli.tools.base import Tool

# Maximum output size in bytes (100 KB).
_MAX_OUTPUT_BYTES = 100 * 1024

# Default command timeout in seconds.
_DEFAULT_TIMEOUT = 120


class BashTool(Tool):
    """Execute shell commands and return their output."""

    def __init__(
        self,
        confirm: Callable[[str], bool] | None = None,
    ) -> None:
        """Create the bash tool.

        Args:
            confirm: Optional callback invoked before running a *risky*
                command (recursive rm, sudo, force push, kill, ...).  It
                receives the command string and returns ``True`` to run it
                or ``False`` to decline.  When ``None`` (the default — and
                the case for sub-agents and GUI front-ends), risky commands
                run without a prompt; outright-dangerous commands are always
                blocked regardless of this callback.
        """
        self._confirm = confirm

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its stdout and stderr. "
            "Use this for running programs, installing packages, "
            "searching with find/grep, or any shell operation."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Maximum execution time in seconds. "
                        f"Defaults to {_DEFAULT_TIMEOUT}."
                    ),
                },
            },
            "required": ["command"],
        }

    def execute(self, **kwargs: object) -> str:
        """Execute a shell command and return combined stdout/stderr.

        Args:
            **kwargs: Must include ``command`` (str).  May include
                ``timeout`` (int, default 120).

        Returns:
            The combined stdout and stderr of the command, or an error
            message if the command was blocked, timed out, or failed.
        """
        command = kwargs.get("command")
        if not isinstance(command, str) or not command.strip():
            return "Error: 'command' parameter is required and must be a non-empty string."

        timeout = kwargs.get("timeout", _DEFAULT_TIMEOUT)
        if not isinstance(timeout, (int, float)):
            timeout = _DEFAULT_TIMEOUT
        timeout = int(timeout)

        # Security check: block dangerous commands outright.
        if is_command_dangerous(command):
            return f"Error: command blocked by security policy: {command}"

        # Risky-but-legitimate commands require confirmation when a confirm
        # callback is wired up (CLI without --yes).  Declining returns a
        # plain message the model can read and adapt to.
        if self._confirm is not None and is_command_risky(command):
            if not self._confirm(command):
                return f"Command declined by user (not run): {command}"

        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=get_sanitized_env(),
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout} seconds."
        except PermissionError as exc:
            return f"Error: permission denied: {exc}"
        except OSError as exc:
            return f"Error: failed to execute command: {exc}"

        # Combine stdout and stderr.
        output = result.stdout
        if result.stderr:
            output = output + result.stderr if output else result.stderr

        # Truncate if output exceeds the maximum size.
        if len(output.encode("utf-8", errors="replace")) > _MAX_OUTPUT_BYTES:
            truncated = output.encode("utf-8", errors="replace")[:_MAX_OUTPUT_BYTES]
            output = truncated.decode("utf-8", errors="replace")
            output += "\n... [output truncated at 100KB]"

        return output
