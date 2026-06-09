"""Security utilities for local-cli.

Provides dangerous command detection, environment variable sanitization,
Ollama host validation, and model name validation.
"""

import os
import re
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Dangerous command patterns
# ---------------------------------------------------------------------------

# Regex patterns for commands that should never be executed, regardless of
# their arguments.  Intentionally broad-but-targeted: a blocklist can never
# be exhaustive (the operation-approval flow is the real backstop), but it
# should catch the common catastrophic one-liners without flagging everyday
# commands.
DANGEROUS_COMMANDS: list[str] = [
    r"\bmkfs\b",                                       # format a filesystem
    r"\bdd\b[^|;&\n]*\bof=/dev/",                      # dd writing to a device
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",       # fork bomb
    r"\bcurl\b[^|;&\n]*\|\s*(?:sudo\s+)?(?:ba)?sh\b",  # curl ... | sh
    r"\bwget\b[^|;&\n]*\|\s*(?:sudo\s+)?(?:ba)?sh\b",  # wget ... | sh
    r"\bchmod\s+-R\s+0?777\s+/",                       # recursive 777 on root
    r">\s*/dev/(?:sd[a-z]|nvme\d|disk\d)",             # overwrite a block device
    r"--no-preserve-root",                             # explicit root-wipe intent
    r"\bchown\s+-R\s+[^\s]+\s+/\s*(?:$|;|&|\|)",       # recursive chown of root
]

# Compiled patterns for efficient repeated matching.
_DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in DANGEROUS_COMMANDS
]

# ``rm`` needs more than a literal pattern: the recursive (-r) and force (-f)
# flags can appear in any order, combined (``-rf`` / ``-fr``) or separately,
# and the catastrophic part is the *target*.  This matches an ``rm`` that has
# both an ``r`` flag and an ``f`` flag (lookaheads, confined to one command
# segment via ``[^|;&]``) whose target is a root-level location: ``/``, a
# top-level dir like ``/etc``, ``~``, ``$HOME``, or ``/*``.  Deeper absolute
# paths (e.g. ``/home/me/project/build``) are treated as intentional and left
# to the approval flow rather than hard-blocked.
_DANGEROUS_RM_RE: re.Pattern[str] = re.compile(
    r"\brm\b"
    r"(?=[^|;&\n]*\s-[a-zA-Z]*r)"               # a flag containing 'r'
    r"(?=[^|;&\n]*\s-[a-zA-Z]*f)"               # a flag containing 'f'
    r"[^|;&\n]*\s"
    r"(?:/[a-zA-Z]*|~|\$HOME|\$\{HOME\}|/\*)"   # root-level target
    r"(?:\s|/\*|$|;|&|\|)"
)


def is_command_dangerous(cmd: str) -> bool:
    """Check whether a shell command matches a known-dangerous pattern.

    Covers a curated set of catastrophic operations (filesystem formatting,
    device overwrites, fork bombs, piping a download into a shell, recursive
    ``rm`` / ``chmod`` / ``chown`` on root locations).  This is a best-effort
    blocklist, not a sandbox: it errs toward the obvious one-liners and
    relies on the operation-approval flow as the real guard.

    Args:
        cmd: The shell command string to check.

    Returns:
        True if the command matches a dangerous pattern, False otherwise.
    """
    if _DANGEROUS_RM_RE.search(cmd):
        return True
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(cmd):
            return True
    return False


# ---------------------------------------------------------------------------
# Environment variable sanitization
# ---------------------------------------------------------------------------

# Environment variables that must be stripped from subprocess environments
# to prevent accidental leakage of secrets.
SANITIZED_ENV_VARS: list[str] = [
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SESSION_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DATABASE_URL",
    "SECRET_KEY",
    "DJANGO_SECRET_KEY",
    "ENCRYPTION_KEY",
    "PRIVATE_KEY",
    "STRIPE_SECRET_KEY",
    "SENDGRID_API_KEY",
    "TWILIO_AUTH_TOKEN",
    "SLACK_TOKEN",
    "SLACK_BOT_TOKEN",
    "NPM_TOKEN",
    "PYPI_TOKEN",
]


def get_sanitized_env() -> dict[str, str]:
    """Return a copy of the current environment with sensitive keys removed.

    Returns:
        A new dictionary based on ``os.environ`` with all keys listed in
        ``SANITIZED_ENV_VARS`` stripped out.
    """
    env = dict(os.environ)
    for key in SANITIZED_ENV_VARS:
        env.pop(key, None)
    return env


# ---------------------------------------------------------------------------
# Ollama host validation
# ---------------------------------------------------------------------------

# Hostnames / IPs that are considered localhost.
_LOCALHOST_HOSTS: set[str] = {
    "localhost",
    "127.0.0.1",
    "::1",
    "[::1]",
    "0.0.0.0",
}


def validate_ollama_host(url: str) -> bool:
    """Validate that an Ollama host URL points to localhost.

    Rejects URLs that:
    - Contain an ``@`` symbol (possible credential injection).
    - Resolve to a non-localhost address.
    - Cannot be parsed as a valid URL.

    Args:
        url: The Ollama host URL to validate.

    Returns:
        True if the URL is a valid localhost URL, False otherwise.
    """
    if not url:
        return False

    # Reject URLs with @ symbol (credential injection / redirect risk).
    if "@" in url:
        return False

    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    # Must have a scheme (http/https).
    if parsed.scheme not in ("http", "https"):
        return False

    # Extract hostname (strip brackets for IPv6).
    hostname = parsed.hostname
    if hostname is None:
        return False

    # Check against known localhost addresses.
    return hostname in _LOCALHOST_HOSTS


# ---------------------------------------------------------------------------
# Model name validation
# ---------------------------------------------------------------------------

# Model names may contain alphanumeric characters, hyphens, underscores,
# dots, colons (for tags like qwen3:8b), and forward slashes (for
# namespaced models like library/model).
_MODEL_NAME_RE: re.Pattern[str] = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._:/-]*$"
)

# Maximum allowed length for a model name.
_MAX_MODEL_NAME_LENGTH = 256


def validate_model_name(name: str) -> bool:
    """Validate a model name to prevent shell injection.

    Accepts names like ``qwen3:8b``, ``all-minilm``, ``library/model:latest``.

    Args:
        name: The model name to validate.

    Returns:
        True if the name is valid, False otherwise.
    """
    if not name:
        return False

    if len(name) > _MAX_MODEL_NAME_LENGTH:
        return False

    return _MODEL_NAME_RE.match(name) is not None
