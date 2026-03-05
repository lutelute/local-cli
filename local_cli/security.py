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

# Regex patterns for commands that should never be executed.
DANGEROUS_COMMANDS: list[str] = [
    r"rm\s+-rf\s+/",
    r"mkfs\.",
    r"dd\s+if=",
    r":\(\)\{\s*:\|:&\s*\};:",      # fork bomb
    r"curl.*\|\s*sh",
    r"wget.*\|\s*sh",
    r"chmod\s+-R\s+777\s+/",
    r">\s*/dev/sd",
]

# Compiled patterns for efficient repeated matching.
_DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in DANGEROUS_COMMANDS
]


def is_command_dangerous(cmd: str) -> bool:
    """Check whether a shell command matches any dangerous pattern.

    Args:
        cmd: The shell command string to check.

    Returns:
        True if the command matches a dangerous pattern, False otherwise.
    """
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
