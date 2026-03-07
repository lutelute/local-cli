"""Configuration management for local-cli.

Priority: CLI args > env vars > config file > defaults.
"""

import os
from pathlib import Path

CONFIG_DEFAULTS: dict[str, object] = {
    "model": "qwen3:8b",
    "sidecar_model": "",
    "ollama_host": "http://localhost:11434",
    "state_dir": "~/.local/state/local-cli",
    "config_file": "~/.config/local-cli/config",
    "auto_approve": False,
    "debug": False,
    "rag": False,
    "rag_path": ".",
    "rag_topk": 5,
    "rag_model": "all-minilm",
    "provider": "ollama",
    "model_registry_file": "",
    "orchestrator_model": "",
}

# Mapping of environment variable names to config keys.
ENV_VAR_MAP: dict[str, str] = {
    "LOCAL_CLI_MODEL": "model",
    "LOCAL_CLI_SIDECAR_MODEL": "sidecar_model",
    "LOCAL_CLI_DEBUG": "debug",
    "LOCAL_CLI_PROVIDER": "provider",
    "OLLAMA_HOST": "ollama_host",
}

# Maximum config file size in bytes (10KB).
_MAX_CONFIG_SIZE = 10 * 1024


def load_config_file(path: str) -> dict[str, str]:
    """Load a key=value config file.

    Security: No eval/source, max 10KB, no symlinks.

    Args:
        path: Path to the config file.

    Returns:
        Dictionary of parsed key-value pairs.
    """
    config_path = Path(path).expanduser()

    if not config_path.exists():
        return {}

    # Reject symlinks to prevent config file manipulation.
    if config_path.is_symlink():
        return {}

    # Reject files that are not regular files.
    if not config_path.is_file():
        return {}

    # Reject oversized config files.
    try:
        file_size = config_path.stat().st_size
    except OSError:
        return {}

    if file_size > _MAX_CONFIG_SIZE:
        return {}

    result: dict[str, str] = {}
    try:
        text = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}

    for line in text.splitlines():
        line = line.strip()
        # Skip empty lines and comments.
        if not line or line.startswith("#"):
            continue
        # Parse key=value (split on first '=' only).
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key:
            result[key] = value

    return result


def _parse_bool(value: object) -> bool:
    """Parse a value as a boolean.

    Accepts bool, str ("1", "true", "yes"), or any truthy value.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes")
    return bool(value)


class Config:
    """Application configuration with layered loading.

    Priority: CLI args > env vars > config file > defaults.
    """

    def __init__(
        self,
        cli_args: object | None = None,
        config_file: str | None = None,
    ) -> None:
        """Initialize configuration.

        Args:
            cli_args: Namespace object from argparse (or any object with
                attributes matching config keys). ``None`` values on the
                namespace are treated as "not set".
            config_file: Override path for the config file. If ``None``,
                uses the default path from CONFIG_DEFAULTS.
        """
        # Start with defaults.
        merged: dict[str, object] = dict(CONFIG_DEFAULTS)

        # Layer 1: config file.
        cfg_path = config_file if config_file is not None else str(merged["config_file"])
        file_values = load_config_file(cfg_path)
        for key, value in file_values.items():
            if key in merged:
                merged[key] = value

        # Layer 2: environment variables.
        for env_var, config_key in ENV_VAR_MAP.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                merged[config_key] = env_value

        # Layer 3: CLI args (highest priority).
        if cli_args is not None:
            args_dict = vars(cli_args) if hasattr(cli_args, "__dict__") else {}
            for key, value in args_dict.items():
                # Only override if the CLI arg was explicitly provided
                # (argparse sets unset args to None).
                if value is not None and key in merged:
                    merged[key] = value

        # Expand paths and convert types.
        self.model: str = str(merged["model"])
        self.sidecar_model: str = str(merged["sidecar_model"])
        self.ollama_host: str = str(merged["ollama_host"]).rstrip("/")
        self.state_dir: str = str(Path(str(merged["state_dir"])).expanduser())
        self.config_file: str = str(Path(cfg_path).expanduser())
        self.auto_approve: bool = _parse_bool(merged["auto_approve"])
        self.debug: bool = _parse_bool(merged["debug"])
        self.rag: bool = _parse_bool(merged["rag"])
        self.rag_path: str = str(merged["rag_path"])
        self.rag_topk: int = int(merged["rag_topk"])
        self.rag_model: str = str(merged["rag_model"])
        self.provider: str = str(merged["provider"])
        self.model_registry_file: str = str(merged["model_registry_file"])
        self.orchestrator_model: str = str(merged["orchestrator_model"])
