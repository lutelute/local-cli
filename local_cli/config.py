"""Configuration management for local-cli.

Priority: CLI args > env vars > config file > defaults.
"""

import os
from pathlib import Path

CONFIG_DEFAULTS: dict[str, object] = {
    "model": "qwen3.5:9b-q4_K_M",
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
    "plan_dir": ".agents/plans",
    "knowledge_dir": ".agents/knowledge",
    "skills_dir": ".agents/skills",
    "default_mode": "agent",
    "num_ctx": 8192,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "think_mode": False,
    "keep_alive": None,
    "llama_server_url": "http://localhost:8090",
}

# Mapping of environment variable names to config keys.
ENV_VAR_MAP: dict[str, str] = {
    "LOCAL_CLI_MODEL": "model",
    "LOCAL_CLI_SIDECAR_MODEL": "sidecar_model",
    "LOCAL_CLI_DEBUG": "debug",
    "LOCAL_CLI_PROVIDER": "provider",
    "OLLAMA_HOST": "ollama_host",
    "LOCAL_CLI_NUM_CTX": "num_ctx",
    "LOCAL_CLI_TEMPERATURE": "temperature",
    "LOCAL_CLI_TOP_P": "top_p",
    "LOCAL_CLI_TOP_K": "top_k",
    "LOCAL_CLI_THINK_MODE": "think_mode",
    "LOCAL_CLI_KEEP_ALIVE": "keep_alive",
    "LLAMA_SERVER_URL": "llama_server_url",
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


def _parse_optional_float(value: object) -> float | None:
    """Parse a value as an optional float.

    Returns ``None`` when the value is ``None`` or an empty string.
    """
    if value is None:
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return float(value)
    return float(value)


def _parse_optional_int(value: object) -> int | None:
    """Parse a value as an optional int.

    Returns ``None`` when the value is ``None`` or an empty string.
    """
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return int(value)
    return int(value)


def _parse_optional_str(value: object) -> str | None:
    """Parse a value as an optional string.

    Returns ``None`` when the value is ``None`` or an empty string.
    """
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


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
        self.plan_dir: str = str(merged["plan_dir"])
        self.knowledge_dir: str = str(merged["knowledge_dir"])
        self.skills_dir: str = str(merged["skills_dir"])
        self.default_mode: str = str(merged["default_mode"])
        self.num_ctx: int = int(merged["num_ctx"])  # type: ignore[arg-type]
        self.temperature: float | None = _parse_optional_float(merged["temperature"])
        self.top_p: float | None = _parse_optional_float(merged["top_p"])
        self.top_k: int | None = _parse_optional_int(merged["top_k"])
        self.think_mode: bool = _parse_bool(merged["think_mode"])
        self.keep_alive: str | None = _parse_optional_str(merged["keep_alive"])
        self.llama_server_url: str = str(merged["llama_server_url"]).rstrip("/")

    @property
    def has_claude_access(self) -> bool:
        """Check whether an Anthropic API key is available.

        Reads ``ANTHROPIC_API_KEY`` from the environment at call time and
        never stores the key value.  Returns ``True`` only when the
        variable is set to a non-empty string.
        """
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
