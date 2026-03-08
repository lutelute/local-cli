"""Model-family-specific inference parameter presets for local-cli.

Provides optimal inference parameters (temperature, top_p, top_k, num_ctx,
etc.) per model family.  Parameters are looked up by prefix-matching the
model name to a known family.  Unknown models receive safe defaults.

The preset system sits between hard-coded defaults and user configuration
in the options merge chain:

    hard defaults < model presets < user config (file/env/CLI)
"""

from __future__ import annotations

from typing import Any

# ------------------------------------------------------------------
# Model family presets
# ------------------------------------------------------------------
# Each entry maps a model family name to an ``options`` dict that will
# be sent to the Ollama ``/api/chat`` endpoint inside the ``options``
# key.  Only Ollama-recognised option keys should appear here.

_PRESETS: dict[str, dict[str, Any]] = {
    "qwen3": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "num_ctx": 8192,
    },
    "qwen2.5": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "num_ctx": 8192,
    },
    "gemma": {
        "temperature": 0.7,
        "num_ctx": 8192,
    },
    "phi": {
        "temperature": 0.7,
        "num_ctx": 8192,
    },
    "llama": {
        "temperature": 0.7,
        "num_ctx": 8192,
    },
}

# Safe defaults applied when the model family is not recognised.
_DEFAULT_PRESET: dict[str, Any] = {
    "temperature": 0.7,
    "num_ctx": 8192,
}

# Minimum temperature for Qwen3 family.  Qwen3 degrades severely with
# temperature=0 (endless repetition), so we enforce a floor.
_QWEN3_MIN_TEMPERATURE: float = 0.1

# ------------------------------------------------------------------
# Thinking-mode support
# ------------------------------------------------------------------
# Set of model family names whose models support the ``think`` parameter
# (Qwen3's built-in reasoning / chain-of-thought mode).

SUPPORTS_THINKING: frozenset[str] = frozenset({"qwen3"})

# ------------------------------------------------------------------
# Ordered prefixes used by ``get_model_family`` for matching.
# Longer prefixes are checked first so that ``qwen2.5`` matches before
# a hypothetical ``qwen2`` entry.
# ------------------------------------------------------------------

_FAMILY_PREFIXES: list[str] = sorted(_PRESETS.keys(), key=len, reverse=True)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def get_model_family(model_name: str) -> str | None:
    """Detect the model family from a model name by prefix matching.

    The model name is lower-cased before comparison.  The longest
    matching prefix wins (e.g. ``"qwen2.5:7b-instruct"`` matches
    ``"qwen2.5"`` rather than a hypothetical ``"qwen2"``).

    Args:
        model_name: Full model name including optional tag
            (e.g. ``"qwen3:8b"``, ``"gemma3:4b"``).

    Returns:
        The family name string (e.g. ``"qwen3"``) if a known family
        matches, or ``None`` for unrecognised models.
    """
    name = model_name.lower()
    for prefix in _FAMILY_PREFIXES:
        if name.startswith(prefix):
            return prefix
    return None


def get_model_preset(model_name: str) -> dict[str, Any]:
    """Return the inference options preset for a model.

    Looks up the model family via :func:`get_model_family` and returns
    a **copy** of the corresponding preset dict.  If the model family
    is not recognised, returns safe defaults.

    For Qwen3 family models the temperature is clamped to a minimum of
    ``0.1`` to prevent the degradation that occurs at temperature=0.

    Args:
        model_name: Full model name including optional tag
            (e.g. ``"qwen3:8b"``).

    Returns:
        A dict of Ollama inference options (e.g.
        ``{"temperature": 0.6, "top_p": 0.95, ...}``).  The caller
        receives a fresh copy and may mutate it freely.
    """
    family = get_model_family(model_name)

    if family is None:
        return dict(_DEFAULT_PRESET)

    preset = dict(_PRESETS[family])

    # Enforce minimum temperature for Qwen3 family.
    if family == "qwen3":
        temp = preset.get("temperature", _DEFAULT_PRESET["temperature"])
        if temp < _QWEN3_MIN_TEMPERATURE:
            preset["temperature"] = _QWEN3_MIN_TEMPERATURE

    return preset
