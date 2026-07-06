"""Adaptive context sizing — stop running 256k models at 8k.

The historical default (``num_ctx: 8192``) was written when small
models shipped small windows.  Today qwen3.5:9b reports a native
context of 262,144 — the old default used 3% of the model's working
memory, and "working memory" is exactly where a local agent hurts most
(multi-file tasks, long tool outputs, compaction churn).

Policy, in order:

1. An explicit user value (config file, LOCAL_CLI_NUM_CTX, flag) wins.
2. Otherwise: ``min(model's native context, RAM tier, 32768)``,
   never below the historical 8192.

The RAM tier keeps the KV cache from swamping small machines:
>= 32 GB -> 32k, >= 16 GB -> 16k, below/unknown -> 8k.  The 32k
ceiling is deliberate even on big machines — prefill latency grows
with the window, and past 32k the wins for interactive agent turns
taper off while the memory bill keeps growing.

Model capability comes from Ollama's ``/api/show`` (the
``<arch>.context_length`` key) and is cached per model name; when the
lookup fails the resolver falls back to the historical 8192, so this
can never make things worse than before.
"""

import subprocess
import sys
from typing import Any

_FLOOR = 8192
_CEILING = 32768

# Model max-context cache: /api/show is local and fast, but the
# resolver runs every turn and failure noise should not repeat.
_max_ctx_cache: dict[str, int] = {}


def _system_ram_gb() -> float:
    """Total physical RAM in GB (0.0 when undetectable)."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            return int(out.stdout.strip()) / 1e9
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except Exception:
        pass
    return 0.0


def _ram_cap(ram_gb: float) -> int:
    if ram_gb >= 32:
        return 32768
    if ram_gb >= 16:
        return 16384
    return _FLOOR


def model_max_context(client: Any, model: str) -> int:
    """The model's native context length (0 when unknown).

    *client* needs a ``show_model(name) -> dict`` method (OllamaClient).
    """
    if model in _max_ctx_cache:
        return _max_ctx_cache[model]
    max_ctx = 0
    try:
        info = client.show_model(model)
        model_info = info.get("model_info") or {}
        for key, value in model_info.items():
            if str(key).endswith(".context_length"):
                max_ctx = int(value)
                break
    except Exception:
        return 0  # do not cache failures — Ollama may come back
    _max_ctx_cache[model] = max_ctx
    return max_ctx


def resolve_num_ctx(client: Any, model: str, configured: int = 0) -> int:
    """The context window to request this turn.

    ``configured > 0`` is an explicit user choice and wins unchanged.
    ``configured <= 0`` means auto.
    """
    if configured > 0:
        return configured
    max_ctx = model_max_context(client, model)
    if max_ctx <= 0:
        return _FLOOR
    target = min(max_ctx, _ram_cap(_system_ram_gb()), _CEILING)
    # Never request more than the model's native window (a 4k model
    # gets 4k, not the floor), never less than the floor otherwise.
    return max(target, min(_FLOOR, max_ctx))
