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


# Grow-on-demand tiers.  Measured (qwen3.5:9b, short task): 8k ran in
# ~25s, >=16k in ~44-58s — blanket-maxing to 32k roughly doubled the
# latency of the common short turn for a working-memory benefit that
# only a large conversation ever uses.  So we start at the floor and
# step up a tier only when the conversation actually approaches the
# current window.  Ollama reloads the model when num_ctx changes, but
# a bump happens at most twice per session (8k->16k->32k), and only
# when genuinely needed, so the reload cost is amortized while every
# short session stays fast.
_TIERS = (8192, 16384, 32768)

# Bump to the next tier once the estimated conversation reaches this
# fraction of the current window — early enough that the turn's own
# tool outputs and reply still fit before compaction kicks in.
_GROW_AT = 0.7


def resolve_num_ctx(
    client: Any,
    model: str,
    configured: int = 0,
    estimated_tokens: int = 0,
) -> int:
    """The context window to request this turn.

    ``configured > 0`` is an explicit user choice and wins unchanged.
    Otherwise the window grows on demand: the smallest tier that holds
    *estimated_tokens* with headroom, capped by the model's native
    window, the machine's RAM tier and the ceiling.  With no estimate
    (session start) this is the fast floor, so short sessions never pay
    the large-context latency tax.
    """
    if configured > 0:
        return configured
    max_ctx = model_max_context(client, model)
    if max_ctx <= 0:
        return _FLOOR
    cap = min(max_ctx, _ram_cap(_system_ram_gb()), _CEILING)
    # Candidate windows: the tiers up to the cap, plus the cap itself so
    # a model whose max falls between tiers (e.g. 20k) can still use its
    # full capacity when a large conversation needs it.
    candidates = [tier for tier in _TIERS if tier <= cap]
    if not candidates or candidates[-1] < cap:
        candidates.append(cap)
    # Smallest candidate whose _GROW_AT fraction still holds the
    # estimate; the largest if the conversation overflows them all.
    target = candidates[0]
    for candidate in candidates:
        target = candidate
        if estimated_tokens <= int(candidate * _GROW_AT):
            break
    return min(max(target, min(_FLOOR, max_ctx)), cap)
