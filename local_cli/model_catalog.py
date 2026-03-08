"""Curated model catalog with categories and metadata.

Provides a catalog of recommended Ollama models organized by category,
with size estimates, descriptions, and capability tags. Used by the
desktop GUI model picker to show available models for download.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CatalogEntry:
    """A model entry in the catalog."""
    name: str
    display: str
    category: str
    params: str
    size_gb: float
    description: str
    tags: list[str]


# Curated catalog of popular/recommended models.
CATALOG: list[CatalogEntry] = [
    # --- Code ---
    CatalogEntry("qwen2.5-coder:7b", "Qwen 2.5 Coder 7B", "Code", "7B", 4.7,
                 "Alibaba code-specialized model. Strong at generation and completion.", ["code", "tools"]),
    CatalogEntry("qwen2.5-coder:14b", "Qwen 2.5 Coder 14B", "Code", "14B", 9.0,
                 "Larger code model with better reasoning.", ["code", "tools"]),
    CatalogEntry("qwen2.5-coder:32b", "Qwen 2.5 Coder 32B", "Code", "32B", 20.0,
                 "Top-tier code model. Near GPT-4 level coding.", ["code", "tools"]),
    CatalogEntry("deepseek-coder-v2:16b", "DeepSeek Coder V2 16B", "Code", "16B", 8.9,
                 "MoE code model, strong at multi-file editing.", ["code", "tools"]),
    CatalogEntry("codellama:7b", "Code Llama 7B", "Code", "7B", 3.8,
                 "Meta's code-focused Llama. Good baseline.", ["code"]),
    CatalogEntry("codellama:13b", "Code Llama 13B", "Code", "13B", 7.4,
                 "Larger Code Llama with better accuracy.", ["code"]),
    CatalogEntry("starcoder2:7b", "StarCoder2 7B", "Code", "7B", 4.0,
                 "BigCode open-source code model.", ["code"]),

    # --- General / Chat ---
    CatalogEntry("qwen3:8b", "Qwen 3 8B", "General", "8B", 5.2,
                 "Latest Qwen. Strong reasoning with tool calling. Japanese OK.", ["general", "tools", "reasoning", "japanese"]),
    CatalogEntry("qwen3:4b", "Qwen 3 4B", "General", "4B", 2.6,
                 "Compact Qwen 3 for lighter hardware.", ["general", "tools"]),
    CatalogEntry("qwen3:14b", "Qwen 3 14B", "General", "14B", 9.0,
                 "Balanced size and quality.", ["general", "tools", "reasoning"]),
    CatalogEntry("qwen3:30b", "Qwen 3 30B", "General", "30B", 18.5,
                 "High-quality reasoning, recommended for agents.", ["general", "tools", "reasoning"]),
    CatalogEntry("llama3.2:3b", "Llama 3.2 3B", "General", "3B", 2.0,
                 "Meta's compact model. Fast and decent.", ["general"]),
    CatalogEntry("llama3.1:8b", "Llama 3.1 8B", "General", "8B", 4.7,
                 "Solid all-around model from Meta.", ["general", "tools"]),
    CatalogEntry("llama3.3:70b", "Llama 3.3 70B", "General", "70B", 43.0,
                 "Flagship Meta model. Requires 48GB+ VRAM.", ["general", "tools", "reasoning"]),
    CatalogEntry("gemma3:4b", "Gemma 3 4B", "General", "4B", 3.3,
                 "Google's compact model. Good for simple tasks.", ["general"]),
    CatalogEntry("gemma3:12b", "Gemma 3 12B", "General", "12B", 8.1,
                 "Larger Gemma with better accuracy. Japanese OK.", ["general", "tools", "japanese"]),
    CatalogEntry("gemma3:27b", "Gemma 3 27B", "General", "27B", 17.0,
                 "Google's largest open model. Japanese OK.", ["general", "tools", "reasoning", "japanese"]),
    CatalogEntry("mistral:7b", "Mistral 7B", "General", "7B", 4.1,
                 "Efficient European model with sliding window attention.", ["general"]),
    CatalogEntry("phi4:14b", "Phi-4 14B", "General", "14B", 9.1,
                 "Microsoft's reasoning-focused model.", ["general", "reasoning"]),

    # --- Small / Edge ---
    CatalogEntry("qwen3:0.6b", "Qwen 3 0.6B", "Small", "0.6B", 0.5,
                 "Ultra-compact. Good for testing and drafts.", ["general"]),
    CatalogEntry("qwen3:1.7b", "Qwen 3 1.7B", "Small", "1.7B", 1.1,
                 "Small but capable for basic tasks.", ["general", "tools"]),
    CatalogEntry("phi4-mini:3.8b", "Phi-4 Mini", "Small", "3.8B", 2.5,
                 "Microsoft's small reasoning model.", ["general", "reasoning"]),
    CatalogEntry("gemma3:1b", "Gemma 3 1B", "Small", "1B", 1.0,
                 "Google's tiniest model. Very fast.", ["general"]),
    CatalogEntry("llama3.2:1b", "Llama 3.2 1B", "Small", "1B", 0.7,
                 "Meta's smallest. Quick responses.", ["general"]),

    # --- Reasoning ---
    CatalogEntry("deepseek-r1:7b", "DeepSeek R1 7B", "Reasoning", "7B", 4.7,
                 "Chain-of-thought reasoning distilled model.", ["reasoning"]),
    CatalogEntry("deepseek-r1:14b", "DeepSeek R1 14B", "Reasoning", "14B", 9.0,
                 "Stronger reasoning with detailed thinking.", ["reasoning"]),
    CatalogEntry("deepseek-r1:32b", "DeepSeek R1 32B", "Reasoning", "32B", 20.0,
                 "Deep reasoning. Good for complex problems.", ["reasoning"]),
    CatalogEntry("qwq:32b", "QwQ 32B", "Reasoning", "32B", 20.0,
                 "Qwen's dedicated reasoning model.", ["reasoning"]),

    # --- Multilingual ---
    CatalogEntry("aya-expanse:8b", "Aya Expanse 8B", "Multilingual", "8B", 4.8,
                 "Cohere's multilingual model. 23 languages.", ["multilingual"]),
    CatalogEntry("aya-expanse:32b", "Aya Expanse 32B", "Multilingual", "32B", 19.0,
                 "Larger multilingual model with better quality.", ["multilingual"]),

    # --- Japanese ---
    CatalogEntry("llm-jp:13b", "LLM-jp 13B", "Japanese", "13B", 7.4,
                 "National Institute of Informatics (NII). Japanese-focused LLM.", ["japanese"]),
    CatalogEntry("hf.co/elyza/Llama-3-ELYZA-JP-8B-GGUF", "ELYZA JP 8B", "Japanese", "8B", 4.7,
                 "ELYZA's Japanese Llama 3. Strong at Japanese NLP tasks.", ["japanese"]),
    CatalogEntry("hf.co/LiquidAI/LFM2.5-1.2B-JP-GGUF", "LFM 2.5 1.2B JP", "Japanese", "1.2B", 0.8,
                 "Liquid AI's Japanese-specialized compact model.", ["japanese"]),
    CatalogEntry("hf.co/mmnga/cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese-gguf", "CyberAgent DeepSeek R1 14B JP", "Japanese", "14B", 9.0,
                 "CyberAgent's Japanese DeepSeek R1 distillation. Reasoning in Japanese.", ["japanese", "reasoning"]),
    CatalogEntry("hf.co/mmnga/Fugaku-LLM-13B-instruct-gguf", "Fugaku LLM 13B", "Japanese", "13B", 7.4,
                 "Fugaku supercomputer project. Japanese instruction-tuned.", ["japanese"]),
    CatalogEntry("hf.co/dahara1/gemma-2-2b-jpn-it-gguf", "Gemma 2 2B JPN IT", "Japanese", "2B", 1.6,
                 "Google Gemma 2 Japanese instruction-tuned. Compact.", ["japanese"]),
]

CATEGORIES = ["Code", "General", "Small", "Reasoning", "Japanese", "Multilingual"]


def get_catalog() -> list[dict]:
    """Return the catalog as a list of dicts."""
    return [
        {
            "name": e.name,
            "display": e.display,
            "category": e.category,
            "params": e.params,
            "size_gb": e.size_gb,
            "description": e.description,
            "tags": e.tags,
        }
        for e in CATALOG
    ]


def get_categories() -> list[str]:
    """Return ordered category list."""
    return list(CATEGORIES)


# ---------------------------------------------------------------------------
# Persistent catalog cache (updated from ollama.com)
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "local-cli"
_CACHE_FILE = _CACHE_DIR / "model_catalog.json"
_CACHE_MAX_AGE = 24 * 60 * 60  # 24 hours


def _classify_model(name: str, tags: list[str], desc: str) -> str:
    """Infer a category from model name, tags, and description."""
    lower_name = name.lower()
    lower_desc = desc.lower()

    if any(k in lower_name for k in ("coder", "code", "starcoder", "deepseek-coder")):
        return "Code"
    if "code" in tags:
        return "Code"

    if "thinking" in tags or any(k in lower_name for k in ("r1", "qwq", "o1")):
        return "Reasoning"

    if any(k in lower_name for k in ("embed", "minilm", "bge")):
        return "Embedding"

    if any(k in lower_name for k in ("elyza", "llm-jp", "japanese", "-jp-", "-jp:")):
        return "Japanese"
    if "japanese" in tags:
        return "Japanese"

    if any(k in lower_name for k in ("aya", "translate")):
        return "Multilingual"

    # Size-based classification from name.
    import re
    size_m = re.search(r"(\d+\.?\d*)[bB]", lower_name)
    if size_m:
        param_b = float(size_m.group(1))
        if param_b <= 3:
            return "Small"

    if any(k in lower_desc for k in ("small", "compact", "tiny", "edge", "on-device")):
        return "Small"

    return "General"


def _infer_params(name: str, sizes: list[str]) -> str:
    """Extract parameter count from model name or sizes."""
    import re
    m = re.search(r"[:\-](\d+\.?\d*[bB])", name)
    if m:
        return m.group(1).upper()
    if sizes:
        return sizes[0].upper()
    return ""


def _infer_size_gb(params: str) -> float:
    """Rough estimate of download size from param count."""
    import re
    m = re.match(r"(\d+\.?\d*)", params)
    if not m:
        return 0.0
    val = float(m.group(1))
    # ~0.6 GB per billion params (Q4 quantized).
    return round(val * 0.65, 1)


def update_catalog() -> dict:
    """Fetch latest models from ollama.com and merge into catalog cache.

    Fetches popular, hot, and tools models, deduplicates, classifies,
    and saves to a local cache file.

    Returns:
        Dict with 'added', 'updated', 'total' counts.
    """
    from local_cli.model_search import search_models

    # Fetch from multiple views.
    all_models: dict[str, dict] = {}
    for sort in ("popular", "hot"):
        for m in search_models(sort=sort):
            if m["name"] not in all_models:
                all_models[m["name"]] = m
    for m in search_models(capability="tools"):
        if m["name"] not in all_models:
            all_models[m["name"]] = m
    for m in search_models(capability="code"):
        if m["name"] not in all_models:
            all_models[m["name"]] = m

    # Load existing cache.
    existing: dict[str, dict] = {}
    if _CACHE_FILE.exists():
        try:
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            for entry in data.get("models", []):
                existing[entry["name"]] = entry
        except (json.JSONDecodeError, KeyError):
            pass

    # Also include built-in catalog entries.
    for entry in get_catalog():
        if entry["name"] not in existing:
            existing[entry["name"]] = entry

    added = 0
    updated = 0

    for name, m in all_models.items():
        # Skip cloud-only models without downloadable sizes.
        if m.get("cloud_only") and not m.get("sizes"):
            continue

        category = _classify_model(name, m.get("tags", []), m.get("description", ""))
        params = _infer_params(name, m.get("sizes", []))
        size_gb = _infer_size_gb(params)

        entry = {
            "name": name,
            "display": name,
            "category": category,
            "params": params,
            "size_gb": size_gb,
            "description": m.get("description", "")[:200],
            "tags": m.get("tags", []),
            "pulls": m.get("pulls", 0),
            "pulls_display": m.get("pulls_display", ""),
            "sizes": m.get("sizes", []),
        }

        if name in existing:
            # Update pulls and description if newer.
            old = existing[name]
            if m.get("pulls", 0) > old.get("pulls", 0):
                old.update({
                    "pulls": m.get("pulls", 0),
                    "pulls_display": m.get("pulls_display", ""),
                    "description": m.get("description", old.get("description", "")),
                })
                updated += 1
        else:
            existing[name] = entry
            added += 1

    # Save cache.
    cache_data = {
        "updated_at": int(time.time()),
        "models": list(existing.values()),
    }

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(
        json.dumps(cache_data, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    return {"added": added, "updated": updated, "total": len(existing)}


def get_cached_catalog() -> list[dict] | None:
    """Load catalog from cache if fresh enough.

    Returns:
        List of model dicts, or None if cache is stale/missing.
    """
    if not _CACHE_FILE.exists():
        return None

    try:
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    updated_at = data.get("updated_at", 0)
    if time.time() - updated_at > _CACHE_MAX_AGE:
        return None

    return data.get("models", [])


def get_merged_catalog() -> tuple[list[dict], list[str]]:
    """Return merged catalog (built-in + cache) and categories.

    Prefers cached data if available; falls back to built-in catalog.
    """
    cached = get_cached_catalog()
    if cached:
        # Merge built-in entries not in cache.
        cached_names = {e["name"] for e in cached}
        merged = list(cached)
        for entry in get_catalog():
            if entry["name"] not in cached_names:
                merged.append(entry)

        # Collect categories in order.
        seen_cats = set()
        cats = []
        for cat in CATEGORIES:
            if any(m.get("category") == cat for m in merged):
                cats.append(cat)
                seen_cats.add(cat)
        for m in merged:
            cat = m.get("category", "General")
            if cat not in seen_cats:
                cats.append(cat)
                seen_cats.add(cat)

        return merged, cats

    return get_catalog(), list(CATEGORIES)
