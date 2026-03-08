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
    context_length: int = 0          # max context window (tokens), 0 = unknown
    release_date: str = ""           # YYYY-MM or YYYY-MM-DD
    strengths: list[str] | None = None  # e.g. ["code", "japanese", "reasoning"]
    ease_of_use: int = 0             # 1-5 rating (0 = unrated)


# Curated catalog of popular/recommended models.
# ease_of_use: 1=難しい 2=やや難 3=普通 4=使いやすい 5=とても使いやすい
CATALOG: list[CatalogEntry] = [
    # --- Code ---
    CatalogEntry("qwen2.5-coder:7b", "Qwen 2.5 Coder 7B", "Code", "7B", 4.7,
                 "Alibaba コード特化。補完・生成・リファクタリングに強い。ツール呼び出し対応。",
                 ["code", "tools"], 32768, "2024-11", ["コード生成", "補完", "リファクタ"], 4),
    CatalogEntry("qwen2.5-coder:14b", "Qwen 2.5 Coder 14B", "Code", "14B", 9.0,
                 "7Bより高精度。複雑なロジックやマルチファイル編集に対応。",
                 ["code", "tools"], 32768, "2024-11", ["コード生成", "複雑なロジック"], 4),
    CatalogEntry("qwen2.5-coder:32b", "Qwen 2.5 Coder 32B", "Code", "32B", 20.0,
                 "コーディング最高峰。GPT-4レベルに迫る精度。16GB+ VRAM推奨。",
                 ["code", "tools"], 32768, "2024-11", ["コード生成", "高精度推論"], 3),
    CatalogEntry("deepseek-coder-v2:16b", "DeepSeek Coder V2 16B", "Code", "16B", 8.9,
                 "MoE構造で効率的。マルチファイル編集・デバッグに強い。",
                 ["code", "tools"], 65536, "2024-06", ["コード生成", "デバッグ", "MoE"], 3),
    CatalogEntry("codellama:7b", "Code Llama 7B", "Code", "7B", 3.8,
                 "Meta製コード特化。軽量だが基本的なコード生成は十分。",
                 ["code"], 16384, "2023-08", ["コード生成"], 3),
    CatalogEntry("codellama:13b", "Code Llama 13B", "Code", "13B", 7.4,
                 "Code Llama大型版。7Bより精度向上。",
                 ["code"], 16384, "2023-08", ["コード生成"], 3),
    CatalogEntry("starcoder2:7b", "StarCoder2 7B", "Code", "7B", 4.0,
                 "BigCode OSS。600以上の言語で学習。補完向き。",
                 ["code"], 16384, "2024-02", ["コード補完", "多言語対応"], 3),

    # --- General / Chat ---
    CatalogEntry("qwen3.5:4b", "Qwen 3.5 4B", "General", "4B", 2.6,
                 "最新Qwen 3.5。Vision+思考+ツール。サイズの割に非常に高性能。",
                 ["general", "tools", "reasoning", "vision"], 32768, "2025-06", ["汎用", "Vision", "推論", "ツール"], 5),
    CatalogEntry("qwen3.5:9b", "Qwen 3.5 9B", "General", "9B", 5.5,
                 "Qwen 3.5中型。Vision対応、バランスの取れた推論力。",
                 ["general", "tools", "reasoning", "vision"], 32768, "2025-06", ["汎用", "Vision", "推論"], 5),
    CatalogEntry("qwen3.5:27b", "Qwen 3.5 27B", "General", "27B", 17.0,
                 "Qwen 3.5大型。フロンティアモデルに迫る品質。16GB+ VRAM推奨。",
                 ["general", "tools", "reasoning", "vision"], 32768, "2025-06", ["汎用", "高精度推論", "Vision"], 4),
    CatalogEntry("qwen3:8b", "Qwen 3 8B", "General", "8B", 5.2,
                 "Qwen 3。ツール呼び出し・推論に強い。日本語もそこそこ。エージェント向き。",
                 ["general", "tools", "reasoning", "japanese"], 32768, "2025-04", ["汎用", "ツール", "推論", "日本語"], 5),
    CatalogEntry("qwen3:4b", "Qwen 3 4B", "General", "4B", 2.6,
                 "Qwen 3軽量版。8GB RAMでも動作。基本的なタスクに十分。",
                 ["general", "tools"], 32768, "2025-04", ["汎用", "軽量"], 5),
    CatalogEntry("qwen3:14b", "Qwen 3 14B", "General", "14B", 9.0,
                 "サイズと品質のバランスが良い。ツール呼び出し安定。",
                 ["general", "tools", "reasoning"], 32768, "2025-04", ["汎用", "推論", "ツール"], 4),
    CatalogEntry("qwen3:30b", "Qwen 3 30B", "General", "30B", 18.5,
                 "高品質推論。エージェントループに最適。24GB+ VRAM推奨。",
                 ["general", "tools", "reasoning"], 32768, "2025-04", ["汎用", "エージェント", "高精度推論"], 4),
    CatalogEntry("llama3.2:3b", "Llama 3.2 3B", "General", "3B", 2.0,
                 "Meta製軽量モデル。高速レスポンス。簡単な質問向き。",
                 ["general"], 131072, "2024-09", ["汎用", "高速"], 4),
    CatalogEntry("llama3.1:8b", "Llama 3.1 8B", "General", "8B", 4.7,
                 "Meta製万能モデル。英語中心だが安定した品質。ツール対応。",
                 ["general", "tools"], 131072, "2024-07", ["汎用", "ツール"], 4),
    CatalogEntry("llama3.3:70b", "Llama 3.3 70B", "General", "70B", 43.0,
                 "Metaフラッグシップ。最高品質だが48GB+ VRAM必須。",
                 ["general", "tools", "reasoning"], 131072, "2024-12", ["汎用", "高精度推論"], 2),
    CatalogEntry("gemma3:4b", "Gemma 3 4B", "General", "4B", 3.3,
                 "Google製軽量。シンプルなタスク向き。日本語は限定的。",
                 ["general"], 32768, "2025-03", ["汎用", "軽量"], 4),
    CatalogEntry("gemma3:12b", "Gemma 3 12B", "General", "12B", 8.1,
                 "Google中型。精度良好、日本語もそこそこ。ツール対応。",
                 ["general", "tools", "japanese"], 32768, "2025-03", ["汎用", "日本語", "ツール"], 4),
    CatalogEntry("gemma3:27b", "Gemma 3 27B", "General", "27B", 17.0,
                 "Google最大OSS。推論・日本語とも高品質。16GB+ VRAM推奨。",
                 ["general", "tools", "reasoning", "japanese"], 32768, "2025-03", ["汎用", "推論", "日本語"], 3),
    CatalogEntry("mistral:7b", "Mistral 7B", "General", "7B", 4.1,
                 "欧州発の効率モデル。英語・仏語に強い。日本語は弱い。",
                 ["general"], 32768, "2023-09", ["汎用", "欧州言語"], 3),
    CatalogEntry("phi4:14b", "Phi-4 14B", "General", "14B", 9.1,
                 "Microsoft製。数学・論理推論に強い。コード生成もそこそこ。",
                 ["general", "reasoning"], 16384, "2024-12", ["推論", "数学", "コード"], 3),

    # --- Small / Edge ---
    CatalogEntry("qwen3.5:0.8b", "Qwen 3.5 0.8B", "Small", "0.8B", 0.5,
                 "最新の超小型Qwen 3.5。テスト・プロトタイプ向き。ツール対応。",
                 ["general", "tools"], 32768, "2025-06", ["テスト用", "超軽量"], 5),
    CatalogEntry("qwen3.5:2b", "Qwen 3.5 2B", "Small", "2B", 1.3,
                 "Qwen 3.5小型。Vision対応。サイズの割に驚くほど有能。",
                 ["general", "tools", "vision"], 32768, "2025-06", ["軽量", "Vision", "ツール"], 5),
    CatalogEntry("qwen3:0.6b", "Qwen 3 0.6B", "Small", "0.6B", 0.5,
                 "超小型。動作確認やドラフト生成向き。精度は限定的。",
                 ["general"], 32768, "2025-04", ["テスト用", "超軽量"], 4),
    CatalogEntry("qwen3:1.7b", "Qwen 3 1.7B", "Small", "1.7B", 1.1,
                 "小さいがツール呼び出し可能。簡単なタスクなら実用的。",
                 ["general", "tools"], 32768, "2025-04", ["軽量", "ツール"], 4),
    CatalogEntry("phi4-mini:3.8b", "Phi-4 Mini", "Small", "3.8B", 2.5,
                 "Microsoft小型推論モデル。数学・論理に特化。",
                 ["general", "reasoning"], 16384, "2025-02", ["推論", "数学"], 3),
    CatalogEntry("gemma3:1b", "Gemma 3 1B", "Small", "1B", 1.0,
                 "Google最小モデル。超高速。精度は最低限。",
                 ["general"], 32768, "2025-03", ["超軽量", "高速"], 4),
    CatalogEntry("llama3.2:1b", "Llama 3.2 1B", "Small", "1B", 0.7,
                 "Meta最小。レスポンス速い。英語の簡単なタスク向き。",
                 ["general"], 131072, "2024-09", ["超軽量", "高速"], 4),

    # --- Reasoning ---
    CatalogEntry("deepseek-r1:7b", "DeepSeek R1 7B", "Reasoning", "7B", 4.7,
                 "思考連鎖（CoT）推論の蒸留版。数学・論理問題に強い。",
                 ["reasoning"], 65536, "2025-01", ["推論", "数学", "CoT"], 3),
    CatalogEntry("deepseek-r1:14b", "DeepSeek R1 14B", "Reasoning", "14B", 9.0,
                 "より深い推論。詳細な思考過程を表示。",
                 ["reasoning"], 65536, "2025-01", ["推論", "詳細思考"], 3),
    CatalogEntry("deepseek-r1:32b", "DeepSeek R1 32B", "Reasoning", "32B", 20.0,
                 "高精度推論。複雑な数学・コーディング問題向き。24GB+ VRAM推奨。",
                 ["reasoning"], 65536, "2025-01", ["推論", "数学", "コード"], 3),
    CatalogEntry("qwq:32b", "QwQ 32B", "Reasoning", "32B", 20.0,
                 "Qwen推論特化。長い思考連鎖で複雑な問題を解く。24GB+ VRAM推奨。",
                 ["reasoning"], 32768, "2024-11", ["推論", "長考"], 3),

    # --- Multilingual ---
    CatalogEntry("aya-expanse:8b", "Aya Expanse 8B", "Multilingual", "8B", 4.8,
                 "Cohere製多言語モデル。23言語対応。翻訳・多言語チャットに最適。",
                 ["multilingual"], 8192, "2024-10", ["多言語", "翻訳"], 4),
    CatalogEntry("aya-expanse:32b", "Aya Expanse 32B", "Multilingual", "32B", 19.0,
                 "大型多言語モデル。高品質な多言語生成。16GB+ VRAM推奨。",
                 ["multilingual"], 8192, "2024-10", ["多言語", "高品質翻訳"], 3),

    # --- Japanese ---
    CatalogEntry("llm-jp:13b", "LLM-jp 13B", "Japanese", "13B", 7.4,
                 "国立情報学研究所（NII）開発。日本語特化。学術・ビジネス文書に強い。",
                 ["japanese"], 4096, "2024-03", ["日本語", "学術", "ビジネス"], 3),
    CatalogEntry("hf.co/elyza/Llama-3-ELYZA-JP-8B-GGUF", "ELYZA JP 8B", "Japanese", "8B", 4.7,
                 "ELYZA製日本語Llama 3。日本語NLPタスク全般に強い。実用的。",
                 ["japanese"], 8192, "2024-06", ["日本語", "NLP", "実用的"], 4),
    CatalogEntry("hf.co/LiquidAI/LFM2.5-1.2B-JP-GGUF", "LFM 2.5 1.2B JP", "Japanese", "1.2B", 0.8,
                 "Liquid AI製日本語特化小型モデル。軽量で高速。簡単な日本語タスク向き。",
                 ["japanese"], 4096, "2024-11", ["日本語", "軽量", "高速"], 4),
    CatalogEntry("hf.co/mmnga/cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese-gguf", "CyberAgent DeepSeek R1 14B JP", "Japanese", "14B", 9.0,
                 "CyberAgent製。DeepSeek R1の日本語蒸留版。日本語での推論に強い。",
                 ["japanese", "reasoning"], 32768, "2025-02", ["日本語", "推論", "CoT"], 3),
    CatalogEntry("hf.co/mmnga/Fugaku-LLM-13B-instruct-gguf", "Fugaku LLM 13B", "Japanese", "13B", 7.4,
                 "富岳スパコンプロジェクト発。日本語命令チューニング済み。",
                 ["japanese"], 4096, "2024-05", ["日本語", "学術"], 2),
    CatalogEntry("hf.co/dahara1/gemma-2-2b-jpn-it-gguf", "Gemma 2 2B JPN IT", "Japanese", "2B", 1.6,
                 "Google Gemma 2日本語版。小型で高速。簡単な日本語会話向き。",
                 ["japanese"], 8192, "2024-08", ["日本語", "軽量", "会話"], 4),
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
            "context_length": e.context_length,
            "release_date": e.release_date,
            "strengths": e.strengths or [],
            "ease_of_use": e.ease_of_use,
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
