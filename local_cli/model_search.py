"""Search Ollama model library for popular and trending models.

Fetches model information from ollama.com/search using stdlib only.
Results include model name, description, pull count, available sizes,
and capability tags.
"""

import html as html_mod
import re
import urllib.request
import urllib.parse
import json
from typing import Any

_USER_AGENT = "local-cli/0.2.0"
_TIMEOUT = 15
_BASE_URL = "https://ollama.com/search"


def _parse_pull_count(text: str) -> int:
    """Parse pull count strings like '922.1K', '1.2M', '53' into integers."""
    text = text.strip().replace(",", "")
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    for suffix, mult in multipliers.items():
        if text.upper().endswith(suffix):
            try:
                return int(float(text[:-1]) * mult)
            except ValueError:
                return 0
    try:
        return int(text)
    except ValueError:
        return 0


def _extract_models_from_html(page_html: str) -> list[dict[str, Any]]:
    """Extract model entries from Ollama search page HTML."""
    cards = re.findall(r"<li[^>]*>(.*?)</li>", page_html, re.DOTALL)
    results: list[dict[str, Any]] = []

    for card in cards:
        # Model name from link.
        name_m = re.search(r'href="/library/([^"]+)"', card)
        if not name_m:
            continue
        name = name_m.group(1)

        # Description.
        desc_m = re.search(r"<p[^>]*>([^<]+)</p>", card)
        desc = html_mod.unescape(desc_m.group(1).strip()) if desc_m else ""

        # Extract all spans for tags and pull counts.
        spans = [
            s.strip()
            for s in re.findall(r"<span[^>]*>([^<]*)</span>", card)
            if s.strip() and s.strip() != "&nbsp;Pulls" and s.strip() != "&nbsp;Tag"
        ]

        # Pull count: look for number patterns like "922.1K" near "Pulls".
        pulls = 0
        pull_m = re.search(
            r'<span[^>]*>([\d,.]+[KMBkmb]?)</span>\s*<span[^>]*>&nbsp;Pulls</span>',
            card,
        )
        if pull_m:
            pulls = _parse_pull_count(pull_m.group(1))

        # Tags: known capability tags.
        known_tags = {"tools", "vision", "thinking", "embedding", "code", "cloud"}
        tags = [s for s in spans if s.lower() in known_tags]

        # Available sizes from the card content.
        size_matches = re.findall(
            r"(?:^|\s)(\d+\.?\d*[bB])(?:\s|$|,)", card
        )
        sizes = list(dict.fromkeys(s.lower() for s in size_matches))

        results.append({
            "name": name,
            "description": desc[:200],
            "pulls": pulls,
            "pulls_display": _format_pulls(pulls),
            "tags": tags,
            "sizes": sizes,
            "cloud_only": "cloud" in [t.lower() for t in tags] and not sizes,
        })

    return results


def _format_pulls(n: int) -> str:
    """Format pull count for display."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def search_models(
    query: str = "",
    sort: str = "popular",
    capability: str = "",
) -> list[dict[str, Any]]:
    """Search Ollama model library.

    Args:
        query: Search query string.
        sort: Sort order — 'popular', 'newest', or 'hot'.
        capability: Filter by capability — 'tools', 'vision',
            'thinking', 'embedding', 'code', or '' for all.

    Returns:
        List of model dicts with name, description, pulls,
        tags, and sizes.
    """
    params: dict[str, str] = {"q": query}
    if sort and sort != "popular":
        params["sort"] = sort
    if capability:
        params["c"] = capability

    url = f"{_BASE_URL}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            page_html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    return _extract_models_from_html(page_html)


def get_popular_models() -> list[dict[str, Any]]:
    """Get popular models sorted by pull count."""
    return search_models(sort="popular")


def get_trending_models() -> list[dict[str, Any]]:
    """Get trending/hot models."""
    return search_models(sort="hot")


def search_code_models() -> list[dict[str, Any]]:
    """Get models with code/tools capabilities."""
    return search_models(capability="tools")
