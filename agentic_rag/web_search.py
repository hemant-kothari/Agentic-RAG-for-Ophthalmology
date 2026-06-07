"""
web_search.py — DuckDuckGo web search fallback for the Agentic RAG system.

Uses the `duckduckgo-search` library (no API key required).
Results are clearly tagged as [WEB SOURCE] to distinguish from KB results.
"""

from __future__ import annotations
from duckduckgo_search import DDGS

MAX_RESULTS   = 4
SEARCH_REGION = "en-us"

# Prepend this to all searches to bias toward clinical/medical results
MEDICAL_PREFIX = "ophthalmology clinical guideline "


def web_search(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """
    Search DuckDuckGo for the query.
    Returns a list of dicts: {title, url, snippet, source: "web"}
    Empty list on failure.
    """
    try:
        biased_query = MEDICAL_PREFIX + query
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(biased_query, region=SEARCH_REGION, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "source":  "web",
                })
        return results
    except Exception as e:
        print(f"[web_search] Warning: search failed: {e}")
        return []


def format_web_results_for_context(results: list[dict]) -> str:
    """
    Format web results as a clearly labeled context block.
    Snippets are capped at 400 chars each to stay within token budget.
    """
    if not results:
        return ""

    lines = ["=== WEB SEARCH RESULTS (not from clinical guidelines) ===\n"]
    for i, r in enumerate(results, 1):
        snippet = r["snippet"][:400].strip()
        lines.append(
            f"[WEB {i}] {r['title']}\n"
            f"Source: {r['url']}\n"
            f"{snippet}\n"
        )
    return "\n".join(lines)
