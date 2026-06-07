"""
context_builder.py — Assembles the LLM context from KB chunks + optional web results.

Responsibilities:
  - Merge KB retrieval results and web search snippets
  - Label each chunk with its source organization
  - Enforce a token budget (~3000 words ≈ ~4000 tokens)
  - Deduplicate near-identical chunks
"""

from __future__ import annotations

MAX_CONTEXT_WORDS = 3000
KB_CHUNK_WORD_CAP = 200    # cap individual KB chunks to avoid single-chunk domination


def _word_count(text: str) -> int:
    return len(text.split())


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [...]"


def _is_near_duplicate(text: str, seen: list[str], threshold: float = 0.7) -> bool:
    """Simple character-level overlap dedup."""
    text_set = set(text.lower().split())
    for s in seen:
        s_set = set(s.lower().split())
        if not text_set or not s_set:
            continue
        overlap = len(text_set & s_set) / min(len(text_set), len(s_set))
        if overlap > threshold:
            return True
    return False


def _format_kb_chunk(result: dict, index: int) -> str:
    """Format a single KB retrieval result."""
    source_org  = result.get("source_org") or "NICE"
    guideline   = result.get("guideline_id", result.get("title", "Unknown"))
    section     = result.get("section_title", "")
    text        = _truncate(result["text"], KB_CHUNK_WORD_CAP)
    score       = result.get("score", 0)

    header = f"[KB {index}] [{source_org}] {guideline}"
    if section:
        header += f" — {section}"
    header += f" (relevance: {score:.3f})"
    return f"{header}\n{text}"


def build_context(
    kb_results:  list[dict],
    web_results: list[dict] | None = None,
) -> str:
    """
    Build a merged context string for the LLM.

    kb_results : list of dicts from retriever.retrieve()
    web_results: list of dicts from web_search.web_search() (optional)

    Returns a single string ready to insert into the user prompt.
    """
    parts         = []
    total_words   = 0
    seen_texts    = []

    # ── KB chunks ──────────────────────────────────────────────────────────
    if kb_results:
        parts.append("=== KNOWLEDGE BASE (Clinical Guidelines) ===\n")
        for i, result in enumerate(kb_results, 1):
            text = result.get("text", "")
            if _is_near_duplicate(text, seen_texts):
                continue
            seen_texts.append(text)

            chunk_str = _format_kb_chunk(result, i)
            chunk_words = _word_count(chunk_str)

            if total_words + chunk_words > MAX_CONTEXT_WORDS:
                break

            parts.append(chunk_str)
            total_words += chunk_words

    # ── Web results ────────────────────────────────────────────────────────
    if web_results:
        remaining = MAX_CONTEXT_WORDS - total_words
        if remaining > 200:
            parts.append("\n=== WEB SEARCH RESULTS (supplementary — not clinical guidelines) ===\n")
            for i, r in enumerate(web_results, 1):
                snippet = _truncate(r.get("snippet", ""), 120)
                title   = r.get("title", "")
                url     = r.get("url", "")
                entry   = f"[WEB {i}] {title}\nURL: {url}\n{snippet}"
                entry_words = _word_count(entry)
                if total_words + entry_words > MAX_CONTEXT_WORDS:
                    break
                parts.append(entry)
                total_words += entry_words

    return "\n\n".join(parts)
