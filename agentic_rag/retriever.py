"""
retriever.py — Enhanced FAISS retriever with source_org/disease filtering
and min_score threshold. Fully backward-compatible with old NICE-only metadata.
"""

import faiss
import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
FAISS_INDEX_FILE = str(BASE_DIR / "faiss.index")
META_FILE        = str(BASE_DIR / "metadata.pkl")
CHUNKS_FILE      = str(BASE_DIR / "chunks.jsonl")
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K    = 10
MIN_SCORE        = 0.30   # cosine similarity threshold

# ── LOAD ONCE AT MODULE IMPORT ───────────────────────────────────────────────
model    = SentenceTransformer(EMBED_MODEL)
index    = faiss.read_index(FAISS_INDEX_FILE)

with open(META_FILE, "rb") as f:
    metadata = pickle.load(f)

chunks = []
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line)["text"])

assert index.ntotal == len(metadata) == len(chunks), (
    f"VDB integrity check failed: index={index.ntotal}, "
    f"meta={len(metadata)}, chunks={len(chunks)}"
)


# ── EMBED ────────────────────────────────────────────────────────────────────
def embed_query(query: str) -> np.ndarray:
    return model.encode([query], normalize_embeddings=True).astype("float32")


# ── RETRIEVE ─────────────────────────────────────────────────────────────────
def retrieve(
    query:          str,
    top_k:          int         = DEFAULT_TOP_K,
    min_score:      float       = MIN_SCORE,
    source_org:     str | None  = None,   # e.g. "AAO", "NICE", "WHO"
    disease_filter: str | None  = None,   # e.g. "Diabetic Retinopathy"
    section_filter: str | None  = None,   # e.g. "Recommendations"
) -> list[dict]:
    """
    Retrieve top_k chunks from FAISS index with optional filters.

    Returns list of dicts:
    {
      score, text, guideline_id, title, section_title, section_type,
      source, source_org, disease, year
    }
    """
    query_emb = embed_query(query)

    # Over-fetch to allow for filtering
    fetch_k = min(top_k * 4, index.ntotal)
    scores, indices = index.search(query_emb, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score < min_score:
            continue

        meta = metadata[idx]

        # Apply filters
        if source_org and (meta.get("source_org") or "NICE") != source_org:
            continue
        if disease_filter:
            doc_disease = (meta.get("disease") or "").lower()
            if disease_filter.lower() not in doc_disease:
                continue
        if section_filter and meta.get("section_type") != section_filter:
            continue

        results.append({
            "score":         float(score),
            "text":          chunks[idx],
            "guideline_id":  meta.get("guideline_id", ""),
            "title":         meta.get("title", ""),
            "section_title": meta.get("section_title", ""),
            "section_type":  meta.get("section_type", ""),
            "source":        meta.get("source", ""),
            "source_org":    meta.get("source_org") or "NICE",
            "disease":       meta.get("disease") or "",
            "year":          meta.get("year"),
        })

        if len(results) >= top_k:
            break

    return results


def max_score(results: list[dict]) -> float:
    """Return the highest similarity score from a result set."""
    if not results:
        return 0.0
    return max(r["score"] for r in results)
