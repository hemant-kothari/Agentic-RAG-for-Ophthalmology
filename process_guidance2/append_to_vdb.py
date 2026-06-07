"""
append_to_vdb.py — Appends Guidance 2 chunks/embeddings to the existing FAISS VDB.

Reads all JSONs from json_g2/, chunks and embeds them using the same model as the
original NICE pipeline (all-MiniLM-L6-v2), then appends to:
  - faiss.index     (incremental index.add())
  - chunks.jsonl    (append mode)
  - metadata.pkl    (extend list)

Run from: capstone/
    python process_guidance2/append_to_vdb.py

Safety check: verifies index.ntotal == len(metadata) == len(chunks) before and after.
"""

import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─── Paths (all relative to capstone/) ───────────────────────────────────────
JSON_G2_DIR   = Path("json_g2")
FAISS_FILE    = Path("faiss.index")
META_FILE     = Path("metadata.pkl")
CHUNKS_FILE   = Path("chunks.jsonl")

EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE    = 400   # words (same as NICE pipeline)
OVERLAP       = 50


# ─── Helpers ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, max_tokens: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list:
    """Identical sliding-window chunker as the NICE pipeline."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def load_all_jsons(json_dir: Path) -> list:
    """Load all JSON files from json_g2/."""
    docs = []
    files = sorted(json_dir.glob("*.json"))
    print(f"Found {len(files)} JSON files in '{json_dir}'")
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            docs.append(json.load(fh))
    return docs


def build_new_chunks(docs: list) -> tuple[list, list]:
    """
    Chunk all sections from all docs.
    Returns (chunks: list[str], metadata: list[dict])
    Uses extended metadata schema (includes source_org, disease, year if present).
    """
    chunks = []
    metadata = []

    for doc in docs:
        guideline_id = doc.get("guideline_id", "UNKNOWN")
        title        = doc.get("title", "")
        source       = doc.get("source", "")
        source_org   = doc.get("source_org", None)
        disease      = doc.get("disease", None)
        year         = doc.get("year", None)

        for section in doc.get("sections", []):
            section_text = section.get("content", "").strip()
            if not section_text:
                continue

            section_chunks = chunk_text(section_text)

            for idx, chunk in enumerate(section_chunks):
                chunk_id = f"{guideline_id}_{section['section_number']}_{idx:03d}"

                chunks.append(chunk)
                metadata.append({
                    "chunk_id":       chunk_id,
                    "guideline_id":   guideline_id,
                    "title":          title,
                    "section_number": section["section_number"],
                    "section_title":  section["section_title"],
                    "section_type":   section["section_type"],
                    "source":         source,
                    # Extended fields (None for old NICE chunks, populated for G2)
                    "source_org":     source_org,
                    "disease":        disease,
                    "year":           year,
                })

    return chunks, metadata


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Load existing VDB
    print("Loading existing FAISS index and metadata...")
    index    = faiss.read_index(str(FAISS_FILE))
    with open(META_FILE, "rb") as f:
        existing_meta = pickle.load(f)

    # Load existing chunks
    existing_chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                existing_chunks.append(json.loads(line)["text"])

    orig_count = index.ntotal
    print(f"Existing index: {orig_count} vectors | {len(existing_meta)} metadata | {len(existing_chunks)} chunks")

    # Integrity check
    if not (orig_count == len(existing_meta) == len(existing_chunks)):
        print("[ERROR] Existing VDB is inconsistent! Aborting.")
        print(f"  index.ntotal={orig_count}, metadata={len(existing_meta)}, chunks={len(existing_chunks)}")
        sys.exit(1)

    # 2. Load and chunk new docs
    docs = load_all_jsons(JSON_G2_DIR)
    if not docs:
        print("[ERROR] No JSONs found in json_g2/. Run process_all.py first.")
        sys.exit(1)

    new_chunks, new_meta = build_new_chunks(docs)
    print(f"\nNew chunks to add: {len(new_chunks)}")

    if not new_chunks:
        print("[WARN] No chunks generated. Check if JSONs have non-empty sections.")
        sys.exit(0)

    # 3. Embed new chunks
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("Generating embeddings for new chunks...")
    new_embeddings = model.encode(
        new_chunks,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    new_embeddings = np.array(new_embeddings).astype("float32")

    print(f"Embedding shape: {new_embeddings.shape}")

    # 4. Append to FAISS
    index.add(new_embeddings)
    print(f"FAISS index size: {orig_count} -> {index.ntotal}")

    # 5. Extend metadata and chunks
    combined_meta   = existing_meta + new_meta
    combined_chunks = existing_chunks + new_chunks

    # 6. Final integrity check
    assert index.ntotal == len(combined_meta) == len(combined_chunks), (
        f"Post-append mismatch: index={index.ntotal}, "
        f"meta={len(combined_meta)}, chunks={len(combined_chunks)}"
    )
    print(f"Integrity check passed: {index.ntotal} total vectors.")

    # 7. Save everything
    print("\nSaving updated FAISS index...")
    faiss.write_index(index, str(FAISS_FILE))

    print("Saving updated metadata...")
    with open(META_FILE, "wb") as f:
        pickle.dump(combined_meta, f)

    print("Saving updated chunks.jsonl...")
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for text in combined_chunks:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"\nDone! VDB updated successfully.")
    print(f"  Previous count : {orig_count:>6}")
    print(f"  New chunks added: {len(new_chunks):>6}")
    print(f"  Total vectors  : {index.ntotal:>6}")


if __name__ == "__main__":
    main()
