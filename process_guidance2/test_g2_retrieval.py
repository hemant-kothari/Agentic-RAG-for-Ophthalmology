"""
test_g2_retrieval.py — Verify that Guidance 2 sources appear in retrieval results.

Run from: capstone/
    python process_guidance2/test_g2_retrieval.py
"""

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from collections import Counter

FAISS_FILE  = "faiss.index"
META_FILE   = "metadata.pkl"
CHUNKS_FILE = "chunks.jsonl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K       = 10

# Load
print("Loading VDB...")
model    = SentenceTransformer(EMBED_MODEL)
index    = faiss.read_index(FAISS_FILE)
with open(META_FILE, "rb") as f:
    metadata = pickle.load(f)
chunks = []
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line)["text"])

assert index.ntotal == len(metadata) == len(chunks), "VDB integrity FAILED"
print(f"VDB loaded: {index.ntotal} vectors | integrity OK\n")

# Source distribution
source_orgs = [m.get("source_org") or "NICE" for m in metadata]
org_counts  = Counter(source_orgs)
print("=== Source distribution in VDB ===")
for org, cnt in sorted(org_counts.items(), key=lambda x: -x[1]):
    print(f"  {org:<20} {cnt:>5} chunks")
print()

# Test queries
TEST_QUERIES = [
    ("HbA1c targets and diabetic retinopathy progression",
     ["ADA", "WHO", "EURETINA", "AAO", "RCOphth", "INDIA"]),
    ("anti-VEGF treatment for diabetic macular oedema",
     ["AAO", "EURETINA", "RCOphth", "ADA"]),
    ("hypertensive retinopathy classification Keith Wagener",
     ["REVIEW"]),
    ("intraocular pressure targets in glaucoma management",
     ["AAO", "RCOphth"]),
    ("diabetic retinopathy screening programme India",
     ["INDIA", "WHO"]),
    ("faricimab brolucizumab neovascular AMD treatment",
     ["AAO", "RCOphth", "EURETINA"]),
]

def retrieve(query: str, top_k: int = TOP_K):
    emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        m = metadata[idx]
        results.append({
            "score":      round(float(score), 4),
            "source_org": m.get("source_org") or "NICE",
            "source":     m.get("source", ""),
            "guideline":  m.get("guideline_id", ""),
            "section":    m.get("section_title", "")[:50],
            "text":       chunks[idx][:120],
        })
    return results

print("=== Retrieval Tests ===\n")
all_passed = True
for query, expected_orgs in TEST_QUERIES:
    results = retrieve(query)
    found_orgs = set(r["source_org"] for r in results)
    hit = bool(found_orgs & set(expected_orgs))

    status = "PASS" if hit else "WARN"
    if not hit:
        all_passed = False

    print(f"[{status}] Query: {query[:65]}")
    print(f"       Expected orgs: {expected_orgs}")
    print(f"       Found orgs:    {sorted(found_orgs)}")
    print(f"       Top result:    [{results[0]['source_org']}] {results[0]['guideline']} | {results[0]['section']}")
    print(f"       Top text:      {results[0]['text'][:100]}...")
    print()

if all_passed:
    print("All retrieval tests passed - both NICE and Guidance 2 sources are retrievable.")
else:
    print("Some queries did not return expected source orgs. Review WARN items above.")
