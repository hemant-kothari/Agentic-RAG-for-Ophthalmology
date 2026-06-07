import json
import pickle
import faiss
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# ---------------- PATHS ----------------
JSON_DIR = Path("json")
CHUNKS_FILE = "chunks.jsonl"
FAISS_INDEX_FILE = "faiss.index"
META_FILE = "metadata.pkl"

# ---------------- MODEL ----------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL)

CHUNK_SIZE = 400   # tokens (approx)
OVERLAP = 50


# ---------------- TOKEN ESTIMATION ----------------
def estimate_tokens(text: str) -> int:
    return len(text.split())


# ---------------- CHUNKING ----------------
def chunk_text(text, max_tokens=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start = end - overlap

    return chunks


# ---------------- LOAD + CHUNK ----------------
chunks = []
metadata = []

chunk_counter = 0

for json_file in sorted(JSON_DIR.glob("*.json")):
    with open(json_file, "r", encoding="utf-8") as f:
        doc = json.load(f)

    for section in doc["sections"]:
        section_text = section["content"]

        section_chunks = chunk_text(section_text)

        for idx, chunk in enumerate(section_chunks):
            chunk_id = f"{doc['guideline_id']}_{section['section_number']}_{idx:03d}"

            chunks.append(chunk)

            metadata.append({
                "chunk_id": chunk_id,
                "guideline_id": doc["guideline_id"],
                "title": doc["title"],
                "section_number": section["section_number"],
                "section_title": section["section_title"],
                "section_type": section["section_type"],
                "source": doc["source"]
            })

            chunk_counter += 1

print(f"Total chunks created: {chunk_counter}")

# ---------------- SAVE CHUNKS ----------------
with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
    for text in chunks:
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

# ---------------- EMBEDDINGS ----------------
print("Generating embeddings...")
embeddings = model.encode(
    chunks,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")

# ---------------- FAISS ----------------
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine similarity
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX_FILE)

# ---------------- SAVE METADATA ----------------
with open(META_FILE, "wb") as f:
    pickle.dump(metadata, f)

print("FAISS index and metadata saved successfully")
