# Retinal Disease — Agentic RAG Clinical Knowledge System

> **Evidence-grounded clinical decision support for ophthalmic diseases using Retrieval-Augmented Generation**

This repository contains the **Agentic RAG** component of a two-part capstone project on AI-assisted retinal disease management. It takes the structured output of a RETFound Vision Transformer classifier and generates comprehensive, multi-source, citation-backed clinical guidance reports — or answers open-ended ophthalmic queries — using a 2,163-vector knowledge base built from 92 clinical guidelines across 8 international organisations.

The companion ViT model repository handles fundus image preprocessing, multi-label disease classification, and DR severity grading.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Knowledge Base](#knowledge-base)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
  - [Web UI (Gradio)](#web-ui-gradio)
  - [CLI — Image Mode](#cli--image-mode)
  - [CLI — Query Mode](#cli--query-mode)
  - [CLI — Interactive Session](#cli--interactive-session)
- [Guardrails](#guardrails)
- [Sample Output](#sample-output)
- [Building the Knowledge Base](#building-the-knowledge-base)
- [Paper](#paper)
- [Dependencies](#dependencies)

---

## Overview

The system operates in two modes:

| Mode | Input | Output |
|------|-------|--------|
| **Image Mode** | `inference_results.json` from the ViT model | Structured per-disease clinical guidance report |
| **Query Mode** | Free-text clinical question | Evidence-grounded answer with inline citations |

Both modes retrieve evidence from a FAISS-indexed knowledge base of international ophthalmic guidelines, assess retrieval confidence, fall back to web search when needed, and enforce clinical safety guardrails on all outputs.

---

## System Architecture

```
Input (ViT JSON / User Query)
         │
         ▼
  ┌─────────────────┐
  │  Guardrail Gate  │  ← off-topic rejection, injection sanitisation
  └────────┬────────┘
           │
     ┌─────┴──────┐
     │            │
  Image         Query
  Mode          Mode
     │            │
     └─────┬──────┘
           │
           ▼
  ┌─────────────────┐
  │   FAISS Retriever│  ← all-MiniLM-L6-v2 embeddings, 2163 vectors
  │  (multi-query)   │
  └────────┬────────┘
           │
     ┌─────┴──────┐
  KB score      KB score
  ≥ 0.35        < 0.35
     │            │
     │        DuckDuckGo
     │        Web Search
     └─────┬──────┘
           │
           ▼
  ┌─────────────────┐
  │ Context Builder  │  ← merge KB + web, dedup, 3000-word budget
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Groq LLM       │  ← llama-3.3-70b-versatile, key rotation
  │  (structured    │
  │   prompts)      │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Response +       │  ← clinical disclaimer appended
  │ Citations        │
  └─────────────────┘
```

---

## Knowledge Base

The FAISS vector database contains **2,163 chunks** from **92 documents** across **8 organisations**:

| Organisation | Documents | Chunks |
|---|---|---|
| NICE (UK) | 72 | 975 |
| AAO (American Academy of Ophthalmology) | 6 | 467 |
| RCOphth (Royal College of Ophthalmologists) | 3 | 323 |
| Review Papers (peer-reviewed journals) | 6 | 130 |
| WHO | 1 | 96 |
| EURETINA | 2 | 89 |
| ADA (American Diabetes Association) | 1 | 48 |
| India Vision 2025 | 1 | 35 |
| **Total** | **92** | **2,163** |

Diseases covered: **Diabetic Retinopathy · Diabetic Macular · Age-Related Macular Degeneration · Glaucoma · Hypertensive Retinopathy**

---

## Project Structure

```
capstone/
│
├── agentic_rag/                    # Core RAG package
│   ├── __init__.py
│   ├── agent.py                    # Main orchestrator (image + query modes)
│   ├── retriever.py                # FAISS retrieval with filtering
│   ├── llm_client.py               # Groq LLM wrapper with key rotation
│   ├── input_parser.py             # ViT JSON parser + confidence labelling
│   ├── guardrails.py               # Off-topic detection, sanitisation, disclaimer
│   ├── web_search.py               # DuckDuckGo fallback
│   ├── context_builder.py          # KB + web context assembly
│   └── prompts/
│       ├── system_image.txt        # System prompt: image mode
│       ├── system_query.txt        # System prompt: query mode
│       ├── user_image.txt          # User prompt template: image mode
│       └── user_query.txt          # User prompt template: query mode
│
├── process_guidance2/              # Knowledge base construction pipeline
│   ├── inspect_pdfs.py             # Step 0: preview PDF text
│   ├── parser_utils.py             # Shared text cleaning utilities
│   ├── parsers/
│   │   ├── aao_ppp_parser.py       # Parser: AAO Preferred Practice Patterns
│   │   ├── society_guideline_parser.py  # Parser: RCOphth, WHO, India
│   │   └── academic_review_parser.py   # Parser: EURETINA, ADA, reviews
│   ├── process_all.py              # Orchestrator: 20 PDFs → json_g2/
│   ├── append_to_vdb.py            # Append json_g2/ to FAISS VDB
│   └── test_g2_retrieval.py        # Retrieval validation (6 queries)
│
├── single_image_results/           # Sample ViT inference output (included)
│   ├── inference_results.json      # Full inference JSON (used as default input)
│   ├── inference_summary.txt       # Human-readable summary
│   ├── attention_overlay.png       # Attention heatmap overlay
│   └── attention_summary_figure.png
|
├── till vdb                        # process NICE documents (guidance folder)
|   ├── chunk_embed_vdb.py
|   ├── download_nice_guidance.py
|   ├── download_nice_products.py
|   ├── loopover_pdf_json.py
|   ├──pdftojson.py
│
├── faiss.index                     # FAISS vector index (2163 vectors)
├── metadata.pkl                    # Chunk metadata (guideline IDs, sources, etc.)
├── chunks.jsonl                    # Raw chunk texts (JSONL)
│
├── run_rag.py                      # CLI entry point
├── rag_app.py                      # Gradio Web UI
│
├── .env                            # API keys (NOT committed)
├── requirements.txt                # Python dependencies
└── paper_agentic_rag.md            # Academic paper (~15 pages)
```

---

## Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/<your-username>/retinal-rag.git
cd ophthalmology-rag

# Install dependencies
pip install -r requirements.txt
```

---

## Setup

Create a `.env` file in the root directory with your Groq API key(s). Multiple keys are supported for automatic rotation if one hits a rate limit:

```env
GROQ_API_KEYS=gsk_your_key_1,gsk_your_key_2,gsk_your_key_3
```

Get a free API key at [console.groq.com](https://console.groq.com).

> **Note:** The `faiss.index`, `metadata.pkl`, and `chunks.jsonl` files are included in this repository. You do not need to rebuild the knowledge base to run the system.

---

## Usage

All commands should be run from the project root directory. Set the encoding environment variable on Windows to avoid character display issues:

```powershell
# Windows PowerShell (run once per session)
$env:PYTHONIOENCODING="utf-8"
```

### Web UI (Gradio)

The recommended interface. Opens a browser tab automatically at `http://localhost:7860`.

```bash
python rag_app.py
```

**Tab 1 — Image Analysis Report:** Click **Use Sample** to generate a report from the included `inference_results.json`, or upload your own.

**Tab 2 — Ask a Question:** Type any question about retinal diseases, screening protocols, or treatment guidelines.

---

### CLI — Image Mode

Generate a clinical guidance report from a ViT inference output file:

```bash
python run_rag.py --mode image --input single_image_results/inference_results.json
```

If `--input` is omitted, the sample file is used by default.

---

### CLI — Query Mode

Ask a single clinical question:

```bash
python run_rag.py --mode query --query "What are the recommended anti-VEGF agents for neovascular AMD?"
```

---

### CLI — Interactive Session

Start a continuous question-and-answer session in the terminal:

```bash
python run_rag.py --mode query --interactive
```

Type `exit` or `quit` to end the session.

---

## Guardrails

The system enforces five safety mechanisms on every request:

| Guardrail | What It Does |
|-----------|-------------|
| **Scope check** | Rejects queries unrelated to ophthalmology and eye disease |
| **Injection sanitisation** | Blocks prompt injection attempts ("ignore instructions", etc.) |
| **Confidence labelling** | Converts raw model certainty scores to `Confirmed` / `Possible — Flagged for Clinical Review` labels |
| **Web result labelling** | Clearly marks supplementary web-sourced content separately from guideline evidence |
| **Clinical disclaimer** | Appended to every response automatically — cannot be bypassed |

All responses include:
> *"This response is generated from published clinical guidelines and is intended for educational and informational purposes only. It does NOT constitute a medical diagnosis, clinical decision, or personal medical advice."*

---

## Sample Output

**Input:** `inference_results.json` — fundus image with predicted Hypertensive Retinopathy (0.787 prob, 0.573 certainty) and AMD (0.516 prob, 0.031 certainty)

**Detected Conditions:**
- `[CONFIRMED]` Hypertensive Retinopathy — High confidence, High certainty
- `[POSSIBLE]` Age-Related Macular Degeneration — Low confidence, Low certainty — flagged for clinical review

**Sources retrieved (11):** AAO_AMD_PPP_2024 · EURETINA_AMD_2023 · RCOphth_AMD_2024 · REVIEW_HTN_RET_CLINICAL · REVIEW_SURV_OPHTH_2026 · TA155 · IPG339 · IPG415 · RCOphth_DR_2024 · REVIEW_HTN_RET_2 · WHO_DR_SCREENING_2020

---

## Building the Knowledge Base

The FAISS VDB is already built and included. These steps are only needed if you want to add new PDF sources.

**Step 1 — Process new PDFs** (place PDFs in `guidance 2/`):
```bash
python process_guidance2/process_all.py
```

**Step 2 — Append to FAISS index:**
```bash
python process_guidance2/append_to_vdb.py
```

**Step 3 — Validate retrieval:**
```bash
python process_guidance2/test_g2_retrieval.py
```

> ⚠️ `append_to_vdb.py` is additive — running it twice will duplicate chunks. If you need to rebuild from scratch, delete `faiss.index`, `metadata.pkl`, and `chunks.jsonl` and re-run from the NICE processing pipeline first.

---

## Dependencies

```
faiss-cpu
sentence-transformers
groq
gradio
duckduckgo-search
python-dotenv
pdfplumber
numpy
```

---

## License

This project is developed as part of a capstone research project. All clinical guideline content used to build the knowledge base belongs to the respective issuing organisations (NICE, AAO, WHO, EURETINA, RCOphth, ADA). This system is for research and educational purposes only.

---

*Part of the Multi-Disease Retinal Screening via Vision Transformer with Agentic RAG-Based Clinical Decision Support*
*Ophthalmology-ViT: (https://github.com/hemant-kothari/Ophthalmology-ViT) (ViT based multi-disease retinal screening system)*
