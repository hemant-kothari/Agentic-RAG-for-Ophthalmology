"""
agent.py — RetinalRAGAgent: Main orchestrator for the Agentic RAG system.

Two modes:
  1. run_image_mode(inference_json_path)  — structured report from ViT output
  2. run_query_mode(user_query)           — open-ended eye care query

Agentic loop:
  Input → Guardrails → Router → Retrieve (per-disease queries) →
  Relevance check → [Web fallback] → Context build → LLM → Format → Output
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_rag.llm_client    import LLMClient, MAX_TOKENS_REPORT, MAX_TOKENS_QUERY
from agentic_rag.input_parser  import parse_inference_results
from agentic_rag.guardrails    import (
    is_off_topic, sanitize_input, add_clinical_disclaimer,
    get_off_topic_response,
)
from agentic_rag.web_search    import web_search
from agentic_rag.context_builder import build_context
from .retriever                  import retrieve, max_score

# ── Constants ─────────────────────────────────────────────────────────────────
WEB_FALLBACK_THRESHOLD = 0.35   # If max KB score < this → add web search
TOP_K_IMAGE            = 8      # Chunks per disease in image mode
TOP_K_QUERY            = 10     # Chunks for query mode

# ── Prompt loading ─────────────────────────────────────────────────────────────
_PROMPT_DIR = Path(__file__).parent / "prompts"

def _load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")

SYSTEM_IMAGE = _load_prompt("system_image.txt")
SYSTEM_QUERY = _load_prompt("system_query.txt")
USER_IMAGE   = _load_prompt("user_image.txt")
USER_QUERY   = _load_prompt("user_query.txt")

# ── Disease → retrieval query mapping ─────────────────────────────────────────
DISEASE_QUERIES: dict[str, list[str]] = {
    "Hypertensive Retinopathy": [
        "hypertensive retinopathy classification grading management",
        "blood pressure retinal damage treatment monitoring",
        "hypertensive retinopathy screening follow-up guidelines",
    ],
    "Age-Related Macular Degeneration (AMD)": [
        "AMD anti-VEGF treatment neovascular macular degeneration",
        "age-related macular degeneration screening monitoring classification",
        "dry AMD geographic atrophy management guidelines",
    ],
    "Diabetic Retinopathy (Presence)": [
        "diabetic retinopathy screening grading classification",
        "diabetic retinopathy treatment laser anti-VEGF guidelines",
        "DR monitoring HbA1c blood pressure control",
    ],
    "Glaucoma": [
        "glaucoma intraocular pressure treatment target guidelines",
        "primary open angle glaucoma screening monitoring management",
        "glaucoma visual field optic nerve assessment",
    ],
    "Diabetic Macular Edema": [
        "diabetic macular edema anti-VEGF treatment guidelines",
        "DME management laser intravitreal injection",
    ],
}

GENERIC_QUERIES = [
    "retinal disease screening monitoring guidelines",
    "ophthalmology clinical recommendations eye disease management",
]


class RetinalRAGAgent:
    def __init__(self):
        self.llm = LLMClient()

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: IMAGE MODE
    # ─────────────────────────────────────────────────────────────────────────
    def run_image_mode(self, inference_json_path: str | Path) -> dict:
        """
        Process ViT inference output and return a structured report.

        Returns:
        {
          "status": "ok" | "error",
          "parsed_input": dict,
          "response": str,           # formatted LLM response
          "sources_used": list[str], # guideline IDs cited
          "web_used": bool,
        }
        """
        # 1. Parse ViT output
        parsed = parse_inference_results(inference_json_path)
        all_findings = parsed["confirmed_findings"] + parsed["possible_findings"]

        if not all_findings:
            return {
                "status":       "ok",
                "parsed_input": parsed,
                "response":     "No positive findings were detected by the AI model. No clinical context to retrieve.",
                "sources_used": [],
                "web_used":     False,
            }

        # 2. Retrieve per-disease context
        all_kb_results = []
        sources_used   = set()

        for finding in all_findings:
            condition = finding["condition"]
            queries   = DISEASE_QUERIES.get(condition, [f"{condition} management guidelines"])

            for q in queries:
                results = retrieve(q, top_k=TOP_K_IMAGE)
                all_kb_results.extend(results)
                for r in results:
                    sources_used.add(r["guideline_id"])

        # Deduplicate by text
        seen, deduped = set(), []
        for r in sorted(all_kb_results, key=lambda x: -x["score"]):
            if r["text"][:80] not in seen:
                seen.add(r["text"][:80])
                deduped.append(r)

        # 3. Web fallback
        web_results = []
        web_used    = False
        top_score   = max_score(deduped)
        if top_score < WEB_FALLBACK_THRESHOLD:
            finding_names = [f["condition"] for f in all_findings]
            web_query     = " ".join(finding_names) + " clinical guidelines treatment"
            web_results   = web_search(web_query)
            web_used      = bool(web_results)

        # 4. Build context
        context = build_context(deduped[:20], web_results)

        # 5. Build user prompt
        findings_summary = parsed["summary_text"]
        conditions_block = _format_conditions_block(parsed)
        dr_block         = _format_dr_block(parsed["dr_severity"])

        user_prompt = USER_IMAGE.format(
            findings_summary = findings_summary,
            conditions_block = conditions_block,
            dr_severity_block = dr_block,
            context          = context,
        )

        # 6. LLM call
        raw_response = self.llm.generate(
            system_prompt = SYSTEM_IMAGE,
            user_prompt   = user_prompt,
            max_tokens    = MAX_TOKENS_REPORT,
        )

        # 7. Add disclaimer
        final_response = add_clinical_disclaimer(raw_response)

        return {
            "status":       "ok",
            "parsed_input": parsed,
            "response":     final_response,
            "sources_used": sorted(sources_used),
            "web_used":     web_used,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: QUERY MODE
    # ─────────────────────────────────────────────────────────────────────────
    def run_query_mode(self, user_query: str) -> dict:
        """
        Answer an open-ended eye care question.

        Returns:
        {
          "status": "ok" | "off_topic" | "blocked",
          "response": str,
          "sources_used": list[str],
          "web_used": bool,
        }
        """
        # 1. Sanitise
        clean_query, injected = sanitize_input(user_query)
        if injected:
            return {
                "status":       "blocked",
                "response":     "Your input contained disallowed content and could not be processed.",
                "sources_used": [],
                "web_used":     False,
            }

        # 2. Off-topic guard
        if is_off_topic(clean_query):
            return {
                "status":       "off_topic",
                "response":     get_off_topic_response(),
                "sources_used": [],
                "web_used":     False,
            }

        # 3. Retrieve from KB
        kb_results   = retrieve(clean_query, top_k=TOP_K_QUERY)
        sources_used = {r["guideline_id"] for r in kb_results}
        top_score    = max_score(kb_results)

        # 4. Web fallback if KB relevance is low
        web_results = []
        web_used    = False
        if top_score < WEB_FALLBACK_THRESHOLD:
            web_results = web_search(clean_query)
            web_used    = bool(web_results)

        # 5. Build context
        context = build_context(kb_results, web_results)

        # 6. User prompt
        user_prompt = USER_QUERY.format(query=clean_query, context=context)

        # 7. LLM call
        raw_response = self.llm.generate(
            system_prompt = SYSTEM_QUERY,
            user_prompt   = user_prompt,
            max_tokens    = MAX_TOKENS_QUERY,
        )

        # 8. Disclaimer
        final_response = add_clinical_disclaimer(raw_response)

        return {
            "status":       "ok",
            "response":     final_response,
            "sources_used": sorted(sources_used),
            "web_used":     web_used,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────
def _format_conditions_block(parsed: dict) -> str:
    lines = []
    for c in parsed["confirmed_findings"]:
        lines.append(
            f"  - {c['condition']}: CONFIRMED | "
            f"Model confidence: {c['prob_label']} | {c['certainty_label']}"
        )
    for c in parsed["possible_findings"]:
        lines.append(
            f"  - {c['condition']}: POSSIBLE (flagged for clinical review) | "
            f"Model confidence: {c['prob_label']} | {c['certainty_label']}"
        )
    return "\n".join(lines) if lines else "  No positive findings."


def _format_dr_block(dr: dict | None) -> str:
    if not dr:
        return "No DR severity assessment available."
    return (
        f"  Grade: {dr['grade']} — {dr['label']}\n"
        f"  Grade certainty: {'High' if dr['certainty'] > 0.9 else 'Moderate' if dr['certainty'] > 0.7 else 'Low'}"
    )
