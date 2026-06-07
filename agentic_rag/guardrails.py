"""
guardrails.py — Input/output safety checks for the Agentic RAG system.

Implements:
  1. Off-topic detection (keyword + scope check)
  2. Input sanitisation (prompt injection)
  3. Clinical disclaimer injection
  4. Scope description for user-facing messages
"""

import re

# ── Scope: diseases and topics covered by the KB ────────────────────────────
IN_SCOPE_KEYWORDS = [
    # Diseases
    "diabetic retinopathy", "diabetic macular", "macular degeneration", "amd",
    "age-related macular", "glaucoma", "hypertensive retinopathy", "hypertension",
    "retinal", "retina", "optic nerve", "optic disc", "fundus", "macula",
    "vitreous", "choroid", "fovea", "drusen", "neovascular",
    # Symptoms & findings
    "visual acuity", "vision loss", "blindness", "scotoma", "floaters",
    "photopsia", "metamorphopsia", "contrast sensitivity", "visual field",
    # Diagnostics
    "oct", "optical coherence tomography", "fluorescein angiography", "ffa",
    "fundus photography", "intraocular pressure", "iop", "tonometry",
    "humphrey", "perimetry", "biomicroscopy",
    # Treatments
    "anti-vegf", "ranibizumab", "bevacizumab", "aflibercept", "faricimab",
    "brolucizumab", "laser", "photocoagulation", "vitrectomy",
    "photodynamic therapy", "intravitreal injection",
    # Screening & management
    "screening", "grading", "classification", "referral", "monitoring",
    "treatment", "management", "guideline", "recommendation",
    # Eye anatomy & general
    "eye", "ocular", "ophthalmology", "optometry", "vision", "sight",
    "cornea", "lens", "pupil", "iris", "cataract",
]

# ── Hard off-topic categories ────────────────────────────────────────────────
OFF_TOPIC_PATTERNS = [
    re.compile(r"\b(weather|restaurant|food|recipe|sport|football|cricket)\b", re.I),
    re.compile(r"\b(stock market|crypto|bitcoin|invest)\b", re.I),
    re.compile(r"\b(politics|election|president|government)\b", re.I),
    re.compile(r"\b(movie|music|celebrity|gossip)\b", re.I),
    re.compile(r"\b(code|programming|software|javascript|python tutorial)\b", re.I),
    re.compile(r"\b(general medicine|cardiology|orthopedic|dermatology)\b", re.I),
]

# ── Prompt injection patterns ────────────────────────────────────────────────
INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?previous instructions?", re.I),
    re.compile(r"system prompt", re.I),
    re.compile(r"you are now", re.I),
    re.compile(r"disregard (your )?instructions?", re.I),
    re.compile(r"act as (a|an|if)", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"dan mode", re.I),
]

CLINICAL_DISCLAIMER = (
    "\n\n---\n"
    "**Clinical Disclaimer:** This response is generated from published clinical "
    "guidelines and is intended for educational and informational purposes only. "
    "It does NOT constitute a medical diagnosis, clinical decision, or personal "
    "medical advice. All findings should be reviewed and interpreted by a qualified "
    "ophthalmologist or healthcare professional."
)

SCOPE_MESSAGE = (
    "I'm a clinical knowledge assistant specialised in **retinal and eye diseases**. "
    "I can help with:\n"
    "- Diabetic Retinopathy & Diabetic Macular Edema\n"
    "- Age-Related Macular Degeneration (AMD)\n"
    "- Glaucoma (Primary Open-Angle)\n"
    "- Hypertensive Retinopathy\n"
    "- General eye screening, diagnostics, and treatment guidelines\n\n"
    "Please ask a question related to these topics."
)

OFF_TOPIC_RESPONSE = (
    "I'm sorry, but that question is outside my area of expertise. "
    "I'm designed to provide information about **retinal and ocular diseases** "
    "based on clinical guidelines (NICE, AAO, WHO, EURETINA, RCOphth, ADA, and more).\n\n"
    + SCOPE_MESSAGE
)


def is_off_topic(query: str) -> bool:
    """
    Returns True if the query is clearly off-topic (not about eye care).
    Uses keyword presence (in-scope wins) + pattern matching (off-topic).
    """
    q_lower = query.lower()

    # If any in-scope keyword is present → not off-topic
    for kw in IN_SCOPE_KEYWORDS:
        if kw in q_lower:
            return False

    # If any off-topic pattern matches → off-topic
    for pat in OFF_TOPIC_PATTERNS:
        if pat.search(query):
            return True

    # Short queries with no eye keywords are ambiguous — treat as possibly in-scope
    # (the LLM router will handle further classification)
    return False


def sanitize_input(query: str) -> tuple[str, bool]:
    """
    Check for prompt injection attempts.
    Returns (sanitized_query, was_injected: bool).
    If injection detected, returns a safe fallback query.
    """
    for pat in INJECTION_PATTERNS:
        if pat.search(query):
            return "[BLOCKED: Invalid input detected]", True
    # Truncate extremely long inputs
    if len(query) > 2000:
        query = query[:2000] + "..."
    return query, False


def add_clinical_disclaimer(response: str) -> str:
    """Append the clinical disclaimer to every response."""
    return response + CLINICAL_DISCLAIMER


def get_off_topic_response() -> str:
    return OFF_TOPIC_RESPONSE


def get_scope_message() -> str:
    return SCOPE_MESSAGE
