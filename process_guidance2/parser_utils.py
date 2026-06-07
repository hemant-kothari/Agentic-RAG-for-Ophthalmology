"""
parser_utils.py — Shared utilities for all Guidance 2 parsers.

Provides:
- clean_text(): strip common junk (page numbers, running headers, ligatures)
- split_into_sections(): regex-based section splitter (numbered or heading-based)
- infer_section_type(): map section title to semantic type
- make_guideline_id(): generate a clean unique ID from source + disease + year
"""

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------------------------

# Unicode ligature fixes (common in PDFs)
LIGATURE_MAP = str.maketrans({
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",   # non-breaking space
    "\u00ad": "",    # soft hyphen
})

# Patterns to strip regardless of source
_STRIP_PATTERNS = [
    re.compile(r"^\d+\s*$"),                                  # lone page numbers
    re.compile(r"^Page\s+\d+\s+of\s+\d+", re.IGNORECASE),   # "Page N of M"
    re.compile(r"^(www\.|http)", re.IGNORECASE),              # URLs as lone lines
    re.compile(r"^\s*[-–—]{3,}\s*$"),                         # horizontal rules
    re.compile(r"^©", re.IGNORECASE),                         # copyright lines
    re.compile(r"^All rights reserved", re.IGNORECASE),
    re.compile(r"^Downloaded from", re.IGNORECASE),
    re.compile(r"^This (guideline|document) (was|is) (produced|published|prepared)", re.IGNORECASE),
]


def clean_text(text: str, extra_strip_patterns: list = None) -> str:
    """
    Clean raw PDF-extracted text:
    - Fix ligatures and unicode issues
    - Remove page numbers, footers, running headers
    - Collapse multiple blank lines to one
    - Optionally strip source-specific patterns
    """
    text = text.translate(LIGATURE_MAP)

    patterns = _STRIP_PATTERNS + (extra_strip_patterns or [])
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        skip = False
        for pat in patterns:
            if pat.search(stripped):
                skip = True
                break
        if not skip:
            cleaned_lines.append(stripped)

    # Collapse 3+ consecutive blank lines into 2
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines))
    return result.strip()


# ---------------------------------------------------------------------------
# SECTION TYPE INFERENCE
# ---------------------------------------------------------------------------

_SECTION_TYPE_RULES = [
    (re.compile(r"\babstract\b", re.IGNORECASE),             "Abstract"),
    (re.compile(r"\bintroduction\b", re.IGNORECASE),         "Introduction"),
    (re.compile(r"\bbackground\b", re.IGNORECASE),           "Background"),
    (re.compile(r"\brecommendation", re.IGNORECASE),         "Recommendation"),
    (re.compile(r"\bdiagnos", re.IGNORECASE),                "Diagnosis"),
    (re.compile(r"\bscreening\b", re.IGNORECASE),            "Screening"),
    (re.compile(r"\bclassification\b", re.IGNORECASE),       "Classification"),
    (re.compile(r"\btreatment\b|\bmanagement\b|\btherapy\b", re.IGNORECASE), "Treatment"),
    (re.compile(r"\bsurgical\b|\bsurgery\b", re.IGNORECASE), "Surgical"),
    (re.compile(r"\bpharmacol|\bdrug\b|\bmedication\b", re.IGNORECASE), "Pharmacology"),
    (re.compile(r"\bepidemiolog|\bprevalence\b|\bincidence\b", re.IGNORECASE), "Epidemiology"),
    (re.compile(r"\bpathophysiol|\bpathogen", re.IGNORECASE),"Pathophysiology"),
    (re.compile(r"\befficacy\b|\bevidence\b", re.IGNORECASE),"Evidence"),
    (re.compile(r"\bsafety\b|\badverse\b|\bside effect", re.IGNORECASE), "Safety"),
    (re.compile(r"\bprognos", re.IGNORECASE),                "Prognosis"),
    (re.compile(r"\bfollow.?up\b|\bmonitoring\b", re.IGNORECASE), "Monitoring"),
    (re.compile(r"\breference", re.IGNORECASE),              "References"),
    (re.compile(r"\bappendix\b|\bannex", re.IGNORECASE),     "Appendix"),
    (re.compile(r"\bmethod", re.IGNORECASE),                 "Methods"),
    (re.compile(r"\bresult", re.IGNORECASE),                 "Results"),
    (re.compile(r"\bdiscussion\b", re.IGNORECASE),           "Discussion"),
    (re.compile(r"\bconclusion", re.IGNORECASE),             "Conclusion"),
    (re.compile(r"\bsummary\b", re.IGNORECASE),              "Summary"),
    (re.compile(r"\brisk factor", re.IGNORECASE),            "Risk Factors"),
    (re.compile(r"\bprevention\b", re.IGNORECASE),           "Prevention"),
]


def infer_section_type(title: str) -> str:
    for pattern, label in _SECTION_TYPE_RULES:
        if pattern.search(title):
            return label
    return "Other"


# ---------------------------------------------------------------------------
# NUMBERED SECTION SPLITTER  (e.g. AAO PPP, RCOphth)
# ---------------------------------------------------------------------------

def split_numbered_sections(full_text: str) -> list:
    """
    Split text on numbered section headers like:
        1. Introduction
        2.1 Diagnosis
        IV. Treatment
    Returns list of dicts: {section_number, section_title, section_type, content}
    """
    # Matches: "1.", "2.1", "2.1.3", "IV.", "A." at line start
    pattern = re.compile(
        r"(?m)^(\d+(?:\.\d+)*|[IVXLC]+|[A-Z])\.\s{1,4}([A-Z][^\n]{2,80})\s*\n"
    )

    matches = list(pattern.finditer(full_text))
    sections = []

    for i, match in enumerate(matches):
        sec_num = match.group(1)
        sec_title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        content = full_text[start:end].strip()

        if len(content) < 30:  # skip near-empty sections
            continue

        sections.append({
            "section_number": sec_num,
            "section_title": sec_title,
            "section_type": infer_section_type(sec_title),
            "content": content,
        })

    return sections


# ---------------------------------------------------------------------------
# HEADING-BASED SECTION SPLITTER  (e.g. EURETINA, ADA, WHO)
# ---------------------------------------------------------------------------

def split_heading_sections(full_text: str, heading_patterns: list = None) -> list:
    """
    Split text on ALL-CAPS or Title-Case headings at line start.
    heading_patterns: list of compiled regex patterns for headings (optional override).
    Returns list of dicts: {section_number, section_title, section_type, content}
    """
    if heading_patterns is None:
        heading_patterns = [
            # ALL CAPS heading (min 4 chars, max 80)
            re.compile(r"(?m)^([A-Z][A-Z\s\-/,]{3,79})$"),
            # Title Case heading (2+ words, no period at end typical of running text)
            re.compile(r"(?m)^([A-Z][a-z]+(?:\s+[A-Z][a-z]*){1,8})$"),
        ]

    # Find all heading positions
    heading_spans = []
    for pat in heading_patterns:
        for m in pat.finditer(full_text):
            heading_spans.append((m.start(), m.end(), m.group(1).strip()))

    if not heading_spans:
        # Fallback: return full text as a single section
        return [{
            "section_number": "1",
            "section_title": "Content",
            "section_type": "Other",
            "content": full_text.strip(),
        }]

    # Sort and deduplicate by position
    heading_spans.sort(key=lambda x: x[0])
    seen_starts = set()
    unique_spans = []
    for start, end, title in heading_spans:
        if start not in seen_starts:
            seen_starts.add(start)
            unique_spans.append((start, end, title))

    sections = []
    for i, (start, end, title) in enumerate(unique_spans):
        content_start = end
        content_end = unique_spans[i + 1][0] if i + 1 < len(unique_spans) else len(full_text)
        content = full_text[content_start:content_end].strip()

        if len(content) < 30:
            continue

        sections.append({
            "section_number": str(i + 1),
            "section_title": title,
            "section_type": infer_section_type(title),
            "content": content,
        })

    return sections


# ---------------------------------------------------------------------------
# ACADEMIC PAPER SECTION SPLITTER
# ---------------------------------------------------------------------------

ACADEMIC_SECTION_KEYWORDS = [
    "Abstract", "Introduction", "Background", "Methods", "Materials and Methods",
    "Results", "Discussion", "Conclusion", "Conclusions", "References",
    "Funding", "Acknowledgements", "Acknowledgments", "Supplementary",
    "Conflict of Interest", "Author Contributions", "Epidemiology",
    "Pathophysiology", "Diagnosis", "Classification", "Treatment",
    "Management", "Screening", "Prevention", "Prognosis", "Summary",
    "Grading", "Imaging", "Risk Factors", "Complications",
]

def split_academic_sections(full_text: str) -> list:
    """
    Split academic paper text on well-known section headings.
    Case-insensitive match at line start.
    """
    keyword_pattern = re.compile(
        r"(?m)^(" + "|".join(re.escape(k) for k in ACADEMIC_SECTION_KEYWORDS) + r")\s*\n",
        re.IGNORECASE
    )

    matches = list(keyword_pattern.finditer(full_text))
    if not matches:
        return [{
            "section_number": "1",
            "section_title": "Content",
            "section_type": "Other",
            "content": full_text.strip(),
        }]

    sections = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        content = full_text[start:end].strip()

        if len(content) < 30:
            continue

        sections.append({
            "section_number": str(i + 1),
            "section_title": title,
            "section_type": infer_section_type(title),
            "content": content,
        })

    return sections


# ---------------------------------------------------------------------------
# ID GENERATOR
# ---------------------------------------------------------------------------

_ORG_CODES = {
    "american academy of ophthalmology": "AAO",
    "american diabetes association": "ADA",
    "euretina": "EURETINA",
    "royal college of ophthalmologists": "RCOphth",
    "world health organization": "WHO",
    "who": "WHO",
    "vision 2025": "INDIA",
    "india": "INDIA",
    "nature": "NATURE",
    "survey of ophthalmology": "SurvOphth",
    "researchgate": "RG",
    "review": "REVIEW",
}

_DISEASE_CODES = {
    "diabetic retinopathy": "DR",
    "diabetic macular": "DMO",
    "age-related macular": "AMD",
    "macular degeneration": "AMD",
    "glaucoma": "GLAUCOMA",
    "hypertensive retinopathy": "HTN_RET",
    "open-angle glaucoma": "POAG",
    "open angle glaucoma": "POAG",
}


def make_guideline_id(source_org: str, disease: str, year: int = None, suffix: str = "") -> str:
    """Generate a clean unique guideline ID like AAO_DR_2024."""
    org = source_org.upper().replace(" ", "_")
    dis = disease.upper().replace(" ", "_").replace("-", "_")
    yr = f"_{year}" if year else ""
    sfx = f"_{suffix}" if suffix else ""
    return f"{org}_{dis}{yr}{sfx}"
