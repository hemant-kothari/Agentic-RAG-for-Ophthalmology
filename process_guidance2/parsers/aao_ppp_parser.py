"""
aao_ppp_parser.py
Parser for AAO Preferred Practice Pattern (PPP) full guidelines.

Format: Heading-based sections (Title Case or ALL CAPS headings on own line).
        Contains financial disclosures pages, committee lists, etc. (skip these).
        Actual clinical content starts after the "HIGHLIGHTS" or "INTRODUCTION" section.

Covers:
- Age-Related Macular Degeneration PPP.pdf
- Diabetic Retinopathy PPP.pdf
- Primary Open-Angle Glaucoma PPP.pdf
- Primary Open-Angle Glaucoma Suspect PPP.pdf
"""

import pdfplumber
import re
from pathlib import Path
from parser_utils import clean_text, infer_section_type

# Lines to skip that are characteristic of AAO PPP boilerplate
_AAO_SKIP_PATTERNS = [
    re.compile(r"preferred practice pattern", re.IGNORECASE),
    re.compile(r"american academy of ophthalmology", re.IGNORECASE),
    re.compile(r"retina/vitreous preferred practice", re.IGNORECASE),
    re.compile(r"^\s*P\d+\s*$"),                          # page labels like "P2", "P14"
    re.compile(r"© \d{4} american academy", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"financial disclosure", re.IGNORECASE),
    re.compile(r"no financial relationships to disclose", re.IGNORECASE),
    re.compile(r"aao\.org", re.IGNORECASE),
    re.compile(r"^\s*S\d+\s*$"),                          # "S262" style page labels
    re.compile(r"Diabetes Care Volume", re.IGNORECASE),
]

# Section headings in AAO PPPs are ALL CAPS (min 4 chars) or Title Case lines of 3-8 words
_HEADING_PATTERN = re.compile(
    r"(?m)^([A-Z][A-Z ,\-/&:]{3,79}|"         # ALL CAPS
    r"(?:[A-Z][a-z]{1,20}\s){2,8}[A-Z][a-z]{1,20})$"  # Title Case multi-word
)

# Skip sections that are purely metadata
_SKIP_SECTION_TITLES = {
    "FINANCIAL DISCLOSURES", "RETINA/VITREOUS PREFERRED PRACTICE",
    "PREFERRED PRACTICE PATTERNS COMMITTEE", "PARTICIPANTS",
    "RETINA/VITREOUS PREFERRED PRACTICE PATTERN COMMITTEE",
    "DEVELOPMENT PROCESS AND PARTICIPANTS",
    "ACADEMY REVIEWERS", "INVITED REVIEWERS",
    "CORRESPONDENCE", "MEDICAL EDITOR",
}


def extract_text_from_pdf(pdf_path: Path) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                cleaned = clean_text(text, extra_strip_patterns=_AAO_SKIP_PATTERNS)
                full_text += cleaned + "\n"
    return full_text


def find_content_start(full_text: str) -> int:
    """Skip front matter: find first major content section."""
    for marker in [
        "HIGHLIGHTS", "INTRODUCTION", "BACKGROUND", "DISEASE DEFINITION",
        "PATIENT POPULATION", "CLINICAL OBJECTIVES", "EPIDEMIOLOGY",
    ]:
        idx = full_text.find(f"\n{marker}\n")
        if idx >= 0:
            return idx
    return 0


def split_aao_sections(full_text: str) -> list:
    """Split on ALL CAPS or Title Case headings."""
    matches = list(_HEADING_PATTERN.finditer(full_text))

    sections = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()

        # Skip metadata sections
        if any(skip in title.upper() for skip in _SKIP_SECTION_TITLES):
            continue

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        content = full_text[start:end].strip()

        # Skip very short sections (likely boilerplate)
        if len(content) < 80:
            continue

        sections.append({
            "section_number": str(len(sections) + 1),
            "section_title": title,
            "section_type": infer_section_type(title),
            "content": content,
        })

    return sections


def parse_aao_ppp(pdf_path: Path, meta: dict) -> dict:
    """
    Parse an AAO PPP PDF into the standard JSON schema.
    meta: dict with keys: guideline_id, title, source_org, disease, year
    """
    full_text = extract_text_from_pdf(pdf_path)
    content_start = find_content_start(full_text)
    content_text = full_text[content_start:]
    sections = split_aao_sections(content_text)

    return {
        "guideline_id": meta["guideline_id"],
        "title": meta["title"],
        "source": meta.get("source", "American Academy of Ophthalmology"),
        "source_org": meta.get("source_org", "AAO"),
        "source_type": meta.get("source_type", "Clinical Practice Guideline"),
        "disease": meta.get("disease", ""),
        "year": meta.get("year"),
        "sections": sections,
    }
