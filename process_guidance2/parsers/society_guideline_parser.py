"""
society_guideline_parser.py
Parser for numbered clinical practice guidelines from:
- Royal College of Ophthalmologists (RCOphth): clear numeric TOC (1., 2., 2.1)
- WHO: numbered sections with dotted TOC (1., 4.1.)
- India Vision 2025: Roman numerals + section keywords

Format: Numbered sections detected via regex at start of line.
Handles both integer sections (1.) and decimal subsections (2.1, 4.1.2).
"""

import pdfplumber
import re
from pathlib import Path
from parser_utils import clean_text, infer_section_type

# Skip patterns specific to RCOphth / WHO / India
_SOCIETY_SKIP_PATTERNS = [
    re.compile(r"© The Royal College of Ophthalmologists", re.IGNORECASE),
    re.compile(r"rcophth\.ac\.uk", re.IGNORECASE),
    re.compile(r"2024/PROF/\d+", re.IGNORECASE),          # RCOphth doc number footers
    re.compile(r"© World Health Organization", re.IGNORECASE),
    re.compile(r"CC BY-NC-SA 3\.0", re.IGNORECASE),
    re.compile(r"^iv\s*iv\s*$", re.IGNORECASE),           # WHO "iv iv" page markers
    re.compile(r"^v\s*v\s*$", re.IGNORECASE),
    re.compile(r"^VISION 2020.*India$", re.IGNORECASE),
    re.compile(r"^Contact:? ", re.IGNORECASE),
    re.compile(r"^\s*\d{4}/\w+/\d+\s*\d+\s*$"),           # RCOphth footer codes
    re.compile(r"^Table of Contents$", re.IGNORECASE),
    re.compile(r"^Contents$", re.IGNORECASE),
]

# Numbered section pattern: "1.", "2.1", "2.1.3", "I.", "II.", "III." etc.
# The title group must NOT end with a bare number (which would indicate a TOC page ref)
_NUM_SECTION_PATTERN = re.compile(
    r"(?m)^(\d+(?:\.\d+)*|[IVX]{1,5})\.\s{1,5}([A-Z][^\n]{3,100}[^\d\s])\s*$"
)

# Minimum characters of content a section must have to be kept
_MIN_SECTION_CONTENT = 200

# Roman numeral section (India format: "I. Introduction")
_ROMAN_SECTION_PATTERN = re.compile(
    r"(?m)^([IVX]{1,4})\.\s+([A-Z][^\n]{3,80})\s*$"
)

# TOC line — skip lines that are clearly table of contents
_TOC_PATTERN = re.compile(r"\.{5,}|\s{3,}\d+\s*$")  # lots of dots or trailing page #


def extract_text_from_pdf(pdf_path: Path) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                cleaned = clean_text(text, extra_strip_patterns=_SOCIETY_SKIP_PATTERNS)
                full_text += cleaned + "\n"
    return full_text


def find_toc_end(full_text: str) -> int:
    """
    Skip the Table of Contents section.
    Look for a transition from TOC-like lines to actual content.
    """
    toc_markers = ["Contents\n", "Table of Contents\n", "TABLE OF CONTENTS\n"]
    toc_start = -1
    for marker in toc_markers:
        idx = full_text.find(marker)
        if idx >= 0:
            toc_start = idx
            break

    if toc_start < 0:
        return 0

    # Find where TOC lines end (lines without dots and page numbers)
    lines = full_text[toc_start:].splitlines()
    consecutive_non_toc = 0
    pos = toc_start
    for line in lines:
        if _TOC_PATTERN.search(line):
            consecutive_non_toc = 0
        elif line.strip() and not _TOC_PATTERN.search(line):
            consecutive_non_toc += 1
        if consecutive_non_toc >= 5:
            return pos
        pos += len(line) + 1
    return toc_start


def split_numbered_sections(full_text: str) -> list:
    """
    Split on lines like: "1. Introduction", "2.1 Screening", "IV. Treatment"
    Skip TOC lines (contain dots or trailing page numbers).
    """
    matches = list(_NUM_SECTION_PATTERN.finditer(full_text))

    # Filter out TOC entries
    content_matches = []
    for m in matches:
        line = m.group(0)
        # If this line looks like a TOC entry (has trailing page numbers or dots), skip
        if _TOC_PATTERN.search(line):
            continue
        content_matches.append(m)

    sections = []
    for i, match in enumerate(content_matches):
        sec_num = match.group(1)
        sec_title = match.group(2).strip()
        start = match.end()
        end = content_matches[i + 1].start() if i + 1 < len(content_matches) else len(full_text)
        content = full_text[start:end].strip()

        # Skip sections with too little content (TOC artifacts, page headers)
        if len(content) < _MIN_SECTION_CONTENT:
            continue

        # Skip appendices and references unless they have substantial content
        if sec_title.lower() in {"references", "index"} and len(content) < 500:
            continue

        # Skip if title itself ends in a digit (leftover TOC "Introduction 7" etc.)
        if re.search(r'\d+\s*$', sec_title):
            continue

        sections.append({
            "section_number": sec_num,
            "section_title": sec_title,
            "section_type": infer_section_type(sec_title),
            "content": content,
        })

    return sections


def parse_society_guideline(pdf_path: Path, meta: dict) -> dict:
    """
    Parse a society clinical guideline PDF (RCOphth, WHO, India) into standard JSON.
    meta: dict with keys: guideline_id, title, source, source_org, source_type, disease, year
    """
    full_text = extract_text_from_pdf(pdf_path)

    # Skip TOC section
    content_start = find_toc_end(full_text)
    content_text = full_text[content_start:]

    sections = split_numbered_sections(content_text)

    if not sections:
        # Fallback: treat entire text as one section
        sections = [{
            "section_number": "1",
            "section_title": "Content",
            "section_type": "Other",
            "content": content_text.strip(),
        }]

    return {
        "guideline_id": meta["guideline_id"],
        "title": meta["title"],
        "source": meta.get("source", ""),
        "source_org": meta.get("source_org", ""),
        "source_type": meta.get("source_type", "Clinical Practice Guideline"),
        "disease": meta.get("disease", ""),
        "year": meta.get("year"),
        "sections": sections,
    }
