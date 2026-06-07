"""
academic_review_parser.py
Parser for academic/journal review papers and EURETINA guidelines.

These are published as journal articles (two-column layout typically degrades to
single column under pdfplumber). Key features:
- Title, authors, abstract all on page 1
- Section headings appear at column breaks (Title Case or ALL CAPS on own line)
- No numbered TOC — use keyword-based section matching

Covers:
- EURETINA diabetic retinopathy guideline pdf.pdf
- EURETINA guidelines AMD pdf.pdf
- ADA standards of care diabetic retinopathy pdf.pdf
- diabetic retinopathy review nature pdf.pdf
- 1-s2.0-S0039625726000160-main.pdf  (Survey of Ophthalmology)
- hypertensive retinopathy clinical review.pdf
- hypertensive retinopathy review pdf.pdf
- primary open angle glaucoma review.pdf
- Glaucoma researrch gate.pdf
- BC-2208_PPPSummaryBenchmarks.17.Retina.pdf  (AAO benchmarks)
- PPP Summary Benchmarks.24.retina.pdf  (AAO benchmarks)
"""

import pdfplumber
import re
from pathlib import Path
from parser_utils import clean_text, infer_section_type, ACADEMIC_SECTION_KEYWORDS

# Skip patterns for journal papers
_JOURNAL_SKIP_PATTERNS = [
    re.compile(r"downloaded from", re.IGNORECASE),
    re.compile(r"by guest on", re.IGNORECASE),
    re.compile(r"^\d+\s+(Evidence|Ophthalmologica|Diabetes Care|Survey)", re.IGNORECASE),  # journal headers
    re.compile(r"DOI:\s*10\.", re.IGNORECASE),
    re.compile(r"Received:\s+\w+ \d+", re.IGNORECASE),
    re.compile(r"Accepted:\s+\w+ \d+", re.IGNORECASE),
    re.compile(r"Published online", re.IGNORECASE),
    re.compile(r"© \d{4}\s+\w", re.IGNORECASE),
    re.compile(r"^E-Mail\s+\w+@", re.IGNORECASE),
    re.compile(r"^www\.\w+\.com", re.IGNORECASE),
    re.compile(r"aao\.org", re.IGNORECASE),
    re.compile(r"American Academy of Ophthalmology", re.IGNORECASE),
    re.compile(r"Preferred Practice Pattern", re.IGNORECASE),
    re.compile(r"^\s*S\d{3}\s*$"),            # page markers like S262
]

# Extended academic section keyword list (ordered by priority for matching)
_SECTION_KEYWORDS = [
    # Standard sections
    "Abstract", "Background and Purpose", "Background",
    "Introduction", "Materials and Methods", "Methods",
    "Results and Conclusions", "Results",
    "Discussion", "Conclusions", "Conclusion",
    "References", "Funding", "Acknowledgements", "Acknowledgments",
    "Conflict of Interest", "Author Contributions", "Supplementary",
    # Clinical content sections (common in reviews)
    "Epidemiology", "Global Epidemiology", "Pathophysiology",
    "Classification", "Clinical Features", "Clinical Presentation",
    "Grading", "Staging", "Natural History",
    "Diagnosis", "Diagnostic Modalities", "Imaging",
    "Screening", "Screening Recommendations",
    "Treatment", "Management", "Treatment Strategies", "Pharmacological Management",
    "Anti-VEGF Therapy", "Laser Treatment", "Surgical Management",
    "Risk Factors", "Prevention", "Prognosis",
    "Summary", "Highlights", "Key Points",
    "Recommendations", "Guidelines",
    # EURETINA-specific
    "Diabetic Macular Edema", "Proliferative DR", "Angiographic Manifestations",
    "Optical Coherence Tomography",
    # ADA-specific
    "Diabetic Retinopathy", "Retinopathy", "Neuropathy",
    "DIABETIC RETINOPATHY",
    # Hypertensive retinopathy
    "Hypertensive Retinopathy", "Hypertension and the Eye",
    # Glaucoma
    "Intraocular Pressure", "Optic Nerve", "Visual Field",
    # Review-specific
    "Antiangiogenic Phytochemicals", "Anti-inflammatory",
    "Oxidative Stress",
]

# Build the pattern dynamically
_KW_PATTERN = re.compile(
    r"(?m)^(" + "|".join(re.escape(k) for k in _SECTION_KEYWORDS) + r")\s*\n",
    re.IGNORECASE
)

# Secondary pattern: ALL CAPS heading on own line (at least 4 chars, max 80)
_CAPS_HEADING = re.compile(r"(?m)^([A-Z][A-Z \-/&:,]{3,79})$")


def extract_text_from_pdf(pdf_path: Path) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                cleaned = clean_text(text, extra_strip_patterns=_JOURNAL_SKIP_PATTERNS)
                full_text += cleaned + "\n"
    return full_text


def split_paragraph_chunks(full_text: str, words_per_chunk: int = 600) -> list:
    """
    Fallback for two-column journal PDFs where headings appear mid-line.
    Split on double newlines (paragraph breaks), then group paragraphs into
    ~words_per_chunk word chunks. Labels each with a generic section number.
    """
    # Split on paragraph boundaries (2+ blank lines or double newline)
    paragraphs = re.split(r'\n{2,}', full_text)
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 80]

    sections = []
    current_words = []
    current_chars = 0
    sec_num = 1

    for para in paragraphs:
        words = para.split()
        current_words.extend(words)
        current_chars += len(para)

        if len(current_words) >= words_per_chunk:
            content = " ".join(current_words)
            sections.append({
                "section_number": str(sec_num),
                "section_title": f"Content Part {sec_num}",
                "section_type": "Other",
                "content": content,
            })
            sec_num += 1
            current_words = []

    # Remaining words
    if current_words:
        content = " ".join(current_words)
        sections.append({
            "section_number": str(sec_num),
            "section_title": f"Content Part {sec_num}",
            "section_type": "Other",
            "content": content,
        })

    return sections


def split_academic_sections(full_text: str) -> list:
    """
    Split on known academic section keywords, then fall back to ALL CAPS headings,
    then fall back to paragraph-based chunking for two-column PDFs.
    """
    # Primary: keyword-based
    kw_matches = list(_KW_PATTERN.finditer(full_text))

    if len(kw_matches) >= 3:
        matches = kw_matches
    else:
        # Fallback 1: ALL CAPS headings
        matches = list(_CAPS_HEADING.finditer(full_text))

    if not matches:
        # Fallback 2: paragraph chunking
        return split_paragraph_chunks(full_text)

    sections = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        content = full_text[start:end].strip()

        if len(content) < 60:
            continue

        if "reference" in title.lower() and len(content) > 5000:
            content = content[:500] + "\n[References truncated for indexing]"

        sections.append({
            "section_number": str(len(sections) + 1),
            "section_title": title,
            "section_type": infer_section_type(title),
            "content": content,
        })

    # If very few sections found for a large document, fall back to paragraph chunking
    total_chars = sum(len(s["content"]) for s in sections)
    if len(sections) <= 2 and total_chars > 20000:
        return split_paragraph_chunks(full_text)

    return sections


def parse_academic_review(pdf_path: Path, meta: dict) -> dict:
    """
    Parse a journal review paper or EURETINA/ADA guideline into standard JSON.
    meta: dict with keys: guideline_id, title, source, source_org, source_type, disease, year
    """
    full_text = extract_text_from_pdf(pdf_path)
    sections = split_academic_sections(full_text)

    return {
        "guideline_id": meta["guideline_id"],
        "title": meta["title"],
        "source": meta.get("source", ""),
        "source_org": meta.get("source_org", ""),
        "source_type": meta.get("source_type", "Review Article"),
        "disease": meta.get("disease", ""),
        "year": meta.get("year"),
        "sections": sections,
    }
