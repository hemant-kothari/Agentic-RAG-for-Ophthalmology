import pdfplumber
import re
import json
from pathlib import Path

PDF_PATH = "ab-externo-canaloplasty-for-primary-openangle-glaucoma-pdf-1899872224084933.pdf"
OUTPUT_JSON = "IPG591.json"


def clean_page_text(text: str) -> str:
    """Remove NICE headers, footers, and page numbers."""
    lines = []
    for line in text.splitlines():
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Remove headers
        if re.match(r".+\(IPG\d+\)", line):
            continue

        # Remove footers
        if line.startswith("© NICE"):
            continue
        if re.match(r"Page \d+ of \d+", line):
            continue

        lines.append(line)

    return "\n".join(lines)


def extract_sections(full_text: str):
    """
    Split text into major numbered sections:
    1 Recommendations
    2 Indications and current treatments
    """
    section_pattern = re.compile(r"\n(?=\d+\s+[A-Z])")
    raw_sections = section_pattern.split("\n" + full_text)

    sections = []

    for sec in raw_sections:
        sec = sec.strip()
        if not sec:
            continue

        header_match = re.match(r"(\d+)\s+(.+)", sec)
        if not header_match:
            continue

        section_number = header_match.group(1)
        section_title = header_match.group(2).strip()

        content = sec[len(header_match.group(0)):].strip()

        sections.append({
            "section_number": section_number,
            "section_title": section_title,
            "section_type": infer_section_type(section_title),
            "content": content
        })

    return sections


def infer_section_type(title: str) -> str:
    title_lower = title.lower()

    if "recommendation" in title_lower:
        return "Recommendation"
    if "efficacy" in title_lower:
        return "Evidence"
    if "safety" in title_lower:
        return "Safety"
    if "procedure" in title_lower:
        return "Procedure"
    if "indications" in title_lower:
        return "Background"
    if "committee" in title_lower:
        return "Committee opinion"
    return "Other"


def extract_pdf_to_json(pdf_path: str, output_path: str):
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += clean_page_text(page_text) + "\n"

    sections = extract_sections(full_text)

    result = {
        "guideline_id": "IPG591",
        "title": "Ab externo canaloplasty for primary open-angle glaucoma",
        "source": "NICE",
        "source_type": "Interventional procedures guidance",
        "sections": sections
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    extract_pdf_to_json(PDF_PATH, OUTPUT_JSON)
    print("JSON generated successfully")
