import pdfplumber
import re
import json
from pathlib import Path

# -------------------- PATHS --------------------
PDF_DIR = Path("guidance")   # folder containing 72 PDFs
OUTPUT_DIR = Path("json")    # output folder for JSONs
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------- CLEANING --------------------
def clean_page_text(text: str) -> str:
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Remove headers like: "... (IPG591)" or "... (TA1093)"
        if re.search(r"\((IPG|TA|NG|DG|MTG|HST)\s*\d+\)", line, re.IGNORECASE):
            continue

        # Remove footers
        if line.startswith("© NICE"):
            continue
        if re.match(r"Page \d+ of \d+", line):
            continue

        lines.append(line)

    return "\n".join(lines)


# -------------------- SECTION TYPE --------------------
def infer_section_type(title: str) -> str:
    t = title.lower()

    if "recommendation" in t:
        return "Recommendation"
    if "efficacy" in t:
        return "Evidence"
    if "safety" in t:
        return "Safety"
    if "procedure" in t:
        return "Procedure"
    if "indication" in t:
        return "Background"
    if "committee" in t:
        return "Committee opinion"

    return "Other"


# -------------------- SECTION EXTRACTION --------------------
def extract_sections(full_text: str):
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


# -------------------- GUIDELINE ID EXTRACTION --------------------
def extract_guideline_id(text: str) -> str:
    match = re.search(
        r"(IPG|TA|NG|DG|MTG|HST)\s*\d+",
        text,
        re.IGNORECASE
    )
    if match:
        return match.group(0).replace(" ", "").upper()
    return "UNKNOWN"


# -------------------- MAIN CONVERSION FUNCTION --------------------
def extract_pdf_to_json(pdf_path: Path):
    print(f"Processing: {pdf_path.name}")

    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += clean_page_text(page_text) + "\n"

    if not full_text.strip():
        print(f"⚠️  No extractable text: {pdf_path.name}")
        return

    guideline_id = extract_guideline_id(full_text)

    # CRITICAL FIX: never overwrite UNKNOWN.json
    safe_id = guideline_id if guideline_id != "UNKNOWN" else pdf_path.stem

    sections = extract_sections(full_text)

    result = {
        "guideline_id": guideline_id,
        "title": pdf_path.stem.replace("-", " "),
        "source": "NICE",
        "source_type": "Guidance",
        "sections": sections
    }

    output_path = OUTPUT_DIR / f"{safe_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated: {output_path.name}")


# -------------------- RUN ON ALL PDFs --------------------
if __name__ == "__main__":
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs\n")

    for pdf_file in pdf_files:
        try:
            extract_pdf_to_json(pdf_file)
        except Exception as e:
            print(f"❌ Failed on {pdf_file.name}: {e}")
