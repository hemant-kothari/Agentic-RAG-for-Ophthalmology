"""
Step 0: inspect_pdfs.py
Dumps the first N lines of extracted text from each PDF in guidance 2/
into a text file in process_guidance2/pdf_previews/ for manual review.

Run from: capstone/
    python process_guidance2/inspect_pdfs.py
"""

import pdfplumber
from pathlib import Path

PDF_DIR = Path("guidance 2")
PREVIEW_DIR = Path("process_guidance2") / "pdf_previews"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

LINES_TO_DUMP = 120  # enough to see headers, structure, section patterns


def dump_preview(pdf_path: Path):
    output_file = PREVIEW_DIR / (pdf_path.stem[:60] + ".txt")
    lines_collected = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        lines_collected.append(f"[P{page_num+1:03d}] {line}")
                if len(lines_collected) >= LINES_TO_DUMP:
                    break

        preview_text = "\n".join(lines_collected[:LINES_TO_DUMP])
        output_file.write_text(preview_text, encoding="utf-8")
        print(f"[OK]  {pdf_path.name[:65]:<65} -> {output_file.name}")

    except Exception as e:
        print(f"[ERR] {pdf_path.name}: {e}")


if __name__ == "__main__":
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in '{PDF_DIR}'\n")
    for pdf in pdfs:
        dump_preview(pdf)
    print(f"\nPreviews saved to: {PREVIEW_DIR.resolve()}")

