"""
process_all.py — Orchestrator for Guidance 2 PDF processing.

Defines the manifest (PDF → parser strategy + metadata) and runs all 20 PDFs.
Outputs JSONs to json_g2/ folder.

Run from: capstone/
    python process_guidance2/process_all.py
"""

import sys
import json
from pathlib import Path

# Add process_guidance2 to path so parsers can import parser_utils
sys.path.insert(0, str(Path(__file__).parent))

from parsers.aao_ppp_parser import parse_aao_ppp
from parsers.society_guideline_parser import parse_society_guideline
from parsers.academic_review_parser import parse_academic_review

# Output directory
OUTPUT_DIR = Path("json_g2")
OUTPUT_DIR.mkdir(exist_ok=True)

PDF_DIR = Path("guidance 2")


# ============================================================
# MANIFEST
# Each entry maps a PDF filename to:
#   parser     : which parser function to use
#   meta       : metadata dict (guideline_id MUST be unique)
# ============================================================

MANIFEST = [
    # ---------- AAO Full PPPs ----------
    {
        "filename": "Age-Related Macular Degeneration PPP.pdf",
        "parser": "aao_ppp",
        "meta": {
            "guideline_id": "AAO_AMD_PPP_2024",
            "title": "Age-Related Macular Degeneration Preferred Practice Pattern",
            "source": "American Academy of Ophthalmology",
            "source_org": "AAO",
            "source_type": "Clinical Practice Guideline",
            "disease": "Age-Related Macular Degeneration",
            "year": 2024,
        },
    },
    {
        "filename": "Diabetic Retinopathy PPP.pdf",
        "parser": "aao_ppp",
        "meta": {
            "guideline_id": "AAO_DR_PPP_2024",
            "title": "Diabetic Retinopathy Preferred Practice Pattern",
            "source": "American Academy of Ophthalmology",
            "source_org": "AAO",
            "source_type": "Clinical Practice Guideline",
            "disease": "Diabetic Retinopathy",
            "year": 2024,
        },
    },
    {
        "filename": "Primary Open-Angle Glaucoma PPP.pdf",
        "parser": "aao_ppp",
        "meta": {
            "guideline_id": "AAO_POAG_PPP_2024",
            "title": "Primary Open-Angle Glaucoma Preferred Practice Pattern",
            "source": "American Academy of Ophthalmology",
            "source_org": "AAO",
            "source_type": "Clinical Practice Guideline",
            "disease": "Primary Open-Angle Glaucoma",
            "year": 2024,
        },
    },
    {
        "filename": "Primary Open-Angle Glaucoma Suspect PPP.pdf",
        "parser": "aao_ppp",
        "meta": {
            "guideline_id": "AAO_POAGS_PPP_2024",
            "title": "Primary Open-Angle Glaucoma Suspect Preferred Practice Pattern",
            "source": "American Academy of Ophthalmology",
            "source_org": "AAO",
            "source_type": "Clinical Practice Guideline",
            "disease": "Glaucoma Suspect",
            "year": 2024,
        },
    },
    # ---------- AAO Benchmark Summaries ----------
    {
        "filename": "BC-2208_PPPSummaryBenchmarks.17.Retina.pdf",
        "parser": "academic_review",
        "meta": {
            "guideline_id": "AAO_RETINA_BENCHMARKS_2017",
            "title": "Retina Summary Benchmarks for Preferred Practice Pattern Guidelines",
            "source": "American Academy of Ophthalmology",
            "source_org": "AAO",
            "source_type": "Summary Benchmarks",
            "disease": "Retinal Diseases",
            "year": 2017,
        },
    },
    {
        "filename": "PPP Summary Benchmarks.24.retina.pdf",
        "parser": "academic_review",
        "meta": {
            "guideline_id": "AAO_RETINA_BENCHMARKS_2024",
            "title": "Retina PPP Summary Benchmarks 2024",
            "source": "American Academy of Ophthalmology",
            "source_org": "AAO",
            "source_type": "Summary Benchmarks",
            "disease": "Retinal Diseases",
            "year": 2024,
        },
    },
    # ---------- ADA ----------
    {
        "filename": "ADA standards of care diabetic retinopathy pdf.pdf",
        "parser": "academic_review",
        "meta": {
            "guideline_id": "ADA_DR_SOC_2026",
            "title": "ADA Standards of Care in Diabetes: Retinopathy, Neuropathy, and Foot Care",
            "source": "American Diabetes Association",
            "source_org": "ADA",
            "source_type": "Standards of Care",
            "disease": "Diabetic Retinopathy",
            "year": 2026,
        },
    },
    # ---------- EURETINA ----------
    {
        "filename": "EURETINA diabetic retinopathy guideline pdf.pdf",
        "parser": "academic_review",
        "meta": {
            "guideline_id": "EURETINA_DR_2017",
            "title": "EURETINA Guidelines for the Management of Diabetic Macular Edema",
            "source": "European Society of Retina Specialists (EURETINA)",
            "source_org": "EURETINA",
            "source_type": "Clinical Practice Guideline",
            "disease": "Diabetic Macular Edema",
            "year": 2017,
        },
    },
    {
        "filename": "EURETINA guidelines AMD pdf.pdf",
        "parser": "academic_review",
        "meta": {
            "guideline_id": "EURETINA_AMD_2023",
            "title": "EURETINA Guidelines on the Management of Age-Related Macular Degeneration",
            "source": "European Society of Retina Specialists (EURETINA)",
            "source_org": "EURETINA",
            "source_type": "Clinical Practice Guideline",
            "disease": "Age-Related Macular Degeneration",
            "year": 2023,
        },
    },
    # ---------- Royal College of Ophthalmologists ----------
    {
        "filename": "Royal College Ophthalmologists AMD guideline pdf.pdf",
        "parser": "society_guideline",
        "meta": {
            "guideline_id": "RCOphth_AMD_2024",
            "title": "Royal College of Ophthalmologists: Age Related Macular Degeneration Services Evidence Base",
            "source": "Royal College of Ophthalmologists",
            "source_org": "RCOphth",
            "source_type": "Commissioning Guidance",
            "disease": "Age-Related Macular Degeneration",
            "year": 2024,
        },
    },
    {
        "filename": "Royal College Ophthalmologists diabetic retinopathy guideline pdf.pdf",
        "parser": "society_guideline",
        "meta": {
            "guideline_id": "RCOphth_DR_2024",
            "title": "Royal College of Ophthalmologists: Diabetic Retinopathy Guideline",
            "source": "Royal College of Ophthalmologists",
            "source_org": "RCOphth",
            "source_type": "Clinical Guideline",
            "disease": "Diabetic Retinopathy",
            "year": 2024,
        },
    },
    {
        "filename": "Royal College Ophthalmologists glaucoma guideline pdf.pdf",
        "parser": "society_guideline",
        "meta": {
            "guideline_id": "RCOphth_GLAUCOMA_2022",
            "title": "Royal College of Ophthalmologists: Glaucoma Guideline",
            "source": "Royal College of Ophthalmologists",
            "source_org": "RCOphth",
            "source_type": "Clinical Guideline",
            "disease": "Glaucoma",
            "year": 2022,
        },
    },
    # ---------- WHO ----------
    {
        "filename": "who diabetic retinopathy.pdf",
        "parser": "society_guideline",
        "meta": {
            "guideline_id": "WHO_DR_SCREENING_2020",
            "title": "WHO: Diabetic Retinopathy Screening - A Short Guide",
            "source": "World Health Organization",
            "source_org": "WHO",
            "source_type": "WHO Guide",
            "disease": "Diabetic Retinopathy",
            "year": 2020,
        },
    },
    # ---------- India ----------
    {
        "filename": "vision 2025 india DR guidelines.pdf",
        "parser": "society_guideline",
        "meta": {
            "guideline_id": "INDIA_DR_VISION2025",
            "title": "Guidelines for Management of Diabetic Retinopathy in India (Vision 2025)",
            "source": "VISION 2020: The Right to Sight - India",
            "source_org": "INDIA",
            "source_type": "National Clinical Guideline",
            "disease": "Diabetic Retinopathy",
            "year": 2025,
        },
    },
    # ---------- Review Papers ----------
    {
        "filename": "diabetic retinopathy review nature pdf.pdf",
        "parser": "academic_review",
        "meta": {
            "guideline_id": "REVIEW_DR_PHYTOCHEM_2022",
            "title": "Nature Against Diabetic Retinopathy: A Review on Antiangiogenic, Antioxidant, and Anti-Inflammatory Phytochemicals",
            "source": "Evidence-Based Complementary and Alternative Medicine",
            "source_org": "REVIEW",
            "source_type": "Review Article",
            "disease": "Diabetic Retinopathy",
            "year": 2022,
        },
    },
    {
        "filename": "1-s2.0-S0039625726000160-main.pdf",
        "parser": "academic_review",
        "meta": {
            "guideline_id": "REVIEW_SURV_OPHTH_2026",
            "title": "Survey of Ophthalmology Review (2026)",
            "source": "Survey of Ophthalmology",
            "source_org": "REVIEW",
            "source_type": "Review Article",
            "disease": "Retinal Diseases",
            "year": 2026,
        },
    },
    {
        "filename": "hypertensive retinopathy clinical review.pdf",
        "parser": "academic_review",
        "meta": {
            "guideline_id": "REVIEW_HTN_RET_CLINICAL",
            "title": "Hypertensive Retinopathy: Clinical Review",
            "source": "Clinical Review",
            "source_org": "REVIEW",
            "source_type": "Review Article",
            "disease": "Hypertensive Retinopathy",
            "year": None,
        },
    },
    {
        "filename": "hypertensive retinopathy review pdf.pdf",
        "parser": "academic_review",
        "meta": {
            "guideline_id": "REVIEW_HTN_RET_2",
            "title": "Hypertensive Retinopathy: Comprehensive Review",
            "source": "Review Article",
            "source_org": "REVIEW",
            "source_type": "Review Article",
            "disease": "Hypertensive Retinopathy",
            "year": None,
        },
    },
    {
        "filename": "primary open angle glaucoma review.pdf",
        "parser": "academic_review",
        "meta": {
            "guideline_id": "REVIEW_POAG_2023",
            "title": "Primary Open-Angle Glaucoma: Comprehensive Review",
            "source": "Review Article",
            "source_org": "REVIEW",
            "source_type": "Review Article",
            "disease": "Primary Open-Angle Glaucoma",
            "year": 2023,
        },
    },
    {
        "filename": "Glaucoma researrch gate.pdf",
        "parser": "academic_review",
        "meta": {
            "guideline_id": "REVIEW_GLAUCOMA_RG",
            "title": "Glaucoma: Review Article (ResearchGate)",
            "source": "ResearchGate",
            "source_org": "REVIEW",
            "source_type": "Review Article",
            "disease": "Glaucoma",
            "year": None,
        },
    },
]


# ============================================================
# PARSER DISPATCH
# ============================================================

PARSER_MAP = {
    "aao_ppp": parse_aao_ppp,
    "society_guideline": parse_society_guideline,
    "academic_review": parse_academic_review,
}


def process_one(entry: dict) -> bool:
    pdf_path = PDF_DIR / entry["filename"]
    guideline_id = entry["meta"]["guideline_id"]
    output_path = OUTPUT_DIR / f"{guideline_id}.json"

    if not pdf_path.exists():
        print(f"[MISSING] {entry['filename']}")
        return False

    parser_fn = PARSER_MAP[entry["parser"]]

    try:
        result = parser_fn(pdf_path, entry["meta"])

        num_sections = len(result.get("sections", []))
        total_content = sum(len(s["content"]) for s in result.get("sections", []))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(
            f"[OK] {guideline_id:<35} | "
            f"{num_sections:>3} sections | "
            f"{total_content:>8,} chars -> {output_path.name}"
        )
        return True

    except Exception as e:
        print(f"[ERR] {guideline_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print(f"Processing {len(MANIFEST)} PDFs from '{PDF_DIR}' -> '{OUTPUT_DIR}'\n")
    print(f"{'Guideline ID':<35} | {'Sections':>8} | {'Chars':>10} | Output")
    print("-" * 85)

    ok, fail = 0, 0
    for entry in MANIFEST:
        success = process_one(entry)
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} succeeded, {fail} failed")
    if fail > 0:
        print("Check [ERR] lines above for failures.")
