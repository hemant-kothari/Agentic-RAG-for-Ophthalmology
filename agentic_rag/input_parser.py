"""
input_parser.py — Parse inference_results.json from the RETFound ViT model.

Produces a clean structured dict used by the agent.
Applies confidence labeling guardrail:
  - certainty >= 0.15  → CONFIRMED positive
  - certainty <  0.15  → POSSIBLE (flagged for clinical review)
"""

import json
from pathlib import Path

# Certainty threshold below which a positive is labeled "Possible"
CERTAINTY_THRESHOLD = 0.15

# Human-readable probability labels
def _prob_label(prob: float) -> str:
    if prob >= 0.75: return "High"
    if prob >= 0.55: return "Moderate"
    return "Low"

def _certainty_label(certainty: float) -> str:
    if certainty >= 0.4:  return "High certainty"
    if certainty >= 0.15: return "Moderate certainty"
    return "Low certainty — flagged for clinical review"


def parse_inference_results(path: str | Path) -> dict:
    """
    Parse inference_results.json into a structured agent input dict.

    Returns:
    {
      "mode": "image",
      "image_path": str,
      "confirmed_findings": [
          {"condition": str, "prob_label": str, "certainty_label": str, "status": "Confirmed"},
          ...
      ],
      "possible_findings": [
          {"condition": str, "prob_label": str, "certainty_label": str, "status": "Possible"},
          ...
      ],
      "all_conditions": [...],   # full list for display
      "dr_severity": {...} | None,
      "summary_text": str,       # one-line natural language summary
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    confirmed = []
    possible  = []
    all_conds = []

    for cond in data.get("conditions_ranked", []):
        if not cond.get("positive") or cond.get("suppressed"):
            all_conds.append({
                "condition":       cond["condition"],
                "prob_label":      _prob_label(cond["probability"]),
                "certainty_label": _certainty_label(cond.get("certainty", 0)),
                "status":          "Negative",
            })
            continue

        prob      = cond["probability"]
        certainty = cond.get("certainty", 0)
        entry = {
            "condition":       cond["condition"],
            "prob_label":      _prob_label(prob),
            "certainty_label": _certainty_label(certainty),
            "raw_prob":        round(prob, 3),
            "raw_certainty":   round(certainty, 3),
        }

        if certainty >= CERTAINTY_THRESHOLD:
            entry["status"] = "Confirmed"
            confirmed.append(entry)
        else:
            entry["status"] = "Possible — flagged for clinical review"
            possible.append(entry)

        all_conds.append(entry)

    # DR severity
    dr = data.get("dr_severity")
    dr_info = None
    if dr:
        dr_info = {
            "grade":       dr.get("predicted_grade", 0),
            "label":       dr.get("grade_label", "Unknown"),
            "certainty":   round(dr.get("grade_certainty", 0), 4),
        }

    # Summary text
    parts = []
    if confirmed:
        parts.append("Confirmed: " + ", ".join(c["condition"] for c in confirmed))
    if possible:
        parts.append("Possible: " + ", ".join(p["condition"] for p in possible))
    if not confirmed and not possible:
        parts.append("No positive findings detected")
    if dr_info:
        parts.append(f"DR Grade: {dr_info['label']}")
    summary_text = " | ".join(parts)

    return {
        "mode":               "image",
        "image_path":         data.get("image_path", ""),
        "confirmed_findings": confirmed,
        "possible_findings":  possible,
        "all_conditions":     all_conds,
        "dr_severity":        dr_info,
        "summary_text":       summary_text,
    }
