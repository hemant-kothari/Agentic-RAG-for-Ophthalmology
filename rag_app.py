"""
rag_app.py — Gradio Web UI for the Retinal Agentic RAG System.

Tabs:
  1. Image Report  — upload / use sample inference_results.json
  2. Ask a Question — open-ended eye care query

Run:
  python rag_app.py
"""

import sys
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from agentic_rag.agent import RetinalRAGAgent

# ── Init agent (shared across tabs) ──────────────────────────────────────────
print("Loading Retinal RAG Agent...")
agent = RetinalRAGAgent()
print("Agent ready.")

SAMPLE_JSON = str(Path("single_image_results/inference_results.json"))


# ── Tab 1: Image Mode ─────────────────────────────────────────────────────────
def run_image(json_file):
    """Process uploaded or sample inference_results.json."""
    if json_file is None:
        path = SAMPLE_JSON
    else:
        path = json_file.name

    try:
        result = agent.run_image_mode(path)
    except Exception as e:
        return (
            f"❌ Error: {e}",
            "—", "—", "—"
        )

    parsed  = result["parsed_input"]
    summary = parsed["summary_text"]

    # Conditions table
    cond_lines = []
    for c in parsed["confirmed_findings"]:
        cond_lines.append(f"✅ **{c['condition']}** — CONFIRMED\n   {c['prob_label']} confidence | {c['certainty_label']}")
    for c in parsed["possible_findings"]:
        cond_lines.append(f"⚠️ **{c['condition']}** — POSSIBLE (flagged for clinical review)\n   {c['prob_label']} confidence | {c['certainty_label']}")
    conditions_md = "\n\n".join(cond_lines) or "No positive findings."

    # DR severity
    dr = parsed["dr_severity"]
    dr_md = f"**{dr['label']}**" if dr else "Not assessed"

    # Sources
    sources_md = "\n".join(f"• {s}" for s in result["sources_used"])
    if result["web_used"]:
        sources_md += "\n• _Web search results included (supplementary)_"

    return result["response"], summary, conditions_md, dr_md, sources_md


# ── Tab 2: Query Mode ─────────────────────────────────────────────────────────
def run_query(query: str):
    """Answer an open-ended eye care question."""
    if not query.strip():
        return "Please enter a question.", "—"

    try:
        result = agent.run_query_mode(query.strip())
    except Exception as e:
        return f"❌ Error: {e}", "—"

    status = result["status"]
    if status in ("off_topic", "blocked"):
        return result["response"], "—"

    sources_md = "\n".join(f"• {s}" for s in result["sources_used"])
    if result["web_used"]:
        sources_md += "\n• _Web search results included (supplementary)_"
    if not sources_md:
        sources_md = "—"

    return result["response"], sources_md


# ── UI Layout ─────────────────────────────────────────────────────────────────
DESCRIPTION = """
# 🔬 Retinal AI — Clinical Knowledge Assistant
**Evidence-based guidance from NICE · AAO · WHO · EURETINA · RCOphth · ADA · India Vision 2025**

> ⚕️ *This tool provides educational clinical context only. It does NOT diagnose or treat patients.
> All AI findings require review by a qualified ophthalmologist.*
"""

EXAMPLES_QUERY = [
    ["What are the recommended screening intervals for diabetic retinopathy?"],
    ["How is neovascular AMD treated according to current guidelines?"],
    ["What are the target intraocular pressure levels in glaucoma management?"],
    ["What causes hypertensive retinopathy and how is it classified?"],
    ["What anti-VEGF agents are recommended for diabetic macular edema?"],
]

with gr.Blocks(
    title="Retinal RAG -- Clinical Knowledge Assistant",
) as app:

    gr.Markdown(DESCRIPTION)

    with gr.Tabs():

        # ── TAB 1: Image Report ───────────────────────────────────────────
        with gr.TabItem("🖼️ Image Analysis Report"):
            gr.Markdown("""
Upload an `inference_results.json` file from the RETFound ViT model, or click
**Use Sample** to run with the included example.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    json_input = gr.File(
                        label="Upload inference_results.json",
                        file_types=[".json"],
                    )
                    with gr.Row():
                        btn_sample = gr.Button("Use Sample Image", variant="secondary")
                        btn_run    = gr.Button("Generate Report", variant="primary")

                    gr.Markdown("### AI Findings")
                    out_summary    = gr.Markdown(label="Summary")
                    out_conditions = gr.Markdown(label="Detected Conditions")
                    out_dr         = gr.Markdown(label="DR Severity")
                    out_sources_img = gr.Markdown(label="Sources Used")

                with gr.Column(scale=2):
                    out_report = gr.Markdown(label="Clinical Guidance Report")

            btn_run.click(
                fn=run_image,
                inputs=[json_input],
                outputs=[out_report, out_summary, out_conditions, out_dr, out_sources_img],
            )
            btn_sample.click(
                fn=lambda: run_image(None),
                inputs=[],
                outputs=[out_report, out_summary, out_conditions, out_dr, out_sources_img],
            )

        # ── TAB 2: Query ──────────────────────────────────────────────────
        with gr.TabItem("💬 Ask a Question"):
            gr.Markdown("""
Ask any question about retinal diseases, eye conditions, screening, diagnosis,
or treatment. Answers are grounded in international clinical guidelines.
""")
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g. What is the recommended follow-up interval after anti-VEGF treatment for AMD?",
                        lines=3,
                    )
                    btn_ask = gr.Button("Ask", variant="primary")
                    out_sources_q = gr.Markdown(label="Sources Used")

                with gr.Column(scale=3):
                    out_answer = gr.Markdown(label="Answer")

            gr.Examples(
                examples=EXAMPLES_QUERY,
                inputs=query_input,
                label="Example Questions",
            )

            btn_ask.click(
                fn=run_query,
                inputs=[query_input],
                outputs=[out_answer, out_sources_q],
            )
            query_input.submit(
                fn=run_query,
                inputs=[query_input],
                outputs=[out_answer, out_sources_q],
            )

    gr.Markdown(
        "_Built with [NICE](https://www.nice.org.uk/), [AAO](https://www.aao.org/), "
        "[WHO](https://www.who.int/), [EURETINA](https://www.euretina.org/), "
        "[RCOphth](https://www.rcophth.ac.uk/), [ADA](https://www.diabetes.org/), "
        "and [Vision 2020 India](https://www.vision2020india.org/) guidelines._"
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )
