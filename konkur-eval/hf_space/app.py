
import json
import os

import gradio as gr
import pandas as pd


def _safe_pct(x: float) -> str:
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return "0.00%"


def _score_str(acc: float, correct: int, total: int) -> str:
    return f"{_safe_pct(acc)} ({int(correct)}/{int(total)})"


def load_leaderboard():
    file_path = "leaderboard_data.json"
    if not os.path.exists(file_path):
        return {"last_updated": None, "models": [], "exam_order": [], "primary_metric": "text_only"}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        models = []
        for i, row in enumerate(data, 1):
            models.append({
                "rank": i,
                "model": row.get("Model", ""),
                "vision_capable": row.get("Vision Capable") == "✅",
                "overall": {
                    "text_only": {
                        "accuracy": row.get("Text-Only Score (Primary)", 0.0),
                        "correct": 0,
                        "total": row.get("Text Questions", 0)
                    },
                    "standard": {
                        "accuracy": row.get("Standard Score (All)", 0.0),
                        "correct": 0,
                        "total": row.get("Total Questions", 0)
                    }
                },
                "per_exam": []
            })
        return {"last_updated": None, "models": models, "exam_order": [], "primary_metric": "text_only"}

    data.setdefault("models", [])
    data.setdefault("exam_order", [])
    data.setdefault("primary_metric", "text_only")
    return data


def overall_dataframe(data):
    rows = []
    for model_data in data.get("models", []):
        rank = model_data.get("rank", "")
        m_name = model_data["model"]
        vis = "✅" if model_data["vision_capable"] else "❌"
        
        # Primary Score (Text Only)
        to_acc = model_data["overall"]["text_only"]["accuracy"]
        to_corr = model_data["overall"]["text_only"]["correct"]
        to_tot = model_data["overall"]["text_only"]["total"]
        primary = f"{to_acc:.2f}%({to_corr}/{to_tot})"
        
        # Standard Score (All)
        std_acc = model_data["overall"]["standard"]["accuracy"]
        std_corr = model_data["overall"]["standard"]["correct"]
        std_tot = model_data["overall"]["standard"]["total"]
        std = f"{std_acc:.2f}% ({std_corr}/{std_tot})"
        
        rows.append([rank, m_name, vis, primary, std])
    
    return pd.DataFrame(rows, columns=["Rank", "Model", "Vision?", "Text-Only Score (Primary)", "Standard Score (All)"])


def exam_dataframe(model_entry) -> pd.DataFrame:
    rows = []
    for r in model_entry.get("per_exam", []):
        text = r.get("text_only", {})
        std = r.get("standard", {})
        rows.append({
            "Exam": r.get("exam_name", ""),
            "Text-Only Acc": _safe_pct(text.get("accuracy", 0.0)),
            "Text-Only (C/T)": f"{int(text.get('correct', 0))}/{int(text.get('total', 0))}",
            "Standard Acc": _safe_pct(std.get("accuracy", 0.0)),
            "Standard (C/T)": f"{int(std.get('correct', 0))}/{int(std.get('total', 0))}",
        })
    return pd.DataFrame(rows)


def model_summary_md(model_entry) -> str:
    name = model_entry.get("model", "")
    vis = "Vision Capable" if model_entry.get("vision_capable") else "Text Only"
    overall = model_entry.get("overall", {})
    text = overall.get("text_only", {})
    std = overall.get("standard", {})
    return "\n".join([
        f"### 🤖 {name}",
        f"- **Type:** {vis}",
        f"- **Text-Only Accuracy:** {_score_str(text.get('accuracy', 0.0), text.get('correct', 0), text.get('total', 0))}",
        f"- **Standard Accuracy:** {_score_str(std.get('accuracy', 0.0), std.get('correct', 0), std.get('total', 0))}",
    ])


data = load_leaderboard()
model_names = [m.get("model", "") for m in data.get("models", [])]
default_model = model_names[0] if model_names else None

# Use a more official/professional theme
theme = gr.themes.Soft(
    text_size="lg",
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "Arial", "sans-serif"],
    primary_hue="indigo",
    secondary_hue="blue",
).set(
    body_text_color="#1f2937",
    block_title_text_weight="600",
    block_border_width="1px",
    block_shadow="0 1px 2px 0 rgba(0, 0, 0, 0.05)"
)

with gr.Blocks(
    title="Konkur 1404 LLM Leaderboard",
    theme=theme,
) as demo:
    gr.Markdown("# 🏆 Konkur 1404 LLM Leaderboard")
    last_updated = data.get("last_updated") or "N/A"
    gr.Markdown(
        "This leaderboard evaluates leading Large Language Models on the Konkur 1404 (Iranian university-entrance exam) First Session (Nobat 1).\n\n"
        "We prioritize open-source models with strong Persian-language support."
    )
    gr.Markdown(
        "### Support the benchmark\n"
        "Running evaluations costs OpenRouter credits. If you would like to sponsor API usage, "
        "please [buy credits on OpenRouter](https://openrouter.ai/pricing) and contact us through "
        "the [project repository](https://github.com/mshojaei77/konkur-eval) so we can coordinate the evaluation. "
        "Donations and compute sponsorships are also welcome through the repository."
    )

    with gr.Tabs():
        with gr.Tab("Overall"):
            overall_table = gr.DataFrame(
                value=overall_dataframe(data),
                interactive=False,
                wrap=True,
                row_count=(0, "dynamic"),
                column_count=(5, "fixed"),
                label="Overall Performance",
            )

        with gr.Tab("Model Details"):
            model_dd = gr.Dropdown(choices=[n for n in model_names if n], value=default_model, label="Model")
            summary = gr.Markdown(value=model_summary_md(next((m for m in data.get("models", []) if m.get("model") == default_model), {})) if default_model else "No model data found.")
            per_exam_table = gr.DataFrame(
                value=exam_dataframe(next((m for m in data.get("models", []) if m.get("model") == default_model), {})) if default_model else pd.DataFrame(),
                interactive=False,
                wrap=True,
                row_count=(0, "dynamic"),
                label="Exam Breakdown",
            )

            def _on_model_change(name):
                m = next((x for x in data.get("models", []) if x.get("model") == name), {})
                return model_summary_md(m), exam_dataframe(m)

            model_dd.change(_on_model_change, inputs=[model_dd], outputs=[summary, per_exam_table])

    gr.Markdown(
        "### Methodology\n"
        "- **Text-Only Score:** Accuracy on questions without images (primary ranking).\n"
        "- **Standard Score:** Accuracy on all questions (text + images).\n"
        "- **Vision?:** Whether the model supports image inputs."
    )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
