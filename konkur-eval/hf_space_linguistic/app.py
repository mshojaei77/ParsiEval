import json
import os
import gradio as gr
import pandas as pd

def load_leaderboard():
    file_path = "leaderboard_data.json"
    if not os.path.exists(file_path):
        return {"last_updated": None, "models": []}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def make_clickable_model(model_name):
    # This could be improved to link to HF model page if known, 
    # but for now just return the name as is or with a simple search link
    link = f"https://huggingface.co/models?search={model_name}"
    return f'<a target="_blank" href="{link}" style="text-decoration: underline; color: inherit;">{model_name}</a>'

def prepare_dataframe(data):
    rows = []
    for m in data.get("models", []):
        scores = m.get("scores", {})
        
        # Format scores: bold the highest? Or just plain numbers.
        # Let's keep it simple first.
        
        rows.append([
            m.get("rank"),
            m.get("model"), # Could be make_clickable_model(m.get("model")), but let's stick to text for now
            scores.get("Overall", 0),
            scores.get("Grammar", 0),
            scores.get("Idiomatic", 0),
            scores.get("Conciseness", 0),
            scores.get("Politeness", 0),
            scores.get("Naturalness", 0),
            scores.get("Instruction Following", 0),
            scores.get("Context Retention", 0),
            scores.get("Safety", 0),
        ])
    
    headers = [
        "Rank", "Model", "Overall", "Grammar", "Idiomatic", 
        "Conciseness", "Politeness", "Naturalness", 
        "Instruction Following", "Context Retention", "Safety"
    ]
    
    df = pd.DataFrame(rows, columns=headers)
    return df

data = load_leaderboard()

# Use a professional theme
theme = gr.themes.Soft(
    text_size="lg",
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "Arial", "sans-serif"],
    primary_hue="teal", # Different color for linguistic
    secondary_hue="green",
).set(
    body_text_color="#1f2937",
    block_title_text_weight="600",
    block_border_width="1px",
    block_shadow="0 1px 2px 0 rgba(0, 0, 0, 0.05)"
)

with gr.Blocks(title="Persian Linguistic Fluency Leaderboard", theme=theme) as demo:
    gr.Markdown("# 🗣️ Persian Linguistic Fluency Leaderboard")
    
    last_updated = data.get("last_updated") or "N/A"
    judge = data.get("judge") or "N/A"
    
    gr.Markdown(
        f"> **Last Updated:** {last_updated} | **Judge:** `{judge}`\n\n"
        "This leaderboard evaluates the linguistic fluency of LLMs across Persian language.\n"
        "Models are ranked by their **Overall** score."
    )

    leaderboard_df = prepare_dataframe(data)
    
    gr.DataFrame(
        value=leaderboard_df,
        headers=leaderboard_df.columns.tolist(),
        datatype=["number", "str"] + ["number"] * 9,
        interactive=False,
        column_count=(11, "fixed"),
        wrap=True,
        label="Leaderboard"
    )
    
    gr.Markdown(
        "### Metrics Description\n"
        "- **Overall**: Aggregate score across all dimensions.\n"
        "- **Grammar**: Grammatical correctness and syntax.\n"
        "- **Idiomatic**: Proper use of idioms and cultural nuances.\n"
        "- **Conciseness**: Ability to convey information efficiently.\n"
        "- **Politeness**: Appropriateness of tone and etiquette.\n"
        "- **Naturalness**: How native-like the text sounds.\n"
        "- **Instruction Following**: Adherence to the prompt constraints.\n"
        "- **Context Retention**: Consistency with prior context.\n"
        "- **Safety**: Avoidance of toxic or harmful content."
    )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
