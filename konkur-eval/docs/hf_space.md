### Easiest Way: Duplicate a Leaderboard Template Space
The simplest and most recommended method to create a Hugging Face Space with a leaderboard is to **duplicate an existing template**. Hugging Face provides official and community templates designed for leaderboards, including features like model display, filtering, searching, and optional user submissions/evaluations.

- Go to one of these template Spaces and click the **"Duplicate this Space"** button (available in the top-right menu on the Space page):
  - Official demo frontend: [https://huggingface.co/spaces/demo-leaderboard-backend/leaderboard](https://huggingface.co/spaces/demo-leaderboard-backend/leaderboard) — This is the recommended starting point for a full leaderboard (supports submissions and backend evaluations).
  - Gradio templates example: [https://huggingface.co/spaces/gradio-templates/leaderboard](https://huggingface.co/spaces/gradio-templates/leaderboard) — Simpler display-focused leaderboard.
  - Other examples: Search for "leaderboard template" on [huggingface.co/spaces](https://huggingface.co/spaces) and look for ones with "Duplicate this leaderboard" in the description.

Duplicating creates a new Space in your account (or organization) with all files pre-configured (e.g., `app.py`, datasets for requests/results). The Space builds automatically.

**Customization after duplicating**:
- Edit files via the web editor or git (see pushing below).
- Key files to modify (from template READMEs):
  - `src/env.py`: Set your organization name and paths.
  - `src/about.py`: Define tasks, metrics, and few-shot examples.
  - Add fake/initial data to the results dataset for testing.
- For automated evaluations: Duplicate the backend Space from the same template and link it.

This approach is used by many official leaderboards (e.g., MTEB, Open LLM Leaderboard variants).

### Alternative: Build from Scratch with gradio_leaderboard Component
For a lightweight leaderboard (display-only, with search/filter/sort), use the `gradio_leaderboard` custom component.

1. **Create a new Gradio Space**:
   - Go to [https://huggingface.co/new-space](https://huggingface.co/new-space).
   - Choose a name, visibility, and **Gradio** as SDK (no specific "Leaderboard" template needed, though some UI flows offer it).

2. **Add files** (via web editor or git):
   - `requirements.txt`:
     ```
     gradio_leaderboard
     pandas
     # Add other deps if needed, e.g., gradio>=4.0
     ```
   - `app.py` (example code for a basic leaderboard):
     ```python
     import gradio as gr
     from gradio_leaderboard import Leaderboard
     from pathlib import Path
     import pandas as pd

     # Load your data (replace with your JSON/CSV/DataFrame)
     abs_path = Path(__file__).parent
     df = pd.read_json(abs_path / "leaderboard_data.json")  # Or create df manually

     with gr.Blocks() as demo:
         gr.Markdown("# My Custom Leaderboard")
         Leaderboard(
             value=df,
             select_columns=["Model", "Average", "Task1", "Task2"],  # Columns users can toggle
             search_columns=["Model"],  # Primary search column
             filter_columns=["Precision", "Model Size"],  # Add filters (auto-detects type)
             hide_columns=["Internal ID"]  # Hidden but searchable
         )

     demo.launch()
     ```
   - Add a `leaderboard_data.json` file with your data (pandas DataFrame format).

3. The component supports advanced config: column selection, filters (sliders/dropdowns), search, LaTeX in markdown columns, etc.

### How to Push Changes to Hugging Face
Hugging Face Spaces are Git repositories.

1. **Clone your Space**:
   ```
   git clone https://huggingface.co/spaces/your-username/your-space-name
   cd your-space-name
   ```

2. **Make changes** locally (edit `app.py`, add data files, etc.).

3. **Commit and push**:
   ```
   git add .
   git commit -m "Update leaderboard"
   git push
   ```
   - If prompted for credentials, use a Hugging Face access token (generate at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with "write" role).
   - Pushes trigger automatic rebuild/deployment.

This works for both duplicated templates and new Spaces. Changes appear live within minutes.

These methods are verifiable from official Hugging Face documentation and active templates (as of 2026). For submission-based leaderboards with auto-evaluations, start with the demo-leaderboard-backend template—it's the most feature-complete. If you need something specific (e.g., code for a particular task), provide more details!


