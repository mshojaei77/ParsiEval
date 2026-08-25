# Konkur Eval

Konkur Eval is a reproducible benchmark runner and leaderboard toolkit for evaluating language and multimodal models on **Konkur 1404**, the 2025 Iranian university entrance examination. It uses an OpenAI-compatible API, resumes interrupted evaluations from per-model result files, and publishes both overall and text-only scores.

The benchmark dataset contains 2,137 Persian and English multiple-choice questions across ten exam sessions. Questions with figures can be sent as multimodal inputs when the selected model supports images.

## What is included

- A resumable benchmark runner for OpenAI-compatible providers.
- Per-exam JSON Lines result artifacts grouped by model.
- Text-only and full-dataset accuracy aggregation.
- A Gradio leaderboard for subject-matter performance.
- A separate Gradio leaderboard for Persian linguistic fluency.
- Dataset extraction, conversion, upload, and Space deployment utilities.

## Repository layout

```text
konkur-eval/
|-- benchmark/                 # Main evaluation runner
|   `-- konkur_bench.py
|-- benchmark_results/         # Versioned model results, grouped by model
|-- docs/                      # Extended workflow and provider notes
|-- hf_space/                  # Konkur benchmark leaderboard Space
|-- hf_space_linguistic/       # Persian linguistic leaderboard Space
|-- scripts/                   # Extraction, publishing, and deployment tools
|-- .env.example               # Safe configuration template
`-- pyproject.toml             # uv project metadata and dependencies
```

The following large or local-only inputs are intentionally excluded from Git:

- `dataset/`: locally extracted dataset files and figures.
- `pdfs/`: source question and answer-key PDFs.
- `etc/`: scratch and intermediate artifacts.

The canonical public dataset is available at [mshojaei77/konkur1404](https://huggingface.co/datasets/mshojaei77/konkur1404).

## Requirements

- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/) for dependency and environment management.
- An API key for an OpenAI-compatible inference provider.
- A Hugging Face write token only when publishing datasets or Spaces.

## Setup

From PowerShell:

```powershell
uv sync
Copy-Item .env.example .env
```

The first successful `uv sync` resolves the declared dependencies and creates `uv.lock`; commit that lockfile to keep later environments reproducible.

Edit `.env` and set at least the provider URL, API key, and one or more model IDs:

```dotenv
EVAL_SUBJECT_BASE_URL=https://openrouter.ai/api/v1
EVAL_SUBJECT_API_KEY=replace-with-your-key
EVAL_SUBJECT_MODELS="
provider/model-name
"
EVAL_EXAMS=ensani_nobat1,ensani_nobat2
EVAL_MAX_SAMPLES=0
```

`EVAL_EXAMS` accepts a comma-separated list or `all`. Set `EVAL_MAX_SAMPLES` to a small positive number for a smoke run, or `0` to evaluate every selected question. Never commit `.env`; it is ignored by Git.

### If uv times out downloading a wheel

The first `uv run` may need to download compiled wheels such as `jiter`. A timeout from `files.pythonhosted.org` is a network/package-index issue, not a benchmark failure. Retry with a longer timeout and serialized downloads:

```powershell
$env:UV_HTTP_TIMEOUT = "180"
$env:UV_CONCURRENT_DOWNLOADS = "1"
uv sync
```

If the public index is slow or blocked on your network, configure a reachable package mirror for the current PowerShell session with `UV_INDEX_URL`, then run `uv sync` again. Keep the mirror URL out of committed project files unless it is an intentional project-wide choice.

## Run an evaluation

Run commands from the repository root because the benchmark and publishing utilities resolve paths from the current working directory:

```powershell
uv run python benchmark/konkur_bench.py
```

For a low-cost smoke test, first set `EVAL_MAX_SAMPLES=5` and select one exam in `.env`.

Each model writes one file per exam under `benchmark_results/<model>/`. Despite the `.json` suffix, these files use JSON Lines format: one result record per line. Existing valid question IDs are skipped on later runs, so interrupted evaluations can resume.

## Metrics

- **Text-only score (primary):** accuracy on questions without figures. This keeps text-only and vision-capable models comparable.
- **Standard score:** accuracy across all attempted text and figure questions.
- **Per-exam score:** both metrics broken down by exam session.

The leaderboard generator deduplicates results by question ID and keeps the latest record.

## Refresh and preview the leaderboard

Generate `hf_space/leaderboard_data.json` from saved benchmark outputs:

```powershell
uv run python scripts/generate_hf_data.py
```

Preview the main Space locally:

```powershell
Push-Location hf_space
uv run --project .. python app.py
Pop-Location
```

Preview the linguistic Space:

```powershell
Push-Location hf_space_linguistic
uv run --project .. python app.py
Pop-Location
```

## Dataset pipeline

The scripts are intentionally kept as explicit workflow stages:

1. `scripts/extract_questions.py` extracts questions from source PDFs.
2. `scripts/extract_keys.py` extracts answer keys.
3. `scripts/process_exam.py` combines and validates an exam.
4. `scripts/upload_to_hf.py` builds and uploads the Hugging Face dataset.

These tools expect local source material in the ignored `pdfs/` and `dataset/` directories. See [Hugging Face Space notes](docs/hf_space.md) and [OpenRouter usage notes](docs/openrouter_usage.md) for the earlier detailed workflow.

## Deploy the Spaces

Set `HF_TOKEN` and `HF_USERNAME` in `.env`, then run:

```powershell
uv run python scripts/deploy_space.py
uv run python scripts/deploy_linguistic_space.py
```

Deployment scripts create or reuse the configured Hugging Face Space and upload the corresponding folder. Review the destination repository IDs in the scripts before publishing.

## Security and data notes

- `.env` and provider credentials are local-only.
- Benchmark outputs may include raw provider errors or responses; inspect new result files before publishing.
- Source exams and extracted local datasets are excluded from Git. Confirm redistribution rights before publishing those materials.
- The runner preserves existing retry, fallback, resumability, and logging behavior.
