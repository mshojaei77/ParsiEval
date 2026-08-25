# ParsiEval — Persian LLM Evaluation Suite

An umbrella repository of benchmarks for evaluating Large Language Models on
Persian: knowledge, language understanding, and linguistic fluency.

## Benchmarks

### 1. ParsiEval (`./`)

The original benchmark: **364 Persian multiple-choice questions** spanning
history, literature, general knowledge, and science, testing comprehension,
fact recall, and logical inference.

- Runner: `run_eval.py` (OpenAI-compatible APIs: OpenAI, AvalAI, Groq, Cerebras, Ollama, LM Studio, OpenRouter)
- Datasets: `parsi-eval-1.csv`, `parsi-eval-2/` (per-subject CSVs)
- Results: `results/`, visualizations: `plots/` (via `create_visuals.py`)
- Legacy provider variants in `etc/`

### 2. Konkur 1404 (`konkur-eval/`)

A reproducible benchmark runner and leaderboard toolkit for **Konkur 1404**,
the 2025 Iranian university entrance examination: **2,137 Persian and English
MCQs across ten exam sessions**, with multimodal support for figure questions.

- Resumable runner over OpenAI-compatible providers (`konkur-eval/benchmark/`)
- Versioned per-model results (`konkur-eval/benchmark_results/`)
- Two Gradio leaderboard Spaces: overall + Persian linguistic fluency
  (`konkur-eval/hf_space/`, `konkur-eval/hf_space_linguistic/`)
- Exam PDFs and dataset tooling (`konkur-eval/pdfs/`, `konkur-eval/dataset/`)

See `konkur-eval/README.md` for full details.

### 3. Persian Fluency Bench (`persian-fluency-bench/`)

Measures how convincingly a model communicates like a real Iranian, with deep
care for **register awareness**: **30 judge-scored cases across 10 everyday
domains × casual (spoken Tehrani) and formal (written standard) tracks**.
Translationese, robotic phrasing, and AI self-references are penalized.

- Subject/judge harness (`persian-fluency-bench/linguistic_fluency/fa/`)
- Leaderboard + methodology (`persian-fluency-bench/bench.linguistic.md`)

See `persian-fluency-bench/README.md` for full details.

## Setup

Each benchmark is self-contained with its own `pyproject.toml` /
`requirements.txt`. Copy the relevant `.env.example` to `.env` and fill in
provider keys. Never commit `.env` or session files.

## Motivation

High-quality benchmarks for languages other than English are crucial for
multilingual NLP. This suite tracks the progress of Persian-capable models
across knowledge (Konkur), understanding (ParsiEval), and naturalness
(Fluency Bench).
