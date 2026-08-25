# Persian Fluency Bench

A benchmark for evaluating the **linguistic fluency of LLMs in Modern Native Persian (Farsi)**.

Persian Fluency Bench measures how convincingly a model communicates like a real Iranian — with **deep care for tone and style** across two registers:

- **Casual** — chats, everyday talking, replying: natural spoken Tehrani Persian (*mahavireh*), Taarof politeness, idioms, warmth, human hesitations.
- **Formal** — writing books, articles, letters: eloquent standard written Persian (*fasih-e ma'yar*), precise grammar, literary elegance, official correspondence tone.

The core skill under test is **register awareness**: casual contexts must never be answered in stiff book-Persian (ketabi), and formal writing must never slide into slang. Translationese, robotic phrasing, and AI self-references are heavily penalized in both tracks.

## Domains

Cases span **10 everyday domains** people most often need help with:

| | | |
|---|---|---|
| Health & Medical | Education & Learning | Shopping & E-commerce |
| Travel & Tourism | Banking & Finance | Food & Restaurants / Cooking |
| Technology & Internet | Career & Workplace | Transportation & Cars |
| Sports & Entertainment | | |

Every domain appears in both the casual and the formal track. See [`CATEGORIES.txt`](CATEGORIES.txt) for the full rationale of each track.

## Repository layout

```
persian-fluency-bench/
├── bench.linguistic.md            # Leaderboard + methodology notes
├── CATEGORIES.txt                 # Track goals & purposes
└── linguistic_fluency/
    ├── run_benchmark.py           # Orchestrator: runs subject + judge
    └── fa/                        # Persian cases + subject/judge scripts
        ├── fluency_prompts_fa.py  # 30 cases: 10 domains x casual + formal
        ├── run_subject_fluency_fa.py
        ├── run_judge_fluency_fa.py
        └── out_v1_real_estate/    # Archived v1 results (legacy protocol)
```

## How it works

- `fluency_prompts_fa.py` – 30 benchmark cases across 10 everyday domains; every case declares a `domain` and a register tag (`casual` / `formal`) plus optional skill tags (`instruction_following`, `context_retention`, ...).
- `run_subject_fluency_fa.py` – queries subject models with a register-aware system prompt ("colloquial when chatting, standard when writing") and appends rows to `out/fa_subject_outputs.jsonl`. Resumable: already-run models are skipped.
- `run_judge_fluency_fa.py` – LLM-judge scoring against a strict native-speaker rubric (0–100 per metric), told each case's expected register so it can punish register mismatch. Deterministic conservative penalties are applied post-hoc (wrong-script responses, bullet lists, AI self-references; spoken-style length/parenthesis penalties only on the casual track). Produces `fa_judged.jsonl`, `fa_summary.json`, `fa_report.md`.

## Results

Leaderboard and methodology notes: [`bench.linguistic.md`](bench.linguistic.md).

The legacy v1 run (casual-only, real-estate operator persona) is archived under
`linguistic_fluency/fa/out_v1_real_estate/`; its scores are not comparable with v2 runs.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in your keys
```

All scripts load `.env` from the repo root (via `python-dotenv`, with a built-in fallback parser if it is not installed).

### Environment variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `EVAL_SUBJECT_API_KEY` (or `OPENAI_API_KEY`) | subject script | API key for the provider serving the subject models |
| `EVAL_SUBJECT_BASE_URL` (or `OPENAI_BASE_URL`) | subject script | OpenAI-compatible base URL (default: `https://openrouter.ai/api/v1`) |
| `EVAL_SUBJECT_MODELS` | subject script | Comma/newline-separated list of OpenRouter-style model IDs |
| `JUDGE_API_KEY` (or `OPENAI_API_KEY`) | judge script | API key for the judge provider |
| `JUDGE_BASE_URL` (or `OPENAI_BASE_URL`) | judge script | Judge base URL (default: `https://openrouter.ai/api/v1`) |
| `JUDGE_MODEL` | judge script | Judge model ID (default: `google/gemini-2.5-flash`) |

## Usage

```bash
# Subject then judge:
python linguistic_fluency/run_benchmark.py

# Only the judge step:
python linguistic_fluency/run_benchmark.py --steps judge

# Subject only:
python linguistic_fluency/run_benchmark.py --steps subject
```

Both steps are resumable: subject models already present in `fa/out/fa_subject_outputs.jsonl` are skipped, and judged rows are deduplicated by `(model, case_id, response)`.

## Methodology notes

- The judge is instructed to be extremely strict and to penalize translationese, robotic tone, register mismatch, formulaic responses, unnecessary lists/markdown, and AI self-references.
- Register rules: casual answers with ketabi phrasing are auto-penalized hard; formal pieces written in slang/textspeak are penalized equally hard. Parentheses and longer prose are legitimate in formal writing, so those penalties apply only to the casual track.
- Scores use a 0–100 scale in 5-point increments.

## License

MIT — see [LICENSE](LICENSE).
