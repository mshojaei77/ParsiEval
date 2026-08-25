# Persian Fluency Bench — Linguistic Fluency Leaderboard

Persian Fluency Bench evaluates the **linguistic fluency of LLMs in Modern Native Persian (Farsi)** across two tracks, spanning 10 everyday domains (health, education, shopping, travel, banking, food, technology, career, transportation, entertainment):

| Track | Contexts | Expected register |
|-------|----------|-------------------|
| **Casual** | chats, everyday talking, short replies | Natural spoken Tehrani Persian (mahavireh) with Taarof and human flow |
| **Formal** | writing books, articles, letters | Eloquent standard written Persian (fasih-e ma'yar), precise and literary |

Models are ranked by their **Overall** score. The judge scores grammar, idiomatic quality, conciseness, politeness, naturalness, instruction following (including **register match**), context retention, and safety.

> **Note (v2)**: The benchmark was redesigned around these two registers and 10 everyday
> domains. The table below is the legacy v1 run — a single-track casual protocol with a
> real-estate operator persona (archived in `linguistic_fluency/fa/out_v1_real_estate/`).
> Scores are **not comparable** with v2 runs.

## Legacy v1 Results — Casual track (real-estate operator persona)

- **Date**: 2026-01-06 00:10:40
- **Judge**: `google/gemini-2.5-flash`

| Model | Overall | Grammar | Idiomatic | Conciseness | Politeness | Naturalness | Instruction Following | Context Retention | Safety |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `qwen/qwen3-30b-a3b-instruct-2507` | **42.1** | 49.8 | 42.7 | **36.6** | 47.7 | 42.9 | 60.5 | 78.5 | 81.5 |
| `mistralai/mistral-saba` | 39.1 | 44.0 | 37.8 | 36.5 | 43.8 | 37.2 | **67.7** | 84.9 | 89.1 |
| `google/gemini-2.5-flash` | 39.1 | **58.6** | **53.1** | 34.4 | **56.2** | **52.8** | 65.5 | 76.7 | 77.3 |
| `google/gemma-3-12b-it` | 37.0 | 56.1 | 50.3 | 29.1 | 54.3 | 50.1 | 65.6 | **85.9** | **89.5** |
| `qwen/qwen3-next-80b-a3b-instruct` | 35.5 | 51.8 | 45.6 | 31.8 | 50.2 | 46.8 | 56.8 | 76.6 | 81.5 |
| `google/gemma-3-4b-it` | 34.6 | 48.5 | 42.5 | 26.4 | 46.7 | 42.0 | 57.7 | 76.5 | 86.4 |
| `google/gemma-3-27b-it` | 32.9 | 50.2 | 45.5 | 24.5 | 48.4 | 44.8 | 56.5 | 73.5 | 83.1 |
| `google/gemma-3n-e4b-it` | 31.7 | 45.1 | 39.7 | 24.4 | 42.8 | 39.1 | 49.1 | 73.1 | 77.0 |
| `baidu/ernie-4.5-21b-a3b` | 27.6 | 30.0 | 25.9 | 25.3 | 30.3 | 25.9 | 47.1 | 59.4 | 70.3 |
| `ibm-granite/granite-4.0-h-micro` | 14.9 | 14.9 | 12.8 | 15.1 | 15.1 | 12.8 | 23.8 | 32.3 | 67.8 |
| `qwen/qwen3-4b:free` | 10.0 | 10.0 | 9.2 | 13.3 | 10.0 | 9.2 | 15.4 | 17.1 | 34.2 |
| `meta-llama/llama-3.2-3b-instruct` | 9.3 | 9.3 | 8.7 | 10.6 | 9.6 | 8.7 | 12.2 | 16.6 | 78.0 |
| `mistralai/ministral-3b` | 8.4 | 8.4 | 7.4 | 8.7 | 8.4 | 7.4 | 11.6 | 13.9 | 75.0 |
| `mistralai/mixtral-8x7b-instruct` | 6.3 | 6.3 | 5.5 | 6.1 | 6.3 | 5.5 | 8.2 | 15.0 | 50.0 |
| `liquid/lfm-2.2-6b` | 4.1 | 4.1 | 4.1 | 4.1 | 4.1 | 4.1 | 4.1 | 4.1 | 68.5 |
| `liquid/lfm2-8b-a1b` | 3.9 | 3.9 | 3.7 | 3.9 | 3.7 | 3.7 | 3.9 | 3.9 | 59.0 |

## v2 Two-Track Results

Pending first full run under the v2 protocol (casual + formal cases, register-aware judging).
