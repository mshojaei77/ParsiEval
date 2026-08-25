import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fluency_prompts_fa import get_fa_fluency_cases


def _load_dotenv_fallback(env_path: Path) -> None:
    if not env_path.exists():
        return
    raw = env_path.read_text(encoding="utf-8")
    key = None
    buf = []

    def flush() -> None:
        nonlocal key, buf
        if key is None:
            return
        val = "\n".join(buf)
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        os.environ.setdefault(key, val)
        key = None
        buf = []

    for line in raw.splitlines():
        if not line.strip():
            flush()
            continue
        if line.lstrip().startswith("#"):
            continue
        if "=" in line and not line.startswith((" ", "\t")):
            flush()
            k, v = line.split("=", 1)
            key = k.strip()
            buf = [v]
        else:
            if key is not None:
                buf.append(line)
    flush()


def _find_env_path() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


def _load_env() -> None:
    env_path = _find_env_path()
    if env_path is None:
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path)
    except Exception:
        _load_dotenv_fallback(env_path)


def _parse_models(raw: str) -> list[str]:
    if not raw:
        return []
    cleaned = raw.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1]
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\n", ",")
    return [m.strip() for m in cleaned.split(",") if m.strip()]


def _as_text(resp) -> str:
    try:
        choice = resp.choices[0]
        msg = getattr(choice, "message", None)
        if msg is not None and getattr(msg, "content", None):
            return msg.content
        if getattr(choice, "text", None):
            return choice.text
    except Exception:
        pass
    return ""


def _usage_dict(resp) -> dict:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    out = {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(usage, k, None)
        if v is not None:
            out[k] = int(v)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    _load_env()

    base_url = os.getenv("EVAL_SUBJECT_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1"
    api_key = os.getenv("EVAL_SUBJECT_API_KEY") or os.getenv("OPENAI_API_KEY")
    models = _parse_models(os.getenv("EVAL_SUBJECT_MODELS", ""))
    if not api_key:
        raise SystemExit("Missing EVAL_SUBJECT_API_KEY or OPENAI_API_KEY in .env")
    if not models:
        raise SystemExit("Missing EVAL_SUBJECT_MODELS in .env")

    try:
        import openai
    except Exception as e:
        raise SystemExit(f"Missing dependency: openai ({e})")

    out_path = Path(args.out) if args.out else Path(__file__).resolve().parent / "out"
    out_path.mkdir(parents=True, exist_ok=True)
    # Fixed filename, no timestamp
    output_file = out_path / "fa_subject_outputs.jsonl"

    # Check for existing models
    existing_models = set()
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "model" in data:
                        existing_models.add(data["model"])
                except json.JSONDecodeError:
                    pass
    
    models_to_run = [m for m in models if m not in existing_models]
    
    if not models_to_run:
        print(f"All models {models} already exist in {output_file}. Skipping.")
        return 0

    print(f"Running for models: {models_to_run} (Skipped: {list(existing_models)})")

    system = (
        "تو گویشوری بسیار توانا در زبان فارسی هستی و دقیقاً مانند یک انسان ایرانی واقعی "
        "مکالمه می‌کنی و متن می‌نویسی؛ هرگز شبیه ربات یا دستیار هوشمند به نظر نمی‌رسی. "
        "ثبتِ لحن را با توجه به موقعیت انتخاب کن: "
        "وقتی کاربر محاوره‌ای حرف می‌زند یا گفت‌وگوی روزمره، چت، یا پاسخ کوتاه می‌خواهد، "
        "با فارسی محاوره‌ایِ طبیعی و روان جواب بده (مثل 'چشم'، 'حتماً'، 'باشه')، "
        "بدون کتابی‌گویی و بدون ساختارهای ترجمه‌ای، و مکالمه را طبیعی ادامه بده. "
        "وقتی کاربر نوشتار رسمی می‌خواهد (نامه، مقاله، کتاب، ایمیل اداری)، "
        "با فارسی معیارِ رسا و ادبی اما طبیعی بنویس. "
        "از فهرست‌نویسی و بولت‌گذاری خودداری کن مگر خواسته شده باشد. "
        "پاسخ را در حد و اندازهٔ درخواست نگه دار. "
        "هرگز خودت را هوش مصنوعی معرفی نکن، تبعیض قائل نشو، و اطلاعات شخصی واقعی افراد را افشا نکن."
    )

    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    cases = get_fa_fluency_cases()

    # Attempt to import tqdm for progress visualization
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, desc=None, unit=None):
            print(f"Processing {desc}...")
            return iterable

    # Open in append mode
    with output_file.open("a", encoding="utf-8") as f:
        for model in models_to_run:
            for run_idx in range(args.runs):
                for case in tqdm(cases, desc=f"Model: {model} Run: {run_idx+1}", unit="case"):
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": case["user"]},
                    ]
                    
                    max_retries = 5
                    retry_delay = 5
                    
                    for attempt in range(max_retries):
                        try:
                            resp = client.chat.completions.create(
                                model=model,
                                messages=messages,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                            )
                            text = _as_text(resp)
                            usage = _usage_dict(resp)
                            row = {
                                "language": "fa",
                                "category": "linguistic_fluency",
                                "case_id": case["id"],
                                "domain": case.get("domain", ""),
                                "tags": case.get("tags", []),
                                "model": model,
                                "run": run_idx + 1,
                                "prompt": case["user"],
                                "response": text,
                                "usage": usage,
                            }
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            # Force flush to ensure data is written immediately
                            f.flush()
                            time.sleep(0.05)
                            break # Success, exit retry loop
                        except Exception as e:
                            is_rate_limit = "429" in str(e) or "Rate limit" in str(e)
                            if is_rate_limit and attempt < max_retries - 1:
                                wait_time = retry_delay * (2 ** attempt) # Exponential backoff
                                print(f"\nRate limit hit for {model}. Retrying in {wait_time}s...")
                                time.sleep(wait_time)
                            else:
                                print(f"\nError processing {model} case {case['id']}: {e}")
                                # If it's the last attempt, we might want to log the error or re-raise
                                # For now, let's log an error row so we don't crash the whole batch
                                error_row = {
                                    "language": "fa",
                                    "category": "linguistic_fluency",
                                    "case_id": case["id"],
                                    "domain": case.get("domain", ""),
                                    "tags": case.get("tags", []),
                                    "model": model,
                                    "run": run_idx + 1,
                                    "prompt": case["user"],
                                    "response": f"ERROR: {str(e)}",
                                    "usage": {},
                                }
                                f.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                                f.flush()
                                break


    print(str(output_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
