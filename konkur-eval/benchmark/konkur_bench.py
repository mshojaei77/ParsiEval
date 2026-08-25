import os
import io
import base64
import re
import csv
import time
import json
import glob
try:
    import msvcrt
except ImportError:
    msvcrt = None
from collections import defaultdict

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

def read_env_values():
    env_path = os.path.join(os.getcwd(), ".env")
    values = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        key = None
        buf = []
        for line in lines:
            s = line.rstrip("\n")
            if not s or s.strip().startswith("#"):
                continue
            if key is None and "=" in s:
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip()
                if v == '"':
                    key = k
                    buf = []
                elif v.startswith('"') and not v.endswith('"'):
                    key = k
                    buf = [v[1:]]
                elif v.startswith('"') and v.endswith('"'):
                    values[k] = v[1:-1]
                else:
                    values[k] = v
            elif key is not None:
                if s.strip() == '"':
                    values[key] = "\n".join(buf)
                    key = None
                    buf = []
                elif s.endswith('"'):
                    buf.append(s[:-1])
                    values[key] = "\n".join(buf)
                    key = None
                    buf = []
                else:
                    buf.append(s)
    return values

def get_exams():
    vals = read_env_values()
    env_exams = vals.get("EVAL_EXAMS") or os.getenv("EVAL_EXAMS")
    
    # List of all known exams in the dataset
    all_exams = [
        "ensani_nobat1", "ensani_nobat2", 
        "tajrobi_nobat1", "tajrobi_nobat2", 
        "riazi_nobat1", "riazi_nobat2", 
        "honar_nobat1", "honar_nobat2", 
        "zaban_nobat1", "zaban_nobat2"
    ]
    
    if env_exams:
        if env_exams.strip().lower() == "all":
            return all_exams
        return [e.strip() for e in env_exams.split(",") if e.strip()]
    return ["ensani_nobat1", "ensani_nobat2"]

USE_IMAGES = True

def get_subject_config():
    vals = read_env_values()
    base_url = vals.get("EVAL_SUBJECT_BASE_URL") or os.getenv("EVAL_SUBJECT_BASE_URL") or "https://openrouter.ai/api/v1"
    api_key = vals.get("EVAL_SUBJECT_API_KEY") or os.getenv("EVAL_SUBJECT_API_KEY") or (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "your-api-key"))
    return base_url, api_key

def get_models():
    vals = read_env_values()
    raw = vals.get("EVAL_SUBJECT_MODELS") or os.getenv("EVAL_SUBJECT_MODELS") or ""
    models = []
    for line in raw.splitlines():
        s = line.strip()
        if s:
            models.append(s)
    return models

BASE_URL, API_KEY = get_subject_config()
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def format_prompt(example):
    prompt = f"Question: {example['question']}\n\n"
    for i, choice in enumerate(example['choices']):
        prompt += f"{i+1}. {choice}\n"
    prompt += "\nAnswer with the number of the correct choice (1, 2, 3, or 4) only."
    return prompt

def extract_answer(response_text):
    text = response_text.strip()
    for k, v in {
        "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "١": "1", "٢": "2", "٣": "3", "٤": "4"
    }.items():
        text = text.replace(k, v)
    for k, v in {"one": "1", "two": "2", "three": "3", "four": "4"}.items():
        if re.search(rf"\b{k}\b", text, flags=re.IGNORECASE):
            text = v
            break
    for k, v in {"یک": "1", "يك": "1", "دو": "2", "سه": "3", "چهار": "4"}.items():
        if k in text:
            text = v
            break
    m = re.search(r"\b([1-4])\b", text)
    return int(m.group(1)) if m else None

def figure_to_base64(figure):
    if not figure:
        return None
    try:
        if hasattr(figure, "save"):
            buf = io.BytesIO()
            figure.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        if isinstance(figure, str):
            path = figure
            if not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)
            from PIL import Image
            img = Image.open(path)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None
    return None

def timed_choice(prompt, timeout=30, default='y'):
    """
    Waits for a single keypress (y/n/r/m) with a timeout.
    Returns the character pressed or default if timeout.
    """
    print(prompt, end="", flush=True)
    
    # If msvcrt is not available (non-Windows), fallback to blocking input
    if msvcrt is None:
        return input().strip().lower()

    start_time = time.time()
    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getwch().lower()
            # Echo the character to confirm receipt
            print(ch)
            return ch
        
        if time.time() - start_time > timeout:
            print(f"\nTimeout ({timeout}s), defaulting to '{default}'")
            return default
            
        time.sleep(0.1)

import openai

class ModelRateLimitedError(RuntimeError):
    """Signal that the current model should be abandoned for this run."""


def chat_with_retries(messages, model_name, max_retries=3):
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
            )
            
            # Validate response content
            if not resp.choices or not resp.choices[0].message.content or not resp.choices[0].message.content.strip():
                raise ValueError("Received empty response from model")
                
            return resp
        except openai.NotFoundError as e:
            # Fail fast for 404 errors related to unsupported features (like images)
            if "support image" in str(e).lower() or "endpoint" in str(e).lower():
                raise e
            # Otherwise retry normally
            print(f"Retry {attempt+1}/{max_retries} due to error: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(8.0, delay * 2)
        except openai.RateLimitError as e:
            raise ModelRateLimitedError(
                f"Model {model_name} is rate-limited; switching to the next model."
            ) from e
        except openai.AuthenticationError as e:
            raise RuntimeError(
                "Provider authentication failed (HTTP 401). Check "
                "EVAL_SUBJECT_BASE_URL and EVAL_SUBJECT_API_KEY in .env."
            ) from e
        except Exception as e:
            print(f"Retry {attempt+1}/{max_retries} due to error: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(8.0, delay * 2)

def deduplicate_results(results_dir):
    """
    Reads all JSON files in the results directory, removes duplicate entries 
    (keeping the latest one per ID), and writes back the clean file.
    """
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    for jf in json_files:
        try:
            unique_records = {}
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        record = json.loads(line)
                        if "id" in record:
                            unique_records[record["id"]] = record
                    except json.JSONDecodeError:
                        continue
            
            # Write back strictly unique records
            with open(jf, "w", encoding="utf-8") as f:
                for record in unique_records.values():
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
        except Exception as e:
            print(f"Error deduplicating {jf}: {e}")

def evaluate_model(model_name):
    safe_model_name = sanitize_filename(model_name)
    results_dir = os.path.join(os.getcwd(), "benchmark_results", safe_model_name)
    os.makedirs(results_dir, exist_ok=True)

    # Force redownload to ensure we have the latest dataset updates (e.g. fixed answer keys)
    #ds = load_dataset("mshojaei77/konkur1404", split="train", download_mode="force_redownload")
    ds = load_dataset("mshojaei77/konkur1404", split="train")
    exams = get_exams()
    
    if exams:
        ds = ds.filter(lambda x: x.get("exam_name") in exams)
    
    vals = read_env_values()
    max_samples = int(vals.get("EVAL_MAX_SAMPLES") or os.getenv("EVAL_MAX_SAMPLES", "0") or "0")

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    # Load existing results to skip processed IDs
    processed_ids = set()
    existing_results_by_exam = defaultdict(list)
    
    # Check all json files in the model directory
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        record = json.loads(line)
                        # Check if answer is valid (non-null). 
                        # Support both new 'llm_answer' and old 'predicted' keys for compatibility.
                        ans = record.get("llm_answer")
                        if ans is None:
                            ans = record.get("predicted")
                        
                        # Only mark as processed if we have a valid answer OR it was explicitly skipped
                        if ans is not None or record.get("skipped"):
                            processed_ids.add(record["id"])
                            
                        existing_results_by_exam[record["exam_name"]].append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

    # Filter out already processed examples
    ds_to_process = ds.filter(lambda x: x["id"] not in processed_ids)
    print(f"Model: {model_name} | Total: {len(ds)} | Processed: {len(processed_ids)} | Remaining: {len(ds_to_process)}")

    if len(ds_to_process) == 0:
        return

    # Process remaining examples
    # Group writes by exam to avoid opening/closing files too often, 
    # but for simplicity and safety, appending per item or small batch is fine.
    # We'll append per item to ensure resumability at item level.
    
    file_handles = {}
    vision_supported = True # Assume supported initially

    try:
        for example in tqdm(ds_to_process):
            prompt = format_prompt(example)
            messages = [{"role": "system", "content": "Answer only with 1, 2, 3, or 4. Be extremely fast and respond instantly. don't do any thinking and resoning, Reasoning: low"}]
            has_image = False
            img_b64 = None
            if USE_IMAGES:
                img_b64 = figure_to_base64(example.get("figure"))
                if img_b64:
                    has_image = True

            # If model is known to not support vision, skip image questions
            if has_image and not vision_supported:
                # Log as skipped
                exam = example.get("exam_name", "unknown")
                result_record = {
                    "id": example["id"],
                    "exam_name": exam,
                    "llm_answer": None,
                    "key": example.get("answer_key"),
                    "correct": 0,
                    "error": "Skipped: Model does not support vision",
                    "model": model_name,
                    "has_image": True,
                    "skipped": True
                }
                if exam not in file_handles:
                    file_path = os.path.join(results_dir, f"{exam}.json")
                    file_handles[exam] = open(file_path, "a", encoding="utf-8")
                file_handles[exam].write(json.dumps(result_record, ensure_ascii=False) + "\n")
                file_handles[exam].flush()
                continue

            if USE_IMAGES and img_b64:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]
                })
            else:
                messages.append({"role": "user", "content": prompt})

            pred = None
            error_msg = ""
            prediction_text = ""

            def run_model_once():
                nonlocal pred, error_msg, prediction_text, vision_supported
                pred = None
                error_msg = ""
                prediction_text = ""
                try:
                    resp = chat_with_retries(messages, model_name)
                    prediction_text = resp.choices[0].message.content.strip()
                    pred = extract_answer(prediction_text)
                except ModelRateLimitedError:
                    raise
                except openai.NotFoundError as e:
                    error_str = str(e).lower()
                    if "support image" in error_str or "endpoint" in error_str:
                        print(f"\nModel {model_name} does not support vision. Switching to text-only mode.")
                        vision_supported = False
                        error_msg = f"Vision not supported: {str(e)}"
                        exam = example.get("exam_name", "unknown")
                        result_record = {
                            "id": example["id"],
                            "exam_name": exam,
                            "llm_answer": None,
                            "key": example.get("answer_key"),
                            "correct": 0,
                            "error": error_msg,
                            "model": model_name,
                            "has_image": has_image,
                            "skipped": True
                        }
                        if exam not in file_handles:
                            file_path = os.path.join(results_dir, f"{exam}.json")
                            file_handles[exam] = open(file_path, "a", encoding="utf-8")
                        file_handles[exam].write(json.dumps(result_record, ensure_ascii=False) + "\n")
                        file_handles[exam].flush()
                        return False
                    error_msg = str(e)
                except Exception as e:
                    error_msg = str(e)
                    if (USE_IMAGES and img_b64) and ("image" in error_msg.lower() or "support image" in error_msg.lower()):
                        try:
                            fallback_msgs = [{"role": "system", "content": "Answer only with 1, 2, 3, or 4."},
                                             {"role": "user", "content": prompt}]
                            resp = chat_with_retries(fallback_msgs, model_name)
                            prediction_text = resp.choices[0].message.content.strip()
                            pred = extract_answer(prediction_text)
                            error_msg = ""
                        except Exception as e2:
                            error_msg = str(e2)
                if not error_msg and pred is None:
                    print(f"\n[WARN] ID={example['id']}: Model returned content but no answer extracted. Content: {prediction_text[:100]!r}...")
                return True

            should_continue_outer = run_model_once()
            if not should_continue_outer:
                continue

            # Handle potentially missing or None answer_key
            raw_key = example.get("answer_key")
            gt = None
            key_is_valid = False
            
            if raw_key is not None:
                try:
                    gt = int(raw_key)
                    key_is_valid = True
                except (ValueError, TypeError):
                    pass
            
            if not key_is_valid:
                print(f"Skipping id={example.get('id')} due to missing/invalid answer_key: {raw_key}")
                # Write failure record so it's logged
                result_record = {
                    "id": example["id"],
                    "exam_name": example.get("exam_name", "unknown"),
                    "llm_answer": None,
                    "key": raw_key,
                    "correct": 0,
                    "error": f"Invalid or missing answer_key in dataset: {raw_key}",
                    "model": model_name,
                    "has_image": has_image
                }
                
                exam = example.get("exam_name", "unknown")
                if exam not in file_handles:
                    file_path = os.path.join(results_dir, f"{exam}.json")
                    file_handles[exam] = open(file_path, "a", encoding="utf-8")
                
                file_handles[exam].write(json.dumps(result_record) + "\n")
                file_handles[exam].flush()
                continue

            exam = example.get("exam_name", "unknown")
            
            skipped_by_user = False
            if pred is None:
                skip_example = False
                while True:
                    print(f"\n[ATTENTION] ID={example.get('id')} has 'llm_answer': null")
                    if prediction_text:
                        print("Raw LLM Response:")
                        print(prediction_text)
                    if error_msg:
                        print(f"Error: {error_msg}")

                    choice = timed_choice(">> Continue to next question? (y/n/r/m): ")
                    if choice == "y":
                        # Mark as skipped so we don't ask again next time
                        skipped_by_user = True
                        break
                    if choice == "n":
                        print("Exiting benchmark by user request.")
                        import sys
                        sys.exit(0)
                    if choice == "r":
                        should_continue_outer = run_model_once()
                        if not should_continue_outer:
                            skip_example = True
                            break
                        if pred is not None:
                            break
                    if choice == "m":
                        print("\n----- RAW LLM RESPONSE START -----")
                        if prediction_text:
                            print(prediction_text)
                        print("----- RAW LLM RESPONSE END -----\n")
                        while True:
                            manual = input(">> Enter answer (1/2/3/4) or blank to cancel: ").strip()
                            if manual == "":
                                break
                            if manual in ("1", "2", "3", "4"):
                                pred = int(manual)
                                break
                        if pred is not None:
                            break
                if skip_example:
                    continue

            if pred is not None and pred not in [1, 2, 3, 4]:
                pred = None

            ok = 1 if (pred is not None and pred == gt) else 0

            result_record = {
                "id": example.get("id"),
                "exam_name": exam,
                "llm_answer": pred,
                "key": gt,
                "correct": ok, 
                "error": error_msg,
                "model": model_name,
                "has_image": has_image,
                "skipped": skipped_by_user
            }
            
            if not error_msg and pred is None:
                result_record["raw_response"] = prediction_text

            if error_msg:
                print(f"Error on id={example.get('id')} exam={exam}: {error_msg}")

            # Write to file immediately
            if exam not in file_handles:
                file_path = os.path.join(results_dir, f"{exam}.json")
                file_handles[exam] = open(file_path, "a", encoding="utf-8")
            
            file_handles[exam].write(json.dumps(result_record, ensure_ascii=False) + "\n")
            file_handles[exam].flush()

    except ModelRateLimitedError as e:
        print(f"\n[SKIP MODEL] {e}")
        return "rate_limited"
    finally:
        for fh in file_handles.values():
            fh.close()
        
        # Clean up duplicates after processing
        print(f"Deduplicating results for {model_name}...")
        deduplicate_results(results_dir)

def generate_leaderboard():
    results_root = os.path.join(os.getcwd(), "benchmark_results")
    if not os.path.exists(results_root):
        print("No results found to generate leaderboard.")
        return

    # Load dataset to determine which questions have images (for backward compatibility)
    print("Loading dataset for leaderboard generation...")
    try:
        ds = load_dataset("mshojaei77/konkur1404", split="train")
        image_lookup = {item['id']: bool(item.get('figure')) for item in ds}
    except Exception as e:
        print(f"Warning: Could not load dataset for image lookup: {e}")
        image_lookup = {}

    # Aggregate all results
    # Use dictionary to store unique results by ID (latest entry wins) to handle retries
    model_results = defaultdict(dict) # model_name -> {id -> record}
    
    # Walk through all model directories
    for model_dir in glob.glob(os.path.join(results_root, "*")):
        if not os.path.isdir(model_dir): continue
        
        json_files = glob.glob(os.path.join(model_dir, "*.json"))
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        r = json.loads(line)
                        m_name = r.get("model", os.path.basename(model_dir))
                        model_results[m_name][r["id"]] = r
                    except json.JSONDecodeError:
                        continue

    # Calculate stats from unique results
    results = []
    for m_name, unique_records in model_results.items():
        stats = {
            "correct": 0, "total": 0, 
            "correct_text": 0, "total_text": 0,
            "per_exam": defaultdict(lambda: {"correct": 0, "total": 0, "correct_text": 0, "total_text": 0})
        }
        
        vision_capable = True # Assume true unless we see a 'skipped' due to vision

        for r in unique_records.values():
            exam = r.get("exam_name", "unknown")
            
            # Determine has_image: check record first, then lookup
            has_image = r.get("has_image")
            if has_image is None:
                has_image = image_lookup.get(r["id"], False)
                
            skipped = r.get("skipped", False)
            
            # Check if model is vision capable based on error logs/skipped status
            if skipped and "support vision" in r.get("error", "").lower():
                vision_capable = False

            # Standard Score (All questions)
            # If skipped due to no vision, it counts as 0 correct but adds to total
            # Wait, user said "Ensure skipped questions don't negatively impact model rankings"
            # But "Standard score (including all questions)" implies they are included.
            # If we exclude them from Standard Score, then Standard Score == Text Score for text models.
            # Let's count them in Total for Standard Score (so they get penalized in overall metric),
            # BUT the primary ranking is Vision-Excluded Score.
            
            # However, if I count them as total=1, correct=0, then accuracy drops.
            # User: "Ensure skipped questions don't negatively impact model rankings"
            # AND "Clearly labeling which score is being used for primary ranking (Vision-excluded score)"
            # So the primary ranking won't be impacted.
            # The "Standard score" will be lower for text-only models, which is fair.
            
            stats["total"] += 1
            stats["correct"] += r["correct"]
            stats["per_exam"][exam]["total"] += 1
            stats["per_exam"][exam]["correct"] += r["correct"]

            # Text-Only Score (Non-image questions)
            if not has_image:
                stats["total_text"] += 1
                stats["correct_text"] += r["correct"]
                stats["per_exam"][exam]["total_text"] += 1
                stats["per_exam"][exam]["correct_text"] += r["correct"]
            
        # Calculate accuracies
        acc_all = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        acc_text = (stats["correct_text"] / stats["total_text"] * 100) if stats["total_text"] > 0 else 0
        
        results.append({
            "model": m_name,
            "vision_capable": vision_capable,
            "accuracy_all": acc_all,
            "correct_all": stats["correct"],
            "total_all": stats["total"],
            "accuracy_text": acc_text,
            "correct_text": stats["correct_text"],
            "total_text": stats["total_text"],
            "per_exam": {
                k: {
                    "accuracy_all": (v["correct"] / v["total"] * 100) if v["total"] > 0 else 0,
                    "accuracy_text": (v["correct_text"] / v["total_text"] * 100) if v["total_text"] > 0 else 0,
                    "correct": v["correct"],
                    "total": v["total"],
                    "correct_text": v["correct_text"],
                    "total_text": v["total_text"]
                } for k, v in stats["per_exam"].items()
            }
        })

    # Sort by Text-Only Accuracy (Primary Ranking)
    results.sort(key=lambda x: x["accuracy_text"], reverse=True)

    last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
    all_exams = set()
    for r in results:
        all_exams.update(r.get("per_exam", {}).keys())
    exam_order = sorted(all_exams)

    out = {
        "last_updated": last_updated,
        "primary_metric": "text_only",
        "exam_order": exam_order,
        "models": []
    }

    for i, r in enumerate(results, 1):
        per_exam_rows = []
        for exam in exam_order:
            d = r["per_exam"].get(exam, {})
            per_exam_rows.append({
                "exam_name": exam,
                "text_only": {
                    "accuracy": float(d.get("accuracy_text", 0.0)),
                    "correct": int(d.get("correct_text", 0)),
                    "total": int(d.get("total_text", 0))
                },
                "standard": {
                    "accuracy": float(d.get("accuracy_all", 0.0)),
                    "correct": int(d.get("correct", 0)),
                    "total": int(d.get("total", 0))
                }
            })

        out["models"].append({
            "rank": i,
            "model": r["model"],
            "vision_capable": bool(r["vision_capable"]),
            "overall": {
                "text_only": {
                    "accuracy": float(r["accuracy_text"]),
                    "correct": int(r["correct_text"]),
                    "total": int(r["total_text"])
                },
                "standard": {
                    "accuracy": float(r["accuracy_all"]),
                    "correct": int(r["correct_all"]),
                    "total": int(r["total_all"])
                }
            },
            "per_exam": per_exam_rows
        })

    out_path = os.path.join(os.getcwd(), "hf_space", "leaderboard_data.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Updated hf_space/leaderboard_data.json ({len(out['models'])} models).")

def run_benchmark():
    models = get_models()
    for m in models:
        print(f"Evaluating model: {m}")
        status = evaluate_model(m)
        if status == "rate_limited":
            print(f"Continuing with the next configured model after {m}.")
    
    generate_leaderboard()

if __name__ == "__main__":
    run_benchmark()
