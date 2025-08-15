import os
import time
import csv
import re
import json
import platform
import subprocess
from typing import Optional
from openai import OpenAI, APITimeoutError, APIError, APIConnectionError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError

load_dotenv()


PROVIDER = "lmstudio"
MODEL_NAME = "qwen/qwen3-4b-thinking-2507"
MODEL_SIZE = "4b"
LICENSE="apache-2.0"

# base urls
BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "avalai": "https://api.avalai.ir/v1",
    "groq": "https://api.groq.com/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
# api keys
API_KEYS = {
    "openai": os.environ.get("OPENAI_API_KEY"),
    "avalai": os.environ.get("AVALAI_API_KEY"),
    "groq": os.environ.get("GROQ_API_KEY"),
    "cerebras": os.environ.get("CEREBRAS_API_KEY"),
    "ollama": os.environ.get("OLLAMA_API_KEY"),
    "lmstudio": os.environ.get("LMSTUDIO_API_KEY"),
    "openrouter": os.environ.get("OPENROUTER_API_KEY"),
}

CONFIG = {
    "base_url": BASE_URLS[PROVIDER],
    "api_key": API_KEYS[PROVIDER],
    "model": MODEL_NAME,
    "temperature": 0.7,
    "max_tokens": 4000,
    "top_p": 0.95,
    "cool_down_time": [1,10],
    "skip_unknown": True,
    "dataset": "parsi-eval-1.csv",
    "output": "results/parsi-eval-1.json"
}



client = OpenAI(
    api_key=CONFIG["api_key"],
    base_url=CONFIG["base_url"]
)

def get_model_response(model_name, messages):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=CONFIG["temperature"],
        max_tokens=CONFIG["max_tokens"],
        top_p=CONFIG["top_p"]
    )
    return chat_completion

def get_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "No GPU"
    except (ImportError, Exception):
        return "Unknown"

def extract_choice_letter(text: Optional[str]) -> str:
    """Extract the first standalone choice letter A-D.

    Accepts matches even when surrounded by non-letter characters such as
    asterisks, parentheses, punctuation, or whitespace (e.g., "**C)**", "(A)", "B.").
    Ensures the extracted letter is not part of a longer ASCII word (e.g., avoids the 'A' in "RSA").

    Args:
        text: The model response text to parse.

    Returns:
        A single uppercase letter among {A, B, C, D} if found; otherwise "Unknown".
    """
    if text is None:
        return "Unknown"

    match = re.search(r"(?<![A-Za-z])[ABCD](?![A-Za-z])", text, flags=re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return "Unknown"

def evaluate_model():
    model_name = CONFIG["model"]
    provider = PROVIDER
    print(f"Evaluating model: {model_name} using {provider} provider")

    questions = []
    correct_answers = []
    with open(CONFIG["dataset"], 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row['question'])
            correct_answers.append(row['answer'])
    
    correct_count = 0
    total = len(questions)
    total_latency = 0
    skipped_count = 0
    evaluated_count = 0
    attempted_count = 0
    stop_requested = False
    
    print(f"Running evaluation on {total} questions...")
    
    for i, (question, correct) in enumerate(zip(questions, correct_answers)):
        messages = [
            {'role': 'system', 'content': 'شما یک دانشجو هستید که به سوالات چند گزینه‌ای پاسخ می‌دهد. شما باید فقط و فقط با یک حرف انگلیسی (A یا B یا C یا D) پاسخ دهید. هیچ توضیح اضافی یا متن دیگری قابل قبول نیست. فقط یک حرف.'},
            {'role': 'user', 'content': question}
        ]
        response = None
        error_occurred = False
        latency = 0
        retry_count = 0
        max_retries = 30
        base_delay = 2
        
        while response is None and retry_count < max_retries:
            start_time = time.time()
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(get_model_response, model_name, messages)
                    response = future.result(timeout=90) # Increased timeout for OpenAI
            except TimeoutError:
                end_time = time.time()
                latency = end_time - start_time
                retry_count += 1
                delay = base_delay * (2 ** (retry_count - 1))
                print(f"Q{i+1}: timed out after {latency:.2f} seconds. Retry {retry_count}/{max_retries} in {delay} seconds...")
                if retry_count < max_retries:
                    time.sleep(delay)
            except APIConnectionError as e:
                end_time = time.time()
                latency = end_time - start_time
                retry_count += 1
                delay = base_delay * (2 ** (retry_count - 1))
                print(f"Q{i+1}: API Connection Error. Retry {retry_count}/{max_retries} in {delay} seconds...")
                if retry_count < max_retries:
                    time.sleep(delay)
            except (APITimeoutError, APIError) as e:
                end_time = time.time()
                latency = end_time - start_time
                print(f"Q{i+1}: API Error ({type(e).__name__}). Marking as failed and skipping.")
                error_occurred = True
                break
        
        # If we exhausted all retries without success
        if response is None and not error_occurred:
            end_time = time.time()
            latency = end_time - start_time
            print(f"Q{i+1}: Failed after {max_retries} retries. Marking as failed and skipping.")
            error_occurred = True
        
        # Count this question as attempted regardless of outcome from the request cycle
        attempted_count += 1

        if error_occurred:
            total_latency += latency
            print(f"Q{i+1}: Model answered Error, Correct: {correct} - ✗ ({latency:.2f}s)")
            continue

        end_time = time.time()
        latency = end_time - start_time
        total_latency += latency
        
        model_response = response.choices[0].message.content
        
        # Extract choice letter robustly, allowing non-letter characters around it
        model_answer = extract_choice_letter(model_response)
        
        # Handle unparseable answers interactively
        if model_answer == "Unknown":
            print(
                f"Q{i+1}: Could not parse a single-letter answer from model response.\n"
                f"Raw response:\n{model_response}\n"
            )
            if CONFIG.get("skip_unknown", False):
                skipped_count += 1
                print(f"Q{i+1}: Skipped automatically due to CONFIG.skip_unknown=True. Correct: {correct} ({latency:.2f}s)")
                # Do not count towards evaluated metrics; proceed to next question
                continue
            else:
                while True:
                    user_choice = input(
                        "Skip this question and continue? (y to skip, n to stop eval): "
                    ).strip().lower()
                    if user_choice in ("y", "n"):
                        break
                    print("Invalid input. Please enter 'y' or 'n'.")

                if user_choice == "n":
                    stop_requested = True
                    print("Stopping evaluation at user's request.")
                    break
                else:
                    skipped_count += 1
                    print(f"Q{i+1}: Skipped by user. Correct: {correct} ({latency:.2f}s)")
                    # Do not count towards evaluated metrics; proceed to next question
                    continue

        is_correct = model_answer == correct
        if is_correct:
            correct_count += 1
        evaluated_count += 1
            
        print(f"Q{i+1}: Model answered {model_answer}, Correct: {correct} - {'✓' if is_correct else '✗'} ({latency:.2f}s)")
        
        # Add cool down time for local providers (ollama or lmstudio)
        if PROVIDER in ['ollama', 'lmstudio'] and (i + 1) % CONFIG["cool_down_time"][0] == 0 and i < total - 1:
            cool_down_seconds = CONFIG["cool_down_time"][1]
            print(f"Cooling down for {cool_down_seconds} seconds after {CONFIG['cool_down_time'][0]} questions...")
            time.sleep(cool_down_seconds)
        
    # Final metrics based on evaluated and attempted questions
    accuracy = (correct_count / evaluated_count) * 100 if evaluated_count > 0 else 0
    avg_latency = total_latency / attempted_count if attempted_count > 0 else 0
    print(f"\nEvaluation complete. Accuracy: {accuracy:.2f}% ({correct_count}/{evaluated_count})")
    print(f"Average latency: {avg_latency:.2f}s, Total latency: {total_latency:.2f}s")
    print(
        f"Attempted: {attempted_count}, Evaluated (non-skipped): {evaluated_count}, "
        f"Skipped: {skipped_count}, Stopped early: {'Yes' if stop_requested else 'No'}"
    )
    
    # Check if provider is using localhost
    is_localhost = "localhost" in CONFIG["base_url"]
    gpu_info = get_gpu_info() if is_localhost else None
    
    new_result = {
        "model": model_name,
        "model_size": MODEL_SIZE,
        "license": LICENSE,
        "provider": PROVIDER,
        "accuracy": f"{accuracy:.2f}%",
        "total_questions": evaluated_count,
        "attempted_questions": attempted_count,
        "skipped_questions": skipped_count,
        "stopped_early": stop_requested,
        "correct_answers": correct_count,
        "avg_latency": f"{avg_latency:.2f}s",
        "total_latency": f"{total_latency:.2f}s",
        "system": {
            "os": platform.system(),
            "gpu": gpu_info if is_localhost else None
        }
    }
    
    try:
        with open(CONFIG["output"], 'r', encoding='utf-8') as json_file:
            results = json.load(json_file)
            
            # Check if model already exists in results
            model_exists = False
            for i, result in enumerate(results):
                if result.get("model") == model_name:
                    # Replace existing model data with new data
                    results[i] = new_result
                    model_exists = True
                    print(f"Updated existing data for model: {model_name}")
                    break
                    
            # If model doesn't exist, append new result
            if not model_exists:
                results.append(new_result)
    except FileNotFoundError:
        results = []
        results.append(new_result)
    
    with open(CONFIG["output"], 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    
    # Run visualization script to update plots and README
    print("\nUpdating visualizations and README...")
    try:
        subprocess.run(["python", "create_visuals.py"], check=True)
        print("Visualizations and README updated successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error updating visualizations: {e}")

if __name__ == "__main__":
    evaluate_model()