import os
import time
import csv
import re
import json
import platform
from openai import OpenAI, APITimeoutError, APIError, APIConnectionError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError

load_dotenv()


PROVIDER = "lmstudio"
MODEL_NAME = "gemma-3-270m-it"

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
    "max_tokens": 100,
    "top_p": 0.95,
    "dataset": "parsi-eval-1.csv",
    "output": "results\\evaluation_results.json"
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
    
    print(f"Running evaluation on {total} questions...")
    
    for i, (question, correct) in enumerate(zip(questions, correct_answers)):
        messages = [
            {'role': 'system', 'content': 'شما یک دستیار هوشمند هستید که به سوالات چند گزینه‌ای پاسخ می‌دهد. شما باید فقط و فقط با یک حرف انگلیسی (A یا B یا C یا D) پاسخ دهید. هیچ توضیح اضافی یا متن دیگری قابل قبول نیست. فقط یک حرف.'},
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
                    response = future.result(timeout=60) # Increased timeout for OpenAI
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
        
        if error_occurred:
            total_latency += latency
            print(f"Q{i+1}: Model answered Error, Correct: {correct} - ✗ ({latency:.2f}s)")
            continue

        end_time = time.time()
        latency = end_time - start_time
        total_latency += latency
        
        model_response = response.choices[0].message.content
        
        # Check if model_response is None to avoid TypeError
        if model_response is None:
            model_answer = "Unknown"
        else:
            match = re.search(r'(?:^|\s)([ABCD])(?:\s|$|\.|\,|\))', model_response)
            model_answer = match.group(1) if match else "Unknown"
        
        is_correct = model_answer == correct
        if is_correct:
            correct_count += 1
            
        print(f"Q{i+1}: Model answered {model_answer}, Correct: {correct} - {'✓' if is_correct else '✗'} ({latency:.2f}s)")
        
    accuracy = (correct_count / total) * 100 if total > 0 else 0
    avg_latency = total_latency / total if total > 0 else 0
    print(f"\nEvaluation complete. Accuracy: {accuracy:.2f}% ({correct_count}/{total})")
    print(f"Average latency: {avg_latency:.2f}s, Total latency: {total_latency:.2f}s")
    
    # Check if provider is using localhost
    is_localhost = "localhost" in CONFIG["base_url"]
    gpu_info = get_gpu_info() if is_localhost else None
    
    new_result = {
        "provider": PROVIDER,
        "model": model_name,
        "accuracy": f"{accuracy:.2f}%",
        "total_questions": total,
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
    except FileNotFoundError:
        results = []

    results.append(new_result)
    
    with open(CONFIG["output"], 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

evaluate_model()