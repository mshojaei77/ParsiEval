import os
import time
import csv
import re
import json
from openai import OpenAI, APITimeoutError, APIError, APIConnectionError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError

load_dotenv()

client = OpenAI(
    api_key="lm-studio", #os.environ.get("OPENAI_API_KEY"),
    base_url="http://localhost:1234/v1",
    timeout=60,
)

def get_model_response(model_name, messages):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
    )
    return chat_completion

def evaluate_model():
    model_name = "gemma-2b-it"
    print(f"Evaluating model: {model_name}")

    questions = []
    correct_answers = []
    with open('cleaned_llm_persian_eval.csv', 'r', encoding='utf-8') as f:
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
        
        # Add a 10-second sleep after every 50 questions
        if (i + 1) % 50 == 0 and i + 1 < total:
            print(f"Completed {i + 1} questions. Taking a 10-second break...")
            time.sleep(10)
        
    accuracy = (correct_count / total) * 100 if total > 0 else 0
    avg_latency = total_latency / total if total > 0 else 0
    print(f"\nEvaluation complete. Accuracy: {accuracy:.2f}% ({correct_count}/{total})")
    print(f"Average latency: {avg_latency:.2f}s, Total latency: {total_latency:.2f}s")
    
    new_result = {
        "model": model_name,
        "accuracy": f"{accuracy:.2f}%",
        "total_questions": total,
        "correct_answers": correct_count,
        "avg_latency": f"{avg_latency:.2f}s",
        "total_latency": f"{total_latency:.2f}s"
    }
    
    try:
        with open('evaluation_results.json', 'r', encoding='utf-8') as json_file:
            results = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Creating new results file or handling corrupted file...")
        results = []

    results.append(new_result)
    
    with open('evaluation_results.json', 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

evaluate_model()