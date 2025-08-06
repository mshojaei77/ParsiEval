import os
import time
import csv
import re
import json
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError

load_dotenv()

client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY"),
)

def get_model_response(model_name, message):
    chat_completion = client.chat.completions.create(
        messages=[message],
        model=model_name,
    )
    return chat_completion

def evaluate_model():
    model_name = "qwen-3-32b"
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
        message = {'role': 'user', 'content': f"{question}\n\nلطفاً فقط با حرف گزینه صحیح (A، B، C یا D) پاسخ دهید. /no_think"}
        
        response = None
        error_occurred = False
        latency = 0
        while response is None:
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(get_model_response, model_name, message)
                try:
                    response = future.result(timeout=60)
                except TimeoutError:
                    end_time = time.time()
                    latency = end_time - start_time
                    print(f"Q{i+1}: timed out after {latency:.2f} seconds. Retrying in 60 seconds...")
                    time.sleep(60)
                except Exception as e:
                    end_time = time.time()
                    latency = end_time - start_time
                    print(f"Q{i+1}: API Error: {e}. Marking as failed and skipping.")
                    error_occurred = True
                    break
        
        if error_occurred:
            total_latency += latency
            print(f"Q{i+1}: Model answered Error, Correct: {correct} - ✗ ({latency:.2f}s)")
            continue

        end_time = time.time()
        latency = end_time - start_time
        total_latency += latency
        
        model_response = response.choices[0].message.content
        
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
    except FileNotFoundError:
        results = []

    results.append(new_result)
    
    with open('evaluation_results.json', 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

evaluate_model()