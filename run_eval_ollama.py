import time
import csv
import re
import json
from ollama import chat
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import keyboard  # Add this import at the top

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_model_response(model_name, message):
    return chat(model=model_name, messages=[message])  # Simple direct call

def evaluate_model():
    # Configure GPU settings
    model_name = 'smollm:135m'  
    gpu_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    print(f"Using GPU device: {gpu_device}")
    print(f"Evaluating model: {model_name}")

    # Load questions and answers from CSV
    questions = []
    correct_answers = []
    with open('cleaned_llm_persian_eval.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row['question'])
            correct_answers.append(row['answer'])
    
    correct_count = 0
    total = len(questions)
    total_latency = 0  # Track total time
    
    print(f"Running evaluation on {total} questions...")
    
    for i, (question, correct) in enumerate(zip(questions, correct_answers)):
        # Check if Ctrl+Q was pressed
        if keyboard.is_pressed('ctrl+alt+c'):
            print(f"Q{i+1}: Skipped by user")
            continue
            
        # Format as a multiple-choice question in Persian
        message = {'role': 'user', 'content': f"{question}\n\nلطفاً فقط با حرف گزینه صحیح (A، B، C یا D) پاسخ دهید."}
        
        start_time = time.time()
        # Use a ThreadPoolExecutor to limit the response time for each question
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_model_response, model_name, message)
            try:
                # Wait for at most 10 seconds for an answer
                response = future.result(timeout=10)
            except TimeoutError:
                end_time = time.time()
                latency = end_time - start_time
                print(f"Q{i+1}: timed out after {latency:.2f} seconds, skipping question")
                continue  # move to the next question
                
        end_time = time.time()
        latency = end_time - start_time
        total_latency += latency
        
        model_response = response['message']['content']
        
        # Extract the letter answer (A, B, C, or D)
        match = re.search(r'(?:^|\s)([ABCD])(?:\s|$|\.|\,|\))', model_response)
        model_answer = match.group(1) if match else "Unknown"
        
        is_correct = model_answer == correct
        if is_correct:
            correct_count += 1
            
        print(f"Q{i+1}: Model answered {model_answer}, Correct: {correct} - {'✓' if is_correct else '✗'} ({latency:.2f}s)")
        
    accuracy = (correct_count / total) * 100
    avg_latency = total_latency / total
    print(f"\nEvaluation complete. Accuracy: {accuracy:.2f}% ({correct_count}/{total})")
    print(f"Average latency: {avg_latency:.2f}s, Total latency: {total_latency:.2f}s")
    
    # Prepare the new result
    new_result = {
        "model": model_name,
        "accuracy": f"{accuracy:.2f}%",
        "gpu_device": gpu_device,
        "total_questions": total,
        "correct_answers": correct_count,
        "avg_latency": f"{avg_latency:.2f}s",
        "total_latency": f"{total_latency:.2f}s"
    }
    
    # Load existing results from the JSON file, if it exists
    try:
        with open('evaluation_results.json', 'r', encoding='utf-8') as json_file:
            results = json.load(json_file)
    except FileNotFoundError:
        results = []

    # Append the new result
    results.append(new_result)
    
    # Save the updated results back to the JSON file
    with open('evaluation_results.json', 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

evaluate_model()