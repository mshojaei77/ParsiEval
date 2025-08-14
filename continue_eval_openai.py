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
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.avalai.ir/v1"
)

def get_model_response(model_name, messages):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
    )
    return chat_completion

def save_partial_results_and_continue():
    model_name = "gemini-2.5-pro"
    print(f"Continuing evaluation for model: {model_name}")

    # Load all questions
    questions = []
    correct_answers = []
    with open('cleaned_llm_persian_eval.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row['question'])
            correct_answers.append(row['answer'])
    
    # Manually input the results from Q1-Q201 based on terminal output
    # From the terminal, we can see Q1-Q201 results
    completed_results = [
        ('C', 'C', True, 9.43),   # Q1
        ('B', 'B', True, 7.86),   # Q2
        ('C', 'C', True, 14.10),  # Q3
        ('C', 'C', True, 10.73),  # Q4
        ('B', 'B', True, 10.77),  # Q5
        ('B', 'B', True, 12.04),  # Q6
        ('D', 'D', True, 4.37),   # Q7
        ('B', 'B', True, 12.65),  # Q8
        ('B', 'B', True, 13.42),  # Q9
        ('A', 'A', True, 35.83),  # Q10
        ('C', 'C', True, 13.49),  # Q11
        ('A', 'A', True, 9.19),   # Q12
        ('B', 'B', True, 13.55),  # Q13
        ('D', 'D', True, 14.39),  # Q14
        ('D', 'B', False, 16.74), # Q15
        ('D', 'D', True, 8.75),   # Q16
        ('C', 'C', True, 15.39),  # Q17
        ('A', 'D', False, 20.76), # Q18
        ('B', 'B', True, 17.68),  # Q19
        ('A', 'C', False, 13.00), # Q20
        ('B', 'B', True, 7.37),   # Q21
        ('B', 'B', True, 22.57),  # Q22
        ('D', 'D', True, 17.56),  # Q23
        ('D', 'C', False, 16.04), # Q24
        ('C', 'D', False, 11.10), # Q25
        ('D', 'A', False, 9.23),  # Q26
        ('A', 'A', True, 9.61),   # Q27
        ('A', 'A', True, 10.80),  # Q28
        ('D', 'D', True, 11.54),  # Q29
        ('C', 'C', True, 17.38),  # Q30
        ('D', 'D', True, 54.60),  # Q31
        ('D', 'D', True, 11.18),  # Q32
        ('B', 'B', True, 6.79),   # Q33
        ('B', 'B', True, 7.41),   # Q34
        ('B', 'B', True, 6.90),   # Q35
        ('B', 'B', True, 6.15),   # Q36
        ('A', 'A', True, 6.08),   # Q37
        ('D', 'D', True, 6.38),   # Q38
        ('A', 'A', True, 5.02),   # Q39
        ('A', 'A', True, 7.52),   # Q40
        ('A', 'A', True, 8.88),   # Q41
        ('A', 'A', True, 22.26),  # Q42
        ('B', 'B', True, 5.49),   # Q43
        ('B', 'B', True, 6.37),   # Q44
        ('B', 'B', True, 6.11),   # Q45
        ('C', 'C', True, 4.85),   # Q46
        ('B', 'B', True, 6.73),   # Q47
        ('A', 'D', False, 6.04),  # Q48
        ('B', 'B', True, 7.09),   # Q49
        ('D', 'D', True, 6.05),   # Q50
        ('D', 'D', True, 4.32),   # Q51
        ('B', 'B', True, 5.65),   # Q52
        ('A', 'A', True, 5.08),   # Q53
        ('C', 'C', True, 6.39),   # Q54
        ('A', 'A', True, 8.80),   # Q55
        ('D', 'B', False, 14.56), # Q56
        ('D', 'D', True, 24.71),  # Q57
        ('C', 'C', True, 7.92),   # Q58
        ('B', 'D', False, 8.97),  # Q59
        ('A', 'A', True, 6.77),   # Q60
        ('C', 'C', True, 6.37),   # Q61
        ('D', 'D', True, 5.37),   # Q62
        ('D', 'D', True, 6.39),   # Q63
        ('A', 'A', True, 6.43),   # Q64
        ('A', 'A', True, 9.33),   # Q65
        ('D', 'A', False, 9.75),  # Q66
        ('D', 'C', False, 23.08), # Q67
        ('A', 'A', True, 7.98),   # Q68
        ('A', 'A', True, 8.70),   # Q69
        ('C', 'C', True, 7.99),   # Q70
        ('D', 'D', True, 6.86),   # Q71
        ('B', 'B', True, 6.42),   # Q72
        ('C', 'C', True, 46.22),  # Q73
        ('D', 'D', True, 6.54),   # Q74
        ('B', 'B', True, 6.37),   # Q75
        ('C', 'C', True, 6.16),   # Q76
        ('A', 'A', True, 4.66),   # Q77
        ('B', 'B', True, 5.27),   # Q78
        ('D', 'D', True, 5.91),   # Q79
        ('C', 'C', True, 5.33),   # Q80
        ('B', 'A', False, 7.45),  # Q81
        ('D', 'B', False, 17.90), # Q82
        ('B', 'B', True, 7.44),   # Q83
        ('C', 'A', False, 6.26),  # Q84
        ('C', 'C', True, 7.02),   # Q85
        ('B', 'B', True, 7.51),   # Q86
        ('B', 'B', True, 5.92),   # Q87
        ('A', 'A', True, 7.28),   # Q88
        ('A', 'A', True, 8.30),   # Q89
        ('C', 'C', True, 6.49),   # Q90
        ('D', 'D', True, 6.01),   # Q91
        ('B', 'B', True, 9.89),   # Q92
        ('C', 'C', True, 6.89),   # Q93
        ('A', 'A', True, 11.20),  # Q94
        ('D', 'D', True, 4.93),   # Q95
        ('C', 'C', True, 7.74),   # Q96
        ('A', 'A', True, 9.24),   # Q97
        ('B', 'B', True, 6.78),   # Q98
        ('D', 'C', False, 14.98), # Q99
        ('A', 'A', True, 4.77),   # Q100
        ('A', 'A', True, 7.41),   # Q101
        ('B', 'C', False, 8.35),  # Q102
        ('A', 'A', True, 5.72),   # Q103
        ('C', 'C', True, 4.89),   # Q104
        ('D', 'D', True, 12.08),  # Q105
        ('B', 'B', True, 19.01),  # Q106
        ('C', 'C', True, 9.34),   # Q107
        ('D', 'A', False, 15.05), # Q108
        ('C', 'C', True, 8.42),   # Q109
        ('D', 'B', False, 7.80),  # Q110
        ('D', 'C', False, 10.37), # Q111
        ('A', 'D', False, 7.12),  # Q112
        ('B', 'B', True, 4.80),   # Q113
        ('C', 'C', True, 4.75),   # Q114
        ('C', 'C', True, 6.58),   # Q115
        ('B', 'B', True, 7.44),   # Q116
        ('C', 'C', True, 6.54),   # Q117
        ('A', 'A', True, 11.42),  # Q118
        ('C', 'C', True, 6.39),   # Q119
        ('D', 'B', False, 13.53), # Q120
        ('C', 'C', True, 12.01),  # Q121
        ('B', 'B', True, 7.32),   # Q122
        ('B', 'B', True, 7.43),   # Q123
        ('D', 'B', False, 15.96), # Q124
        ('C', 'C', True, 6.28),   # Q125
        ('D', 'D', True, 7.24),   # Q126
        ('D', 'B', False, 3.99),  # Q127
        ('C', 'C', True, 8.02),   # Q128
        ('A', 'A', True, 24.72),  # Q129
        ('C', 'C', True, 8.37),   # Q130
        ('A', 'A', True, 6.72),   # Q131
        ('B', 'B', True, 7.65),   # Q132
        ('B', 'B', True, 6.54),   # Q133
        ('C', 'C', True, 6.06),   # Q134
        ('D', 'B', False, 12.37), # Q135
        ('A', 'A', True, 7.60),   # Q136
        ('D', 'D', True, 14.93),  # Q137
        ('D', 'D', True, 7.45),   # Q138
        ('D', 'D', True, 7.45),   # Q139
        ('D', 'D', True, 16.47),  # Q140
        ('A', 'D', False, 6.88),  # Q141
        ('B', 'B', True, 8.69),   # Q142
        ('C', 'C', True, 14.45),  # Q143
        ('D', 'D', True, 7.19),   # Q144
        ('C', 'C', True, 6.38),   # Q145
        ('D', 'D', True, 5.75),   # Q146
        ('C', 'C', True, 6.23),   # Q147
        ('A', 'A', True, 5.86),   # Q148
        ('D', 'D', True, 5.72),   # Q149
        ('D', 'D', True, 7.55),   # Q150
        ('A', 'A', True, 5.29),   # Q151
        ('C', 'C', True, 6.58),   # Q152
        ('B', 'C', False, 17.00), # Q153
        ('A', 'A', True, 5.84),   # Q154
        ('C', 'C', True, 8.48),   # Q155
        ('B', 'C', False, 15.57), # Q156
        ('A', 'D', False, 13.05), # Q157
        ('C', 'C', True, 4.35),   # Q158
        ('D', 'D', True, 6.24),   # Q159
        ('B', 'B', True, 5.82),   # Q160
        ('C', 'C', True, 6.55),   # Q161
        ('A', 'A', True, 6.91),   # Q162
        ('D', 'D', True, 20.12),  # Q163
        ('D', 'C', False, 11.45), # Q164
        ('B', 'B', True, 5.93),   # Q165
        ('B', 'B', True, 7.56),   # Q166
        ('D', 'C', False, 14.21), # Q167
        ('D', 'A', False, 11.27), # Q168
        ('D', 'D', True, 6.02),   # Q169
        ('D', 'B', False, 11.31), # Q170
        ('C', 'C', True, 10.12),  # Q171
        ('B', 'B', True, 6.13),   # Q172
        ('D', 'C', False, 7.11),  # Q173
        ('C', 'C', True, 4.65),   # Q174
        ('D', 'D', True, 6.57),   # Q175
        ('A', 'A', True, 8.54),   # Q176
        ('D', 'D', True, 7.57),   # Q177
        ('C', 'C', True, 7.92),   # Q178
        ('D', 'D', True, 10.85),  # Q179
        ('A', 'B', False, 23.06), # Q180
        ('C', 'C', True, 9.48),   # Q181
        ('D', 'D', True, 32.11),  # Q182 (after retry)
        ('B', 'B', True, 7.54),   # Q183
        ('D', 'D', True, 6.04),   # Q184
        ('A', 'A', True, 5.59),   # Q185
        ('C', 'C', True, 5.86),   # Q186
        ('A', 'A', True, 4.96),   # Q187
        ('C', 'C', True, 8.84),   # Q188
        ('D', 'D', True, 5.43),   # Q189
        ('A', 'A', True, 6.16),   # Q190
        ('D', 'B', False, 18.70), # Q191
        ('D', 'D', True, 10.68),  # Q192
        ('B', 'B', True, 7.55),   # Q193
        ('C', 'C', True, 5.66),   # Q194
        ('B', 'B', True, 6.72),   # Q195
        ('C', 'C', True, 5.91),   # Q196
        ('A', 'A', True, 5.82),   # Q197
        ('D', 'D', True, 5.37),   # Q198
        ('C', 'C', True, 7.79),   # Q199
        ('A', 'A', True, 4.90),   # Q200
        ('A', 'A', True, 5.72),   # Q201
    ]
    
    # Calculate stats for completed questions
    correct_count = sum(1 for _, _, is_correct, _ in completed_results if is_correct)
    total_latency = sum(latency for _, _, _, latency in completed_results)
    completed_questions = len(completed_results)
    
    print(f"Loaded {completed_questions} completed results")
    print(f"Current accuracy: {(correct_count/completed_questions)*100:.2f}% ({correct_count}/{completed_questions})")
    print(f"Current total latency: {total_latency:.2f}s")
    
    # Continue from question 202
    total = len(questions)
    start_index = completed_questions  # Start from Q202 (index 201)
    
    print(f"Continuing evaluation from Q{start_index + 1} to Q{total}...")
    
    for i in range(start_index, total):
        question = questions[i]
        correct = correct_answers[i]
        
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
                    response = future.result(timeout=60)
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
        
    # Calculate final results
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

if __name__ == "__main__":
    save_partial_results_and_continue()