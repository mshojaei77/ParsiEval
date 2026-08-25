---
license: mit
task_categories:
- question-answering
- multiple-choice
language:
- fa
- en
tags:
- konkur
- entrance-exam
- education
size_categories:
- 1K<n<10K
pretty_name: Konkur1404 (Persian MCQ)
dataset_name: konkur1404
multimodal: true
llm_eval_ready: true
---

# Dataset Card for Konkur1404

## Dataset Description

This dataset contains questions from the Konkur (Iranian University Entrance Exam) for the year 1404. It is designed for evaluating models on Persian multiple-choice questions across various subjects.

### Dataset Summary

- **Total Examples**: 2137
- **Splits**: train
- **Languages**: Persian (fa)

## Dataset Structure

### Data Instances

An example from the dataset looks like this:

```json
{
  "id": "ensani_nobat1_1",
  "exam_name": "ensani_nobat1",
  "question": "اگر شعاع دایره شکل زیر برابر $x = \\frac{1}{\\sqrt{2\\pi}}$ و مجموع مساحتهای دو شکل برابر ۱۶ باشد، محیط دایره کدام است؟",
  "choices": [
    "$\\sqrt{\\pi}$",
    "$2\\sqrt{\\pi}$",
    "$3\\sqrt{\\pi}$",
    "$4\\sqrt{\\pi}$"
  ],
  "answer_key": 4,
  "figure": "<Image: PNG, (437, 231)>"
}
```

### Data Fields

The dataset contains the following fields:

- **id** (string): Description of id.
- **exam_name** (string): Description of exam_name.
- **question** (string): Description of question.
- **choices** (List(Value('string'))): Description of choices.
- **answer_key** (int32): Description of answer_key.
- **figure** (PIL.Image.Image): Description of figure.

## Dataset Statistics

### Split: train
- Count: 2137
- **exam_name Distribution**:
  - zaban_nobat1: 400
  - zaban_nobat2: 350
  - ensani_nobat1: 280
  - tajrobi_nobat1: 225
  - ensani_nobat2: 221
  - tajrobi_nobat2: 185
  - riazi_nobat1: 145
  - honar_nobat1: 126
  - riazi_nobat2: 105
  - honar_nobat2: 100
- **answer_key Distribution**:
  - 1.0: 551
  - 2.0: 539
  - 3.0: 538
  - 4.0: 508

## Evaluation with OpenAI-Compatible API

- Deterministic settings (temperature=0) are recommended.
- Normalize Persian digits and English number words.
- Report both overall accuracy and per-exam accuracy.
- Use multimodal input for questions with figures if your model supports images.

### Evaluation Script

```python
import os
import io
import base64
import re
import csv
import time
from collections import defaultdict

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "your-api-key")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-5.2")
USE_IMAGES = True
EXAMS = ["ensani_nobat1", "ensani_nobat2"]

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def format_prompt(example):
    prompt = f"Question: {example['question']}\n\n"
    for i, choice in enumerate(example['choices']):
        prompt += f"{i+1}. {choice}\n"
    prompt += "\nAnswer with the number of the correct choice (1, 2, 3, or 4) only."
    return prompt

def extract_answer(response_text):
    text = response_text.strip()
    for k, v in {"۱": "1", "۲": "2", "۳": "3", "۴": "4"}.items():
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

def chat_with_retries(messages, max_retries=3):
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,
                max_tokens=10
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(8.0, delay * 2)

def evaluate():
    ds = load_dataset("mshojaei77/konkur1404", split="train")
    if EXAMS:
        ds = ds.filter(lambda x: x.get("exam_name") in EXAMS)

    totals = defaultdict(int)
    corrects = defaultdict(int)
    rows = []

    for example in tqdm(ds):

        prompt = format_prompt(example)
        messages = [{"role": "system", "content": "Answer only with 1, 2, 3, or 4."}]

        img_b64 = None
        if USE_IMAGES:
            img_b64 = figure_to_base64(example.get("figure"))

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
        try:
            resp = chat_with_retries(messages)
            prediction_text = resp.choices[0].message.content.strip()
            pred = extract_answer(prediction_text)
        except Exception as e:
            error_msg = str(e)

        gt = int(example["answer_key"])
        exam = example.get("exam_name", "unknown")
        totals[exam] += 1
        ok = int(pred == gt)
        corrects[exam] += ok
        rows.append({"id": example.get("id"), "exam_name": exam, "predicted": pred, "ground_truth": gt, "correct": ok, "error": error_msg})
        if error_msg:
            print(f"Error on id={example.get('id')} exam={exam}: {error_msg}")

    total = sum(totals.values())
    correct = sum(corrects.values())
    if total:
        print(f"Accuracy: {100*correct/total:.2f}% ({correct}/{total})")
        for exam, t in totals.items():
            if t:
                print(f"- {exam}: {100*corrects[exam]/t:.2f}% ({corrects[exam]}/{t})")
    else:
        print("No examples evaluated.")

    if rows:
        with open("konkur1404_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id","exam_name","predicted","ground_truth","correct","error"])
            w.writeheader()
            w.writerows(rows)
        print("Saved konkur1404_results.csv")

if __name__ == "__main__":
    evaluate()
```
### Data Notes

- Choices are always 4 options; answer_key is 1–4 (1-based).
- Figures are PNGs referenced by relative paths; when loaded via HF Datasets, figure may be an image object.
- Text may include LaTeX-style math and Persian digits; normalize for robust parsing.

### Ethics and Usage

- For evaluation and research use; respect exam policies and local regulations.
- Random baseline is 25% accuracy; report per-exam breakdown for interpretability.
