import os
import json
import base64
import requests
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import io
import time
import glob

# Load environment variables
load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_keys_from_pdf(pdf_path):
    print(f"Extracting keys from: {pdf_path}")
    doc = fitz.open(pdf_path)
    # Assume keys are on the first page for now
    page = doc[0]
    
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    base64_image = encode_image_to_base64(img)
    
    prompt = """
    You are an expert data extractor.
    This image is an answer key table for an exam.
    The table lists Question Numbers and their Correct Options (1, 2, 3, or 4).
    
    Extract ALL answer keys from the table.
    Return a single flat JSON object mapping the question ID (string) to the correct option (integer).
    
    Example format:
    {
        "1": 3,
        "2": 1,
        ...
    }
    """
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://konkur-eval.local",
        "X-Title": "Konkur Eval"
    }
    
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Key extraction failed: {response.text}")
        
    result = response.json()
    content = result['choices'][0]['message']['content']
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
        
    return json.loads(content)

def process_page(page, page_num, answer_keys, output_dir, img_output_dir):
    print(f"Processing Page {page_num}...")
    
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    base64_image = encode_image_to_base64(img)
    
    prompt = """
    You are an expert at digitizing exam questions.
    Extract all questions from this image.
    The image contains Persian text and Math formulas.
    
    For each question, extract:
    1. id: The question number (integer).
    2. question: The question text in Markdown. Use LaTeX for math (wrapped in $...$). Preserve Persian text exactly.
    3. choices: A list of the 4 answer choices (strings).
    4. answer_key: Set to null (unknown).
    5. figure: If the question has an associated figure/diagram, provide the bounding box of the figure as [ymin, xmin, ymax, xmax] where coordinates are normalized 0-1000 (top-left is 0,0). If no figure, set to null.
    
    Output strictly valid JSON with the following schema:
    {
        "questions": [
            {
                "id": 1,
                "question": "...",
                "choices": ["...", "...", "...", "..."],
                "answer_key": null,
                "figure": [ymin, xmin, ymax, xmax] or null
            }
        ]
    }
    """
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://konkur-eval.local",
        "X-Title": "Konkur Eval"
    }
    
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "response_format": {"type": "json_object"}
    }
    
    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                break
            print(f"Attempt {attempt+1} failed: {response.status_code}")
            time.sleep(2)
        except Exception as e:
            print(f"Attempt {attempt+1} error: {e}")
            time.sleep(2)
    else:
        print(f"Failed to process page {page_num} after retries.")
        return []

    result = response.json()
    try:
        content = result['choices'][0]['message']['content']
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        data = json.loads(content)
        questions = data.get("questions", [])
        
        processed_questions = []
        for q in questions:
            # Populate answer key
            q_id = str(q.get("id"))
            if q_id in answer_keys:
                q["answer_key"] = answer_keys[q_id]
            
            # Process figures
            if q.get("figure"):
                try:
                    bbox = q["figure"]
                    ymin, xmin, ymax, xmax = bbox
                    width, height = img.size
                    
                    left = xmin * width / 1000
                    top = ymin * height / 1000
                    right = xmax * width / 1000
                    bottom = ymax * height / 1000
                    
                    # Increased Padding for better cropping
                    padding = 50 
                    left = max(0, left - padding)
                    top = max(0, top - padding)
                    right = min(width, right + padding)
                    bottom = min(height, bottom + padding)
                    
                    crop = img.crop((left, top, right, bottom))
                    figure_filename = f"q{q['id']}_fig.png"
                    figure_path = os.path.join(img_output_dir, figure_filename)
                    crop.save(figure_path)
                    
                    # Store relative path in JSON
                    q["figure"] = f"figures/{figure_filename}"
                except Exception as e:
                    print(f"Error extracting figure for Q{q.get('id')}: {e}")
                    q["figure"] = None
            
            processed_questions.append(q)
            
        return processed_questions

    except Exception as e:
        print(f"Error parsing JSON for page {page_num}: {e}")
        return []

def process_exam(exam_name, questions_pdf_path, keys_pdf_path):
    print(f"\n{'='*50}")
    print(f"Starting processing for exam: {exam_name}")
    print(f"{'='*50}")
    
    # Create exam-specific output directory
    output_dir = os.path.join("dataset", exam_name)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Extract Keys (or load if exists)
    keys_output_path = os.path.join(output_dir, "answer_keys.json")
    answer_keys = {}
    
    if os.path.exists(keys_output_path):
        print(f"Loading existing answer keys from {keys_output_path}")
        try:
            with open(keys_output_path, "r", encoding="utf-8") as f:
                answer_keys = json.load(f)
        except Exception as e:
            print(f"Error loading keys: {e}. Re-extracting.")
            
    if not answer_keys:
        try:
            if os.path.exists(keys_pdf_path):
                answer_keys = extract_keys_from_pdf(keys_pdf_path)
                with open(keys_output_path, "w", encoding="utf-8") as f:
                    json.dump(answer_keys, f, ensure_ascii=False, indent=4)
                print(f"Extracted and saved {len(answer_keys)} answer keys.")
            else:
                print(f"Warning: Key PDF not found at {keys_pdf_path}")
        except Exception as e:
            print(f"Critical error extracting keys: {e}")
            return

    # 2. Process Questions PDF
    if not os.path.exists(questions_pdf_path):
        print(f"Error: Questions PDF not found at {questions_pdf_path}")
        return

    doc = fitz.open(questions_pdf_path)
    all_questions = []
    
    print(f"Found {len(doc)} pages in questions PDF.")
    
    for i in range(len(doc)):
        page_num = i + 1
        page_output_path = os.path.join(output_dir, f"page_{page_num}.json")
        
        # Check if page already processed
        if os.path.exists(page_output_path):
            print(f"Page {page_num} already processed. Loading from file...")
            try:
                with open(page_output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    questions = data.get("questions", [])
                    all_questions.extend(questions)
            except Exception as e:
                print(f"Error loading existing page {page_num}: {e}. Reprocessing...")
                # Reprocess if load fails
                page = doc[i]
                questions = process_page(page, page_num, answer_keys, output_dir, figures_dir)
                if questions:
                    all_questions.extend(questions)
                    with open(page_output_path, "w", encoding="utf-8") as f:
                        json.dump({"questions": questions}, f, ensure_ascii=False, indent=4)
        else:
            # Process new page
            page = doc[i]
            questions = process_page(page, page_num, answer_keys, output_dir, figures_dir)
            if questions:
                all_questions.extend(questions)
                with open(page_output_path, "w", encoding="utf-8") as f:
                    json.dump({"questions": questions}, f, ensure_ascii=False, indent=4)
        
    # 3. Save Final Dataset
    final_dataset = {
        "metadata": {
            "exam_name": exam_name,
            "source": questions_pdf_path,
            "total_questions": len(all_questions)
        },
        "questions": all_questions
    }
    
    final_output_path = os.path.join(output_dir, f"{exam_name}_full.json")
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)
        
    print(f"Full processing complete for {exam_name}. Saved to {final_output_path}")

def main():
    pdfs_dir = "pdfs"
    questions_dir = os.path.join(pdfs_dir, "questions")
    keys_dir = os.path.join(pdfs_dir, "keys")
    
    # Find all question PDFs
    question_pdfs = glob.glob(os.path.join(questions_dir, "*_questions.pdf"))
    
    print(f"Found {len(question_pdfs)} exams to process.")
    
    for q_pdf in question_pdfs:
        # q_pdf is like "pdfs/questions/ensani_nobat1_questions.pdf"
        basename = os.path.basename(q_pdf)
        # exam_name is "ensani_nobat1"
        exam_name = basename.replace("_questions.pdf", "")
        
        # Construct key path
        key_filename = f"{exam_name}_key.pdf"
        key_pdf = os.path.join(keys_dir, key_filename)
        
        process_exam(exam_name, q_pdf, key_pdf)

if __name__ == "__main__":
    main()
