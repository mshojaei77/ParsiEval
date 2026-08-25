import os
import json
import base64
import requests
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import io

# Load environment variables
load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_questions_from_page(pdf_path, page_num, output_dir):
    # Open PDF
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1] # 0-based index
    
    # Render page to image
    zoom = 2.0 # Higher resolution for better OCR
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Prepare API request
    base64_image = encode_image_to_base64(img)
    
    # Load answer keys
    answer_keys = {}
    keys_path = os.path.join(output_dir, "answer_keys.json")
    if os.path.exists(keys_path):
        with open(keys_path, "r", encoding="utf-8") as f:
            answer_keys = json.load(f)
    
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
    
    print(f"Sending request to {OPENAI_BASE_URL} with model {OPENAI_MODEL}...")
    response = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return
    
    result = response.json()
    try:
        content = result['choices'][0]['message']['content']
        # Clean up code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
            
        data = json.loads(content)
        
        # Process figures
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
        
        for q in data.get("questions", []):
            # Populate answer key
            q_id = str(q.get("id"))
            if q_id in answer_keys:
                q["answer_key"] = answer_keys[q_id]

            if q.get("figure"):
                # Crop figure
                bbox = q["figure"]
                # Un-normalize bbox
                # ymin, xmin, ymax, xmax (0-1000)
                ymin, xmin, ymax, xmax = bbox
                
                width, height = img.size
                left = xmin * width / 1000
                top = ymin * height / 1000
                right = xmax * width / 1000
                bottom = ymax * height / 1000
                
                # Add some padding
                padding = 5
                left = max(0, left - padding)
                top = max(0, top - padding)
                right = min(width, right + padding)
                bottom = min(height, bottom + padding)
                
                crop = img.crop((left, top, right, bottom))
                figure_filename = f"q{q['id']}_fig.png"
                figure_path = os.path.join(output_dir, "figures", figure_filename)
                crop.save(figure_path)
                
                q["figure"] = f"figures/{figure_filename}"
        
        # Save JSON
        output_json_path = os.path.join(output_dir, f"page_{page_num}.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        print(f"Successfully saved results to {output_json_path}")
        
    except Exception as e:
        print(f"Failed to parse response: {e}")
        print("Raw response:", content)

if __name__ == "__main__":
    pdf_file = "pdfs/questions/ensani_nobat1_questions.pdf"
    output_directory = "dataset"
    os.makedirs(output_directory, exist_ok=True)
    
    # Page 2
    extract_questions_from_page(pdf_file, 2, output_directory)
