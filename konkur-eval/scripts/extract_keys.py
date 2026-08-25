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

def extract_keys_from_pdf(pdf_path, output_path):
    print(f"Processing key PDF: {pdf_path}")
    
    # Open PDF and get first page
    doc = fitz.open(pdf_path)
    page = doc[0] 
    
    # Render page to image
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
        "3": 4,
        ...
    }
    
    Ensure you capture ALL questions visible in the table.
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
        
        # Save JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        print(f"Successfully saved answer keys to {output_path}")
        
    except Exception as e:
        print(f"Failed to parse response: {e}")
        print("Raw response:", content)

if __name__ == "__main__":
    key_pdf = "pdfs/keys/ensani_nobat1_key.pdf"
    output_file = "dataset/answer_keys.json"
    os.makedirs("dataset", exist_ok=True)
    
    extract_keys_from_pdf(key_pdf, output_file)
