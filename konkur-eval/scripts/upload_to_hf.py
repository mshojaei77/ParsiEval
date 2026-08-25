import os
import json
import glob
from dotenv import load_dotenv
from datasets import Dataset, Features, Value, Sequence, Image

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")

if not HF_TOKEN or not HF_USERNAME:
    print("Error: HF_TOKEN or HF_USERNAME not found in .env")
    # Try to read directly if load_dotenv fails (fallback)
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('HF_TOKEN='):
                    HF_TOKEN = line.strip().split('=')[1]
                elif line.startswith('HF_USERNAME='):
                    HF_USERNAME = line.strip().split('=')[1]
    except Exception as e:
        pass

if not HF_TOKEN or not HF_USERNAME:
    raise ValueError("Could not load HF_TOKEN or HF_USERNAME from .env")

DATASET_DIR = "dataset"
REPO_ID = f"{HF_USERNAME}/konkur1404"

def load_data():
    all_questions = []
    
    # Find all *_full.json files
    # Structure: dataset/{exam_name}/{exam_name}_full.json
    pattern = os.path.join(DATASET_DIR, "*", "*_full.json")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} exam files: {[os.path.basename(f) for f in files]}")
    
    for file_path in files:
        print(f"Processing {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        exam_name = data.get('metadata', {}).get('exam_name', os.path.basename(os.path.dirname(file_path)))
        exam_dir = os.path.dirname(file_path)
        
        for q in data.get('questions', []):
            # Handle figure path
            figure_path = None
            if q.get('figure'):
                # The JSON has "figures/qX.png", which is relative to the JSON file location
                # We need absolute path for datasets library to load it reliably
                abs_figure_path = os.path.abspath(os.path.join(exam_dir, q['figure']))
                if os.path.exists(abs_figure_path):
                    figure_path = abs_figure_path
                else:
                    print(f"Warning: Figure not found at {abs_figure_path} for question {q['id']}")
            
            # Create record
            record = {
                "id": f"{exam_name}_{q['id']}",
                "exam_name": exam_name,
                "question": q['question'],
                "choices": q['choices'],
                "answer_key": q['answer_key'],
                "figure": figure_path 
            }
            all_questions.append(record)
            
    return all_questions

def main():
    print("Loading data...")
    data = load_data()
    print(f"Total questions collected: {len(data)}")
    
    if not data:
        print("No data found to upload.")
        return

    # Define features
    features = Features({
        "id": Value("string"),
        "exam_name": Value("string"),
        "question": Value("string"),
        "choices": Sequence(Value("string")),
        "answer_key": Value("int32"), # 1-4
        "figure": Image()
    })
    
    print("Creating dataset...")
    # Create dataset
    ds = Dataset.from_list(data, features=features)
    
    print(f"Pushing to Hub: {REPO_ID}...")
    try:
        ds.push_to_hub(REPO_ID, token=HF_TOKEN)
        print(f"Successfully uploaded dataset to https://huggingface.co/datasets/{REPO_ID}")
    except Exception as e:
        print(f"Error pushing to Hub: {e}")

if __name__ == "__main__":
    main()
