
import os
import json
import glob
from collections import defaultdict
from datasets import load_dataset

def generate_json():
    results_root = os.path.join(os.getcwd(), "benchmark_results")
    output_file = os.path.join(os.getcwd(), "hf_space", "leaderboard_data.json")
    
    if not os.path.exists(results_root):
        print("No results found.")
        return

    # Load dataset for image lookup
    print("Loading dataset for image lookup...")
    try:
        ds = load_dataset("mshojaei77/konkur1404", split="train")
        image_lookup = {item['id']: bool(item.get('figure')) for item in ds}
    except Exception as e:
        print(f"Warning: Could not load dataset: {e}")
        image_lookup = {}

    model_results = defaultdict(dict)
    
    for model_dir in glob.glob(os.path.join(results_root, "*")):
        if not os.path.isdir(model_dir): continue
        
        json_files = glob.glob(os.path.join(model_dir, "*.json"))
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        r = json.loads(line)
                        m_name = r.get("model", os.path.basename(model_dir))
                        model_results[m_name][r["id"]] = r
                    except json.JSONDecodeError:
                        continue

    data = []
    
    for m_name, unique_records in model_results.items():
        stats = {
            "correct": 0, "total": 0, 
            "correct_text": 0, "total_text": 0,
            # Per exam stats could be added if we want detailed view
        }
        
        vision_capable = True
        
        for r in unique_records.values():
            has_image = r.get("has_image")
            if has_image is None:
                has_image = image_lookup.get(r["id"], False)
                
            skipped = r.get("skipped", False)
            if skipped and "support vision" in r.get("error", "").lower():
                vision_capable = False

            stats["total"] += 1
            stats["correct"] += r["correct"]
            
            if not has_image:
                stats["total_text"] += 1
                stats["correct_text"] += r["correct"]

        # Calculate Accuracies
        std_acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        text_acc = (stats["correct_text"] / stats["total_text"] * 100) if stats["total_text"] > 0 else 0
        
        data.append({
            "Model": m_name,
            "Vision Capable": "✅" if vision_capable else "❌",
            "Text-Only Score (Primary)": round(text_acc, 2),
            "Standard Score (All)": round(std_acc, 2),
            "Total Questions": stats["total"],
            "Text Questions": stats["total_text"]
        })

    # Sort by Text-Only Score (Primary)
    data.sort(key=lambda x: x["Text-Only Score (Primary)"], reverse=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {output_file}")

if __name__ == "__main__":
    generate_json()
