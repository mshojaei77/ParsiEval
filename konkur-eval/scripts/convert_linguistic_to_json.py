import os
import json
import re

def parse_markdown_table(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract Date and Judge
    date_match = re.search(r'- \*\*Date\*\*: (.*)', content)
    judge_match = re.search(r'- \*\*Judge\*\*: `(.*)`', content)
    
    last_updated = date_match.group(1) if date_match else "Unknown"
    judge = judge_match.group(1) if judge_match else "Unknown"

    # Find the table
    # Looking for the header row
    header_regex = r'\| Model \| Overall \| Grammar \| Idiomatic \| Conciseness \| Politeness \| Naturalness \| Instruction Following \| Context Retention \| Safety \|'
    match = re.search(header_regex, content)
    
    if not match:
        print("Table header not found")
        return None

    # Process lines after header
    lines = content[match.end():].strip().split('\n')
    
    models = []
    # Skip separator line like | :--- | :---: | ...
    start_idx = 0
    if lines[0].strip().startswith('| :---'):
        start_idx = 1
        
    rank = 1
    for line in lines[start_idx:]:
        line = line.strip()
        if not line.startswith('|'):
            break
            
        # Parse row
        # Example: | `qwen/qwen3-30b-a3b-instruct-2507` | **42.1** | 49.8 | ...
        parts = [p.strip() for p in line.split('|')[1:-1]]
        
        if len(parts) < 10:
            continue
            
        model_name = parts[0].replace('`', '')
        
        # Helper to clean score (remove bold, etc)
        def clean_score(s):
            s = s.replace('*', '')
            try:
                return float(s)
            except ValueError:
                return 0.0

        scores = {
            "Overall": clean_score(parts[1]),
            "Grammar": clean_score(parts[2]),
            "Idiomatic": clean_score(parts[3]),
            "Conciseness": clean_score(parts[4]),
            "Politeness": clean_score(parts[5]),
            "Naturalness": clean_score(parts[6]),
            "Instruction Following": clean_score(parts[7]),
            "Context Retention": clean_score(parts[8]),
            "Safety": clean_score(parts[9]),
        }
        
        models.append({
            "rank": rank,
            "model": model_name,
            "scores": scores
        })
        rank += 1
        
    return {
        "last_updated": last_updated,
        "judge": judge,
        "models": models
    }

def main():
    source_file = os.path.join(os.getcwd(), "bench.linguistic.md")
    output_file = os.path.join(os.getcwd(), "hf_space_linguistic", "leaderboard_data.json")
    
    data = parse_markdown_table(source_file)
    
    if data:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully generated {output_file}")
    else:
        print("Failed to parse data")

if __name__ == "__main__":
    main()
