import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run linguistic fluency benchmarks for Persian (Farsi).")
    parser.add_argument("--langs", nargs="+", default=["fa"], choices=["fa"], help="Languages to run (default: fa)")
    parser.add_argument("--steps", nargs="+", default=["subject", "judge"], choices=["subject", "judge"], help="Steps to run (default: subject judge)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    # Map languages to their script directories and fixed output files
    lang_config = {
        "fa": {
            "dir": base_dir / "fa",
            "subject_script": "run_subject_fluency_fa.py",
            "judge_script": "run_judge_fluency_fa.py",
            "subject_output": "out/fa_subject_outputs.jsonl"
        }
    }

    for lang in args.langs:
        config = lang_config[lang]
        lang_dir = config["dir"]
        
        print(f"\n{'='*20} Processing Language: {lang.upper()} {'='*20}")

        # 1. Run Subject Model
        if "subject" in args.steps:
            print(f"--- Running Subject Model for {lang} ---")
            subject_script = lang_dir / config["subject_script"]
            cmd = [sys.executable, str(subject_script)]
            run_command(cmd)

        # 2. Run Judge Model
        if "judge" in args.steps:
            print(f"--- Running Judge Model for {lang} ---")
            judge_script = lang_dir / config["judge_script"]
            # The subject script outputs to {lang_dir}/out/{filename} by default
            # We construct the path to pass to the judge script
            input_file = lang_dir / config["subject_output"]
            
            if not input_file.exists():
                print(f"Warning: Input file for judge not found: {input_file}")
                print("Skipping judge step for this language.")
                continue

            cmd = [sys.executable, str(judge_script), "--in", str(input_file)]
            run_command(cmd)

    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    main()
