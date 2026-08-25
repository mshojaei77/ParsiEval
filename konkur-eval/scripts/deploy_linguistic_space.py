import os
import time
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file. Please add it.")

def deploy():
    api = HfApi(token=HF_TOKEN)
    repo_id = "mshojaei77/Persian-linguistic-llm-leaderboard"
    space_folder = "hf_space_linguistic"

    print(f"Creating/Checking Space: {repo_id}")
    
    # Retry logic for creation
    for attempt in range(5):
        try:
            create_repo(
                repo_id=repo_id,
                token=HF_TOKEN,
                repo_type="space",
                space_sdk="gradio",
                exist_ok=True
            )
            print("Space created or already exists.")
            break
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"Create attempt {attempt+1}/5 failed: {e}\nRetrying in {wait_time}s...")
            time.sleep(wait_time)
    else:
        print("Failed to create repo after 5 attempts.")
        return

    print("Uploading files...")
    # Retry logic for upload
    for attempt in range(5):
        try:
            api.upload_folder(
                folder_path=space_folder,
                repo_id=repo_id,
                repo_type="space",
                ignore_patterns=["__pycache__", "*.pyc"]
            )
            print(f"Successfully uploaded files to https://huggingface.co/spaces/{repo_id}")
            break
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"Upload attempt {attempt+1}/5 failed: {e}\nRetrying in {wait_time}s...")
            time.sleep(wait_time)
    else:
        print("Failed to upload files after 5 attempts.")

if __name__ == "__main__":
    deploy()
