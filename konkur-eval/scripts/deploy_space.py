
import os
import time
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

def deploy():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    username = os.getenv("HF_USERNAME")
    
    if not token or not username:
        print("Error: HF_TOKEN or HF_USERNAME not found in .env")
        return

    api = HfApi(token=token)
    repo_name = "konkur1404-llm-leaderboard"
    repo_id = f"{username}/{repo_name}"

    # Optional: Delete old space if needed (uncomment if you want to clean up)
    # try:
    #     api.delete_repo(repo_id=f"{username}/konkur1404-leaderboard", repo_type="space")
    #     print("Deleted old space.")
    # except:
    #     pass

    print(f"Creating/Checking Space: {repo_id}")
    last_err = None
    for attempt in range(1, 6):
        try:
            create_repo(
                repo_id=repo_id,
                token=token,
                repo_type="space",
                space_sdk="gradio",
                exist_ok=True
            )
            print("Space created or already exists.")
            last_err = None
            break
        except Exception as e:
            last_err = e
            wait_s = min(30, 2 ** (attempt - 1))
            print(f"Create attempt {attempt}/5 failed: {e}")
            if attempt < 5:
                print(f"Retrying in {wait_s}s...")
                time.sleep(wait_s)
    if last_err is not None:
        print(f"Error creating space: {last_err}")
        return

    print("Uploading files...")
    folder_path = os.path.join(os.getcwd(), "hf_space")
    
    last_err = None
    for attempt in range(1, 6):
        try:
            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type="space",
                commit_message="Update leaderboard data and app"
            )
            print(f"Successfully uploaded files to https://huggingface.co/spaces/{repo_id}")
            last_err = None
            break
        except Exception as e:
            last_err = e
            wait_s = min(30, 2 ** (attempt - 1))
            print(f"Upload attempt {attempt}/5 failed: {e}")
            if attempt < 5:
                print(f"Retrying in {wait_s}s...")
                time.sleep(wait_s)
    if last_err is not None:
        print(f"Error uploading files: {last_err}")

if __name__ == "__main__":
    deploy()
