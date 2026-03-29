import os
from pathlib import Path
from huggingface_hub import create_repo, upload_folder

HF_SPACE_REPO = os.getenv("HF_SPACE_REPO")
HF_TOKEN = os.getenv("HF_TOKEN")
DEPLOYMENT_DIR = Path(__file__).resolve().parent

def main():
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found.")
    if not HF_SPACE_REPO:
        raise ValueError("HF_SPACE_REPO not found.")

    create_repo(
        repo_id=HF_SPACE_REPO,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        token=HF_TOKEN
    )

    upload_folder(
        folder_path=str(DEPLOYMENT_DIR),
        repo_id=HF_SPACE_REPO,
        repo_type="space",
        token=HF_TOKEN
    )
    print(f"Deployment files uploaded to {HF_SPACE_REPO}")

if __name__ == "__main__":
    main()