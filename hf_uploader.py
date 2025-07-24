#hf_uploader.py
import json
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, Repository, create_repo, upload_file
import tempfile

def convert_json_to_jsonl(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    jsonl_path = json_path.replace(".json", ".jsonl")
    with open(jsonl_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    return jsonl_path


def push_dataset_to_hub(json_data: list, repo_name: str, hf_token: str):
    """
    Converts list of dicts to JSONL and uploads it to Hugging Face dataset repo.
    """
    # 1. Create repo if it doesn't exist
    create_repo(repo_id=repo_name, token=hf_token, repo_type="dataset", exist_ok=True, private=True)

    # 2. Write JSONL data to a temp file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".jsonl") as temp_file:
        for entry in json_data:
            temp_file.write(json.dumps(entry) + "\n")
        temp_file_path = temp_file.name

    # 3. Upload the temp file directly
    upload_file(
        path_or_fileobj=temp_file_path,
        path_in_repo="data.jsonl",
        repo_id=repo_name,
        token=hf_token,
        repo_type="dataset"
    )

    # 4. Cleanup temp file
    os.remove(temp_file_path)

    print(f"✅ Dataset pushed to: https://huggingface.co/datasets/{repo_name}")