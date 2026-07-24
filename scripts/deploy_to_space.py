import os
from pathlib import Path

from huggingface_hub import HfApi

SPACE_ID = "karimhoucem/text_summarization"
REPO_ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "app.py": "app.py",
    "requirements.txt": "requirements.txt",
    "deploy/space_README.md": "README.md",
}


def main() -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])

    for local_path, remote_path in FILES.items():
        api.upload_file(
            path_or_fileobj=str(REPO_ROOT / local_path),
            path_in_repo=remote_path,
            repo_id=SPACE_ID,
            repo_type="space",
            commit_message="Sync from GitHub Actions",
        )

    api.upload_folder(
        folder_path=str(REPO_ROOT / "summarizer"),
        path_in_repo="summarizer",
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="Sync summarizer package from GitHub Actions",
    )


if __name__ == "__main__":
    main()
