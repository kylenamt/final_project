#!/usr/bin/env python3
"""Download a DDSP model checkpoint and gin configs from Hugging Face."""

import argparse
import os
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: huggingface_hub. "
        "Install with `pip install huggingface_hub`."
    ) from exc


def download_files(repo_id: str, path_in_repo: str, local_dir: Path, token: str | None) -> None:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
    prefix = path_in_repo.rstrip("/") + "/"
    matched = [f for f in files if f.startswith(prefix)]

    if not matched:
        raise FileNotFoundError(
            f"No files found under '{path_in_repo}/' in repo {repo_id}"
        )

    local_dir.mkdir(parents=True, exist_ok=True)

    for remote_path in matched:
        filename = remote_path[len(prefix):]
        local_path = local_dir / filename
        api.hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="model",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
        # hf_hub_download preserves the repo folder structure; move to flat dir
        downloaded = local_dir / remote_path
        if downloaded != local_path:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded.rename(local_path)
        print(f"Downloaded {filename}")

    # Clean up any leftover empty subdirectories from the repo structure
    for d in sorted(local_dir.rglob("*"), reverse=True):
        if d.is_dir() and not list(d.iterdir()):
            d.rmdir()

    print(f"Done. Downloaded {len(matched)} files to {local_dir}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download DDSP checkpoint and gin configs from Hugging Face."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Hugging Face repo id, e.g. username/model-name",
    )
    parser.add_argument(
        "--path-in-repo",
        required=True,
        help="Folder in the repo to download from",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Local directory to save files to",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(args.model_dir).expanduser().resolve()
    download_files(args.repo, args.path_in_repo, local_dir, args.token)


if __name__ == "__main__":
    main()
