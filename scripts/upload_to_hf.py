#!/usr/bin/env python3
"""Upload a DDSP model checkpoint and gin configs to Hugging Face."""

import argparse
import os
import re
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: huggingface_hub. "
        "Install with `pip install huggingface_hub`."
    ) from exc


CKPT_INDEX_RE = re.compile(r"ckpt-(\d+)\.index$")


def find_latest_checkpoint_step(model_dir: Path) -> str:
    candidates = []
    for path in model_dir.glob("ckpt-*.index"):
        match = CKPT_INDEX_RE.search(path.name)
        if match:
            candidates.append(int(match.group(1)))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint index files found in {model_dir}")
    return str(max(candidates))


def build_upload_list(model_dir: Path, include_checkpoint_file: bool) -> list[Path]:
    latest_step = find_latest_checkpoint_step(model_dir)
    files: list[Path] = []

    data_file = model_dir / f"ckpt-{latest_step}.data-00000-of-00001"
    index_file = model_dir / f"ckpt-{latest_step}.index"

    if index_file.exists():
        files.append(index_file)
    if data_file.exists():
        files.append(data_file)

    for gin_file in sorted(model_dir.glob("*.gin")):
        files.append(gin_file)

    if include_checkpoint_file:
        ckpt_file = model_dir / "checkpoint"
        if ckpt_file.exists():
            files.append(ckpt_file)

    if not files:
        raise FileNotFoundError(f"No files found to upload in {model_dir}")

    return files


def upload_files(repo_id: str, files: list[Path], path_in_repo: str, token: str | None) -> None:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    for path in files:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=str(Path(path_in_repo) / path.name),
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        print(f"Uploaded {path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload latest DDSP checkpoint and gin configs to Hugging Face."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Hugging Face repo id, e.g. username/model-name",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--path-in-repo",
        default="model",
        help="Folder in the repo to place uploaded files",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--include-checkpoint-file",
        action="store_true",
        help="Also upload the TensorFlow 'checkpoint' file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    files = build_upload_list(model_dir, args.include_checkpoint_file)

    upload_files(args.repo, files, args.path_in_repo, args.token)

    print(
        f"Done. Uploaded {len(files)} files from {model_dir} to {args.repo}"
        f" under {args.path_in_repo}."
    )


if __name__ == "__main__":
    main()
