#!/usr/bin/env python3
"""Download the Bach Violin Dataset (v1.0) from Zenodo.

Source: https://zenodo.org/record/6050245
Paper:  "Deep Performer: Score-to-Audio Music Performance Synthesis"
        Dong et al., ICASSP 2022

Dataset: 6.5 hours of Bach solo violin (BWV 1001-1006), 17 violinists, ~265 MB.
"""

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ZENODO_URL = (
    "https://zenodo.org/records/6050245/files/bach-violin-dataset-v1.0.zip?download=1"
)
ZIP_FILENAME = "bach-violin-dataset-v1.0.zip"
EXPECTED_MD5 = "7cb08f01bf1c887386a9c425c996bac6"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        bar = int(pct / 2)
        print(f"\r  [{'#' * bar}{' ' * (50 - bar)}] {pct:5.1f}%", end="", flush=True)


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url}\n  -> {dest}")
    urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress bar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-dir",
        default="data/raw/bach-violin",
        help="Directory to extract dataset into (default: data/raw/bach-violin)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip MD5 checksum verification",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip file after extraction",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / ZIP_FILENAME

    # --- Download ---
    if zip_path.exists():
        print(f"Zip already exists at {zip_path}, skipping download.")
    else:
        download(ZENODO_URL, zip_path)

    # --- Verify ---
    if not args.no_verify:
        print("Verifying MD5 checksum...", end=" ", flush=True)
        actual = md5(zip_path)
        if actual != EXPECTED_MD5:
            print(f"FAILED\n  Expected: {EXPECTED_MD5}\n  Got:      {actual}")
            sys.exit(1)
        print("OK")

    # --- Extract ---
    print(f"Extracting to {out_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for i, name in enumerate(members, 1):
            print(f"\r  {i}/{len(members)}: {name[:70]:<70}", end="", flush=True)
            zf.extract(name, out_dir)
    print()

    # --- Clean up ---
    if not args.keep_zip:
        zip_path.unlink()
        print(f"Removed {zip_path.name}")

    print(f"\nDone. Dataset extracted to: {out_dir}")
    print("Directory layout:")
    for p in sorted(out_dir.rglob("*"))[:30]:
        print(f"  {p.relative_to(out_dir)}")


if __name__ == "__main__":
    main()
