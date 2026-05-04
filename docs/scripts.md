# Scripts

This folder contains command-line helpers for data and model management.

## scripts/train_ddsp.sh

Wrapper around ddsp_run. Uses Makefile variables for gin configs and
TFRecord paths. Exits early if ddsp_run is missing or TFRecords are not found.

## scripts/inference.py

CLI stub for inference. Parses args and prints them; wire it to
TimbreTransfer if you want a full CLI runner.

## scripts/download_from_hf.py

Download checkpoints and gin configs from Hugging Face.

Example:

```bash
python scripts/download_from_hf.py \
  --repo username/model-name \
  --path-in-repo model \
  --model-dir artifacts/model
```

## scripts/upload_to_hf.py

Upload the latest checkpoint and gin files to Hugging Face.

Example:

```bash
python scripts/upload_to_hf.py \
  --repo username/model-name \
  --model-dir artifacts/ae \
  --path-in-repo model
```

## scripts/download_bach_violin.py

Downloads and extracts the Bach Violin dataset from Zenodo.
