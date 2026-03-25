# DDSP Timbre Transfer

DDSP-based timbre transfer experiments — train a synthesizer on one instrument
and resynthesize audio from another source.

## Quick start

### 1. Environment setup

#### Option A: Docker (recommended for reproducibility)

Requires [Docker](https://docs.docker.com/get-docker/) and
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
(any NVIDIA driver >= 450.80).

```bash
docker compose run --rm ddsp          # build image & drop into a shell
# or for VS Code: open the repo and select "Reopen in Container"
```

The image bundles CUDA 11.2 + cuDNN 8.1 + TF 2.11.1 — works on any GPU.

#### Option B: Conda (local install)

Requires [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html)
and an NVIDIA GPU with CUDA 11.2 support.

```bash
make setup                  # creates conda env with Python 3.10, CUDA 11.2, cuDNN 8.1, and all pip deps
conda activate conda_env3.10
```

Or manually:

```bash
conda env create -f environment.yml
conda activate conda_env3.10
```

### 2. Prepare data

Place mono `.wav` files in `data/preprocessed/solo_violin/` (or change
`INPUT_PATTERN`), then:

```bash
make prepare
```

This creates TFRecord shards in `data/tfrecords/solo_violin/`.

### 3. Train

```bash
make train
```

Training checkpoints are saved to `artifacts/ae/`. Override defaults:

```bash
make train SAVE_DIR=artifacts/my_run/ BATCH_SIZE=16
```

### 4. Evaluate

```bash
make eval
```

### 5. Upload to Hugging Face

```bash
make upload-hf HF_REPO=username/repo-name MODEL_DIR=artifacts/ae/ PATH_IN_REPO=trained_noreverb
```

## Makefile targets

| Target           | Description                                                              |
| ---------------- | ------------------------------------------------------------------------ |
| `make setup`     | Create/update conda env from `environment.yml`                           |
| `make lock`      | Regenerate `conda-lock.txt` and `requirements-lock.txt` from current env |
| `make prepare`   | Build TFRecord dataset from `INPUT_PATTERN`                              |
| `make train`     | Train DDSP model                                                         |
| `make eval`      | Evaluate DDSP model                                                      |
| `make sample`    | Sample from DDSP model                                                   |
| `make upload-hf` | Upload checkpoint + gin config to Hugging Face                           |
| `make help`      | Print all targets and variables                                          |

## Overridable variables

| Variable          | Default                                           | Description                       |
| ----------------- | ------------------------------------------------- | --------------------------------- |
| `INPUT_PATTERN`   | `data/preprocessed/solo_violin/*.wav`             | Glob for input audio files        |
| `OUTPUT_TFRECORD` | `data/tfrecords/solo_violin/solo_violin.tfrecord` | TFRecord output path              |
| `TFRECORD_PATH`   | `data/tfrecords/solo_violin/*.tfrecord`           | TFRecord glob for training        |
| `SAVE_DIR`        | `artifacts/ae/`                                   | Checkpoint save directory         |
| `BATCH_SIZE`      | `8`                                               | Training batch size               |
| `GIN_MODEL`       | `models/ae.gin`                                   | Gin config for model architecture |
| `GIN_DATASET`     | `datasets/tfrecord.gin`                           | Gin config for dataset            |
| `GIN_EVAL`        | `eval/basic_f0_ld.gin`                            | Gin config for evaluation         |
| `GIN_SEARCH_PATH` | `configs/ddsp_gin`                                | Gin file search path              |

## Project structure

```
src/               # Python modules (feature_utils, loss, model_loading, etc.)
src/demo/          # Jupyter notebooks (timbre_transfer, playground)
configs/ddsp_gin/  # Gin configuration files
scripts/           # Shell scripts (train_ddsp.sh, upload_to_hf.py)
data/              # Audio data and TFRecords
artifacts/         # Trained model checkpoints
```
