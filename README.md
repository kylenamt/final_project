# DDSP Timbre Transfer

DDSP-based timbre transfer experiments — train a synthesizer on one instrument
and resynthesize audio from another source.

**How it works:** A DDSP autoencoder is trained on a target instrument's audio.
At inference, pitch (F0) and loudness are extracted from a source signal and fed
through the trained decoder to resynthesize the audio in the target instrument's
timbre. Two deterministic vocoder baselines (WORLD and SMS/HPS) are provided for
comparison. See [docs/architecture.md](docs/architecture.md) for the full picture.

## Quick start

### 1. Environment setup

#### Option A: Docker (recommended for reproducibility)

Requires [Docker](https://docs.docker.com/get-docker/) and
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
(any NVIDIA driver >= 450.80).

```bash
# Build once
docker compose build

# Start a shell inside the container
docker compose run --rm ddsp

# or via Makefile shortcuts
make docker-build
make docker

# or for VS Code: open the repo and select "Reopen in Container"
```

The image bundles CUDA 11.2 + cuDNN 8.1 + TF 2.11.1 — works on any GPU.

#### Option B: Conda (local install)

Requires [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html)
and an NVIDIA GPU with CUDA 11 support.

```bash
make setup                  # creates conda env with Python 3.10, CUDA 11.8, cuDNN 8.9.2, and all pip deps
conda activate conda_env3.10
```

Or manually:

```bash
conda env create -f environment.yml
conda activate conda_env3.10
```

### 2. Prepare data

Place mono `.wav` files in `data/raw/solo_violin/` (or change
`INPUT_PATTERN`), then:

```bash
make prepare
```

This creates TFRecord shards in `data/tfrecords/solo_violin/`.

### 3. Train

```bash
make train
```

Training checkpoints are saved to the preset's `SAVE_DIR`. Override defaults:

```bash
make train SAVE_DIR=artifacts/my_run/ BATCH_SIZE=32
```

Or use a preset (see below):

```bash
make train ae
make train solo_instrument
```

### 4. Evaluate

```bash
make eval
```

### 5. Upload to Hugging Face

```bash
make upload-hf HF_REPO=username/repo-name MODEL_DIR=artifacts/ae/ PATH_IN_REPO=ae
```

## Presets

Presets are defined in `configs/training_config/preset.mk` and set `SAVE_DIR`,
`MODEL_DIR`, and `PATH_IN_REPO` automatically:

| Preset | SAVE_DIR | Model |
|--------|----------|-------|
| `ae` | `artifacts/ae/` | Full autoencoder (encoder + decoder) |
| `solo_instrument` | `artifacts/solo_instrument/` | Decoder-only with trainable reverb |

Usage: `make train ae`, `make eval solo_instrument`, etc.

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

## Overridable variables

| Variable          | Default                                           | Description                       |
| ----------------- | ------------------------------------------------- | --------------------------------- |
| `INPUT_PATTERN`   | `data/raw/solo_violin/*.wav`                      | Glob for input audio files        |
| `OUTPUT_TFRECORD` | `data/tfrecords/solo_violin/solo_violin.tfrecord` | TFRecord output path              |
| `TFRECORD_PATH`   | `data/tfrecords/solo_violin/*.tfrecord`           | TFRecord glob for training        |
| `SAVE_DIR`        | (set by preset)                                   | Checkpoint save directory         |
| `BATCH_SIZE`      | `16`                                              | Training batch size               |
| `GIN_MODEL`       | `models/solo_instrument.gin`                      | Gin config for model architecture |
| `GIN_DATASET`     | `datasets/tfrecord.gin`                           | Gin config for dataset            |
| `GIN_EVAL`        | `eval/basic_f0_ld.gin`                            | Gin config for evaluation         |
| `GIN_SEARCH_PATH` | `configs/ddsp_gin`                                | Gin file search path              |

## Project structure

```
src/                          Python package
  ├── utils.py                Audio I/O helpers
  ├── data_preprocessing.py   Silence trimming, resampling, splitting
  ├── feature_utils.py        F0/loudness extraction and manipulation
  ├── model_loading.py        Checkpoint/gin discovery, model restoration
  ├── timbre_transfer.py      Inference script (CLI + module)
  ├── baseline.py             WORLD and SMS/HPS vocoder baselines
  ├── visualize.py            Spectrogram and feature plotting
  ├── evaluation/             Evaluation subpackage
  │   ├── loss.py             MMD and Wasserstein distance metrics
  │   ├── segment.py          Audio segmentation
  │   └── timbre_metrics.py   Spectral feature extraction (pytimbre)
  └── notebooks/              Jupyter notebooks
      ├── demo/
      ├── experiment/
      └── preprocessing/
configs/
  ├── ddsp_gin/               Gin configuration files
  │   ├── models/             Model architectures (ae.gin, solo_instrument.gin)
  │   ├── datasets/           Dataset providers (tfrecord.gin, nsynth.gin)
  │   ├── eval/               Evaluation configs
  │   └── optimization/       Training hyperparameters
  └── training_config/
      └── preset.mk           Named experiment presets
scripts/
  ├── train_ddsp.sh           Training entry point (wraps ddsp_run)
  ├── download_from_hf.py     Download checkpoints from Hugging Face
  ├── download_bach_violin.py Download the Bach Violin dataset
  └── upload_to_hf.py         Upload checkpoints to Hugging Face
data/                         Audio data and TFRecords
artifacts/                    Trained model checkpoints
docs/                         Documentation (see docs/index.md)
  ├── setup.md                Environment setup and commands
  ├── architecture.md         Codebase architecture and module reference
  ├── data.md                 Data layout and preprocessing
  ├── evaluation.md           Evaluation pipeline reference
  └── scripts.md              Script reference
Dockerfile                    GPU container (TF 2.11.1-gpu base)
docker-compose.yml            Docker Compose with GPU support
environment.yml               Conda environment definition
conda-lock.txt                Conda lock file (generated)
requirements-lock.txt         Pip lock file (generated)
pyproject.toml                Package metadata and pip dependencies
```

## Documentation

- [docs/index.md](docs/index.md) — documentation entry point.
- [docs/setup.md](docs/setup.md) — environment setup and commands.
- [docs/architecture.md](docs/architecture.md) — codebase architecture, module reference, gin configuration, and training orchestration.
- [docs/data.md](docs/data.md) — data layout, preprocessing, and TFRecords.
- [docs/evaluation.md](docs/evaluation.md) — evaluation pipeline: segmentation, timbre feature extraction, and distributional distance metrics.
- [docs/scripts.md](docs/scripts.md) — script reference and examples.
