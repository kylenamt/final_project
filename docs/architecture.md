# Architecture

This repo implements DDSP-based timbre transfer plus deterministic vocoder
baselines. The core flow is:

1) Load input audio
2) Extract features (F0, loudness)
3) Run DDSP model to synthesize target timbre
4) Save audio and compute evaluation metrics

## Key modules

- src/timbre_transfer.py: model loader and inference wrapper
- src/feature_utils.py: feature extraction and adjustments
- src/model_loading.py: checkpoint and gin discovery
- src/baseline.py: WORLD and SMS/HPS baselines
- src/visualize.py: plots for analysis and reporting
- src/evaluation/: metrics and batch pipelines

## Configuration

- configs/ddsp_gin/: gin configs for models, datasets, and eval
- configs/training_config/preset.mk: training presets

## Entry points

- Makefile targets: setup, prepare, train, eval, sample, upload-hf
- scripts/train_ddsp.sh: training wrapper around ddsp_run
- scripts/inference.py: CLI stub for inference (wire to TimbreTransfer as needed)
