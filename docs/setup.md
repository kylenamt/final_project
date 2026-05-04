# Setup

This project supports two primary setups: Docker (recommended) and Conda.

## Docker

Requirements:
- Docker
- nvidia-container-toolkit (NVIDIA driver >= 450.80)

Commands:

```bash
docker compose build
docker compose run --rm ddsp

# Makefile shortcuts
make docker-build
make docker
```

The image includes CUDA 11.2, cuDNN 8.1, and TensorFlow 2.11.1.

## Conda

Requirements:
- Miniconda or Anaconda
- NVIDIA GPU with CUDA 11 support

Commands:

```bash
make setup
conda activate conda_env3.10
```

## Locking dependencies

```bash
make lock
```

This regenerates conda-lock.txt and requirements-lock.txt.
