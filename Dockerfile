# --------------------------------------------------------------------------
# DDSP Timbre Transfer — reproducible GPU environment
#
# Base image bundles CUDA 11.2 + cuDNN 8.1 + TensorFlow 2.11.1-gpu,
# so the host only needs nvidia-container-toolkit (any driver >= 450.80).
# --------------------------------------------------------------------------
FROM tensorflow/tensorflow:2.11.1-gpu

LABEL maintainer="namkhanh"
LABEL description="DDSP timbre-transfer experiments (TF 2.11.1, CUDA 11.2)"

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for soundfile, librosa, pyworld, and audio I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace

# Copy full project (data/artifacts excluded via .dockerignore)
COPY . .

# Install project + dependencies in editable mode
RUN pip install --no-cache-dir -e .

# Default command: interactive shell
CMD ["/bin/bash"]
