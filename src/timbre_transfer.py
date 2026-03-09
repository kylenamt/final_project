#!/usr/bin/env python
import argparse
import os
import time
import sys
from pathlib import Path

import gin
import numpy as np
import soundfile as sf

import ddsp.training

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model_loading import find_model_dir, find_latest_checkpoint
from feature_utils import (
    auto_adjust_features,
    compute_features,
    load_dataset_stats,
    shift_f0,
    shift_loudness,
    trim_features,
)


DEFAULT_SAMPLE_RATE = 16000


def load_model(model_dir: str, gin_file: str, audio, audio_features):
    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)

    time_steps_train = gin.query_parameter("F0LoudnessPreprocessor.time_steps")
    n_samples_train = gin.query_parameter("Harmonic.n_samples")
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = int(audio.shape[1] / hop_size)
    n_samples = time_steps * hop_size

    gin_params = [
        f"Harmonic.n_samples = {n_samples}",
        f"FilteredNoise.n_samples = {n_samples}",
        f"F0LoudnessPreprocessor.time_steps = {time_steps}",
        "oscillator_bank.use_angular_cumsum = True",
    ]
    with gin.unlock_config():
        gin.parse_config(gin_params)

    audio_features = trim_features(audio_features, time_steps, n_samples)

    model = ddsp.training.models.Autoencoder()
    ckpt = find_latest_checkpoint(model_dir)
    model.restore(ckpt)

    start_time = time.time()
    _ = model(audio_features, training=False)
    print(f"Restoring model took {time.time() - start_time:.1f} seconds")
    return model, audio_features


def resynthesize(model, audio_features):
    start_time = time.time()
    outputs = model(audio_features, training=False)
    audio_gen = model.get_audio_from_outputs(outputs)
    print(f"Prediction took {time.time() - start_time:.1f} seconds")
    audio_gen = np.array(audio_gen)[0]
    return audio_gen


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a local DDSP model and resynthesize audio."
    )
    parser.add_argument("--model-dir", default="artifacts/trained_models/")
    parser.add_argument("--gin-file", default="")
    parser.add_argument("--input-audio", required=True)
    parser.add_argument("--output-audio", default="artifacts/outputs/timbre_transfer.wav")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--pitch-shift", type=float, default=0.0)
    parser.add_argument("--loudness-shift", type=float, default=0.0)
    parser.add_argument("--auto-adjust", action="store_true")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--quiet", type=float, default=20.0)
    parser.add_argument("--autotune", type=float, default=0.0)
    return parser.parse_args()

