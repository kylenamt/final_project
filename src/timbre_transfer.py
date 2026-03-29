#!/usr/bin/env python
"""DDSP Timbre Transfer inference script.

Can be run standalone from the command line or imported as a module.
Supports both local checkpoints and pretrained models from GCS.
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import time
import warnings
from pathlib import Path

from typing import Any, Dict

import gin
import librosa
import numpy as np
import soundfile as sf
import tensorflow.compat.v2 as tf  # type: ignore

import ddsp
import ddsp.training

from model_loading import (
    find_model_dir,
    load_pretrained_model,
    restore_autoencoder,
    PRETRAINED_MODELS,
)
from feature_utils import (
    trim_features,
)


DEFAULT_SAMPLE_RATE = 16000


def load_model(
    model_dir: str, gin_file: str, audio: np.ndarray, audio_features: Dict[str, Any]
) -> tuple:
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

    model = restore_autoencoder(model_dir)

    start_time = time.time()
    _ = model(audio_features, training=False) # type: ignore
    print(f"Restoring model took {time.time() - start_time:.1f} seconds")
    return model, audio_features


def resynthesize(model: Any, audio_features: Dict[str, Any]) -> np.ndarray:
    start_time = time.time()
    outputs = model(audio_features, training=False)
    audio_gen = model.get_audio_from_outputs(outputs)
    print(f"Prediction took {time.time() - start_time:.1f} seconds")
    audio_gen = np.array(audio_gen)[0]
    return audio_gen


def parse_args():
    parser = argparse.ArgumentParser(
        description="DDSP timbre transfer inference."
    )
    parser.add_argument("--audio-path", required=True,
                        help="Path to input audio (.wav or .npy)")
    parser.add_argument("--output-dir", default="artifacts/outputs",
                        help="Directory for output files")
    parser.add_argument("--model", default="Violin",
                        help=f"Pretrained model ({', '.join(PRETRAINED_MODELS)}) or path to local checkpoint dir")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--pitch-shift", type=float, default=0.0,
                        help="Pitch shift in octaves")
    parser.add_argument("--loudness-shift", type=float, default=0.0,
                        help="Loudness shift in dB")
    parser.add_argument("--auto-adjust", type=int, default=1,
                        help="Auto-adjust pitch/loudness to training data (1=on, 0=off)")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Note detection sensitivity")
    parser.add_argument("--quiet", type=float, default=20.0,
                        help="Suppress silent regions (dB)")
    parser.add_argument("--autotune", type=float, default=0.0,
                        help="Autotune amount (0=off, 1=full)")
    parser.add_argument("--f0-confidence-threshold", type=float, default=0.0,
                        help="Zero out f0 below this confidence")
    parser.add_argument("--start-sec", type=float, default=None,
                        help="Start time in seconds (optional segment extraction)")
    parser.add_argument("--end-sec", type=float, default=None,
                        help="End time in seconds (optional segment extraction)")
    return parser.parse_args()


