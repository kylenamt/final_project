#!/usr/bin/env python
import argparse
import glob
import os
import pickle
import re
import time

import gin
import librosa
import numpy as np
import soundfile as sf
import tensorflow.compat.v2 as tf

import ddsp
import ddsp.training
from ddsp.training.postprocessing import detect_notes, fit_quantile_transform

try:
    from ddsp.colab.colab_utils import auto_tune, get_tuning_factor
except Exception:  # Colab utilities are optional in local runs.
    auto_tune = None
    get_tuning_factor = None


DEFAULT_SAMPLE_RATE = 16000


def find_model_dir(path: str) -> str:
    if os.path.isfile(path) and path.endswith(".gin"):
        return os.path.dirname(path)
    gin_files = glob.glob(os.path.join(path, "*.gin"))
    if gin_files:
        return path
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith(".gin") and not name.startswith("."):
                return root
    raise FileNotFoundError(f"No gin file found under {path}")


def find_latest_checkpoint(model_dir: str) -> str:
    ckpt_indexes = glob.glob(os.path.join(model_dir, "ckpt-*.index"))
    if not ckpt_indexes:
        raise FileNotFoundError(f"No checkpoint found under {model_dir}")
    steps = []
    for path in ckpt_indexes:
        match = re.search(r"ckpt-(\d+)\.index$", path)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"No checkpoint steps parsed under {model_dir}")
    step = max(steps)
    return os.path.join(model_dir, f"ckpt-{step}")


def load_dataset_stats(model_dir: str):
    stats_path = os.path.join(model_dir, "dataset_statistics.pkl")
    if tf.io.gfile.exists(stats_path):
        with tf.io.gfile.GFile(stats_path, "rb") as handle:
            return pickle.load(handle)
    return None


def shift_loudness(audio_features, ld_shift_db: float):
    audio_features["loudness_db"] += ld_shift_db
    return audio_features


def shift_f0(audio_features, pitch_shift: float):
    audio_features["f0_hz"] *= 2.0 ** pitch_shift
    audio_features["f0_hz"] = np.clip(
        audio_features["f0_hz"], 0.0, librosa.midi_to_hz(110.0)
    )
    return audio_features


def auto_adjust_features(
    audio_features,
    dataset_stats,
    threshold: float,
    quiet: float,
    autotune_amount: float,
):
    mask_on, note_on_value = detect_notes(
        audio_features["loudness_db"],
        audio_features["f0_confidence"],
        threshold,
    )

    if np.any(mask_on):
        target_mean_pitch = dataset_stats["mean_pitch"]
        pitch = ddsp.core.hz_to_midi(audio_features["f0_hz"])
        mean_pitch = np.mean(pitch[mask_on])
        p_diff = target_mean_pitch - mean_pitch
        p_diff_octave = p_diff / 12.0
        round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
        p_diff_octave = round_fn(p_diff_octave)
        audio_features = shift_f0(audio_features, p_diff_octave)

        _, loudness_norm = fit_quantile_transform(
            audio_features["loudness_db"],
            mask_on,
            inv_quantile=dataset_stats["quantile_transform"],
        )

        mask_off = np.logical_not(mask_on)
        loudness_norm[mask_off] -= quiet * (1.0 - note_on_value[mask_off][:, np.newaxis])
        loudness_norm = np.reshape(loudness_norm, audio_features["loudness_db"].shape)
        audio_features["loudness_db"] = loudness_norm

        if autotune_amount and auto_tune and get_tuning_factor:
            f0_midi = np.array(ddsp.core.hz_to_midi(audio_features["f0_hz"]))
            tuning_factor = get_tuning_factor(
                f0_midi, audio_features["f0_confidence"], mask_on
            )
            f0_midi_at = auto_tune(
                f0_midi, tuning_factor, mask_on, amount=autotune_amount
            )
            audio_features["f0_hz"] = ddsp.core.midi_to_hz(f0_midi_at)
    return audio_features


def compute_features(audio, sample_rate: int):
    ddsp.spectral_ops.reset_crepe()
    audio_features = ddsp.training.metrics.compute_audio_features(audio)
    audio_features["loudness_db"] = audio_features["loudness_db"].astype(np.float32)
    return audio_features


def trim_features(audio_features, time_steps: int, n_samples: int):
    for key in ["f0_hz", "f0_confidence", "loudness_db"]:
        audio_features[key] = audio_features[key][:time_steps]
    audio_features["audio"] = audio_features["audio"][:, :n_samples]
    return audio_features


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

