import os
import pickle
from typing import Optional

import gin
import librosa
import numpy as np
import tensorflow.compat.v2 as tf

import ddsp
import ddsp.training
from ddsp.training.postprocessing import detect_notes, fit_quantile_transform

try:
    from ddsp.colab.colab_utils import auto_tune, get_tuning_factor
except Exception:  # Colab utilities are optional in local runs.
    auto_tune = None
    get_tuning_factor = None


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
    auto_tune_fn=None,
    get_tuning_factor_fn=None,
):
    auto_tune_fn = auto_tune if auto_tune_fn is None else auto_tune_fn
    get_tuning_factor_fn = get_tuning_factor if get_tuning_factor_fn is None else get_tuning_factor_fn

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

        if autotune_amount and auto_tune_fn and get_tuning_factor_fn:
            f0_midi = np.array(ddsp.core.hz_to_midi(audio_features["f0_hz"]))
            tuning_factor = get_tuning_factor_fn(
                f0_midi, audio_features["f0_confidence"], mask_on
            )
            f0_midi_at = auto_tune_fn(
                f0_midi, tuning_factor, mask_on, amount=autotune_amount
            )
            audio_features["f0_hz"] = ddsp.core.midi_to_hz(f0_midi_at)
    return audio_features


def _to_numpy(value):
    if isinstance(value, tf.Tensor):
        return value.numpy()
    return value


def compute_features(audio, sample_rate: Optional[int] = None):
    _ = sample_rate
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)
    ddsp.spectral_ops.reset_crepe()
    audio_features = ddsp.training.metrics.compute_audio_features(audio)
    audio_features = tf.nest.map_structure(_to_numpy, audio_features)
    audio_features["loudness_db"] = audio_features["loudness_db"].astype(np.float32)
    return audio_features


def trim_features(audio_features, time_steps: int, n_samples: int):
    for key in ["f0_hz", "f0_confidence", "loudness_db"]:
        audio_features[key] = audio_features[key][:time_steps]
    audio_features["audio"] = audio_features["audio"][:, :n_samples]
    return audio_features
