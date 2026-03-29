"""Audio-feature utilities for DDSP timbre transfer.

Handles feature extraction (via DDSP / CREPE), pitch & loudness
manipulation, auto-adjustment to match a target dataset, and simple
feature trimming.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

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


# ---------------------------------------------------------------------------
# Local fallbacks for auto-tune helpers (used when ddsp.colab is absent)
# Ported from the official DDSP timbre-transfer Colab demo.
# ---------------------------------------------------------------------------

def _get_tuning_factor(
    f0_midi: np.ndarray,
    f0_confidence: np.ndarray,
    mask_on: np.ndarray,
) -> float:
    """Estimate the tuning offset (in semitones) that best aligns *f0_midi*
    to chromatic intervals.

    Sweeps offsets in [-0.5, 0.5] semitones and picks the one that minimises
    a combined cost of per-frame deviation from the nearest integer MIDI
    value and the number of inter-frame jumps.
    """
    tuning_factors = np.linspace(-0.5, 0.5, 101)
    midi_diffs = (f0_midi[mask_on][:, np.newaxis] - tuning_factors[np.newaxis, :]) % 1.0
    midi_diffs[midi_diffs > 0.5] -= 1.0
    weights = f0_confidence[mask_on][:, np.newaxis]

    cost_diffs = np.mean(weights * np.abs(midi_diffs), axis=0)

    f0_at = f0_midi[mask_on][:, np.newaxis] - midi_diffs
    f0_at_diffs = np.diff(f0_at, axis=0)
    deltas = (f0_at_diffs != 0.0).astype(float)
    cost_deltas = np.mean(weights[:-1] * deltas, axis=0)

    _norm = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8)
    cost = _norm(cost_deltas) + _norm(cost_diffs)
    return float(tuning_factors[np.argmin(cost)])


def _auto_tune(
    f0_midi: np.ndarray,
    tuning_factor: float,  # noqa: ARG001 – kept for API parity
    mask_on: np.ndarray,  # noqa: ARG001
    amount: float = 0.0,
) -> np.ndarray:
    """Snap *f0_midi* toward the nearest major-scale note.

    Parameters
    ----------
    f0_midi : array
        Per-frame MIDI pitch values.
    tuning_factor : float
        Estimated tuning offset (unused directly, but part of the
        official API signature).
    mask_on : array
        Boolean mask of voiced frames.
    amount : float
        Blend factor in [0, 1].  0 = no correction, 1 = full snap.
    """
    major_scale = np.ravel(
        [np.array([0, 2, 4, 5, 7, 9, 11]) + 12 * i for i in range(10)]
    )
    all_scales = np.stack([major_scale + i for i in range(12)])

    f0_on = f0_midi[mask_on]
    f0_diff_tsn = f0_on[:, np.newaxis, np.newaxis] - all_scales[np.newaxis, :, :]
    f0_diff_ts = np.min(np.abs(f0_diff_tsn), axis=-1)
    f0_diff_s = np.mean(f0_diff_ts, axis=0)
    scale_idx = np.argmin(f0_diff_s)

    f0_diff_tn = f0_midi[:, np.newaxis] - all_scales[scale_idx][np.newaxis, :]
    note_idx = np.argmin(np.abs(f0_diff_tn), axis=-1)
    midi_diff = np.take_along_axis(f0_diff_tn, note_idx[:, np.newaxis], axis=-1)[:, 0]
    return f0_midi - amount * midi_diff

# Pre-compute the upper clamp for F0 (MIDI 110 ≈ 14080 Hz).
_MAX_F0_HZ: float = float(librosa.midi_to_hz(110.0))

# Type alias for the feature dict used throughout the pipeline.
AudioFeatures = Dict[str, Any]


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def load_dataset_stats(model_dir: str) -> Optional[Dict[str, Any]]:
    """Load ``dataset_statistics.pkl`` from *model_dir*, or ``None``."""
    stats_path = os.path.join(model_dir, "dataset_statistics.pkl")
    if tf.io.gfile.exists(stats_path):
        with tf.io.gfile.GFile(stats_path, "rb") as fh:
            return pickle.load(fh)
    return None


# ---------------------------------------------------------------------------
# Feature manipulation
# ---------------------------------------------------------------------------

def shift_loudness(audio_features: AudioFeatures, ld_shift_db: float) -> AudioFeatures:
    """Shift loudness by a fixed amount (dB)."""
    audio_features["loudness_db"] = audio_features["loudness_db"] + ld_shift_db
    return audio_features


def shift_f0(audio_features: AudioFeatures, pitch_shift: float) -> AudioFeatures:
    """Shift F0 by *pitch_shift* octaves, clamped to ``[0, _MAX_F0_HZ]``."""
    audio_features["f0_hz"] = np.clip(
        audio_features["f0_hz"] * 2.0 ** pitch_shift,
        0.0,
        _MAX_F0_HZ,
    )
    return audio_features


def trim_features(
    audio_features: AudioFeatures, time_steps: int, n_samples: int
) -> AudioFeatures:
    """Truncate feature arrays to the given lengths."""
    for key in ("f0_hz", "f0_confidence", "loudness_db"):
        audio_features[key] = audio_features[key][:time_steps]
    audio_features["audio"] = audio_features["audio"][:, :n_samples]
    return audio_features


def compute_alignment(audio: np.ndarray) -> tuple:
    """Compute temporally-aligned ``(time_steps, n_samples, hop_size)``
    from the active gin config.

    Reads ``F0LoudnessPreprocessor.time_steps`` and ``Harmonic.n_samples``
    to derive the model's hop size, then rounds the audio length down to
    an exact multiple of that hop.

    Returns
    -------
    time_steps : int
        Number of feature frames that fit in *audio*.
    n_samples : int
        Number of audio samples (``time_steps * hop_size``).
    hop_size : int
        Samples per feature frame.
    """
    import gin  # local import – gin may not be configured at module level

    time_steps_train = gin.query_parameter("F0LoudnessPreprocessor.time_steps")
    n_samples_train = gin.query_parameter("Harmonic.n_samples")
    hop_size = int(n_samples_train / time_steps_train)

    n_audio = audio.shape[1] if audio.ndim == 2 else audio.shape[0]
    time_steps = int(n_audio / hop_size)
    n_samples = time_steps * hop_size
    return time_steps, n_samples, hop_size


# ---------------------------------------------------------------------------
# Auto-adjustment to target dataset
# ---------------------------------------------------------------------------

def auto_adjust_features(
    audio_features: AudioFeatures,
    dataset_stats: Dict[str, Any],
    threshold: float,
    quiet: float,
    autotune_amount: float,
    auto_tune_fn=None,
    get_tuning_factor_fn=None,
) -> AudioFeatures:
    """Match pitch & loudness range to a target dataset.

    Steps (when voiced frames exist):
    1. Shift F0 by the nearest whole-octave difference to the target mean
       pitch.
    2. Normalise loudness via the dataset's quantile transform.
    3. Optionally auto-tune to the nearest semitone.
    """
    auto_tune_fn = auto_tune_fn or auto_tune or _auto_tune
    get_tuning_factor_fn = get_tuning_factor_fn or get_tuning_factor or _get_tuning_factor

    mask_on, note_on_value = detect_notes(
        audio_features["loudness_db"],
        audio_features["f0_confidence"],
        threshold,
    )
    if not np.any(mask_on):
        return audio_features

    # --- Pitch adjustment (whole octaves) ---
    pitch = ddsp.core.hz_to_midi(audio_features["f0_hz"])
    p_diff = dataset_stats["mean_pitch"] - np.mean(pitch[mask_on])
    p_diff_octave = p_diff / 12.0
    round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
    audio_features = shift_f0(audio_features, round_fn(p_diff_octave))

    # --- Loudness normalisation ---
    _, loudness_norm = fit_quantile_transform(
        audio_features["loudness_db"],
        mask_on,
        inv_quantile=dataset_stats["quantile_transform"],
    )
    mask_off = ~mask_on
    loudness_norm[mask_off] -= quiet * (
        1.0 - note_on_value[mask_off][:, np.newaxis]
    )
    audio_features["loudness_db"] = loudness_norm.reshape(
        audio_features["loudness_db"].shape
    )

    # --- Optional auto-tune ---
    if autotune_amount and auto_tune_fn and get_tuning_factor_fn:
        f0_midi = np.asarray(ddsp.core.hz_to_midi(audio_features["f0_hz"]))
        tuning_factor = get_tuning_factor_fn(
            f0_midi, audio_features["f0_confidence"], mask_on
        )
        f0_midi_at = auto_tune_fn(
            f0_midi, tuning_factor, mask_on, amount=autotune_amount
        )
        audio_features["f0_hz"] = ddsp.core.midi_to_hz(f0_midi_at)

    return audio_features


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_features(
    audio: np.ndarray
) -> AudioFeatures:
    """Extract F0, loudness, and audio features via DDSP/CREPE.

    Parameters
    ----------
    audio : array-like
        Raw waveform (1-D or batched ``[1, N]``).

    Returns
    -------
    AudioFeatures
        Dict with ``f0_hz``, ``f0_confidence``, ``loudness_db``, ``audio``,
        etc., all as NumPy arrays.
    """
    audio_t = tf.convert_to_tensor(audio, dtype=tf.float32)
    ddsp.spectral_ops.reset_crepe()

    features = ddsp.training.metrics.compute_audio_features(audio_t)
    features = {
        k: v.numpy() if isinstance(v, tf.Tensor) else v
        for k, v in features.items()
    }
    features["loudness_db"] = features["loudness_db"].astype(np.float32)
    return features
