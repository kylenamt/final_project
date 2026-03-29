"""Experiment pipeline: batch DDSP inference and WORLD vocoder baseline.

Provides two main functions:
- ``run_synthesize_dir`` – DDSP timbre transfer over a directory of WAV files.
- ``run_vocoder_dir``    – WORLD vocoder baseline over the same directory using
                           a randomly-sampled "source bank" of solo instrument audio.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import numpy as np

from data_preprocessing import trim_silence
from utils import load_audio, save_audio
from feature_utils import (
    auto_adjust_features,
    compute_features,
    load_dataset_stats,
    shift_f0,
    shift_loudness,
)
from timbre_transfer import load_model, resynthesize, DEFAULT_SAMPLE_RATE
from baseline import Baseline

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed return values
# ---------------------------------------------------------------------------

class PipelineResult(TypedDict):
    """Return type for ``run_synthesize_dir`` and ``run_vocoder_dir``."""
    processed: int
    failed: int
    failed_files: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_wav_files(root_dir: str | Path) -> List[Path]:
    """Recursively collect all ``.wav`` files under *root_dir*, sorted."""
    root = Path(root_dir)
    return sorted(root.rglob("*.wav"))


def make_output_path(
    input_path: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    suffix: str = "_TRANSFERED",
) -> Path:
    """Build an output path that mirrors the input directory structure.

    Example::

        make_output_path(
            "data/raw/voice/FULL/male5/long_tones/forte/m5.wav",
            "data/raw/voice/FULL",
            "data/processed/voice/Full_transfered",
        )
        # → Path("data/processed/voice/Full_transfered/male5/long_tones/forte/m5_TRANSFERED.wav")
    """
    rel = Path(input_path).relative_to(input_root)
    stem = rel.stem + suffix
    return Path(output_root) / rel.with_name(stem + rel.suffix)


def build_source_bank(
    source_dir: str | Path,
    sr: int = DEFAULT_SAMPLE_RATE,
    silence_threshold_db: float = -40.0,
) -> List[np.ndarray]:
    """Load all WAV files in *source_dir*, resample, trim silence, and return
    a list of 1-D float64 arrays (the "source bank").
    """
    paths = collect_wav_files(source_dir)
    if not paths:
        raise FileNotFoundError(f"No .wav files found in {source_dir}")

    bank: List[np.ndarray] = []
    for p in paths:
        audio, _ = load_audio(p, sr=sr, mono=True)
        trimmed, _ = trim_silence(audio, threshold_db=silence_threshold_db)
        if len(trimmed) > 0:
            bank.append(trimmed.astype(np.float64))
            logger.info("Source bank: %s → %d samples", p.name, len(trimmed))

    if not bank:
        raise ValueError("Source bank is empty after silence trimming.")
    return bank


def select_source_segment(
    source_bank: List[np.ndarray],
    target_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly select a source segment of *target_length* from the bank.

    Picks a random file from the bank, then a random offset within that file.
    If the chosen file is shorter than *target_length*, pads by wrapping.
    """
    idx = rng.integers(0, len(source_bank))
    src = source_bank[idx]

    if len(src) >= target_length:
        start = rng.integers(0, len(src) - target_length + 1)
        return src[start : start + target_length]

    return np.pad(src, (0, target_length - len(src)), mode="wrap")


# ---------------------------------------------------------------------------
# DDSP inference pipeline
# ---------------------------------------------------------------------------


def run_synthesize_dir(
    model_dir: str | Path,
    gin_file: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    sr: int = DEFAULT_SAMPLE_RATE,
    auto_adjust: bool = True,
    pitch_shift: float = 0.0,
    loudness_shift: float = 0.0,
    threshold: float = 1.0,
    quiet: float = 20.0,
    autotune_amount: float = 0.0,
) -> PipelineResult:
    """Run DDSP timbre transfer on every WAV file under *input_dir*.

    The model is loaded **once** with a fixed 4-second window
    (``n_samples=64000``, ``time_steps=1000``).  Longer files are processed
    by chunking features into 1000-frame slices and concatenating the
    synthesised output.

    Parameters
    ----------
    model_dir, gin_file : paths to the DDSP checkpoint directory and gin config.
    input_dir  : root directory of input WAV files (searched recursively).
    output_dir : root directory for output; subdirectory structure is preserved.
    sr : sample rate.
    auto_adjust : apply ``auto_adjust_features`` using dataset statistics.
    pitch_shift : manual pitch shift in octaves.
    loudness_shift : manual loudness shift in dB.
    threshold, quiet, autotune_amount : parameters for ``auto_adjust_features``.

    Returns
    -------
    PipelineResult
    """
    import gin as gin_module  # local import — avoids TF init at module level

    model_dir = str(model_dir)
    gin_file = str(gin_file)
    wav_files = collect_wav_files(input_dir)
    logger.info("DDSP pipeline: %d files in %s", len(wav_files), input_dir)

    # --- Load model once with a 4-second reference ---
    ref_audio = np.zeros((1, 64000), dtype=np.float32)
    ref_features = compute_features(ref_audio)
    model, _ = load_model(model_dir, gin_file, ref_audio, ref_features)

    # Read the model's fixed chunk dimensions from gin
    time_steps = int(gin_module.query_parameter("F0LoudnessPreprocessor.time_steps"))
    n_samples = int(gin_module.query_parameter("Harmonic.n_samples"))

    dataset_stats = load_dataset_stats(model_dir) if auto_adjust else None

    processed, failed = 0, 0
    failed_files: List[str] = []
    items = tqdm(wav_files, desc="DDSP inference") if tqdm else wav_files

    for wav_path in items:
        try:
            audio, _ = load_audio(wav_path, sr=sr, mono=True)
            original_len = len(audio)
            audio_2d = audio[np.newaxis, :]  # (1, N)

            features = compute_features(audio_2d)

            if dataset_stats is not None:
                features = auto_adjust_features(
                    features, dataset_stats, threshold, quiet, autotune_amount
                )
            if pitch_shift != 0.0:
                features = shift_f0(features, pitch_shift)
            if loudness_shift != 0.0:
                features = shift_loudness(features, loudness_shift)

            # --- Chunk features and synthesise ---
            total_frames = features["f0_hz"].shape[-1]
            n_chunks = max(1, int(np.ceil(total_frames / time_steps)))

            chunks_out: List[np.ndarray] = []
            for c in range(n_chunks):
                f_start = c * time_steps
                f_end = min(f_start + time_steps, total_frames)
                s_start = c * n_samples
                s_end = min(s_start + n_samples, features["audio"].shape[-1])

                chunk_feat: Dict[str, Any] = {}
                for key in ("f0_hz", "f0_confidence", "loudness_db"):
                    sl = features[key][..., f_start:f_end]
                    pad_width = time_steps - sl.shape[-1]
                    if pad_width > 0:
                        pad_cfg = [(0, 0)] * (sl.ndim - 1) + [(0, pad_width)]
                        sl = np.pad(sl, pad_cfg)
                    chunk_feat[key] = sl

                audio_sl = features["audio"][..., s_start:s_end]
                pad_width = n_samples - audio_sl.shape[-1]
                if pad_width > 0:
                    pad_cfg = [(0, 0)] * (audio_sl.ndim - 1) + [(0, pad_width)]
                    audio_sl = np.pad(audio_sl, pad_cfg)
                chunk_feat["audio"] = audio_sl

                audio_gen = resynthesize(model, chunk_feat)

                actual_samples = s_end - s_start
                if actual_samples < n_samples:
                    audio_gen = audio_gen[:actual_samples]

                chunks_out.append(audio_gen)

            full_audio = np.concatenate(chunks_out)[:original_len]

            out_path = make_output_path(wav_path, input_dir, output_dir)
            save_audio(out_path, full_audio, sr)
            processed += 1

        except Exception:
            failed += 1
            failed_files.append(str(wav_path))
            logger.warning("Failed: %s", wav_path, exc_info=True)

    logger.info("DDSP done: %d processed, %d failed", processed, failed)
    return {"processed": processed, "failed": failed, "failed_files": failed_files}


# ---------------------------------------------------------------------------
# WORLD vocoder baseline pipeline
# ---------------------------------------------------------------------------


def run_vocoder_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    source_dir: str | Path,
    sr: int = DEFAULT_SAMPLE_RATE,
    method: str = "f0",
    silence_threshold_db: float = -40.0,
    seed: int = 42,
) -> PipelineResult:
    """Run WORLD vocoder baseline on every WAV file under *input_dir*.

    For each target file a random source segment is drawn from a "source bank"
    built from the solo instrument recordings in *source_dir*.

    Parameters
    ----------
    input_dir  : root directory of target WAV files (searched recursively).
    output_dir : root directory for output; subdirectory structure is preserved.
    source_dir : directory of source instrument WAV files (e.g. solo violin).
    sr : sample rate.
    method : ``"f0"`` for F0 transfer, ``"f0_ap"`` for F0 + AP transfer.
    silence_threshold_db : threshold for silence trimming in source audio.
    seed : random seed for reproducible source segment selection.

    Returns
    -------
    PipelineResult
    """
    wav_files = collect_wav_files(input_dir)
    logger.info("Baseline pipeline: %d files in %s", len(wav_files), input_dir)

    source_bank = build_source_bank(source_dir, sr, silence_threshold_db)
    rng = np.random.default_rng(seed)
    bl = Baseline(sample_rate=sr)

    transfer_fn = bl.f0_transfer if method == "f0" else bl.f0_and_ap_transfer

    processed, failed = 0, 0
    failed_files: List[str] = []
    items = tqdm(wav_files, desc="Baseline inference") if tqdm else wav_files

    for wav_path in items:
        try:
            target_audio, _ = load_audio(wav_path, sr=sr, mono=True)
            target_audio = target_audio.astype(np.float64)

            source_audio = select_source_segment(
                source_bank, len(target_audio), rng
            )
            target_audio, source_audio = Baseline.match_length(
                target_audio, source_audio
            )

            result = transfer_fn(target_audio, source_audio)

            out_path = make_output_path(
                wav_path, input_dir, output_dir, suffix="_BASELINE"
            )
            save_audio(out_path, result.astype(np.float32), sr)
            processed += 1

        except Exception:
            failed += 1
            failed_files.append(str(wav_path))
            logger.warning("Failed: %s", wav_path, exc_info=True)

    logger.info("Baseline done: %d processed, %d failed", processed, failed)
    return {"processed": processed, "failed": failed, "failed_files": failed_files}
