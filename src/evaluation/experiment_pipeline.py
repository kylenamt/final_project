"""Experiment pipeline: batch DDSP inference, WORLD vocoder baseline, and evaluation.

Provides three main functions:
- ``run_synthesize_dir``  – DDSP timbre transfer over a directory of WAV files.
- ``run_vocoder_dir``     – WORLD vocoder baseline over the same directory using
                            a randomly-sampled "source bank" of solo instrument audio.
- ``run_evaluation_dir``  – Per-file timbre loss computation between original and
                            transferred audio folders.

Resume behaviour
----------------
All functions skip files whose output already exists and is valid.
To pause mid-run just interrupt the kernel (Ctrl-C / Kernel > Interrupt).
Already-written files / CSV rows are preserved and will be skipped automatically
on the next run — no separate checkpoint file is needed.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd

from data_preprocessing import trim_silence
from .loss import Loss
from .timbre_metrics import TimbreMetrics
from utils import load_audio, save_audio
from feature_utils import (
    auto_adjust_features,
    compute_features,
    load_dataset_stats,
    shift_f0,
    shift_loudness,
)
from timbre_transfer import load_model, resynthesize
from utils import DEFAULT_SAMPLE_RATE
from baseline import Baseline

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineResult",
    "collect_wav_files",
    "make_output_path",
    "build_source_bank",
    "select_source_segment",
    "run_synthesize_dir",
    "run_vocoder_dir",
    "run_evaluation_dir",
]


# ---------------------------------------------------------------------------
# Typed return values
# ---------------------------------------------------------------------------

class PipelineResult(TypedDict):
    """Return type for ``run_synthesize_dir`` and ``run_vocoder_dir``."""
    processed: int
    failed: int
    skipped: int
    failed_files: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_valid_output(path: Path) -> bool:
    """Return True if *path* exists and is not a corrupt partial write.

    A valid WAV file is at least 44 bytes (header). Anything smaller was left
    behind by an interrupted write and should be reprocessed.
    """
    return path.exists() and path.stat().st_size > 44


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

    Files whose output already exists and is valid are skipped automatically,
    so re-running the function resumes from where it left off.

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

    processed, failed, skipped = 0, 0, 0
    failed_files: List[str] = []
    items = tqdm(wav_files, desc="DDSP inference") if tqdm else wav_files

    for wav_path in items:
        out_path = make_output_path(wav_path, input_dir, output_dir)

        if _is_valid_output(out_path):
            skipped += 1
            continue

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

            save_audio(out_path, full_audio, sr)
            processed += 1

        except Exception:
            failed += 1
            failed_files.append(str(wav_path))
            logger.warning("Failed: %s", wav_path, exc_info=True)

    logger.info(
        "DDSP done: %d processed, %d skipped, %d failed",
        processed, skipped, failed,
    )
    return {"processed": processed, "failed": failed, "skipped": skipped, "failed_files": failed_files}


# ---------------------------------------------------------------------------
# WORLD vocoder baseline pipeline
# ---------------------------------------------------------------------------


def _process_one_baseline(
    wav_path: Path,
    out_path: Path,
    source_bank: List[np.ndarray],
    sr: int,
    method: str,
    file_seed: int,
) -> Optional[str]:
    """Process a single file for the WORLD vocoder baseline.

    Returns ``None`` on success, or the failing file path as a string on error.
    This is a module-level function so it can be pickled by
    :class:`~concurrent.futures.ProcessPoolExecutor`.
    """
    try:
        bl = Baseline(sample_rate=sr)
        transfer_fn = bl.f0_transfer if method == "f0" else bl.f0_and_ap_transfer

        file_rng = np.random.default_rng(file_seed)

        target_audio, _ = load_audio(wav_path, sr=sr, mono=True)
        target_audio = target_audio.astype(np.float64)

        source_audio = select_source_segment(
            source_bank, len(target_audio), file_rng
        )
        target_audio, source_audio = Baseline.match_length(
            target_audio, source_audio
        )

        result = transfer_fn(target_audio, source_audio)

        save_audio(out_path, result.astype(np.float32), sr)
        return None
    except Exception:
        logger.warning("Failed: %s", wav_path, exc_info=True)
        return str(wav_path)


def run_vocoder_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    source_dir: str | Path,
    sr: int = DEFAULT_SAMPLE_RATE,
    method: str = "f0",
    silence_threshold_db: float = -40.0,
    seed: int = 42,
    n_workers: int = 1,
) -> PipelineResult:
    """Run WORLD vocoder baseline on every WAV file under *input_dir*.

    For each target file a random source segment is drawn from a "source bank"
    built from the solo instrument recordings in *source_dir*.

    Each file's source segment is selected using a deterministic per-file seed
    (``seed + file_index``) so that interrupted and resumed runs produce
    identical output.

    Files whose output already exists and is valid are skipped automatically,
    so re-running the function resumes from where it left off.

    Parameters
    ----------
    input_dir  : root directory of target WAV files (searched recursively).
    output_dir : root directory for output; subdirectory structure is preserved.
    source_dir : directory of source instrument WAV files (e.g. solo violin).
    sr : sample rate.
    method : ``"f0"`` for F0 transfer, ``"f0_ap"`` for F0 + AP transfer.
    silence_threshold_db : threshold for silence trimming in source audio.
    seed : base random seed; file at index *i* uses ``seed + i``.
    n_workers : number of parallel workers.  ``1`` (default) runs sequentially.
        Set to ``os.cpu_count()`` or a specific integer to parallelise via
        :class:`~concurrent.futures.ProcessPoolExecutor`.

    Returns
    -------
    PipelineResult
    """
    wav_files = collect_wav_files(input_dir)
    logger.info("Baseline pipeline: %d files in %s", len(wav_files), input_dir)

    source_bank = build_source_bank(source_dir, sr, silence_threshold_db)

    # --- Build work items, skipping already-completed files ----------------
    work: List[Dict[str, Any]] = []
    skipped = 0
    for file_idx, wav_path in enumerate(wav_files):
        out_path = make_output_path(wav_path, input_dir, output_dir, suffix="_BASELINE")
        if _is_valid_output(out_path):
            skipped += 1
            continue
        work.append({
            "wav_path": wav_path,
            "out_path": out_path,
            "file_seed": seed + file_idx,
        })

    logger.info(
        "Baseline: %d to process, %d skipped, %d workers",
        len(work), skipped, n_workers,
    )

    processed, failed = 0, 0
    failed_files: List[str] = []

    if n_workers <= 1:
        # ---- Sequential path (default) -----------------------------------
        items = tqdm(work, desc="Baseline inference") if tqdm else work
        for item in items:
            err = _process_one_baseline(
                item["wav_path"], item["out_path"],
                source_bank, sr, method, item["file_seed"],
            )
            if err is None:
                processed += 1
            else:
                failed += 1
                failed_files.append(err)
    else:
        # ---- Parallel path ------------------------------------------------
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _process_one_baseline,
                    item["wav_path"], item["out_path"],
                    source_bank, sr, method, item["file_seed"],
                ): item["wav_path"]
                for item in work
            }
            items = (
                tqdm(as_completed(futures), desc="Baseline inference", total=len(futures))
                if tqdm else as_completed(futures)
            )
            for future in items:
                err = future.result()
                if err is None:
                    processed += 1
                else:
                    failed += 1
                    failed_files.append(err)

    logger.info(
        "Baseline done: %d processed, %d skipped, %d failed",
        processed, skipped, failed,
    )
    return {"processed": processed, "failed": failed, "skipped": skipped, "failed_files": failed_files}


# ---------------------------------------------------------------------------
# Evaluation pipeline — reference-based comparison
# ---------------------------------------------------------------------------


def _evaluate_one_file_against_reference(
    file_path: Path,
    ref_2d: np.ndarray,
    feat_keys: List[str],
    rel_path: str,
    sr: int,
) -> Optional[Dict[str, Any]]:
    """Compare one synthesised file's feature distribution to the reference.

    Returns a dict row on success, or ``None`` on error.
    Module-level function for :class:`~concurrent.futures.ProcessPoolExecutor`.
    """
    try:
        tm = TimbreMetrics(sample_rate=sr)
        loss = Loss()

        series = tm.extract_series_from_file(file_path)
        if not series:
            logger.warning("Empty features: %s", rel_path)
            return None

        available = sorted(set(series.keys()) & set(feat_keys))
        if not available:
            logger.warning("No matching features: %s", rel_path)
            return None

        # Build column indices that match feat_keys ordering
        col_idx = [feat_keys.index(k) for k in available]
        file_2d = np.column_stack([series[k] for k in available])
        ref_subset = ref_2d[:, col_idx]

        distances = loss.evaluate(file_2d, ref_subset)

        return {"file": rel_path, **distances}

    except Exception:
        logger.warning("Pairwise evaluation failed: %s", rel_path, exc_info=True)
        return None


def run_evaluation_dir(
    reference_dir: str | Path,
    synthesized_dirs: Dict[str, str | Path],
    output_csv: str | Path,
    sr: int = DEFAULT_SAMPLE_RATE,
    n_workers: int = 1,
    max_frames: int = 10000,
) -> Dict[str, pd.DataFrame]:
    """Compare synthesised audio folders against a reference set.

    Uses the solo instrument recordings in *reference_dir* as the ground-truth
    timbre distribution.  For each method folder in *synthesized_dirs*, two
    comparison strategies are applied:

    **Strategy 1 — distribution-level**: concatenate all per-frame features
    from the folder into one array and compare against the concatenated
    reference.  Produces one row per method.

    **Strategy 2 — pairwise**: compare each individual file's feature
    distribution against the full reference.  Produces one row per file.

    Both strategies use :meth:`~evaluation.loss.Loss.evaluate` (MMD +
    Wasserstein).

    Resume behaviour
    ----------------
    Pairwise results are appended to ``{output_csv}_pairwise.csv``.  Files
    already present (keyed by ``(method, file)``) are skipped on re-run.

    Parameters
    ----------
    reference_dir : directory of reference instrument WAV files (e.g. solo
        violin).
    synthesized_dirs : mapping of method name to directory, e.g.
        ``{"ddsp": "path/to/ddsp", "baseline": "path/to/baseline"}``.
    output_csv : base path for output CSVs.  Two files are written:
        ``{stem}_distribution.csv`` and ``{stem}_pairwise.csv``.
    sr : sample rate.
    n_workers : number of parallel workers for feature extraction and
        pairwise evaluation.
    max_frames : maximum frames to use for Strategy 1 distribution
        comparison (sub-sampled for computational efficiency, since MMD is
        O(n^2)).

    Returns
    -------
    dict
        ``{"distribution": pd.DataFrame, "pairwise": pd.DataFrame}``
    """
    output_csv = Path(output_csv)
    dist_csv = output_csv.parent / f"{output_csv.stem}_distribution.csv"
    pair_csv = output_csv.parent / f"{output_csv.stem}_pairwise.csv"

    # ------------------------------------------------------------------
    # 1. Build reference feature set
    # ------------------------------------------------------------------
    logger.info("Building reference feature set from %s", reference_dir)
    tm = TimbreMetrics(sample_rate=sr)
    ref_features = tm.extract_from_dir(reference_dir, n_workers=n_workers)
    feat_keys = sorted(ref_features.keys())
    ref_2d = np.column_stack([ref_features[k] for k in feat_keys])

    logger.info(
        "Reference: %d frames, %d features", ref_2d.shape[0], ref_2d.shape[1],
    )

    # ------------------------------------------------------------------
    # 2. Resume: load existing pairwise results
    # ------------------------------------------------------------------
    done_pairs: set[tuple[str, str]] = set()
    existing_pair_rows: List[Dict[str, Any]] = []
    if pair_csv.exists():
        df_existing = pd.read_csv(pair_csv)
        done_pairs = set(
            zip(df_existing["method"].tolist(), df_existing["file"].tolist())
        )
        existing_pair_rows = df_existing.to_dict("records")
        logger.info("Resuming pairwise: %d rows in %s", len(done_pairs), pair_csv)

    dist_rows: List[Dict[str, Any]] = []
    new_pair_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(42)
    loss = Loss()

    for method_name, synth_dir in synthesized_dirs.items():
        logger.info("Evaluating method '%s' from %s", method_name, synth_dir)

        # --------------------------------------------------------------
        # Strategy 1 — distribution-level comparison
        # --------------------------------------------------------------
        synth_features = tm.extract_from_dir(synth_dir, n_workers=n_workers)
        synth_2d = np.column_stack([synth_features[k] for k in feat_keys])

        # Sub-sample for computational efficiency
        ref_sub = ref_2d
        synth_sub = synth_2d
        if ref_sub.shape[0] > max_frames:
            idx = rng.choice(ref_sub.shape[0], max_frames, replace=False)
            ref_sub = ref_sub[idx]
        if synth_sub.shape[0] > max_frames:
            idx = rng.choice(synth_sub.shape[0], max_frames, replace=False)
            synth_sub = synth_sub[idx]

        dist_result = loss.evaluate(synth_sub, ref_sub)
        dist_rows.append({
            "method": method_name,
            "n_files": len(collect_wav_files(synth_dir)),
            "total_frames": synth_2d.shape[0],
            **dist_result,
        })
        logger.info(
            "Strategy 1 [%s]: mmd=%.6f, wasserstein=%.6f",
            method_name, dist_result["mmd"], dist_result["wasserstein"],
        )

        # --------------------------------------------------------------
        # Strategy 2 — pairwise file-vs-reference comparison
        # --------------------------------------------------------------
        synth_dir_path = Path(synth_dir)
        wav_files = collect_wav_files(synth_dir_path)
        work = []
        skipped = 0
        for wf in wav_files:
            rel = str(wf.relative_to(synth_dir_path))
            if (method_name, rel) in done_pairs:
                skipped += 1
                continue
            work.append({"path": wf, "rel": rel})

        logger.info(
            "Strategy 2 [%s]: %d to process, %d skipped",
            method_name, len(work), skipped,
        )

        processed, failed = 0, 0
        if n_workers <= 1:
            items = tqdm(work, desc=f"Pairwise [{method_name}]") if tqdm else work
            for item in items:
                row = _evaluate_one_file_against_reference(
                    item["path"], ref_2d, feat_keys, item["rel"], sr,
                )
                if row is not None:
                    row["method"] = method_name
                    new_pair_rows.append(row)
                    processed += 1
                else:
                    failed += 1
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        _evaluate_one_file_against_reference,
                        item["path"], ref_2d, feat_keys, item["rel"], sr,
                    ): item
                    for item in work
                }
                items = (
                    tqdm(
                        as_completed(futures),
                        desc=f"Pairwise [{method_name}]",
                        total=len(futures),
                    )
                    if tqdm else as_completed(futures)
                )
                for future in items:
                    row = future.result()
                    if row is not None:
                        row["method"] = method_name
                        new_pair_rows.append(row)
                        processed += 1
                    else:
                        failed += 1

        logger.info(
            "Strategy 2 [%s] done: %d processed, %d skipped, %d failed",
            method_name, processed, skipped, failed,
        )

    # ------------------------------------------------------------------
    # 3. Save results
    # ------------------------------------------------------------------
    df_dist = pd.DataFrame(dist_rows)
    if not df_dist.empty:
        dist_csv.parent.mkdir(parents=True, exist_ok=True)
        df_dist.to_csv(dist_csv, index=False)
        logger.info("Distribution results saved → %s", dist_csv)

    all_pair_rows = existing_pair_rows + new_pair_rows
    df_pair = pd.DataFrame(all_pair_rows)
    if not df_pair.empty:
        pair_csv.parent.mkdir(parents=True, exist_ok=True)
        df_pair.to_csv(pair_csv, index=False)
        logger.info("Pairwise results saved → %s", pair_csv)

    return {"distribution": df_dist, "pairwise": df_pair}
