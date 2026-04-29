"""Batch inference: DDSP timbre transfer, WORLD vocoder baseline, and evaluation.

Provides three main functions:
- ``synthesize_dir``  – DDSP timbre transfer over a directory of WAV files.
- ``vocode_dir``      – WORLD vocoder baseline over the same directory using
                        a randomly-sampled "source bank" of solo instrument audio.
- ``evaluate_dir``    – Distributional and pairwise timbre loss computation
                        from pre-computed feature parquet files.

Resume behaviour
----------------
All functions skip files whose output already exists and is valid.
To pause mid-run just interrupt the kernel (Ctrl-C / Kernel > Interrupt).
Already-written files / CSV rows are preserved and will be skipped automatically
on the next run — no separate checkpoint file is needed.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd

from data_preprocessing import trim_silence
from .loss import Loss
from utils import load_audio, save_audio
from feature_utils import (
    auto_adjust_features,
    compute_features,
    load_dataset_stats,
    shift_f0,
    shift_loudness,
)
from timbre_transfer import TimbreTransfer
from utils import DEFAULT_SAMPLE_RATE
from baseline import Baseline, BaselineSMS

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
    "synthesize_dir",
    "vocode_dir",
    "vocode_dir_sms",
    "load_features",
    "evaluate_dir",
]


# ---------------------------------------------------------------------------
# Typed return values
# ---------------------------------------------------------------------------

class PipelineResult(TypedDict):
    """Return type for ``synthesize_dir`` and ``vocode_dir``."""
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


def synthesize_dir(
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
    model_dir = str(model_dir)
    gin_file = str(gin_file)
    wav_files = collect_wav_files(input_dir)
    logger.info("DDSP pipeline: %d files in %s", len(wav_files), input_dir)

    # --- Load model once ---
    tt = TimbreTransfer(model_dir, gin_file)
    tt.load()
    time_steps = tt.time_steps
    n_samples = tt.n_samples

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

            full_audio = tt.synthesize(features)[:original_len]

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
    use_crepe: bool = False,
    precomputed_f0: Optional[np.ndarray] = None,
) -> Optional[str]:
    """Process a single file for the WORLD vocoder baseline.

    Returns ``None`` on success, or the failing file path as a string on error.
    This is a module-level function so it can be pickled by
    :class:`~concurrent.futures.ProcessPoolExecutor`.

    If *precomputed_f0* is provided the worker skips the CREPE call and
    passes the F0 array straight into the WORLD synthesis step — used when
    the main process pre-runs CREPE on the GPU so that workers stay
    CPU-only and can be parallelised safely.
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

        result = transfer_fn(
            target_audio,
            source_audio,
            use_crepe=use_crepe and precomputed_f0 is None,
            target_f0=precomputed_f0,
        )

        save_audio(out_path, result.astype(np.float32), sr)
        return None
    except Exception:
        logger.warning("Failed: %s", wav_path, exc_info=True)
        return str(wav_path)


def _precompute_crepe_f0(
    wav_paths: List[Path],
    sr: int,
) -> Dict[str, np.ndarray]:
    """Run CREPE sequentially on every target file and return its F0 contour.

    Runs in the main process so there is exactly one TensorFlow / GPU
    context.  The resulting dict is keyed by ``str(wav_path)`` so workers
    can look up their file's F0 array without re-entering CREPE.
    """
    bl = Baseline(sample_rate=sr)
    out: Dict[str, np.ndarray] = {}
    items = tqdm(wav_paths, desc="CREPE F0 precompute") if tqdm else wav_paths
    for wav_path in items:
        try:
            audio, _ = load_audio(wav_path, sr=sr, mono=True)
            f0 = bl.crepe_f0(audio.astype(np.float64))[0]
            out[str(wav_path)] = np.asarray(f0, dtype=np.float64)
        except Exception:
            logger.warning("CREPE precompute failed: %s", wav_path, exc_info=True)
    return out


def vocode_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    source_dir: str | Path,
    sr: int = DEFAULT_SAMPLE_RATE,
    method: str = "f0_ap",
    silence_threshold_db: float = -40.0,
    seed: int = 42,
    n_workers: int = 1,
    use_crepe: bool = False,
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
    use_crepe : if ``True``, estimate the target F0 contour with CREPE instead
        of WORLD's built-in Harvest/Stonemask estimator.  Passed through to
        :meth:`Baseline.f0_transfer` / :meth:`Baseline.f0_and_ap_transfer`.

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
            "precomputed_f0": None,
        })

    # --- Phase 1: precompute CREPE F0 on the main process (GPU) ------------
    # Parallel CREPE workers would each load their own TF graph and fight
    # over GPU memory.  Running it once here and handing the F0 contours
    # off to CPU workers keeps phase 2 safely parallelisable.
    if use_crepe and work:
        logger.info(
            "Precomputing CREPE F0 for %d files (sequential, main process)",
            len(work),
        )
        crepe_f0 = _precompute_crepe_f0([item["wav_path"] for item in work], sr)
        for item in work:
            item["precomputed_f0"] = crepe_f0.get(str(item["wav_path"]))

    logger.info(
        "Baseline: %d to process, %d skipped, %d workers, use_crepe=%s",
        len(work), skipped, n_workers, use_crepe,
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
                use_crepe, item["precomputed_f0"],
            )
            if err is None:
                processed += 1
            else:
                failed += 1
                failed_files.append(err)
    else:
        # ---- Parallel path ------------------------------------------------
        # When use_crepe=True the main process has already initialised
        # TensorFlow / CUDA.  Forking workers from that state is unsafe
        # (can hang or corrupt the CUDA context), so force "spawn" which
        # starts each worker from a clean Python.
        pool_ctx = mp.get_context("spawn") if use_crepe else None
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=pool_ctx) as pool:
            futures = {
                pool.submit(
                    _process_one_baseline,
                    item["wav_path"], item["out_path"],
                    source_bank, sr, method, item["file_seed"],
                    use_crepe, item["precomputed_f0"],
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
# SMS/HPS baseline pipeline
# ---------------------------------------------------------------------------


def _process_one_sms(
    wav_path: Path,
    out_path: Path,
    source_bank: List[np.ndarray],
    sr: int,
    method: str,
    file_seed: int,
    alpha: float,
    sms_kwargs: Dict[str, Any],
    use_crepe: bool = False,
    precomputed_f0: Optional[np.ndarray] = None,
) -> Optional[str]:
    """Process a single file for the SMS/HPS baseline.

    Returns ``None`` on success, or the failing file path as a string on error.
    This is a module-level function so it can be pickled by
    :class:`~concurrent.futures.ProcessPoolExecutor`.

    If *precomputed_f0* is provided (raw CREPE contour at ~5 ms hop), the
    worker resamples it to the HPS frame rate before passing it into
    the transfer method.
    """
    try:
        bl = BaselineSMS(sample_rate=sr, **sms_kwargs)
        transfer_fn = bl.f0_transfer if method == "f0" else bl.f0_and_ap_transfer

        file_rng = np.random.default_rng(file_seed)

        target_audio, _ = load_audio(wav_path, sr=sr, mono=True)
        target_audio = target_audio.astype(np.float64)

        source_audio = select_source_segment(
            source_bank, len(target_audio), file_rng
        )
        target_audio, source_audio = BaselineSMS.match_length(
            target_audio, source_audio
        )

        # Resample precomputed CREPE F0 to approximate HPS frame count
        f0_override = None
        if precomputed_f0 is not None:
            n_hps_est = len(target_audio) // bl._H + 1
            idx = np.round(
                np.linspace(0, len(precomputed_f0) - 1, n_hps_est)
            ).astype(int)
            f0_override = precomputed_f0[idx].astype(np.float64)

        result = transfer_fn(
            target_audio,
            source_audio,
            alpha=alpha,
            use_crepe=use_crepe and precomputed_f0 is None,
            target_f0=f0_override,
        )

        save_audio(out_path, result.astype(np.float32), sr)
        return None
    except Exception:
        logger.warning("Failed: %s", wav_path, exc_info=True)
        return str(wav_path)


def vocode_dir_sms(
    input_dir: str | Path,
    output_dir: str | Path,
    source_dir: str | Path,
    sr: int = DEFAULT_SAMPLE_RATE,
    method: str = "f0_ap",
    alpha: float = 1.0,
    silence_threshold_db: float = -40.0,
    seed: int = 42,
    n_workers: int = 1,
    use_crepe: bool = False,
    **sms_kwargs,
) -> PipelineResult:
    """Run SMS/HPS baseline on every WAV file under *input_dir*.

    Same interface as :func:`vocode_dir` but uses :class:`BaselineSMS`
    (Harmonic Plus Stochastic model from ``sms-tools``) instead of the
    WORLD-based :class:`Baseline`.

    Files whose output already exists and is valid are skipped automatically,
    so re-running the function resumes from where it left off.

    Parameters
    ----------
    input_dir  : root directory of target WAV files (searched recursively).
    output_dir : root directory for output; subdirectory structure is preserved.
    source_dir : directory of source instrument WAV files (e.g. solo violin).
    sr : sample rate.
    method : ``"f0"`` for F0 transfer, ``"f0_ap"`` for F0 + AP transfer.
    alpha : blend strength 0 (no transfer) → 1 (full transfer).
    silence_threshold_db : threshold for silence trimming in source audio.
    seed : base random seed; file at index *i* uses ``seed + i``.
    n_workers : number of parallel workers.  ``1`` (default) runs sequentially.
    use_crepe : if ``True``, estimate F0 with CREPE (main process) and
        distribute precomputed contours to workers.
    **sms_kwargs : forwarded to :class:`BaselineSMS`
        (e.g. ``window``, ``M``, ``N``, ``H``, ``Ns``, ``stocf``, ``t``, …).

    Returns
    -------
    PipelineResult
    """
    wav_files = collect_wav_files(input_dir)
    logger.info("SMS pipeline: %d files in %s", len(wav_files), input_dir)

    source_bank = build_source_bank(source_dir, sr, silence_threshold_db)

    # --- Build work items, skipping already-completed files ----------------
    work: List[Dict[str, Any]] = []
    skipped = 0
    for file_idx, wav_path in enumerate(wav_files):
        out_path = make_output_path(wav_path, input_dir, output_dir, suffix="_SMS")
        if _is_valid_output(out_path):
            skipped += 1
            continue
        work.append({
            "wav_path": wav_path,
            "out_path": out_path,
            "file_seed": seed + file_idx,
            "precomputed_f0": None,
        })

    # --- Phase 1: precompute CREPE F0 on the main process (GPU) -----------
    if use_crepe and work:
        logger.info(
            "Precomputing CREPE F0 for %d files (sequential, main process)",
            len(work),
        )
        crepe_f0 = _precompute_crepe_f0([item["wav_path"] for item in work], sr)
        for item in work:
            item["precomputed_f0"] = crepe_f0.get(str(item["wav_path"]))

    logger.info(
        "SMS: %d to process, %d skipped, %d workers, use_crepe=%s",
        len(work), skipped, n_workers, use_crepe,
    )

    processed, failed = 0, 0
    failed_files: List[str] = []

    if n_workers <= 1:
        # ---- Sequential path (default) -----------------------------------
        items = tqdm(work, desc="SMS inference") if tqdm else work
        for item in items:
            err = _process_one_sms(
                item["wav_path"], item["out_path"],
                source_bank, sr, method, item["file_seed"],
                alpha, sms_kwargs, use_crepe, item["precomputed_f0"],
            )
            if err is None:
                processed += 1
            else:
                failed += 1
                failed_files.append(err)
    else:
        # ---- Parallel path ------------------------------------------------
        pool_ctx = mp.get_context("spawn") if use_crepe else None
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=pool_ctx) as pool:
            futures = {
                pool.submit(
                    _process_one_sms,
                    item["wav_path"], item["out_path"],
                    source_bank, sr, method, item["file_seed"],
                    alpha, sms_kwargs, use_crepe, item["precomputed_f0"],
                ): item["wav_path"]
                for item in work
            }
            items = (
                tqdm(as_completed(futures), desc="SMS inference", total=len(futures))
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
        "SMS done: %d processed, %d skipped, %d failed",
        processed, skipped, failed,
    )
    return {"processed": processed, "failed": failed, "skipped": skipped, "failed_files": failed_files}


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def load_features(path: str | Path) -> pd.DataFrame:
    """Load pre-computed features from a parquet file.

    The parquet is expected to contain a ``filename`` column (identifying the
    source audio file for each frame) plus one or more numeric feature columns
    (e.g. ``spectral_centroid``, ``spectral_spread``, …).

    These files are produced by the *feature_extraction* notebook.
    """
    return pd.read_parquet(Path(path))


# ---------------------------------------------------------------------------
# Evaluation pipeline — pre-computed features
# ---------------------------------------------------------------------------


def evaluate_dir(
    ref_features: pd.DataFrame | str | Path,
    synth_features: Dict[str, pd.DataFrame | str | Path],
    output_csv: str | Path,
    feature_cols: Optional[List[str]] = None,
    max_frames: int = 10_000,
) -> Dict[str, pd.DataFrame]:
    """Compare pre-computed feature distributions against a reference.

    Accepts DataFrames (or paths to parquet files) produced by the
    *feature_extraction* notebook.  No audio loading or feature extraction
    is performed here.

    Two comparison strategies are applied for each method:

    **Strategy 1 — distribution-level**: compare the full concatenated
    feature matrix of each method against the reference.  One row per method.

    **Strategy 2 — pairwise**: compare each individual file's frames against
    the full reference distribution.  One row per file.

    Both strategies use :class:`~evaluation.loss.Loss` (MMD + Wasserstein).

    Parameters
    ----------
    ref_features : DataFrame or path to parquet
        Reference feature set.  Must have a ``filename`` column and numeric
        feature columns.
    synth_features : dict
        ``{method_name: DataFrame_or_path}`` for each synthesis method.
    output_csv : path
        Base path for output CSVs.  Two files are written:
        ``{stem}_distribution.csv`` and ``{stem}_pairwise.csv``.
    feature_cols : list of str, optional
        Feature columns to compare.  If *None*, all numeric columns
        (excluding ``filename``) present in **both** the reference and
        the first synth DataFrame are used.
    max_frames : int
        Cap for distribution-level comparison (sub-sampled because MMD is
        O(n²)).

    Returns
    -------
    dict
        ``{"distribution": pd.DataFrame, "pairwise": pd.DataFrame}``
    """
    output_csv = Path(output_csv)
    dist_csv = output_csv.parent / f"{output_csv.stem}_distribution.csv"
    pair_csv = output_csv.parent / f"{output_csv.stem}_pairwise.csv"

    # ------------------------------------------------------------------
    # 1. Load DataFrames if paths were given
    # ------------------------------------------------------------------
    if not isinstance(ref_features, pd.DataFrame):
        ref_features = load_features(ref_features)

    loaded_synth: Dict[str, pd.DataFrame] = {}
    for name, data in synth_features.items():
        if isinstance(data, pd.DataFrame):
            loaded_synth[name] = data
        else:
            loaded_synth[name] = load_features(data)

    # ------------------------------------------------------------------
    # 2. Determine feature columns
    # ------------------------------------------------------------------
    meta_cols = {"filename"}
    if feature_cols is None:
        feature_cols = sorted(
            c for c in ref_features.columns
            if c not in meta_cols
            and ref_features[c].dtype.kind in ("f", "i")
        )

    logger.info("Evaluating %d features: %s", len(feature_cols), feature_cols)

    ref_2d = ref_features[feature_cols].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(42)
    loss = Loss()

    dist_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    for method_name, synth_df in loaded_synth.items():
        logger.info("Evaluating method '%s'", method_name)

        # Only use features present in both ref and synth
        available = sorted(set(feature_cols) & set(synth_df.columns))
        if not available:
            logger.warning(
                "No common features for method '%s', skipping", method_name,
            )
            continue

        col_idx = [feature_cols.index(c) for c in available]
        synth_2d = synth_df[available].to_numpy(dtype=np.float64)
        ref_subset = ref_2d[:, col_idx]

        # --------------------------------------------------------------
        # Strategy 1 — distribution-level comparison
        # --------------------------------------------------------------
        ref_sub = ref_subset
        synth_sub = synth_2d
        if ref_sub.shape[0] > max_frames:
            idx = rng.choice(ref_sub.shape[0], max_frames, replace=False)
            ref_sub = ref_sub[idx]
        if synth_sub.shape[0] > max_frames:
            idx = rng.choice(synth_sub.shape[0], max_frames, replace=False)
            synth_sub = synth_sub[idx]

        dist_result = loss.evaluate(synth_sub, ref_sub)
        n_files = (
            synth_df["filename"].nunique()
            if "filename" in synth_df.columns
            else 0
        )
        dist_rows.append({
            "method": method_name,
            "n_files": n_files,
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
        if "filename" not in synth_df.columns:
            logger.warning(
                "No 'filename' column for method '%s', skipping pairwise",
                method_name,
            )
            continue

        groups = synth_df.groupby("filename")
        items = (
            tqdm(groups, desc=f"Pairwise [{method_name}]")
            if tqdm
            else groups
        )
        processed, failed = 0, 0
        for fname, group in items:
            try:
                file_2d = group[available].to_numpy(dtype=np.float64)
                if file_2d.shape[0] < 2:
                    continue
                distances = loss.evaluate(file_2d, ref_subset)
                pair_rows.append({
                    "method": method_name,
                    "file": fname,
                    **distances,
                })
                processed += 1
            except Exception:
                failed += 1
                logger.warning(
                    "Pairwise failed: %s/%s", method_name, fname, exc_info=True,
                )

        logger.info(
            "Strategy 2 [%s] done: %d processed, %d failed",
            method_name, processed, failed,
        )

    # ------------------------------------------------------------------
    # 3. Save results
    # ------------------------------------------------------------------
    df_dist = pd.DataFrame(dist_rows)
    if not df_dist.empty:
        dist_csv.parent.mkdir(parents=True, exist_ok=True)
        df_dist.to_csv(dist_csv, index=False)
        logger.info("Distribution results saved → %s", dist_csv)

    df_pair = pd.DataFrame(pair_rows)
    if not df_pair.empty:
        pair_csv.parent.mkdir(parents=True, exist_ok=True)
        df_pair.to_csv(pair_csv, index=False)
        logger.info("Pairwise results saved → %s", pair_csv)

    return {"distribution": df_dist, "pairwise": df_pair}
