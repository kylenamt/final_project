# Codebase Architecture

## Overview

This project trains a DDSP autoencoder on a target instrument's audio. At inference
time, pitch (F0) and loudness are extracted from a source signal and fed through the
trained decoder to resynthesize the audio in the target instrument's timbre. Two
deterministic vocoder baselines (WORLD and SMS/HPS) are provided for comparison.

```
Raw audio (.wav)
  |
  v
[data_preprocessing.py]  clip silence, resample to 16 kHz, split into 4 s segments
  |
  v
Preprocessed .wav files
  |
  v
[ddsp_prepare_tfrecord]   extract F0 + loudness, write TFRecord shards
  |
  v
TFRecord dataset  ──────>  [ddsp_run --mode=train]  ──>  Checkpoint (artifacts/)
                                                              |
                                                              v
Source audio  ──>  [feature_utils.py]  ──>  [timbre_transfer.py]  ──>  Transferred audio
                   extract F0, loudness       load model, resynthesize
                                                              |
                                                              v
                                              [evaluation/]  segment, extract timbre
                                                             features, compute MMD /
                                                             Wasserstein distances
```

---

## Pipeline Stages

### 1. Data Preprocessing (`src/data_preprocessing.py`)

Prepares raw recordings for TFRecord generation.

| Function | Purpose |
|----------|---------|
| `clip_silence(in_path, threshold_db=-40.0)` | Trim leading/trailing silence via frame-RMS detection |
| `downsample_audio(in_path, out_path, target_sr=16000)` | Resample with polyphase filtering (`scipy.signal.resample_poly`) |
| `split_audio(in_path, out_dir, n_seconds)` | Fixed-length segmentation; drops remainder shorter than `n_seconds` |

Typical flow: raw WAV &rarr; silence-trimmed &rarr; 16 kHz mono &rarr; 4-second segments &rarr; `ddsp_prepare_tfrecord`.

### 2. Audio Utilities (`src/utils.py`)

Low-level audio I/O shared across the codebase.

| Function | Purpose |
|----------|---------|
| `load_audio(path, sr, mono, normalize)` | Load via librosa with optional resample and normalization |
| `save_audio(path, audio, sr)` | Write via soundfile; auto-creates parent directories |
| `to_mono(audio)` | Average channels to mono |
| `normalize_audio(audio, peak=0.99)` | Peak-normalize to target amplitude |
| `clip_audio(audio, end_time, start_time, sr)` | Extract a time range in seconds |

### 3. Feature Extraction (`src/feature_utils.py`)

Extracts and manipulates the conditioning signals (F0, loudness) that drive DDSP synthesis.

**Core types and functions:**

- `AudioFeatures = Dict[str, Any]` &mdash; dict with keys `f0_hz`, `f0_confidence`, `loudness_db`, `audio`.
- `compute_features(audio)` &mdash; extract F0 (CREPE) and loudness via `ddsp.training.metrics`.
- `compute_alignment(audio)` &mdash; derive `(time_steps, n_samples, hop_size)` from the active gin config.

**Feature manipulation:**

| Function | Purpose |
|----------|---------|
| `shift_f0(features, octaves)` | Shift pitch by N octaves, clamped to `[0, ~14 kHz]` |
| `shift_loudness(features, dB)` | Offset loudness by a fixed amount |
| `trim_features(features, time_steps, n_samples)` | Truncate arrays to model dimensions |

**Auto-adjustment** (`auto_adjust_features`):

1. Shift F0 by the nearest whole octave to match the target dataset's mean pitch.
2. Normalize loudness via the dataset's quantile transform.
3. Optionally auto-tune to the nearest major-scale note.

Relies on `dataset_statistics.pkl` saved during training (loaded by `load_dataset_stats`).

### 4. Model Loading (`src/model_loading.py`)

Discovers checkpoints and gin configs, then restores a `ddsp.training.models.Autoencoder`.

| Function | Purpose |
|----------|---------|
| `find_model_dir(path)` | Walk directories to find one containing `.gin` files |
| `find_latest_checkpoint(model_dir)` | Return `ckpt-<max_step>` prefix from `.index` files |
| `restore_autoencoder(model_dir)` | Instantiate and restore model weights |
| `load_ddsp_model(model_path)` | Full pipeline: discover &rarr; parse gin &rarr; restore &rarr; return dict |
| `load_models(model_paths)` | Batch-load multiple models keyed by directory basename |

**Gin file resolution** (`_select_gin_file`):

1. Explicit `gin_file` argument.
2. `model_path` itself if it ends in `.gin`.
3. `operative_config-<highest_step>.gin`.
4. First `.gin` alphabetically.

**Pretrained models** (from Google Cloud Storage):

`Violin`, `Flute`, `Flute2`, `Trumpet`, `Tenor_Saxophone` &mdash; downloaded via
`load_pretrained_model(name)` and cached under `artifacts/pretrained/`.

### 5. Timbre Transfer Inference (`src/timbre_transfer.py`)

Orchestrates the full inference pipeline, usable as a CLI or imported as a module.

- `load_model(model_dir, gin_file, audio, audio_features)` &mdash; parse gin, override
  `n_samples` / `time_steps` for the input length, restore weights.
- `resynthesize(model, audio_features)` &mdash; forward pass &rarr; extract generated audio.

**CLI usage:**

```bash
python src/timbre_transfer.py \
  --audio-path input.wav \
  --model Violin \
  --pitch-shift 0.0 \
  --loudness-shift 0.0 \
  --auto-adjust 1 \
  --autotune 0.0
```

Key flags: `--model` (pretrained name or local checkpoint path), `--pitch-shift` (octaves),
`--loudness-shift` (dB), `--auto-adjust` (match training data stats), `--threshold`,
`--quiet`, `--autotune` (0&ndash;1), `--start-sec` / `--end-sec` (clip input).

### 6. Baseline Methods (`src/baseline.py`)

Two deterministic vocoder baselines, both offering the same transfer API.

#### `Baseline` &mdash; WORLD vocoder

Decomposes audio into F0 (pitch), spectral envelope (SP), and aperiodicity (AP),
then recombines components from source and target signals.

| Method | Components kept from target | Components from source |
|--------|---------------------------|----------------------|
| `f0_transfer` | F0 | SP + AP |
| `f0_and_ap_transfer` | F0 + AP | SP |

#### `BaselineSMS` &mdash; SMS/HPS vocoder

Drop-in replacement using harmonic + stochastic decomposition instead of WORLD.
Requires the optional `sms_tools` package. Adds an `alpha` parameter for partial
blending between source and target components.

Both classes provide `decompose(y)` &rarr; `(f0, sp, ap)` and `synthesize(f0, sp, ap, fs)`.

### 7. Visualization (`src/visualize.py`)

Plotting helpers used primarily in notebooks.

| Function | Purpose |
|----------|---------|
| `plot_spectrograms(audios, fs)` | STFT log-magnitude spectrograms for a list of signals |
| `plot_features(audio_features)` | Loudness, F0 (MIDI), and confidence time-series |
| `plot_feature_from_audio(audio, sr)` | Extract features then plot |
| `plot_series(feature_series)` | Single feature trajectory |
| `plot_transfer_comparison(src, tgt, out, fs)` | 3-panel source / target / output spectrograms |
| `plot_envelope_comparison(src_analysis, tgt_analysis, fs, N)` | Overlay source and target mean spectral envelopes |

---

## Evaluation Subpackage (`src/evaluation/`)

See [evaluation.md](evaluation.md) for the full reference. In brief:

- **`AudioSegmenter`** &mdash; non-overlapping segmentation (BPM-driven or fixed-length).
- **`TimbreMetrics`** &mdash; frame-based spectral feature extraction via pytimbre.
- **`Loss`** &mdash; distributional distances (MMD and Wasserstein) between feature arrays.

---

## Gin Configuration System

DDSP uses [gin-config](https://github.com/google/gin-config) for model, dataset, and
training hyperparameters. All gin files live under `configs/ddsp_gin/`.

### Models

**`models/ae.gin`** &mdash; Full autoencoder (encoder + decoder, no reverb):

| Component | Setting |
|-----------|---------|
| Encoder | `MfccTimeDistributedRnnEncoder`: 512-ch GRU, z_dims=16, z_time_steps=125 |
| Decoder | `RnnFcDecoder`: 512-ch GRU, 3 layers/stack, inputs=(ld_scaled, f0_scaled, z) |
| Outputs | amps (1), harmonic_distribution (100), noise_magnitudes (65) |
| Synthesizers | `Harmonic` + `FilteredNoise` &rarr; `Add` |
| Loss | Spectral L1 + log-magnitude (equal weight) |
| Audio | 16 kHz, 64000 samples (4 s), 1000 time steps |

**`models/solo_instrument.gin`** &mdash; Decoder-only with trainable reverb:

- Extends `ae.gin` but sets `Autoencoder.encoder = None`.
- Decoder inputs: `(ld_scaled, f0_scaled)` only (no latent z).
- Adds `Reverb` processor (trainable, reverb_length=48000).
- Harmonic distribution reduced from 100 to 60 components.

### Datasets

| File | Description |
|------|-------------|
| `datasets/base.gin` | Common settings (eval batch=32, sample batch=16) |
| `datasets/tfrecord.gin` | `TFRecordProvider` for sharded datasets |
| `datasets/nsynth.gin` | NSynth dataset format |

### Evaluation

| File | Evaluators |
|------|-----------|
| `eval/basic_f0_ld.gin` | BasicEvaluator + F0LdEvaluator (default) |
| `eval/basic_f0_ld_twm.gin` | BasicEvaluator + F0LdEvaluator + ToneWeightedMfccEvaluator |
| `eval/basic.gin` | BasicEvaluator only |

### Optimization

`optimization/base.gin` &mdash; standard training hyperparameters (learning rate, steps, etc.).

---

## Training & Orchestration

### Makefile presets

Presets are defined in `configs/training_config/preset.mk` as whitespace-delimited rows:

```
# name              SAVE_DIR                    MODEL_DIR                   PATH_IN_REPO
ae                  artifacts/ae/               artifacts/ae/               ae
solo_instrument     artifacts/solo_instrument/  artifacts/solo_instrument/  solo_instrument
```

Usage: `make train ae` or `make train solo_instrument` sets `SAVE_DIR`, `MODEL_DIR`,
and `PATH_IN_REPO` automatically.

### `scripts/train_ddsp.sh`

Bash wrapper around `ddsp_run`. Key behavior:

1. Sets `LD_LIBRARY_PATH` to `$CONDA_PREFIX/lib` so TF finds CUDA libraries.
2. Constructs gin search paths and parameter flags from environment variables.
3. Calls `ddsp_run --mode={train,eval,sample}` with the assembled flags.

### `scripts/upload_to_hf.py`

Uploads the latest checkpoint (`.index` + `.data`) and all `.gin` configs to a
Hugging Face Hub repository. Finds the highest-step checkpoint automatically.

```bash
make upload-hf HF_REPO=username/repo MODEL_DIR=artifacts/ae/ PATH_IN_REPO=ae
```

---

## Notebooks (`src/demo/`)

| Notebook | Purpose |
|----------|---------|
| `timbre_transfer.ipynb` | End-to-end DDSP timbre transfer demo |
| `baseline_demo.ipynb` | WORLD / SMS baseline comparison |
| `timbre_metrics_demo.ipynb` | Timbre feature extraction and visualization |
| `run_preprocessing.ipynb` | Data preprocessing pipeline (trim, resample, split) |
| `playground.ipynb` | Experimental sandbox |
| `check_gpu.ipynb` | GPU / CUDA / TensorFlow verification |

---

## Dependency Graph

```
utils.py
  ^
  |--- data_preprocessing.py
  |--- baseline.py        (+ pyworld, sms_tools[opt])
  |--- visualize.py       (+ matplotlib)

feature_utils.py          (+ ddsp, tensorflow)
  ^
  |--- timbre_transfer.py (+ model_loading.py)
  |--- visualize.py

evaluation/
  |- loss.py              (numpy, scipy)
  |- segment.py           (librosa, numpy)
  |- timbre_metrics.py    (pytimbre, numpy)
```
