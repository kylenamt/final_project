# Evaluation Pipeline

## Overview

The evaluation pipeline quantifies how well timbre transfer preserves the target
instrument's character. It works by comparing spectral feature distributions between
original instrument audio and transferred audio using statistical distance metrics.

```
Audio signal (original or transferred)
  |
  v
[AudioSegmenter]   split into non-overlapping segments
  |
  v
Per-segment arrays
  |
  v
[TimbreMetrics]    extract per-frame spectral features, aggregate to one scalar per metric
  |
  v
Feature vectors  (n_segments x n_features)
  |
  v
[Loss]             compute MMD and/or Wasserstein distance between two feature matrices
  |
  v
Scalar distance    (lower = more similar timbres)
```

All three classes are exported from `src/evaluation/__init__.py`:

```python
from evaluation import Loss, AudioSegmenter, TimbreMetrics
```

---

## Audio Segmentation (`src/evaluation/segment.py`)

`AudioSegmenter` splits a mono signal into non-overlapping windows of equal length.

### Configuration

```python
segmenter = AudioSegmenter(sr=16000, bpm=120, beats=1)  # BPM-driven
segmenter = AudioSegmenter(sr=16000, L=16000)            # fixed-length (1 second)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sr` | (required) | Sample rate |
| `bpm` | `None` | Beats per minute; estimated from signal if omitted |
| `beats` | `1` | Number of beats per segment |
| `L` | `None` | Fixed segment length in samples (overrides BPM if set) |

When `bpm` is `None` and `L` is `None`, tempo is estimated via `librosa.beat.beat_track`
during `fit()`.

### API

```python
segments = segmenter.fit_transform(signal)  # list of np.ndarray

# Or step by step:
segmenter.fit(signal)
segments = segmenter.transform(signal)

# Properties (available after fit):
segmenter.n_segments   # number of complete segments
segmenter.tail_samples # samples discarded at the end
```

The API follows the sklearn `fit` / `transform` / `fit_transform` pattern.

---

## Timbre Feature Extraction (`src/evaluation/timbre_metrics.py`)

`TimbreMetrics` extracts frame-based spectral descriptors using the
[pytimbre](https://pypi.org/project/pytimbre/) library.

### Configuration

```python
tm = TimbreMetrics(sample_rate=16000, frame_width_sec=0.25, overlap_pct=0.5)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | `16000` | Expected sample rate of input signals |
| `frame_width_sec` | `0.25` | Analysis frame width in seconds |
| `overlap_pct` | `0.5` | Frame overlap (0.5 = 50%) |

### Extraction Modes

**Scalar aggregation** (one value per metric, averaged over frames):

```python
features = tm.extract_from_array(signal)
# {'spectral_centroid': 2341.5, 'spectral_spread': 891.2, ...}
```

**Per-frame trajectories** (one array per metric):

```python
series = tm.extract_series_from_array(signal)
# {'spectral_centroid': array([2100, 2300, ...]), ...}
```

**From file:**

```python
series = tm.extract_series_from_file("audio.wav")
```

### Feature Pipeline

1. Mono numpy array &rarr; `pytimbre.Waveform`
2. `FrameBuilder.from_waveform(...)` &rarr; windowed frames
3. `SpectralTimeHistory.from_fourier_transform(...)` &rarr; per-frame spectra
4. `SpectralMetrics.from_spectrum(spec)` &rarr; per-frame feature dict
5. Aggregate across frames (mean) or return raw trajectories

### Extracted Features

Features come from `pytimbre.SpectralMetrics` and typically include:
spectral centroid, spectral spread, spectral skewness, spectral kurtosis,
spectral flatness, spectral irregularity, and sound levels (normalized as
`spec_la`, `spec_lc`, `spec_lz`).

Extraction is best-effort per frame &mdash; if a metric fails on a particular frame,
it is skipped and extraction continues.

---

## Distributional Distance Metrics (`src/evaluation/loss.py`)

`Loss` compares two feature distributions (e.g., original vs. transferred) using
kernel-based and optimal-transport distances.

### Configuration

```python
loss = Loss(kernel="rbf", bandwidth=None, n_projections=50)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel` | `"rbf"` | Kernel for MMD (only RBF supported) |
| `bandwidth` | `None` | RBF sigma; `None` uses the median heuristic |
| `n_projections` | `50` | Random projections for sliced Wasserstein |

### MMD (Maximum Mean Discrepancy)

```python
distance = loss.mmd(X, Y)
```

- Uses the unbiased quadratic-time estimator with an RBF kernel.
- Bandwidth is set via the **median heuristic**: median of all pairwise Euclidean
  distances in the concatenated sample, ensuring a data-adaptive scale.
- Returns `sqrt(max(MMD^2, 0))` to avoid negative values from estimation variance.
- Inputs: 1-D `(n,)` or 2-D `(n, d)` arrays. 1-D arrays are promoted to `(n, 1)`.

### Wasserstein-1 Distance

```python
distance = loss.wasserstein(X, Y)
```

- **1-D inputs**: exact solution via `scipy.stats.wasserstein_distance`.
- **Multi-dimensional inputs**: sliced Wasserstein approximation &mdash; projects both
  distributions onto 50 random unit vectors, computes 1-D Wasserstein on each
  projection, and returns the mean.
- Interpretation: the "earth mover's distance" &mdash; minimum cost of transforming
  one distribution into the other.

### Input Requirements

- Both `X` and `Y` must contain at least 2 samples.
- Feature dimensions must match (`X.shape[1] == Y.shape[1]`).
- Arrays are cast to `float64` internally.

---

## Typical Evaluation Workflow

```python
from evaluation import AudioSegmenter, TimbreMetrics, Loss

sr = 16000

# 1. Segment both signals identically
segmenter = AudioSegmenter(sr=sr, L=sr)  # 1-second segments
original_segments = segmenter.fit_transform(original_audio)
transferred_segments = segmenter.fit_transform(transferred_audio)

# 2. Extract timbre features from each segment
tm = TimbreMetrics(sample_rate=sr)
orig_features = [tm.extract_from_array(seg) for seg in original_segments]
xfer_features = [tm.extract_from_array(seg) for seg in transferred_segments]

# 3. Stack into matrices
import numpy as np
keys = sorted(orig_features[0].keys())
X = np.array([[f[k] for k in keys] for f in orig_features])
Y = np.array([[f[k] for k in keys] for f in xfer_features])

# 4. Compute distances
loss = Loss()
print("MMD:", loss.mmd(X, Y))
print("Wasserstein:", loss.wasserstein(X, Y))
```

---

## Related Notebooks

| Notebook | What it demonstrates |
|----------|---------------------|
| `src/demo/timbre_metrics_demo.ipynb` | Feature extraction, per-frame trajectories, visualization |
| `src/demo/baseline_demo.ipynb` | Comparing DDSP vs. WORLD baseline outputs with metrics |
