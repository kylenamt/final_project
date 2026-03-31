# pytimbre Package Reference

Quick-reference for how the `pytimbre` package is structured, what it can
compute, and how to wire its classes together.

---

## Package layout

```
pytimbre/
├── audio.py                          # Waveform
├── spectral/
│   ├── spectra.py                    # Spectrum
│   ├── spectral_frame_builder.py     # FrameBuilder
│   ├── time_histories.py             # SpectralTimeHistory, OverallLevelTimeHistory
│   ├── fundamental_frequency.py      # FundamentalFrequencyCalculator
│   └── swipe.py                      # Swipe pitch detection
├── timbre_features/
│   ├── features.py                   # TimbreFeatures, ImpulseFeatures (convenience)
│   └── metrics/
│       ├── spectral.py               # SpectralMetrics   (11 features)
│       ├── temporal.py               # TemporalMetrics   (23 features)
│       ├── level.py                  # LevelMetrics
│       ├── harmonic.py               # HarmonicMetrics   (8 features)
│       ├── sound_quality.py          # SoundQualityMetrics
│       ├── equivalent_level.py       # EquivalentLevelMetrics
│       ├── impulse.py                # ImpulseMetrics
│       └── room_acoustics.py         # RoomAcousticsMetrics
└── utilities/
    ├── acoustic_weights.py           # AcousticWeights
    ├── audio_filtering.py            # WaveformFilter
    ├── fractional_octave_band.py     # FractionalOctaveBandTools
    └── yin.py                        # YIN pitch detection
```

---

## Core data classes

### `Waveform`

Thin wrapper around a 1-D float array + sample rate + start time.

```python
from pytimbre.audio import Waveform

wfm = Waveform(samples_array, sample_rate, start_time=0.0)
wfm = Waveform.from_wave_file("clip.wav")
```

Key properties: `.samples`, `.sample_rate`, `.duration`, `.is_continuous`,
`.is_impulsive`.

### `FrameBuilder`

Slices a waveform into overlapping frames for frame-by-frame analysis.

```python
from pytimbre.spectral.spectral_frame_builder import FrameBuilder

fb = FrameBuilder.from_waveform(wfm, overlap_pct=0.5, frame_width_sec=0.25)
fb.complete_frame_count   # number of full frames
fb.get_next_waveform_subset(wfm)  # returns next frame as a Waveform
```

### `Spectrum`

Single-frame frequency representation. Created via FFT or digital filters.

```python
from pytimbre.spectral.spectra import Spectrum

spec = Spectrum.from_fourier_transform(frame_wfm, fft_size=2048)
spec = Spectrum.from_digital_filters(frame_wfm, frequency_resolution=3)
```

Key properties: `.frequencies`, `.pressures_pascals`, `.pressures_decibels`,
`.waveform` (the source frame — **not None** when built from a waveform).

### `SpectralTimeHistory`

Ordered collection of `Spectrum` objects with frame timing metadata.

```python
from pytimbre.spectral.time_histories import SpectralTimeHistory

sth = SpectralTimeHistory.from_fourier_transform(wfm, fb)
sth = SpectralTimeHistory.from_digital_filters(wfm, fb, frequency_resolution=3)
```

Key properties: `.spectra` (array of `Spectrum`), `.frequencies`, `.times`,
`.spectrogram_array_decibels` (2-D array, shape `[n_frames, n_freqs]`).

---

## Metric classes

Every metric class exposes a `.get_features() -> dict` method.

### `SpectralMetrics` — frequency-domain shape descriptors

Built from a **single `Spectrum`**; applied per-frame in a time history.

```python
from pytimbre.timbre_features.metrics.spectral import SpectralMetrics

sm = SpectralMetrics.from_spectrum(spec)
sm.get_features()
```

| Feature key | Description |
|---|---|
| `spectral_centroid` | Centre of gravity of the spectrum (Hz) |
| `spectral_spread` | Std-dev around the centroid (Hz) |
| `spectral_skewness` | Asymmetry of spectral shape |
| `spectral_kurtosis` | Peakiness of spectral shape |
| `spectral_slope` | Linear regression slope over amplitude |
| `spectral_decrease` | Low-frequency-emphasised slope |
| `spectral_roll_off` | Frequency below which 95 % of energy lies |
| `spectral_energy` | Sum of squared pressures |
| `spectral_flatness` | Geometric / arithmetic mean (0 = tonal, 1 = noise) |
| `spectral_crest` | Peak / mean pressure |
| `mean_center` | Integration variable minus centroid |

### `TemporalMetrics` — time-domain / envelope descriptors

Built from a **`Waveform`** (not a `Spectrum`). Computes the analytic signal
envelope (Hilbert transform + 5 Hz low-pass) and derives features from it.

```python
from pytimbre.timbre_features.metrics.temporal import TemporalMetrics

tm = TemporalMetrics.from_waveform(wfm)
tm.get_features()
```

**Global (single-value) features — computed over the entire waveform:**

| Feature key | Description |
|---|---|
| `attack` | Start time of attack (s) |
| `log_attack` | log10 of attack duration (LAT) |
| `attack slope` | Weighted-average temporal slope during attack (dB/s) |
| `decrease` | End time of attack / start of decay (s) |
| `decrease slope` | Exponential decay rate (dB/s) |
| `release` | Release phase time (s) |
| `temporal centroid` | Energy centre-of-gravity in time (s) |
| `effective duration` | Duration above 40 % energy threshold (s) |
| `amplitude modulation` | Median amplitude variation during sustain (0–1) |
| `frequency modulation` | Median instantaneous frequency variation (Hz) |

**Instantaneous (per-frame) features — computed with a short sliding window,
then averaged in `get_features()`:**

| Feature key(s) | Description |
|---|---|
| `auto-correlation_01` … `auto-correlation_12` | Mean autocorrelation coefficients (12 lags) |
| `zero crossing rate` | Mean zero-crossing rate (Hz) |

Raw per-frame arrays are available directly:

```python
tm.auto_correlation       # shape (n_frames, 12)
tm.zero_crossing_rate     # shape (n_frames,)
```

Default frame parameters: hop = 2.9 ms, window = 23.2 ms (configurable via
`.hop_size_seconds` and `.window_size_seconds`).

### Other metric classes (brief)

| Class | Factory | Scope |
|---|---|---|
| `LevelMetrics` | `.from_spectrum(s)` / `.from_waveform(w)` | LA, LC, LZ levels |
| `HarmonicMetrics` | `HarmonicMetrics(spec)` | Harmonic ratio, inharmonicity, OER, tristimulus, etc. (8 features) |
| `SoundQualityMetrics` | `.from_waveform(w)` | Loudness, roughness, sharpness, boominess |
| `EquivalentLevelMetrics` | `.from_waveform(w)` | Leq variants |
| `RoomAcousticsMetrics` | various | RT60, clarity, definition |

---

## Convenience aggregator: `TimbreFeatures`

Combines multiple metric classes automatically based on what data is available.

```python
from pytimbre.timbre_features.features import TimbreFeatures
```

| Method | Input | Output | Includes temporal? |
|---|---|---|---|
| `TimbreFeatures.from_waveform(wfm)` | `Waveform` | `dict` | Yes (temporal + level + sound quality) |
| `TimbreFeatures.from_spectra(spec)` | `Spectrum` | `dict` | Yes, if `spec.waveform is not None` |
| `TimbreFeatures.from_time_history(sth)` | `SpectralTimeHistory` | `pd.DataFrame` | Yes — calls `from_spectra` per frame |

### `from_time_history` detail

Iterates over every `Spectrum` in the time history and calls `from_spectra`
on each one. Because `SpectralTimeHistory.from_fourier_transform` passes the
frame's waveform subset into each `Spectrum`, **temporal features are
included per-frame** (attack, centroid, ZCR, autocorrelation, etc., all
recomputed on the short frame waveform).

The result is a pandas `DataFrame` with one row per frame and one column per
feature (spectral + level + harmonic + temporal + sound quality).

---

## Spectral vs temporal: how they differ

| Aspect | Spectral path | Temporal path |
|---|---|---|
| Domain | Frequency (FFT magnitudes) | Time (waveform envelope / samples) |
| Input | `Spectrum` | `Waveform` |
| Frame handling | External — `FrameBuilder` slices, one `Spectrum` per frame | Internal — its own sliding window (hop 2.9 ms, win 23.2 ms) for ZCR & autocorrelation; global features use the full waveform |
| Typical use | Per-frame spectral shape (centroid, flatness, …) | Whole-signal dynamics (attack, decay, centroid) **plus** per-frame texture (ZCR, autocorrelation) |

### Can temporal features be used for time-varying extraction?

**Yes, with caveats:**

1. **Via `TimbreFeatures.from_time_history`:** temporal features are extracted
   per frame (each frame's waveform subset is passed to
   `TemporalMetrics.from_waveform`). This gives frame-level temporal features,
   but the global features (attack, release, temporal centroid) are recomputed
   on each short frame, so their meaning changes — they describe the micro
   dynamics *within* each frame rather than the macro envelope of the whole
   signal.

2. **Via raw properties:** `tm.auto_correlation` and `tm.zero_crossing_rate`
   already return per-frame arrays using TemporalMetrics' own sliding window
   (finer hop than a typical spectral frame). These can be used directly as
   time-varying features without going through the spectral frame pipeline.

3. **Bottom line:** for time-varying *spectral* descriptors, use
   `SpectralTimeHistory` + per-frame `SpectralMetrics`. For time-varying
   *temporal texture* (ZCR, autocorrelation), use `TemporalMetrics` raw arrays
   directly — they have their own internal framing. The global envelope features
   (attack, decay, temporal centroid, effective duration) are designed to
   describe an entire signal, not individual frames.

---

## Typical workflow

```python
from pytimbre.audio import Waveform
from pytimbre.spectral.spectral_frame_builder import FrameBuilder
from pytimbre.spectral.time_histories import SpectralTimeHistory
from pytimbre.timbre_features.metrics.spectral import SpectralMetrics
from pytimbre.timbre_features.metrics.temporal import TemporalMetrics
from pytimbre.timbre_features.features import TimbreFeatures

# 1. Load audio
wfm = Waveform(signal, sr, 0.0)

# 2a. Spectral features per frame (manual loop)
fb  = FrameBuilder.from_waveform(wfm, overlap_pct=0.5, frame_width_sec=0.25)
sth = SpectralTimeHistory.from_fourier_transform(wfm, fb)
for spec in sth.spectra:
    feats = SpectralMetrics.from_spectrum(spec).get_features()

# 2b. All features per frame (convenience — includes temporal)
df = TimbreFeatures.from_time_history(sth)   # DataFrame, one row per frame

# 3. Temporal features on the whole signal
tm = TemporalMetrics.from_waveform(wfm)
tm.get_features()              # dict with 23 features (globals averaged)
tm.zero_crossing_rate          # raw per-frame array
tm.auto_correlation            # raw per-frame array (n_frames, 12)
```
