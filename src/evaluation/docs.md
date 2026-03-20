Here's a cleaner, higher-level version:

---

# Evaluation Module

Analyzes timbre from audio signals by segmenting, extracting features, and computing statistics — entirely in memory, no temp files.

---

## Files

**`segment.py` — `AudioSegmenter`**
Splits a signal into fixed-length non-overlapping segments. Configure with either an explicit length `L` (in samples) or a `bpm` + `beats` pair which computes `L` automatically. Call `fit(signal)` then `transform(signal)` to get a list of numpy arrays.

**`timbre_metrics.py` — `TimbreMetrics`**
Extracts temporal and spectral features from a single segment. The primary method is `extract_from_array(signal)` which accepts a numpy array and returns a flat feature dictionary. A legacy `_extract_features(file_path)` method is kept for backward compatibility.

**`pipeline.py` — `TimbreAnalysisPipeline`**
Orchestrates the full workflow: segment → extract → aggregate. Call `run(signal)` to get per-segment features plus summary statistics (`mean`, `median`, `std`, `iqr`) across all segments. Use the static `compare(result_a, result_b)` to get absolute differences in median features between two runs.

---

## Usage

```python
from evaluation import TimbreAnalysisPipeline

pipeline = TimbreAnalysisPipeline(sr=16000, L=8000)  # or bpm=120
result = pipeline.run(signal)

# per-segment features
result["per_segment"]   # list of dicts

# summary statistics across segments
result["median"]        # recommended by Peeters et al. (2011)
result["iqr"]

# compare two signals
diff = TimbreAnalysisPipeline.compare(result_a, result_b)
```