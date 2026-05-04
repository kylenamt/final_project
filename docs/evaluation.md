# Evaluation

The evaluation pipeline computes distributional and pairwise metrics for
synthesized audio. It supports batch inference and metrics such as MMD,
Wasserstein distance, and pitch/timbre coupling analyses.

## Core components

- src/evaluation/loss.py: MMD and Wasserstein
- src/evaluation/timbre_metrics.py: spectral and temporal descriptors
- src/evaluation/pitch_metrics.py: pitch regression and coupling
- src/evaluation/noise_metrics.py: noise-related metrics
- src/evaluation/batch_inference.py: batch synthesis, baselines, evaluation

## Common flow

1) Run synthesis or baseline generation
2) Extract features for reference and generated sets
3) Compute distributional and pairwise metrics

Batch functions are exposed via the evaluation package and can be called from
scripts or notebooks.
