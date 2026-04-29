"""Evaluation utilities for timbre-transfer quality assessment."""

from .loss import Loss

__all__ = [
    "Loss",
    "NoiseMetrics",
    "PitchMetrics",
    "TimbreMetrics",
    "collect_wav_files",
    "load_features",
    "evaluate_dir",
    "synthesize_dir",
    "vocode_dir",
]

_LAZY = {
    "NoiseMetrics": ("noise_metrics", "NoiseMetrics"),
    "PitchMetrics": ("pitch_metrics", "PitchMetrics"),
    "TimbreMetrics": ("timbre_metrics", "TimbreMetrics"),
    "collect_wav_files": ("batch_inference", "collect_wav_files"),
    "load_features": ("batch_inference", "load_features"),
    "evaluate_dir": ("batch_inference", "evaluate_dir"),
    "synthesize_dir": ("batch_inference", "synthesize_dir"),
    "vocode_dir": ("batch_inference", "vocode_dir"),
}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        module_name, attr = _LAZY[name]
        module = import_module(f".{module_name}", __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
