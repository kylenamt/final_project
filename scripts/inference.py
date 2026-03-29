#!/usr/bin/env python
"""CLI entry point for DDSP timbre transfer inference.

Usage::

    python scripts/inference.py --audio-path input.wav --model Violin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without pip install
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_loading import PRETRAINED_MODELS
from utils import DEFAULT_SAMPLE_RATE


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for timbre transfer inference."""
    parser = argparse.ArgumentParser(
        description="DDSP timbre transfer inference."
    )
    parser.add_argument(
        "--audio-path", required=True,
        help="Path to input audio (.wav or .npy)",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/outputs",
        help="Directory for output files",
    )
    parser.add_argument(
        "--model", default="Violin",
        help=f"Pretrained model ({', '.join(PRETRAINED_MODELS)}) "
             f"or path to local checkpoint dir",
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument(
        "--pitch-shift", type=float, default=0.0,
        help="Pitch shift in octaves",
    )
    parser.add_argument(
        "--loudness-shift", type=float, default=0.0,
        help="Loudness shift in dB",
    )
    parser.add_argument(
        "--auto-adjust", type=int, default=1,
        help="Auto-adjust pitch/loudness to training data (1=on, 0=off)",
    )
    parser.add_argument(
        "--threshold", type=float, default=1.0,
        help="Note detection sensitivity",
    )
    parser.add_argument(
        "--quiet", type=float, default=20.0,
        help="Suppress silent regions (dB)",
    )
    parser.add_argument(
        "--autotune", type=float, default=0.0,
        help="Autotune amount (0=off, 1=full)",
    )
    parser.add_argument(
        "--f0-confidence-threshold", type=float, default=0.0,
        help="Zero out f0 below this confidence",
    )
    parser.add_argument(
        "--start-sec", type=float, default=None,
        help="Start time in seconds (optional segment extraction)",
    )
    parser.add_argument(
        "--end-sec", type=float, default=None,
        help="End time in seconds (optional segment extraction)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Args: {args}")
    # Full inference logic can be wired here using TimbreTransfer class.
