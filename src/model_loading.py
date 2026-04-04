"""DDSP model discovery and loading utilities.

Provides helpers to locate gin config files and checkpoints within a
model directory, and to restore a ``ddsp.training.models.Autoencoder``.
"""

import glob
import logging
import os
import re
from typing import Optional, TypedDict

import tensorflow.compat.v2 as tf  # type: ignore

import gin
import ddsp.training

logger = logging.getLogger(__name__)

__all__ = [
    "PretrainedResult",
    "find_model_dir",
    "find_gin_file",
    "find_latest_checkpoint",
    "restore_autoencoder",
    "load_pretrained_model",
    "PRETRAINED_MODELS",
]


class PretrainedResult(TypedDict):
    """Return type for :func:`load_pretrained_model`."""
    model_dir: str
    gin_file: str


# ---------------------------------------------------------------------------
# Gin / checkpoint discovery
# ---------------------------------------------------------------------------

def find_gin_file(model_dir: str) -> str:
    """Find the best gin config in *model_dir*.

    Prefers ``operative_config-<step>.gin`` with the highest step number.
    Falls back to the first ``.gin`` file alphabetically.
    """
    gin_candidates = sorted(glob.glob(os.path.join(model_dir, "*.gin")))
    if not gin_candidates:
        raise FileNotFoundError(f"No gin file found under {model_dir}")

    operative = [
        (int(m.group(1)), path)
        for path in gin_candidates
        if (m := re.search(r"operative_config-(\d+)\.gin$", os.path.basename(path)))
    ]
    if operative:
        return max(operative, key=lambda t: t[0])[1]

    return gin_candidates[0]


def find_model_dir(path: str) -> str:
    """Locate the directory containing ``.gin`` config files.

    *path* may point to a ``.gin`` file, a directory that directly contains
    ``.gin`` files, or a parent directory that is walked recursively.

    Raises ``FileNotFoundError`` when no gin file can be found.
    """
    if os.path.isfile(path) and path.endswith(".gin"):
        return os.path.dirname(path)

    if os.path.isdir(path):
        if glob.glob(os.path.join(path, "*.gin")):
            return path
        for root, _, files in os.walk(path):
            if any(f.endswith(".gin") and not f.startswith(".") for f in files):
                return root

    raise FileNotFoundError(f"No gin file found under {path}")


def find_latest_checkpoint(model_dir: str) -> str:
    """Return the path (without extension) of the highest-step checkpoint.

    Looks for ``ckpt-<step>.index`` files and returns the prefix
    ``<model_dir>/ckpt-<max_step>``.
    """
    ckpt_indexes = glob.glob(os.path.join(model_dir, "ckpt-*.index"))
    if not ckpt_indexes:
        raise FileNotFoundError(f"No checkpoint found under {model_dir}")

    steps = [
        int(m.group(1))
        for path in ckpt_indexes
        if (m := re.search(r"ckpt-(\d+)\.index$", path))
    ]
    if not steps:
        raise FileNotFoundError(f"No checkpoint steps parsed under {model_dir}")

    return os.path.join(model_dir, f"ckpt-{max(steps)}")


def restore_autoencoder(model_dir: str):
    """Instantiate and restore a DDSP Autoencoder from ``model_dir``.

    Expects gin config to be parsed by the caller when shape-dependent
    overrides are needed (e.g., timbre transfer inference).
    """
    model = ddsp.training.models.Autoencoder() # type: ignore
    ckpt_path = find_latest_checkpoint(model_dir)
    model.restore(ckpt_path) # type: ignore
    return model


# ---------------------------------------------------------------------------
# Pretrained model loading (from GCS)
# ---------------------------------------------------------------------------

PRETRAINED_MODELS = ("Violin", "Flute", "Flute2", "Trumpet", "Tenor_Saxophone")
GCS_CKPT_DIR = "gs://ddsp/models/timbre_transfer_colab/2021-07-08"


def load_pretrained_model(
    model_name: str,
    local_dir: str = "artifacts/pretrained",
) -> PretrainedResult:
    """Download a pretrained DDSP model from Google Cloud Storage.

    Uses ``tf.io.gfile`` for GCS access (no ``gsutil`` needed).
    Files are cached locally; subsequent calls skip the download.

    Parameters
    ----------
    model_name : str
        One of ``Violin``, ``Flute``, ``Flute2``, ``Trumpet``,
        ``Tenor_Saxophone``.
    local_dir : str
        Local directory to cache downloaded files.

    Returns
    -------
    dict
        Keys: ``model_dir``, ``gin_file``.
    """
    import shutil

    if model_name not in PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {PRETRAINED_MODELS}"
        )

    model_dir = os.path.join(local_dir, model_name)
    gcs_path = f"{GCS_CKPT_DIR}/solo_{model_name.lower()}_ckpt"
    gin_file = os.path.join(model_dir, "operative_config-0.gin")

    # Download from GCS if not already cached
    if not os.path.isfile(gin_file):
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir, exist_ok=True)

        remote_files = tf.io.gfile.listdir(gcs_path)
        for fname in remote_files:
            src = f"{gcs_path}/{fname}"
            dst = os.path.join(model_dir, fname)
            tf.io.gfile.copy(src, dst, overwrite=True)
        logger.info("Downloaded %d files from %s", len(remote_files), gcs_path)

    return {"model_dir": model_dir, "gin_file": gin_file}
