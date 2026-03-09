"""DDSP model discovery and loading utilities.

Provides helpers to locate gin config files and checkpoints within a
model directory, and to restore a ``ddsp.training.models.Autoencoder``.
"""

import glob
import os
import re
from typing import Dict, Iterable, Optional

import gin
import ddsp.training


# ---------------------------------------------------------------------------
# Gin / checkpoint discovery
# ---------------------------------------------------------------------------

def _select_gin_file(model_path: str, model_dir: str, gin_file: Optional[str] = None) -> str:
    """Return the gin config path to use.

    Resolution order:
    1. Explicit *gin_file* if given.
    2. *model_path* itself, if it is a ``.gin`` file.
    3. The ``operative_config-<step>.gin`` with the highest step number.
    4. The first ``.gin`` found alphabetically.
    """
    if gin_file:
        return gin_file
    if os.path.isfile(model_path) and model_path.endswith(".gin"):
        return model_path

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


# ---------------------------------------------------------------------------
# Full model loading
# ---------------------------------------------------------------------------

def load_ddsp_model(
    model_path: str, gin_file: Optional[str] = None
) -> Dict[str, object]:
    """Load a DDSP Autoencoder model from a path or gin file.

    Returns a dict with keys ``model``, ``model_dir``, ``gin_file``, and
    ``checkpoint``.
    """
    model_dir = find_model_dir(model_path)
    gin_path = _select_gin_file(model_path, model_dir, gin_file)

    with gin.unlock_config():
        gin.clear_config()
        gin.parse_config_file(gin_path, skip_unknown=True)

    model = ddsp.training.models.Autoencoder()
    ckpt_path = find_latest_checkpoint(model_dir)
    model.restore(ckpt_path)

    return {
        "model": model,
        "model_dir": model_dir,
        "gin_file": gin_path,
        "checkpoint": ckpt_path,
    }


def load_models(
    model_paths: Iterable[str], gin_file: Optional[str] = None
) -> Dict[str, Dict[str, object]]:
    """Load multiple DDSP models keyed by directory basename."""
    return {
        os.path.basename(os.path.abspath(p)): load_ddsp_model(p, gin_file=gin_file)
        for p in model_paths
    }
