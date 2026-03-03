import glob
import os
import re
from time import time
from typing import Dict, Iterable, Optional

import gin
import ddsp.training


def _select_gin_file(model_path: str, model_dir: str, gin_file: Optional[str]) -> str:
    if gin_file:
        return gin_file
    if os.path.isfile(model_path) and model_path.endswith(".gin"):
        return model_path

    gin_candidates = sorted(glob.glob(os.path.join(model_dir, "*.gin")))
    if not gin_candidates:
        raise FileNotFoundError(f"No gin file found under {model_dir}")

    operative = []
    for path in gin_candidates:
        match = re.search(r"operative_config-(\d+)\.gin$", os.path.basename(path))
        if match:
            operative.append((int(match.group(1)), path))
    if operative:
        return max(operative, key=lambda item: item[0])[1]

    return gin_candidates[0]


def find_model_dir(path: str) -> str:
    if os.path.isfile(path) and path.endswith(".gin"):
        return os.path.dirname(path)
    if os.path.isdir(path):
        gin_files = glob.glob(os.path.join(path, "*.gin"))
        if gin_files:
            return path
        for root, _, files in os.walk(path):
            for name in files:
                if name.endswith(".gin") and not name.startswith("."):
                    return root
    raise FileNotFoundError(f"No gin file found under {path}")


def find_latest_checkpoint(model_dir: str) -> str:
    ckpt_indexes = glob.glob(os.path.join(model_dir, "ckpt-*.index"))
    if not ckpt_indexes:
        raise FileNotFoundError(f"No checkpoint found under {model_dir}")
    steps = []
    for path in ckpt_indexes:
        match = re.search(r"ckpt-(\d+)\.index$", path)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"No checkpoint steps parsed under {model_dir}")
    step = max(steps)
    return os.path.join(model_dir, f"ckpt-{step}")


def load_ddsp_model(model_path: str, gin_file: Optional[str] = None) -> Dict[str, object]:
    """Load a DDSP Autoencoder model from a path or gin file.

    Returns metadata along with the instantiated model for downstream use.
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


def load_models(model_paths: Iterable[str], gin_file: Optional[str] = None) -> Dict[str, Dict[str, object]]:
    """Load multiple DDSP models and return a mapping by basename."""
    models: Dict[str, Dict[str, object]] = {}
    for path in model_paths:
        key = os.path.basename(os.path.abspath(path))
        models[key] = load_ddsp_model(path, gin_file=gin_file)
    return models
