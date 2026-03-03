import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import tensorflow.compat.v2 as tf

import ddsp.losses

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

from loss import FAD, Pitch_centRMSE, SNR  # noqa: E402

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff")


def _read_audio_mono(path: str, target_sr: Optional[int]) -> Tuple[np.ndarray, int]:
	audio, sr = sf.read(path)
	if audio.ndim > 1:
		audio = np.mean(audio, axis=1)
	audio = audio.astype(np.float32, copy=False)
	if target_sr is not None and sr != target_sr:
		audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
		sr = target_sr
	return audio, sr


def _list_audio_files(root_dir: str, audio_exts: Iterable[str]) -> Dict[str, str]:
	files: Dict[str, str] = {}
	for name in os.listdir(root_dir):
		path = os.path.join(root_dir, name)
		if os.path.isfile(path) and name.lower().endswith(tuple(ext.lower() for ext in audio_exts)):
			stem = os.path.splitext(name)[0]
			files[stem] = path
	return files


def _resolve_dirs(data_dir: str, ref_subdir: str, est_subdir: str) -> Tuple[str, str]:
	ref_dir = ref_subdir if os.path.isabs(ref_subdir) else os.path.join(data_dir, ref_subdir)
	est_dir = est_subdir if os.path.isabs(est_subdir) else os.path.join(data_dir, est_subdir)

	if not os.path.isdir(ref_dir):
		raise FileNotFoundError(f"Reference directory not found: {ref_dir}")
	if not os.path.isdir(est_dir):
		raise FileNotFoundError(f"Estimated directory not found: {est_dir}")

	return ref_dir, est_dir


def _common_stems(ref_dir: str, est_dir: str, audio_exts: Iterable[str]) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
	ref_files = _list_audio_files(ref_dir, audio_exts)
	est_files = _list_audio_files(est_dir, audio_exts)
	common = sorted(set(ref_files) & set(est_files))
	if not common:
		raise FileNotFoundError("No matching audio filenames between reference and estimated directories")
	return ref_files, est_files, common


def _load_pair(ref_path: str, est_path: str, sample_rate: Optional[int]) -> Tuple[np.ndarray, np.ndarray, int]:
	ref_audio, ref_sr = _read_audio_mono(ref_path, sample_rate)
	est_audio, est_sr = _read_audio_mono(est_path, sample_rate)

	if ref_sr != est_sr:
		est_audio = librosa.resample(est_audio, orig_sr=est_sr, target_sr=ref_sr)
		est_sr = ref_sr

	n_samples = min(len(ref_audio), len(est_audio))
	ref_audio = ref_audio[:n_samples]
	est_audio = est_audio[:n_samples]
	return ref_audio, est_audio, ref_sr


def run_snr_tests(
	data_dir: str,
	ref_subdir: str = "ref",
	est_subdir: str = "est",
	audio_exts: Iterable[str] = AUDIO_EXTS,
	sample_rate: Optional[int] = None,
) -> Dict[str, object]:
	"""Run SNR on paired audio files."""
	ref_dir, est_dir = _resolve_dirs(data_dir, ref_subdir, est_subdir)
	ref_files, est_files, common_stems = _common_stems(ref_dir, est_dir, audio_exts)

	snr_metric = SNR()
	per_file: List[Dict[str, object]] = []
	values: List[float] = []

	for stem in common_stems:
		ref_audio, est_audio, _ = _load_pair(ref_files[stem], est_files[stem], sample_rate)
		snr_val = snr_metric(ref_audio, est_audio)
		per_file.append({"file": stem, "snr": snr_val})
		values.append(snr_val)

	return {
		"reference_dir": ref_dir,
		"estimated_dir": est_dir,
		"num_files": len(common_stems),
		"averages": {"snr": float(np.mean(values)) if values else 0.0},
		"per_file": per_file,
	}


def _loss_to_float(loss_output) -> float:
	if isinstance(loss_output, dict):
		if "loss" in loss_output:
			return float(np.asarray(loss_output["loss"]))
		values = [np.asarray(value) for value in loss_output.values()]
		summed = np.sum(values)
		return float(np.asarray(summed))
	return float(np.asarray(loss_output))


def run_spectral_loss_tests(
	data_dir: str,
	ref_subdir: str = "ref",
	est_subdir: str = "est",
	audio_exts: Iterable[str] = AUDIO_EXTS,
	sample_rate: Optional[int] = None,
	loss_type: str = "L1",
	mag_weight: float = 1.0,
	logmag_weight: float = 1.0,
) -> Dict[str, object]:
	"""Run DDSP SpectralLoss on paired audio files."""
	ref_dir, est_dir = _resolve_dirs(data_dir, ref_subdir, est_subdir)
	ref_files, est_files, common_stems = _common_stems(ref_dir, est_dir, audio_exts)

	spectral_loss = ddsp.losses.SpectralLoss(
		loss_type=loss_type,
		mag_weight=mag_weight,
		logmag_weight=logmag_weight,
	)

	per_file: List[Dict[str, object]] = []
	values: List[float] = []

	for stem in common_stems:
		ref_audio, est_audio, _ = _load_pair(ref_files[stem], est_files[stem], sample_rate)
		ref_tensor = tf.convert_to_tensor(ref_audio[None, :], dtype=tf.float32)
		est_tensor = tf.convert_to_tensor(est_audio[None, :], dtype=tf.float32)
		loss_val = spectral_loss(ref_tensor, est_tensor)
		loss_float = _loss_to_float(loss_val)
		per_file.append({"file": stem, "spectral_loss": loss_float})
		values.append(loss_float)

	return {
		"reference_dir": ref_dir,
		"estimated_dir": est_dir,
		"num_files": len(common_stems),
		"averages": {"spectral_loss": float(np.mean(values)) if values else 0.0},
		"per_file": per_file,
	}



def run_pitch_cent_rmse_tests(
	data_dir: str,
	ref_subdir: str = "ref",
	est_subdir: str = "est",
	audio_exts: Iterable[str] = AUDIO_EXTS,
	sample_rate: Optional[int] = None,
	conf_threshold: float = 0.5,
	voiced_threshold_hz: float = 1.0,
) -> Dict[str, object]:
	"""Run pitch RMSE (cents) on paired audio files using CREPE."""
	ref_dir, est_dir = _resolve_dirs(data_dir, ref_subdir, est_subdir)
	ref_files, est_files, common_stems = _common_stems(ref_dir, est_dir, audio_exts)

	rmse_metric = Pitch_centRMSE()
	per_file: List[Dict[str, object]] = []
	values: List[float] = []

	for stem in common_stems:
		ref_audio, est_audio, sr = _load_pair(ref_files[stem], est_files[stem], sample_rate)
		rmse_val = rmse_metric(
			ref_audio,
			est_audio,
			sr,
			conf_threshold=conf_threshold,
			voiced_threshold_hz=voiced_threshold_hz,
		)
		per_file.append({"file": stem, "pitch_cent_rmse": rmse_val})
		values.append(rmse_val)

	return {
		"reference_dir": ref_dir,
		"estimated_dir": est_dir,
		"num_files": len(common_stems),
		"averages": {"pitch_cent_rmse": float(np.mean(values)) if values else 0.0},
		"per_file": per_file,
	}


def run_fad_test(
	data_dir: str,
	ref_subdir: str = "ref",
	est_subdir: str = "est",
	sample_rate: int = 16000,
) -> Dict[str, object]:
	"""Run Frechet Audio Distance on a directory pair."""
	ref_dir, est_dir = _resolve_dirs(data_dir, ref_subdir, est_subdir)
	fad_metric = FAD(sample_rate=sample_rate)
	return {
		"reference_dir": ref_dir,
		"estimated_dir": est_dir,
		"fad": fad_metric(ref_dir, est_dir),
	}


# def run_loss_tests(
# 	data_dir: str,
# 	ref_subdir: str = "ref",
# 	est_subdir: str = "est",
# 	audio_exts: Iterable[str] = AUDIO_EXTS,
# 	sample_rate: Optional[int] = None,
# 	compute_fad: bool = False,
# ) -> Dict[str, object]:
# 	"""Run SNR and pitch RMSE on paired audio files.

# 	This wraps the per-metric runners for convenience.
# 	"""
# 	snr_results = run_snr_tests(
# 		data_dir=data_dir,
# 		ref_subdir=ref_subdir,
# 		est_subdir=est_subdir,
# 		audio_exts=audio_exts,
# 		sample_rate=sample_rate,
# 	)
# 	rmse_results = run_pitch_cent_rmse_tests(
# 		data_dir=data_dir,
# 		ref_subdir=ref_subdir,
# 		est_subdir=est_subdir,
# 		audio_exts=audio_exts,
# 		sample_rate=sample_rate,
# 	)

# 	merged: Dict[str, Dict[str, object]] = {}
# 	for item in snr_results["per_file"]:
# 		merged[item["file"]] = {"file": item["file"], "snr": item["snr"]}
# 	for item in rmse_results["per_file"]:
# 		merged.setdefault(item["file"], {"file": item["file"]})
# 		merged[item["file"]]["pitch_cent_rmse"] = item["pitch_cent_rmse"]

# 	averages = {
# 		**snr_results["averages"],
# 		**rmse_results["averages"],
# 	}
# 	if compute_fad:
# 		fad_result = run_fad_test(data_dir, ref_subdir, est_subdir)
# 		averages["fad"] = fad_result["fad"]

# 	return {
# 		"reference_dir": snr_results["reference_dir"],
# 		"estimated_dir": snr_results["estimated_dir"],
# 		"num_files": snr_results["num_files"],
# 		"averages": averages,
# 		"per_file": list(merged.values()),
# 	}

