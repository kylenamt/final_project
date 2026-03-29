"""Visualization helpers for audio spectrograms and feature trajectories."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import librosa
import librosa.display
from scipy import signal as scipy_signal
from scipy.signal import get_window

from feature_utils import compute_features
from utils import DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)

__all__ = [
    "plot_spectrograms",
    "plot_features",
    "plot_feature_from_audio",
    "plot_series",
    "plot_transfer_comparison",
    "plot_envelope_comparison",
]


def plot_spectrograms(
    audios: List[np.ndarray],
    sr: int,
    vmin: float = -5,
    vmax: float = 1,
    size: int = 512 + 256,
) -> None:
    """Plot STFT log-magnitude spectrograms for a list of audio arrays."""
    if not isinstance(audios, list) or len(audios) == 0:
        raise ValueError("audios must be a list with one or more audio arrays")

    for i, audio in enumerate(audios):
        if len(audio.shape) == 2:
            audio = audio[0]
        _, _, Sxx = scipy_signal.stft(audio, fs=sr, nperseg=size,
                                      noverlap=size * 3 // 4)
        logmag = np.log10(np.abs(Sxx) + 1e-7)
        logmag = np.flipud(logmag)
        plt.matshow(logmag, vmin=vmin, vmax=vmax, cmap=plt.cm.magma, aspect='auto')  # type: ignore
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title(f'Audio {i+1}')
        plt.show()


def plot_features(audio_features: Dict[str, Any], trim: int = -15) -> None:
    """Plot loudness, F0, and F0 confidence from extracted audio features."""
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 6))
    ax[0].plot(audio_features['loudness_db'][:trim])
    ax[0].set_ylabel('loudness_db')
    ax[1].plot(librosa.hz_to_midi(audio_features['f0_hz'][:trim]))
    ax[1].set_ylabel('f0 [midi]')
    ax[2].plot(audio_features['f0_confidence'][:trim])
    ax[2].set_ylabel('f0 confidence')
    ax[2].set_xlabel('frame')
    plt.show()


def plot_feature_from_audio(
    audio: np.ndarray, sr: int = DEFAULT_SAMPLE_RATE, trim: int = -15
) -> None:
    """Extract features from raw audio and plot them."""
    features = compute_features(audio)
    plot_features(features, trim=trim)


def plot_series(
    feature_series: np.ndarray, name: Optional[str] = None, size: int = 512 + 256
) -> None:
    """Plot a single feature time-series trajectory."""
    plt.figure(figsize=(12, 4))
    plt.plot(feature_series, label=name if name else "Feature")
    plt.xlabel('Frame')
    plt.ylabel('Value')
    title = f"Feature Series for {name}" if name else "Feature Series"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _stft(x: np.ndarray, N: int = 2048, H: int = 256, window: str = 'hann') -> np.ndarray:
    """Compute a simple STFT magnitude spectrogram in dB."""
    w = get_window(window, N)
    hN = N // 2 + 1
    n_frames = (len(x) - N) // H + 1
    S = np.zeros((hN, n_frames))
    for i in range(n_frames):
        frame = x[i * H: i * H + N]
        if len(frame) < N:
            frame = np.pad(frame, (0, N - len(frame)))
        S[:, i] = 20 * np.log10(np.abs(np.fft.rfft(frame * w)) + 1e-12)
    return S


def plot_transfer_comparison(
    src: np.ndarray,
    tgt: np.ndarray,
    out: np.ndarray,
    sr: int,
    N: int = 2048,
    H: int = 256,
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side spectrograms: source | target | output."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('Timbre Transfer — Spectral Comparison', fontsize=14, fontweight='bold')

    pairs = [('Source', src), ('Target (original)', tgt), ('Output (transferred)', out)]
    for ax, (label, sig) in zip(axes, pairs):
        S = _stft(sig, N=N, H=H)
        hN = N // 2 + 1
        times = np.arange(S.shape[1]) * H / sr
        ax.imshow(S, origin='lower', aspect='auto', cmap='magma',
                  extent=[0, times[-1], 0, sr / 2],
                  vmin=-100, vmax=0)
        ax.set_title(label)
        ax.set_xlabel('Time (s)')
        ax.set_ylim(0, min(8000, sr / 2))
    axes[0].set_ylabel('Frequency (Hz)')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved plot → %s", save_path)
    plt.close(fig)


def plot_envelope_comparison(
    src_analysis: Dict[str, Any],
    tgt_analysis: Dict[str, Any],
    sr: int,
    N: int,
    save_path: Optional[str] = None,
) -> None:
    """Overlay source and target mean spectral envelopes to show the transfer delta."""
    hN = N // 2 + 1
    freqs = np.linspace(0, sr / 2, hN)

    def mean_env(analysis: Dict[str, Any]) -> np.ndarray:
        voiced = analysis['frames_f0'] > 0
        if voiced.sum() > 0:
            return np.median(analysis['frames_env'][voiced], axis=0)
        return np.mean(analysis['frames_env'], axis=0)

    src_env = mean_env(src_analysis)
    tgt_env = mean_env(tgt_analysis)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, src_env, color='tomato',    lw=2, label='Source envelope (timbre donor)')
    ax.plot(freqs, tgt_env, color='steelblue', lw=2, label='Target envelope (before transfer)')
    ax.fill_between(freqs, tgt_env, src_env, alpha=0.15, color='gold', label='Transfer delta')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Spectral Envelope Comparison: Source vs Target')
    ax.set_xlim(0, min(8000, sr / 2))
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved plot → %s", save_path)
    plt.close(fig)
