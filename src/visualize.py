"""Visualization helpers for audio spectrograms, feature trajectories, and evaluation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import matplotlib
# Only set Agg backend if not in interactive environment (e.g., Jupyter)
import sys
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
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
    "plot_world_params",
    "plot_transfer_comparison",
    "plot_envelope_comparison",
    "plot_loss_by_group",
    "plot_loss_boxplot",
    "plot_feature_comparison",
    "plot_feature_radar",
    "plot_loss_heatmap",
    "plot_method_comparison",
    "plot_distribution_comparison",
]


def plot_spectrograms(
    audios: List[np.ndarray],
    sr: int = DEFAULT_SAMPLE_RATE,
    vmin: float = -5,
    vmax: float = 1,
    size: int = 512 + 256,
    titles: Optional[List[str]] = None,
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
        title = titles[i] if titles and i < len(titles) else f'Audio {i+1}'
        plt.title(title)
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


def plot_world_params(
    f0_list: List[np.ndarray],
    sp_list: List[np.ndarray],
    ap_list: List[np.ndarray],
    sr: int = DEFAULT_SAMPLE_RATE,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot WORLD vocoder parameters (F0, SP, AP) for multiple audio signals.

    Each signal gets one row of three plots:
    - F0: line plot (1-D pitch contour)
    - SP: heatmap (2-D spectral envelope)
    - AP: heatmap (2-D aperiodicity)

    Parameters
    ----------
    f0_list : list of 1-D arrays, shape (n_frames,)
    sp_list : list of 2-D arrays, shape (n_frames, n_freq_bins)
    ap_list : list of 2-D arrays, shape (n_frames, n_freq_bins)
    sr : sample rate, used to label the frequency axis.
    labels : optional display names for each signal.
    save_path : if given, save the figure to this path.
    """
    # Accept raw arrays (single signal) or lists of arrays (multiple signals)
    if isinstance(f0_list, np.ndarray) and f0_list.ndim == 1:
        f0_list, sp_list, ap_list = [f0_list], [sp_list], [ap_list]

    n = len(f0_list)
    if labels is None:
        labels = [f"Audio {i + 1}" for i in range(n)]

    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n), squeeze=False)

    for i in range(n):
        f0, sp, ap = f0_list[i], sp_list[i], ap_list[i]
        n_freq_bins = sp.shape[1]
        freqs = np.linspace(0, sr / 2, n_freq_bins)

        # -- F0: line plot --
        ax_f0 = axes[i, 0]
        frames = np.arange(len(f0))
        ax_f0.plot(frames, f0, color="steelblue", lw=1)
        ax_f0.set_ylabel("F0 (Hz)")
        ax_f0.set_xlabel("Frame")
        ax_f0.set_title(f"{labels[i]} — F0")

        # -- SP: heatmap (log-scale spectral envelope) --
        ax_sp = axes[i, 1]
        sp_db = 10 * np.log10(sp.T + 1e-12)
        ax_sp.imshow(
            sp_db, origin="lower", aspect="auto", cmap="magma",
            extent=[0, sp.shape[0], freqs[0], freqs[-1]],
        )
        ax_sp.set_ylabel("Frequency (Hz)")
        ax_sp.set_xlabel("Frame")
        ax_sp.set_title(f"{labels[i]} — Spectral Envelope")

        # -- AP: heatmap (aperiodicity) --
        ax_ap = axes[i, 2]
        ax_ap.imshow(
            ap.T, origin="lower", aspect="auto", cmap="inferno",
            extent=[0, ap.shape[0], freqs[0], freqs[-1]],
        )
        ax_ap.set_ylabel("Frequency (Hz)")
        ax_ap.set_xlabel("Frame")
        ax_ap.set_title(f"{labels[i]} — Aperiodicity")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot → %s", save_path)
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


# ---------------------------------------------------------------------------
# Evaluation visualizations
# ---------------------------------------------------------------------------


def plot_loss_by_group(
    df: pd.DataFrame,
    group_col: str,
    metric: str = "mmd",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart of mean loss per group with std error bars.

    Parameters
    ----------
    df : DataFrame with at least *group_col* and *metric* columns.
    group_col : column to group by (e.g. ``"speaker"``, ``"exercise"``).
    metric : loss column name (``"mmd"`` or ``"wasserstein"``).
    ax : optional matplotlib Axes; a new figure is created when *None*.
    save_path : if given, save the figure to this path.
    """
    grouped = df.groupby(group_col)[metric].agg(["mean", "std"]).sort_values("mean")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(8, len(grouped) * 0.6), 5))
    else:
        fig = ax.figure

    ax.bar(range(len(grouped)), grouped["mean"], yerr=grouped["std"],
           capsize=3, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped.index, rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by {group_col}")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved plot → %s", save_path)
    if own_fig:
        plt.close(fig)


def plot_loss_boxplot(
    df: pd.DataFrame,
    group_col: str,
    metric: str = "mmd",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> None:
    """Box plot of loss distribution per group.

    Parameters
    ----------
    df : DataFrame with at least *group_col* and *metric* columns.
    group_col : column to group by.
    metric : loss column name.
    ax : optional matplotlib Axes.
    save_path : if given, save the figure to this path.
    """
    groups = sorted(df[group_col].unique())
    data = [df.loc[df[group_col] == g, metric].dropna().values for g in groups]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.6), 5))
    else:
        fig = ax.figure

    bp = ax.boxplot(data, patch_artist=True, labels=groups)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} distribution by {group_col}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved plot → %s", save_path)
    if own_fig:
        plt.close(fig)


def plot_feature_comparison(
    df: pd.DataFrame,
    feature_name: str,
    group_col: str,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> None:
    """Grouped bar chart comparing original vs transferred feature means.

    Parameters
    ----------
    df : DataFrame with ``{feature_name}_orig`` and ``{feature_name}_trans`` columns.
    feature_name : base feature name (e.g. ``"spectral_centroid"``).
    group_col : column to group by.
    ax : optional matplotlib Axes.
    save_path : if given, save the figure to this path.
    """
    col_orig = f"{feature_name}_orig"
    col_trans = f"{feature_name}_trans"

    grouped = df.groupby(group_col)[[col_orig, col_trans]].mean()
    groups = grouped.index.tolist()
    x = np.arange(len(groups))
    width = 0.35

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.8), 5))
    else:
        fig = ax.figure

    ax.bar(x - width / 2, grouped[col_orig], width, label="Original", color="steelblue")
    ax.bar(x + width / 2, grouped[col_trans], width, label="Transferred", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel(feature_name)
    ax.set_title(f"{feature_name} — Original vs Transferred by {group_col}")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved plot → %s", save_path)
    if own_fig:
        plt.close(fig)


def plot_feature_radar(
    df: pd.DataFrame,
    group: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Radar chart of z-scored feature means (original vs transferred).

    Parameters
    ----------
    df : DataFrame from ``run_evaluation_dir`` with ``*_orig`` and ``*_trans`` columns.
    group : if given, filter to rows where any metadata column matches this value.
    save_path : if given, save the figure to this path.
    """
    sub = df if group is None else df[df.isin([group]).any(axis=1)]

    orig_cols = sorted([c for c in sub.columns if c.endswith("_orig")])
    trans_cols = [c.replace("_orig", "_trans") for c in orig_cols]
    labels = [c.replace("_orig", "") for c in orig_cols]

    means_orig = sub[orig_cols].mean().values
    means_trans = sub[trans_cols].mean().values

    # Z-score normalize using the original distribution as reference
    std_orig = sub[orig_cols].std().values
    std_orig[std_orig == 0] = 1.0
    z_orig = (means_orig - means_orig) / std_orig  # zeros by definition
    z_trans = (means_trans - means_orig) / std_orig

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    z_orig = np.concatenate([z_orig, z_orig[:1]])
    z_trans = np.concatenate([z_trans, z_trans[:1]])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.plot(angles, z_orig, "o-", lw=2, label="Original", color="steelblue")
    ax.fill(angles, z_orig, alpha=0.1, color="steelblue")
    ax.plot(angles, z_trans, "o-", lw=2, label="Transferred", color="tomato")
    ax.fill(angles, z_trans, alpha=0.1, color="tomato")
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    title = "Feature Radar (z-scored)"
    if group:
        title += f" — {group}"
    ax.set_title(title, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved plot → %s", save_path)
    plt.close(fig)


def plot_loss_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    metric: str = "mmd",
    save_path: Optional[str] = None,
) -> None:
    """2-D heatmap of mean loss values (e.g. speaker x technique).

    Parameters
    ----------
    df : DataFrame with at least *row_col*, *col_col*, and *metric* columns.
    row_col : column for heatmap rows (e.g. ``"speaker"``).
    col_col : column for heatmap columns (e.g. ``"technique"``).
    metric : loss column name.
    save_path : if given, save the figure to this path.
    """
    pivot = df.pivot_table(values=metric, index=row_col, columns=col_col, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.8),
                                    max(5, len(pivot.index) * 0.4)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Mean {metric.upper()} — {row_col} vs {col_col}")
    fig.colorbar(im, ax=ax, label=metric.upper())

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved plot → %s", save_path)
    plt.close(fig)


def plot_method_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str = "mmd",
    group_col: str = "speaker",
    label_a: str = "DDSP",
    label_b: str = "Baseline",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side grouped bar chart comparing two methods.

    Parameters
    ----------
    df_a, df_b : DataFrames from ``run_evaluation_dir`` for each method.
    metric : loss column name.
    group_col : column to group by.
    label_a, label_b : legend labels for each method.
    ax : optional matplotlib Axes.
    save_path : if given, save the figure to this path.
    """
    mean_a = df_a.groupby(group_col)[metric].mean()
    mean_b = df_b.groupby(group_col)[metric].mean()

    groups = sorted(set(mean_a.index) | set(mean_b.index))
    x = np.arange(len(groups))
    width = 0.35

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.8), 5))
    else:
        fig = ax.figure

    vals_a = [mean_a.get(g, 0) for g in groups]
    vals_b = [mean_b.get(g, 0) for g in groups]

    ax.bar(x - width / 2, vals_a, width, label=label_a, color="steelblue")
    ax.bar(x + width / 2, vals_b, width, label=label_b, color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} — {label_a} vs {label_b} by {group_col}")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved plot → %s", save_path)
    if own_fig:
        plt.close(fig)


def plot_distribution_comparison(
    df_distribution: pd.DataFrame,
    metric: str = "mmd",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> None:
    """Bar chart of distribution-level distance per method.

    Parameters
    ----------
    df_distribution : DataFrame with columns ``method`` and at least *metric*.
        Typically one row per method from Strategy 1 evaluation.
    metric : distance column name (e.g. ``"mmd"`` or ``"wasserstein"``).
    ax : optional matplotlib Axes.
    save_path : if given, save the figure to this path.
    """
    methods = df_distribution["method"].tolist()
    values = df_distribution[metric].tolist()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.5), 5))
    else:
        fig = ax.figure

    colors = ["steelblue", "tomato", "seagreen", "darkorange"]
    bar_colors = [colors[i % len(colors)] for i in range(len(methods))]

    bars = ax.bar(methods, values, color=bar_colors)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Distribution-level {metric.upper()} vs Reference")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.4f}", ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved plot → %s", save_path)
    if own_fig:
        plt.close(fig)
