import numpy as np
import librosa

import matplotlib.pyplot as plt
import librosa.display
from scipy import signal as scipy_signal

DEFAULT_SAMPLE_RATE = 16000

def plot_spectrograms(audios, fs, vmin=-5, vmax=1, size=512 + 256):
    if not isinstance(audios, list) or len(audios) == 0:
        raise ValueError("audios must be a list with one or more audio arrays")

    for i, audio in enumerate(audios):
        if len(audio.shape) == 2:
            audio = audio[0]
        _, _, Sxx = scipy_signal.stft(audio, fs=fs, nperseg=size,
                                      noverlap=size * 3 // 4)
        logmag = np.log10(np.abs(Sxx) + 1e-7)
        logmag = np.flipud(logmag)
        plt.matshow(logmag, vmin=vmin, vmax=vmax, cmap=plt.cm.magma, aspect='auto') # type: ignore
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title(f'Audio {i+1}')
        plt.show()
    
def plot_features(audio_features, trim=-15):
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 6))
    ax[0].plot(audio_features['loudness_db'][:trim])
    ax[0].set_ylabel('loudness_db')
    ax[1].plot(librosa.hz_to_midi(audio_features['f0_hz'][:trim]))
    ax[1].set_ylabel('f0 [midi]')
    ax[2].plot(audio_features['f0_confidence'][:trim])
    ax[2].set_ylabel('f0 confidence')
    ax[2].set_xlabel('frame')
    plt.show()

from feature_utils import compute_features
def plot_feature_from_audio(audio, sr=DEFAULT_SAMPLE_RATE, trim=-15):
    features = compute_features(audio, sr)
    plot_features(features, trim=trim)

def plot_series(feature_series,name=None):
    plt.figure(figsize=(6, 4))
    for key, series in feature_series.items():
        plt.plot(series, label=key)
    plt.xlabel('Frame')
    plt.ylabel('Value')
    title = f"Feature Series for {name}" if name else "Feature Series"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()