import numpy as np
import librosa

import matplotlib.pyplot as plt
import librosa.display

def plot_spectrograms(audios, fs):
    if not isinstance(audios, list) or len(audios) == 0:
        raise ValueError("audios must be a list with one or more audio arrays")
    
    fig, ax = plt.subplots(nrows=len(audios), ncols=1, sharex=False, figsize=(10, 4 * len(audios)))
    if len(audios) == 1:
        ax = [ax]

    for i, audio in enumerate(audios):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=fs, ax=ax[i])
        ax[i].set_ylim([0, 8000])
        duration = len(audio) / fs
        ax[i].set_ylabel('Frequency [Hz]')
        ax[i].set_xlabel(f'Time [s] (Duration: {duration:.2f}s)')
    
    plt.tight_layout()
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
