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
        duration = len(audio) / fs
        ax[i].set_ylabel('Frequency [Hz]')
        ax[i].set_xlabel(f'Time [s] (Duration: {duration:.2f}s)')
    
    plt.tight_layout()
    plt.show()
    
