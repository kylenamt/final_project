import numpy as np
import librosa

import matplotlib.pyplot as plt
import librosa.display

def create_spectrogram_plot(audios, fs):
    if not isinstance(audios, list) or len(audios) == 0:
        raise ValueError("audios must be a list with one or more audio arrays")
    
    fig, ax = plt.subplots(nrows=len(audios), ncols=1, sharex=True, figsize=(10, 8))
    if len(audios) == 1:
        ax = [ax]

    for i, audio in enumerate(audios):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=fs, ax=ax[i])
        ax[i].set_ylabel('Frequency [Hz]')
        ax[i].set_xlabel('Time [s]')
    # Original audio

    plt.tight_layout()
    plt.show()
    
