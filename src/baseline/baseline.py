# Deterministic Timbre Transfer Baseline Implementations using pyworld and DSP tools
# Includes: Source-Filter Vocoder, Spectral Envelope Transfer, Spectral Modeling Synthesis (SMS), Additive, Subtractive,
# Phase Vocoder Modification, Formant Shifting, and Concatenative Unit Selection (simplified)

import numpy as np
import pyworld as pw
import librosa
import scipy.signal
import soundfile as sf
import os

class Baseline:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate


    def load_audio(self, path, sr=16000):
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float64), sr

    
    def match_length(self, y, target_len):
        if len(y) < target_len:
            return np.pad(y, (0, target_len - len(y)), 'wrap')
        else:
            return y[:target_len]

    def decompose(self, y, fs):
        f0, timeaxis = pw.dio(y, fs)  # type: ignore
        f0 = pw.stonemask(y, f0, timeaxis, fs)  # type: ignore
        sp = pw.cheaptrick(y, f0, timeaxis, fs)  # type: ignore
        ap = pw.d4c(y, f0, timeaxis, fs)  # type: ignore
        return f0, sp, ap


    def synthesize(self, f0, sp, ap, fs):
        return pw.synthesize(f0, sp, ap, fs)  # type: ignore
    
    
    def f0_transfer(self, source_y, target_y, output_path = None):

        if len(target_y) != len(source_y):
            raise ValueError("Target audio must be at least as long as source audio for F0 transfer.")

        source_f0, source_sp, source_ap = self.decompose(source_y, self.sample_rate)
        target_f0, target_sp, target_ap = self.decompose(target_y, self.sample_rate)

        synthesized_y = self.synthesize(source_f0, target_sp, target_ap, self.sample_rate)

        if output_path is not None:
            sf.write(output_path, synthesized_y, self.sample_rate)
        return synthesized_y
    
    def f0_and_ap_transfer(self, source_y, target_y, output_path = None):

        if len(target_y) != len(source_y):
            raise ValueError("Target audio must be at least as long as source audio for F0 transfer.")

        source_f0, source_sp, source_ap = self.decompose(source_y, self.sample_rate)
        target_f0, target_sp, target_ap = self.decompose(target_y, self.sample_rate)

        synthesized_y = self.synthesize(source_f0, target_sp, source_ap, self.sample_rate)

        if output_path is not None:
            sf.write(output_path, synthesized_y, self.sample_rate)
        return synthesized_y
    

    

