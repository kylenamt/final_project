# Deterministic Timbre Transfer Baseline Implementations using pyworld and DSP tools
# Includes: Source-Filter Vocoder, Spectral Envelope Transfer, Spectral Modeling Synthesis (SMS), Additive, Subtractive,
# Phase Vocoder Modification, Formant Shifting, and Concatenative Unit Selection (simplified)

import numpy as np
import pyworld as pw
import librosa
import scipy.signal
import soundfile as sf
import os


def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr


def world_decompose(y, fs):
    f0, timeaxis = pw.harvest(y, fs)
    sp = pw.cheaptrick(y, f0, timeaxis, fs)
    ap = pw.d4c(y, f0, timeaxis, fs)
    return f0, sp, ap


def world_synthesize(f0, sp, ap, fs):
    return pw.synthesize(f0, sp, ap, fs)


# 1. Source-Filter Vocoder Timbre Transfer

def source_filter_transfer(f0_tgt, ap_tgt, sp_src):
    return world_synthesize(f0_tgt, sp_src, ap_tgt, fs=len(f0_tgt))


# 2. Spectral Envelope Transfer (Cross-Synthesis)

def spectral_envelope_transfer(sp_source, sp_target):
    eps = 1e-6
    envelope_source = scipy.signal.medfilt(np.log(sp_source + eps), kernel_size=(1, 9))
    envelope_target = scipy.signal.medfilt(np.log(sp_target + eps), kernel_size=(1, 9))
    sp_target_flat = sp_target / (np.exp(envelope_target) + eps)
    sp_transferred = sp_target_flat * np.exp(envelope_source)
    return sp_transferred


# 3. Spectral Modeling Synthesis (Additive + Noise)

def sms_sinusoidal_model(f0, harmonic_amp, fs=16000, num_harmonics=40):
    T = len(f0)
    t = np.linspace(0, T / fs, T)
    signal = np.zeros_like(t)
    for k in range(1, num_harmonics + 1):
        freq = k * f0
        amp = harmonic_amp[:, k - 1] if harmonic_amp.ndim > 1 else harmonic_amp
        phase = 2 * np.pi * freq * t
        signal += amp * np.sin(phase)
    return signal


# 4. Additive Synthesis via Harmonic Resynthesis

def additive_synthesis(f0, harmonic_distribution, amplitude, fs=16000):
    T = len(f0)
    t = np.linspace(0, T / fs, T)
    signal = np.zeros_like(t)
    for k in range(harmonic_distribution.shape[1]):
        freq = (k + 1) * f0
        amp = amplitude * harmonic_distribution[:, k]
        phase = 2 * np.pi * freq * t
        signal += amp * np.sin(phase)
    return signal


# 5. Subtractive Synthesis (filtered excitation)

def subtractive_synthesis(envelope, excitation):
    return scipy.signal.fftconvolve(excitation, envelope, mode='same')


# 6. Phase Vocoder-Based Timbre Modification

def phase_vocoder_transfer(y_src, y_tgt, sr):
    S_src = librosa.stft(y_src)
    S_tgt = librosa.stft(y_tgt)
    mag_src = np.abs(S_src)
    mag_tgt = np.abs(S_tgt)
    phase_tgt = np.angle(S_tgt)
    mag_mix = mag_src
    S_new = mag_mix * np.exp(1j * phase_tgt)
    y_out = librosa.istft(S_new)
    return y_out


# 7. Concatenative Synthesis (simplified frame matching)

def mfcc_concatenate(source_frames, target_frames):
    matched = []
    for f_src in source_frames:
        dists = [np.linalg.norm(f_src - f_tgt) for f_tgt in target_frames]
        best = np.argmin(dists)
        matched.append(target_frames[best])
    return np.concatenate(matched, axis=0)


def save_audio(path, y, sr):
    sf.write(path, y, sr)

# Placeholder utilities for feature extraction and matching (for unit selection, needs frames and MFCCs extracted)
# For each method, plug into an evaluation or testing harness to process audio pairs
