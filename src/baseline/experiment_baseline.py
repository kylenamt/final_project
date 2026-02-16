import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import median_filter

# ----------------------------
# Utilities
# ----------------------------

def hz_to_midi_safe(f):
    """Hz -> MIDI, but handles 0/NaN."""
    f = np.asarray(f)
    out = np.full_like(f, np.nan, dtype=float)
    mask = np.isfinite(f) & (f > 0)
    out[mask] = librosa.hz_to_midi(f[mask])
    return out

def smooth_envelope_from_mag(mag, n_fft, sr, cep_lifter=40):
    """
    Cepstral smoothing of log magnitude spectrum to obtain a smooth spectral envelope.
    mag: (n_freq,) linear magnitude
    Returns env: (n_freq,) linear magnitude envelope (positive).
    """
    mag = np.maximum(mag, 1e-8)
    log_mag = np.log(mag)

    # Real cepstrum via irfft (log magnitude is real and even-ish)
    cep = np.fft.irfft(log_mag, n=n_fft)

    # Low-quefrency lifter: keep only first cep_lifter coeffs
    cep_s = np.zeros_like(cep)
    keep = min(len(cep_s), cep_lifter)
    cep_s[:keep] = cep[:keep]

    # Back to smoothed log spectrum
    log_env = np.fft.rfft(cep_s, n=n_fft).real

    env = np.exp(log_env)
    return np.maximum(env, 1e-8)

def build_envelope_codebook(
    violin_wav,
    sr=16000,
    n_fft=2048,
    hop=256,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
    midi_bin_size=1.0,
    cep_lifter=40,
    min_conf=0.7,
):
    """
    Build a dictionary: pitch_bin -> median spectral envelope (linear mag, per STFT bin).
    Uses librosa.pyin for f0 estimate.
    """
    x, sr = librosa.load(violin_wav, sr=sr, mono=True)

    # STFT magnitude
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    mag = np.abs(S)  # (n_freq, n_frames)

    # Pitch tracking on time frames consistent with hop
    f0, voiced_flag, voiced_prob = librosa.pyin(
        x, fmin=fmin, fmax=fmax, sr=sr, frame_length=n_fft, hop_length=hop
    )
    midi = hz_to_midi_safe(f0)

    # Bin by MIDI
    codebook_lists = {}  # pitch_bin -> list of envelopes

    for t in range(mag.shape[1]):
        if not np.isfinite(midi[t]):
            continue
        if voiced_prob is not None and np.isfinite(voiced_prob[t]) and voiced_prob[t] < min_conf:
            continue
        pitch_bin = int(np.round(midi[t] / midi_bin_size) * midi_bin_size)

        env = smooth_envelope_from_mag(mag[:, t], n_fft=n_fft, sr=sr, cep_lifter=cep_lifter)
        codebook_lists.setdefault(pitch_bin, []).append(env)

    if not codebook_lists:
        raise RuntimeError("No voiced frames found in violin reference. Try a cleaner violin clip or relax min_conf.")

    # Aggregate to a single prototype per bin (median for robustness)
    codebook = {}
    for k, envs in codebook_lists.items():
        E = np.stack(envs, axis=1)  # (n_freq, n_envs)
        codebook[k] = np.median(E, axis=1)

    return codebook, sr, n_fft, hop

def nearest_key(codebook, k):
    keys = np.array(sorted(codebook.keys()))
    idx = np.argmin(np.abs(keys - k))
    return int(keys[idx])

def apply_codebook_envelopes(
    voice_wav,
    codebook,
    sr=16000,
    n_fft=2048,
    hop=256,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
    midi_bin_size=1.0,
    cep_lifter=40,
    min_conf=0.7,
    smooth_time=5,
):
    """
    Voice -> "violinized" magnitude:
    - Compute voice STFT magnitude and smooth envelope
    - Flatten voice magnitude by dividing its envelope
    - Multiply by looked-up violin envelope per pitch-bin
    - Resynthesize with original phase (fast)
    """
    x, sr = librosa.load(voice_wav, sr=sr, mono=True)

    S = librosa.stft(x, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    mag = np.abs(S)
    phase = np.angle(S)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        x, fmin=fmin, fmax=fmax, sr=sr, frame_length=n_fft, hop_length=hop
    )
    midi = hz_to_midi_safe(f0)

    # Prepare output magnitude
    out_mag = np.zeros_like(mag)

    for t in range(mag.shape[1]):
        # Voice envelope (for flattening)
        env_voice = smooth_envelope_from_mag(mag[:, t], n_fft=n_fft, sr=sr, cep_lifter=cep_lifter)
        flat = mag[:, t] / np.maximum(env_voice, 1e-8)

        # If unvoiced, just keep flatter spectrum lightly shaped by a neutral bin
        if not np.isfinite(midi[t]) or (voiced_prob is not None and np.isfinite(voiced_prob[t]) and voiced_prob[t] < min_conf):
            # choose a mid key (or nearest available)
            k = nearest_key(codebook, int(np.median(list(codebook.keys()))))
        else:
            pitch_bin = int(np.round(midi[t] / midi_bin_size) * midi_bin_size)
            k = nearest_key(codebook, pitch_bin)

        env_violin = codebook[k]
        out_mag[:, t] = flat * env_violin

    # Optional temporal smoothing to reduce "dictionary jitter"
    if smooth_time and smooth_time > 1:
        out_mag = median_filter(out_mag, size=(1, smooth_time))

    # Recompose complex STFT using original phase
    S_out = out_mag * np.exp(1j * phase)

    y = librosa.istft(S_out, hop_length=hop, window="hann", center=True, length=len(x))
    y = y / (np.max(np.abs(y)) + 1e-8)
    return y, sr

# ----------------------------
# Quick demo runner
# ----------------------------
if __name__ == "__main__":
    # 1) Provide paths
    violin_ref = "violin_reference.wav"   # clean monophonic violin
    voice_in   = "voice_input.wav"        # monophonic voice / singing
    out_path   = "voice_to_violin_baseline.wav"

    # 2) Build codebook from violin
    codebook, sr, n_fft, hop = build_envelope_codebook(
        violin_ref,
        sr=16000,
        n_fft=2048,
        hop=256,
        midi_bin_size=1.0,
        cep_lifter=40,
        min_conf=0.7,
    )
    print(f"Codebook bins: {len(codebook)}")

    # 3) Apply to voice
    y, sr = apply_codebook_envelopes(
        voice_in,
        codebook,
        sr=sr,
        n_fft=n_fft,
        hop=hop,
        midi_bin_size=1.0,
        cep_lifter=40,
        min_conf=0.7,
        smooth_time=5,
    )

    # 4) Save
    sf.write(out_path, y, sr)
    print("Wrote:", out_path)
