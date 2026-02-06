import os
import argparse
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np

from src.utils.config import load_config

def preprocess_audio(config, split):
    """
    Preprocesses audio files for a given split:
    - Resamples to target sample rate.
    - Normalizes audio.
    - Segments into fixed-length clips.
    - Skips silent clips.
    - Saves processed clips and a new manifest.
    """
    splits_dir = config['paths']['splits']
    processed_dir = config['paths']['data_processed']
    
    # Define output directories
    output_audio_dir = os.path.join(processed_dir, 'audio', split)
    output_manifest_dir = os.path.join(processed_dir, 'manifests')
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(output_manifest_dir, exist_ok=True)

    # Load the split manifest
    split_manifest_path = os.path.join(splits_dir, f"{split}.csv")
    if not os.path.exists(split_manifest_path):
        print(f"Error: Split manifest not found at {split_manifest_path}")
        return
    
    df = pd.read_csv(split_manifest_path)
    print(f"Processing {len(df)} files for the '{split}' split...")

    # Get config parameters
    target_sr = config['audio']['sr']
    clip_len_sec = config['audio']['clip_seconds']
    clip_len_samples = int(target_sr * clip_len_sec)
    rms_threshold = 1e-4 # Corresponds to -80 dB, a common silence threshold

    processed_records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        try:
            # Load, resample, and convert to mono
            wav, sr = librosa.load(row['path'], sr=target_sr, mono=True)

            # Peak normalization
            wav /= np.max(np.abs(wav)) + 1e-7

            # Segment into clips
            for i in range(0, len(wav) - clip_len_samples, clip_len_samples):
                clip = wav[i : i + clip_len_samples]

                # Skip silent clips
                rms = np.sqrt(np.mean(clip**2))
                if rms < rms_threshold:
                    continue
                
                clip_idx = i // clip_len_samples
                output_filename = f"{row['uid']}__{clip_idx}.wav"
                output_path = os.path.join(output_audio_dir, output_filename)

                # Save the clip
                sf.write(output_path, clip, target_sr)

                processed_records.append({
                    'path': output_path,
                    'original_uid': row['uid'],
                    'clip_idx': clip_idx,
                    'duration_sec': clip_len_sec,
                    'sr': target_sr
                })

        except Exception as e:
            print(f"Warning: Failed to process {row['path']}. Error: {e}")

    # Save the new manifest for the processed clips
    if not processed_records:
        print(f"No clips were generated for the '{split}' split. Check RMS threshold and audio files.")
        return

    processed_df = pd.DataFrame(processed_records)
    output_manifest_path = os.path.join(output_manifest_dir, f"{split}.csv")
    processed_df.to_csv(output_manifest_path, index=False)

    print(f"\nPreprocessing for '{split}' split complete.")
    print(f"Generated {len(processed_df)} clips.")
    print(f"Processed clips saved in: {output_audio_dir}")
    print(f"Processed manifest saved to: {output_manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio clips from a data split.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the config file.")
    parser.add_argument("--split", required=True, choices=['train', 'val', 'test'], help="Which data split to process.")
    parser.add_argument("--override", nargs='*', default=[], help="Override config values.")
    
    args = parser.parse_args()
    
    config = load_config(args.config, args.override)
    preprocess_audio(config, args.split)

if __name__ == "__main__":
    main()
