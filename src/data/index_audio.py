import os
import argparse
import pandas as pd
from tqdm import tqdm
import librosa
import soundfile as sf

from src.utils.config import load_config

def get_audio_metadata(file_path):
    """Extracts metadata from an audio file."""
    try:
        info = sf.info(file_path)
        duration = info.duration
        sr = info.samplerate
        return duration, sr
    except Exception as e:
        print(f"Warning: Could not read {file_path}. Error: {e}")
        return None, None

def index_audio_files(config):
    """
    Scans the raw data directory, extracts metadata, and saves it to a manifest CSV.
    """
    raw_data_dir = config['paths']['data_raw']
    manifest_dir = config['paths']['manifests']
    dataset_name = os.path.basename(os.path.dirname(raw_data_dir)) # e.g., 'VocalSet'

    os.makedirs(manifest_dir, exist_ok=True)
    
    print(f"Scanning for .wav files in {raw_data_dir}...")
    
    records = []
    for root, _, files in os.walk(raw_data_dir):
        for file in tqdm(files, desc="Indexing files"):
            if file.lower().endswith('.wav'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, raw_data_dir)
                
                duration, sr = get_audio_metadata(full_path)
                if duration is None:
                    continue

                # Create a unique ID from the file path
                uid = os.path.splitext(rel_path.replace(os.sep, '_'))[0]

                records.append({
                    'path': full_path,
                    'dataset': dataset_name,
                    'uid': uid,
                    'duration_sec': duration,
                    'sr': sr
                })

    if not records:
        print("No .wav files found. Exiting.")
        return

    manifest_df = pd.DataFrame(records)
    output_path = os.path.join(manifest_dir, f"{dataset_name}.csv")
    manifest_df.to_csv(output_path, index=False)
    
    print(f"Successfully created manifest with {len(manifest_df)} entries.")
    print(f"Manifest saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Index audio files and create a manifest.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the config file.")
    parser.add_argument("--override", nargs='*', default=[], help="Override config values.")
    
    args = parser.parse_args()
    
    config = load_config(args.config, args.override)
    index_audio_files(config)

if __name__ == "__main__":
    main()
