import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.config import load_config

def get_singer_id(uid):
    """
    Extracts a singer/speaker ID from the UID.
    This assumes a UID structure like 'female1_arpeggios_belt_a_1_1'
    and extracts 'female1'. You may need to adjust this for your data.
    """
    return uid.split('_')[0]

def split_manifest(config):
    """
    Splits the main data manifest into train, validation, and test sets.
    Ensures that all files from a single singer belong to the same split.
    """
    manifest_dir = config['paths']['manifests']
    splits_dir = config['paths']['splits']
    dataset_name = os.path.basename(os.path.dirname(config['paths']['data_raw'])) # VocalSet
    
    os.makedirs(splits_dir, exist_ok=True)

    # Load the main manifest
    manifest_path = os.path.join(manifest_dir, f"{dataset_name}.csv")
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at {manifest_path}")
        print("Please run the data indexing script first.")
        return
        
    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df)} entries.")

    # Extract singer IDs to ensure speaker-independent splits
    df['singer_id'] = df['uid'].apply(get_singer_id)
    unique_singers = df['singer_id'].unique()
    
    print(f"Found {len(unique_singers)} unique singers.")

    # Get split ratios
    ratios = config['splits']['ratios']
    test_size = ratios['test']
    val_size = ratios['val'] / (1.0 - test_size) # Adjust val size for the second split

    # Split singers, not individual files
    seed = config['splits']['seed']
    train_singers, test_singers = train_test_split(unique_singers, test_size=test_size, random_state=seed)
    train_singers, val_singers = train_test_split(train_singers, test_size=val_size, random_state=seed)

    # Create dataframes for each split
    train_df = df[df['singer_id'].isin(train_singers)]
    val_df = df[df['singer_id'].isin(val_singers)]
    test_df = df[df['singer_id'].isin(test_singers)]

    # Save the splits
    train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)

    print("\nSplits created successfully:")
    print(f"  Train: {len(train_df)} files ({len(train_singers)} singers)")
    print(f"  Val:   {len(val_df)} files ({len(val_singers)} singers)")
    print(f"  Test:  {len(test_df)} files ({len(test_singers)} singers)")
    print(f"Splits saved in: {splits_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split data manifest into train/val/test sets.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the config file.")
    parser.add_argument("--override", nargs='*', default=[], help="Override config values.")
    
    args = parser.parse_args()
    
    config = load_config(args.config, args.override)
    split_manifest(config)

if __name__ == "__main__":
    main()
