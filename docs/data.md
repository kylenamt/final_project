# Data and preprocessing

## Expected layout

- data/raw/: input audio
- data/processed/: trimmed and resampled audio
- data/tfrecords/: TFRecord shards for training
- artifacts/: model checkpoints and outputs

## Prepare TFRecords

Place mono WAV files under data/raw/solo_violin/ (or update INPUT_PATTERN),
then run:

```bash
make prepare
```

This calls ddsp_prepare_tfrecord and writes TFRecords under data/tfrecords/.

## Preprocessing helpers

- src/data_preprocessing.py: silence trimming, resampling, splitting
- src/feature_utils.py: feature extraction and auto-adjustment
