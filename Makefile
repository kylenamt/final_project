SHELL := /bin/bash

INPUT_PATTERN ?= data/raw/solo_violin/*.wav
OUTPUT_TFRECORD ?= data/tfrecords/solo_violin/solo_violin.tfrecord
TFRECORD_PATH ?= data/tfrecords/solo_violin/*.tfrecord

# SAMPLE_PATH ?= data/voice/forte/*.wav
# SAMPLE_TFRECORD ?= data/voice/*.tfrecord

SAVE_DIR ?= artifacts/ae2/
MODEL_DIR ?= artifacts/ae2/
# PATH_IN_REPO ?= trained_noreverb
# HF_REPO ?=

BATCH_SIZE ?= 8

GIN_MODEL ?= models/ae.gin
GIN_DATASET ?= datasets/tfrecord.gin
GIN_EVAL ?= eval/basic_f0_ld.gin
GIN_SEARCH_PATH ?= configs/ddsp_gin

PYTHON ?= python

.PHONY: help setup lock prepare prepare-input train eval sample

setup:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Done. Activate with:  conda activate conda_env3.10"

lock:
	conda list --export > conda-lock.txt
	pip freeze > requirements-lock.txt
	@echo "Lock files updated: conda-lock.txt, requirements-lock.txt"

help:
	@echo "Targets:"
	@echo "  setup    - Create/update conda env from environment.yml"
	@echo "  lock     - Regenerate conda-lock.txt & requirements-lock.txt"
	@echo "  prepare  - Build TFRecord dataset from INPUT_PATTERN"
	@echo "  train    - Train DDSP model"
	@echo "  eval     - Evaluate DDSP model"
	@echo "  sample   - Sample from DDSP model"
	@echo "  upload-hf - Upload latest checkpoint + gin to Hugging Face"
	@echo "Variables (override with VAR=...):"
	@echo "  INPUT_PATTERN, OUTPUT_TFRECORD, TFRECORD_PATH, SAVE_DIR, BATCH_SIZE"
	@echo "  GIN_MODEL, GIN_DATASET, GIN_EVAL, GIN_SEARCH_PATH, PYTHON"
	@echo "  HF_REPO, MODEL_DIR, PATH_IN_REPO"


prepare:
	ddsp_prepare_tfrecord \
		--input_audio_filepatterns="$(INPUT_PATTERN)" \
		--output_tfrecord_path="$(OUTPUT_TFRECORD)" \
		--num_shards=10 \
		--alsologtostderr

prepare-sample:
	ddsp_prepare_tfrecord \
		--input_audio_filepatterns="$(SAMPLE_PATH)" \
		--output_tfrecord_path="$(SAMPLE_TFRECORD)" \
		--num_shards=10 \
		--alsologtostderr

# train-reverb:
# 	TFRECORD_PATH="$(TFRECORD_PATH)" \
# 	SAVE_DIR="$(SAVE_DIR)" \
# 	BATCH_SIZE="$(BATCH_SIZE)" \
# 	GIN_MODEL="$(GIN_MODEL)" \
# 	GIN_DATASET="$(GIN_DATASET)" \
# 	GIN_EVAL="$(GIN_EVAL)" \
# 	GIN_SEARCH_PATH="$(GIN_SEARCH_PATH)" \
# 	PYTHON_BIN="$(PYTHON)" \
# 	MODE=train \
# 	bash scripts/train_ddsp.sh

train:
	TFRECORD_PATH="$(TFRECORD_PATH)" \
	SAVE_DIR="$(SAVE_DIR)" \
	BATCH_SIZE="$(BATCH_SIZE)" \
	GIN_MODEL="$(GIN_MODEL)" \
	GIN_DATASET="$(GIN_DATASET)" \
	GIN_EVAL="$(GIN_EVAL)" \
	GIN_SEARCH_PATH="$(GIN_SEARCH_PATH)" \
	PYTHON_BIN="$(PYTHON)" \
	MODE=train \
	bash scripts/train_ddsp.sh

eval:
	TFRECORD_PATH="$(TFRECORD_PATH)" \
	SAVE_DIR="$(SAVE_DIR)" \
	BATCH_SIZE="$(BATCH_SIZE)" \
	GIN_MODEL="$(GIN_MODEL)" \
	GIN_DATASET="$(GIN_DATASET)" \
	GIN_EVAL="$(GIN_EVAL)" \
	GIN_SEARCH_PATH="$(GIN_SEARCH_PATH)" \
	PYTHON_BIN="$(PYTHON)" \
	MODE=eval \
	bash scripts/train_ddsp.sh

sample:
	TFRECORD_PATH="$(SAMPLE_TFRECORD)" \
	SAVE_DIR="$(SAVE_DIR)" \
	BATCH_SIZE="$(BATCH_SIZE)" \
	GIN_MODEL="$(GIN_MODEL)" \
	GIN_DATASET="$(GIN_DATASET)" \
	GIN_EVAL="$(GIN_EVAL)" \
	GIN_SEARCH_PATH="$(GIN_SEARCH_PATH)" \
	PYTHON_BIN="$(PYTHON)" \
	MODE=sample \
	bash scripts/train_ddsp.sh

upload-hf:
	$(PYTHON) scripts/upload_to_hf.py \
		--repo "$(HF_REPO)" \
		--model-dir "$(MODEL_DIR)" \
		--path-in-repo "$(PATH_IN_REPO)"
