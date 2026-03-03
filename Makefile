SHELL := /bin/bash

INPUT_PATTERN ?= data/preprocessed/solo_violin/*.wav
OUTPUT_TFRECORD ?= data/tfrecords/solo_violin/
TFRECORD_PATH ?= data/tfrecords/solo_violin/*.tfrecord

# SAMPLE_PATH ?= data/voice/forte/*.wav
# SAMPLE_TFRECORD ?= data/voice/*.tfrecord

SAVE_DIR ?= artifacts/trained_models/
SAVE_DIR_NO_REVERB ?= artifacts/trained_noreverb/

BATCH_SIZE ?= 8

GIN_MODEL ?= models/solo_instrument.gin
GIN_DATASET ?= datasets/tfrecord.gin
GIN_EVAL ?= eval/basic_f0_ld.gin
GIN_SEARCH_PATH ?= configs/ddsp_gin

PYTHON ?= python

.PHONY: help prepare prepare-input train eval sample

help:
	@echo "Targets:"
	@echo "  prepare  - Build TFRecord dataset from INPUT_PATTERN"
	@echo "  prepare-input  - Build TFRecord dataset from INPUT_PATTERN (input-only)"
	@echo "  train    - Train DDSP model"
	@echo "  eval     - Evaluate DDSP model"
	@echo "  sample   - Sample from DDSP model"
	@echo "Variables (override with VAR=...):"
	@echo "  INPUT_PATTERN, OUTPUT_TFRECORD, TFRECORD_PATH, SAVE_DIR, BATCH_SIZE"
	@echo "  GIN_MODEL, GIN_DATASET, GIN_EVAL, GIN_SEARCH_PATH, PYTHON"


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

train-reverb:
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

train-no-reverb:
	TFRECORD_PATH="$(TFRECORD_PATH)" \
	SAVE_DIR="$(SAVE_DIR_NO_REVERB)" \
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
