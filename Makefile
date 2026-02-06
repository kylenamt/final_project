SHELL := /bin/bash

INPUT_PATTERN ?= data/ddsp_data/raw_audio/*.wav
OUTPUT_TFRECORD ?= data/ddsp_data/train/vocalset.tfrecord
TFRECORD_PATH ?= data/ddsp_data/train/*.tfrecord
SAVE_DIR ?= artifacts/trained_models/
BATCH_SIZE ?= 4
GIN_MODEL ?= models/solo_instrument.gin
GIN_DATASET ?= datasets/tfrecord.gin
GIN_EVAL ?= eval/basic_f0_ld.gin
GIN_SEARCH_PATH ?= $(shell $(PYTHON) -c "import os,sys;\
import ddsp;\
print(os.path.join(os.path.dirname(ddsp.__file__), 'training', 'gin'))" 2>/dev/null)
PYTHON ?= python

.PHONY: help prepare train eval sample

help:
	@echo "Targets:"
	@echo "  prepare  - Build TFRecord dataset from INPUT_PATTERN"
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
	TFRECORD_PATH="$(TFRECORD_PATH)" \
	SAVE_DIR="$(SAVE_DIR)" \
	BATCH_SIZE="$(BATCH_SIZE)" \
	GIN_MODEL="$(GIN_MODEL)" \
	GIN_DATASET="$(GIN_DATASET)" \
	GIN_EVAL="$(GIN_EVAL)" \
	GIN_SEARCH_PATH="$(GIN_SEARCH_PATH)" \
	PYTHON_BIN="$(PYTHON)" \
	MODE=sample \
	bash scripts/train_ddsp.sh
