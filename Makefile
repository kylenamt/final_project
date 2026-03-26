SHELL := /bin/bash

# Load .env file if present
-include .env
export HF_REPO HF_TOKEN

# Ensure TF can find CUDA/cuDNN libs from the conda environment
ifdef CONDA_PREFIX
export LD_LIBRARY_PATH := $(CONDA_PREFIX)/lib$(if $(LD_LIBRARY_PATH),:$(LD_LIBRARY_PATH))
endif

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS := $(shell awk '!/^\s*\#/ && NF {print $$1}' configs/training_config/presets.mk)

PRESET ?=
ifneq ($(filter train eval sample upload-hf download-hf,$(firstword $(MAKECMDGOALS))),)
    ifeq ($(PRESET),)
        PRESET := $(word 2,$(MAKECMDGOALS))
    endif
endif

ifneq ($(PRESET),)
    ifeq ($(filter $(PRESET),$(PRESETS)),)
        $(error Unknown preset '$(PRESET)'. Valid presets: $(PRESETS))
    endif

    SAVE_DIR     ?= $(shell awk '/^$(PRESET)\s/ {print $$2}' configs/training_config/presets.mk)
    MODEL_DIR    ?= $(shell awk '/^$(PRESET)\s/ {print $$3}' configs/training_config/presets.mk)
    PATH_IN_REPO ?= $(shell awk '/^$(PRESET)\s/ {print $$4}' configs/training_config/presets.mk)

    $(info Using preset '$(PRESET)': SAVE_DIR=$(SAVE_DIR), MODEL_DIR=$(MODEL_DIR))
endif

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_PATTERN   ?= data/raw/solo_violin/*.wav
OUTPUT_TFRECORD ?= data/tfrecords/solo_violin/solo_violin.tfrecord
TFRECORD_PATH   ?= data/tfrecords/solo_violin/*.tfrecord

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
BATCH_SIZE      ?= 16
GIN_MODEL       ?= models/solo_instrument.gin
GIN_DATASET     ?= datasets/tfrecord.gin
GIN_EVAL        ?= eval/basic_f0_ld.gin
GIN_SEARCH_PATH ?= configs/ddsp_gin
PYTHON          ?= python

TRAIN_ENV = \
    SAVE_DIR="$(SAVE_DIR)" \
    BATCH_SIZE="$(BATCH_SIZE)" \
    GIN_MODEL="$(GIN_MODEL)" \
    GIN_DATASET="$(GIN_DATASET)" \
    GIN_EVAL="$(GIN_EVAL)" \
    GIN_SEARCH_PATH="$(GIN_SEARCH_PATH)" \
    PYTHON_BIN="$(PYTHON)"

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: help setup lock docker docker-build prepare prepare-sample train eval sample upload-hf download-hf $(PRESETS)

# Preset names are consumed as no-op targets so `make train ae` works
$(PRESETS):
	@:

setup:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Done. Activate with:  conda activate conda_env3.10"

docker-build:
	docker compose build

docker:
	docker compose run --rm ddsp

lock:
	conda list --export > conda-lock.txt
	pip freeze > requirements-lock.txt
	@echo "Lock files updated: conda-lock.txt, requirements-lock.txt"

define prepare_tfrecord
	ddsp_prepare_tfrecord \
		--input_audio_filepatterns="$(1)" \
		--output_tfrecord_path="$(2)" \
		--num_shards=10 \
		--alsologtostderr
endef

prepare:
	$(call prepare_tfrecord,$(INPUT_PATTERN),$(OUTPUT_TFRECORD))

prepare-sample:
	$(call prepare_tfrecord,$(SAMPLE_PATH),$(SAMPLE_TFRECORD))

train eval sample:
	$(TRAIN_ENV) \
	TFRECORD_PATH="$(if $(filter sample,$@),$(SAMPLE_TFRECORD),$(TFRECORD_PATH))" \
	MODE=$@ \
	bash scripts/train_ddsp.sh

upload-hf:
	$(PYTHON) scripts/upload_to_hf.py \
		--repo "$(HF_REPO)" \
		--model-dir "$(MODEL_DIR)" \
		--path-in-repo "$(PATH_IN_REPO)" \
		--include-checkpoint-file

download-hf:
	$(PYTHON) scripts/download_from_hf.py \
		--repo "$(HF_REPO)" \
		--path-in-repo "$(PATH_IN_REPO)" \
		--model-dir "$(MODEL_DIR)"