# Project To-Do Tasks — DDSP Timbre Transfer

---

## Phase 1 — Data Preparation
**Goal:** Obtain clean, well-structured datasets ready for modeling.

- [ ] Identify target **monophonic instrument dataset** (e.g., violin, flute)
- [ ] Identify **source dataset** for timbre transfer (e.g., singing voice, other instrument)
- [ ] Verify licensing and usage permissions
- [ ] Convert all audio to a **common format** (sample rate, mono, bit depth)
- [ ] Remove corrupted or low-quality samples
- [ ] Segment long recordings into fixed-length clips
- [ ] Trim silence and clean noise if needed
- [ ] Organize dataset directory structure
- [ ] Create **train / validation / test splits**
- [ ] Document dataset statistics (duration, clip count, pitch range)

---

## Phase 2 — Feature Extraction & Preprocessing
**Goal:** Produce conditioning features compatible with baseline and DDSP.

- [ ] Extract **F0 (pitch)** for all samples
- [ ] Extract **loudness** features
- [ ] Extract **spectral envelope / MFCCs**
- [ ] Validate extracted features (ranges, NaNs)
- [ ] Align feature frame rate with audio frames
- [ ] Normalize features (mean/std or min/max)
- [ ] Save features in standardized format (NumPy / TFRecord)
- [ ] Implement reproducible preprocessing scripts
- [ ] Log preprocessing configuration

---

## Phase 3 — Data Adapter & Pipeline
**Goal:** Build a reusable data-loading pipeline.

- [ ] Define unified data format for **DDSP and baseline**
- [ ] Implement dataset loader module
- [ ] Implement batching and shuffling
- [ ] Ensure deterministic validation/test loading
- [ ] Test pipeline on small subset
- [ ] Add data sanity checks

---

## Phase 4 — Baseline Model Setup
**Goal:** Establish a fair comparison reference.

- [ ] Select baseline method (**WORLD vocoder**)
- [ ] Implement or integrate WORLD pipeline
- [ ] Configure baseline conditioning (F0, loudness)
- [ ] Generate baseline outputs on test set
- [ ] Store baseline audio outputs
- [ ] Document baseline configuration

---

## Phase 5 — DDSP Model Configuration
**Goal:** Prepare the DDSP timbre-transfer system.

- [ ] Select DDSP model variant (supervised)
- [ ] Define encoder inputs (F0, loudness, optional residual)
- [ ] Configure synthesizer components (harmonic + noise)
- [ ] Define loss functions (multi-scale spectral loss)
- [ ] Set training hyperparameters
- [ ] Implement logging (losses, audio samples)
- [ ] Enable checkpointing and config versioning

---

## Phase 6 — Model Training
**Goal:** Train models and collect artifacts.

- [ ] Run initial training sanity check
- [ ] Train DDSP model on full dataset
- [ ] Monitor convergence and stability
- [ ] Save checkpoints periodically
- [ ] Generate intermediate audio samples
- [ ] Archive logs and model versions
- [ ] Select best checkpoint via validation loss

---

## Phase 7 — Evaluation & Analysis
**Goal:** Quantitative and qualitative assessment.

- [ ] Run DDSP inference on test set
- [ ] Run baseline inference on test set
- [ ] Compute **spectral distance**
- [ ] Compute **F0 error**
- [ ] Compute **loudness error**
- [ ] Compare DDSP vs baseline results
- [ ] Prepare samples for listening tests
- [ ] Conduct limited subjective evaluation (if feasible)
- [ ] Analyze artifacts and failure cases
- [ ] Summarize observations and insights

---

## Phase 8 — Reporting & Documentation
**Goal:** Final deliverables.

- [ ] Compile figures and result tables
- [ ] Document methodology and setup
- [ ] Write evaluation and discussion
- [ ] Reflect on limitations and future work
- [ ] Finalize report or presentation slides
- [ ] Prepare materials for advisor review

---

## Ongoing — Project Management
- [ ] Track progress in bi-weekly advisor meetings
- [ ] Update task priorities as scope evolves
- [ ] Maintain experiment and decision log

---

*Note: The schedule is tentative; future development tasks will be agreed upon during bi-weekly meetings with the advisor.*
8cuR68rqcDQr5-