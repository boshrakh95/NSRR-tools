# NSRR Preprocessing Pipeline — Complete Step-by-Step Guide

*Written 2026-06-03. Reference for paper writing and reproducibility.*
*Primary source of truth for how raw NSRR data becomes SleepFM embeddings.*

---

## Pipeline Overview

```
Step 0  Validate adapters (test scripts)
Step 1  Extract channel inventory (extract_nsrr_channels.py)
Step 2  Extract metadata (extract_metadata.py)
Step 3  Test preprocessing (test_preprocessing.py)
Step 4  Preprocess EDF → HDF5 (preprocess_signals.py)
Step 5  Extract task targets (extract_targets_*.py)
Step 6  Build master targets (create_master_targets.py)
Step 7  Build task subject lists (create_task_subject_lists.py)
Step 8  Extract SleepFM embeddings (extract_sleepfm_embeddings.py)
Step 9  Test dataset (test_context_window_dataset.py)
Step 10 Train models (train_context_sweep.py)
```

All steps are idempotent (safe to re-run; existing outputs are skipped by default).

---

## Step 0 — Validate Dataset Adapters

**Scripts:** `scripts/test_stages_adapter.py`, `test_shhs_adapter.py`, `test_apples_adapter.py`, `test_mros_adapter.py`

These scripts test that each dataset's adapter can:
- Find EDF files in the expected directory structure
- Parse annotation files (XML for SHHS/MrOS, CSV for STAGES, .annot for APPLES)
- Load metadata CSV files and resolve subject IDs

Run these before any heavy processing to catch path or format issues early.

---

## Step 1 — Extract Channel Inventory

**Script:** `scripts/extract_nsrr_channels.py`  
**Purpose:** Scans all EDF files in each dataset and records every channel name found.  
**Output:** `output/channel_analysis/{dataset}_channels.csv`, `all_unique_channels.txt`

This step is informational — it produces the channel inventory used to build and verify `configs/channel_definitions.yaml`. The channel definitions config is derived from this analysis plus `sleepfm-clinical/sleepfm/configs/channel_groups.json`.

**Important:** We do NOT use SleepFM's `channel_groups.json` directly in our pipeline. The SleepFM model is channel-agnostic (it assigns channels by position in a pre-built tensor, not by name). Our preprocessing pipeline builds the tensor manually using our own channel mapping. See §Channel Mapping Architecture below.

---

## Step 2 — Extract Metadata

**Script:** `scripts/extract_metadata.py`  
**Config:** `configs/paths.yaml`  
**Class:** `src/nsrr_tools/core/metadata_builder.py`  
**Output:** `/scratch/boshra95/psg/unified/metadata/unified_metadata.parquet`

Consolidates subject-level metadata (age, sex, BMI, AHI, etc.) from each dataset's CSV files into a unified parquet. This file is used by:
- `preprocess_signals.py` (to find which subjects have EDF files)
- `extract_targets_*.py` (to resolve subject IDs across dataset-specific formats)

Subject IDs vary by dataset:
- SHHS: `nsrrid` (numeric)
- MrOS: `nsrrid` (numeric)
- APPLES: `appleid` (in main CSV) / `nsrrid` (in harmonized CSV); matched via a merge/rename step
- STAGES: `subject_code` (string, NOT `nsrrid`)

---

## Step 3 — Test Preprocessing

**Script:** `scripts/test_preprocessing.py` (or `scripts/test_channel_config.py`)

Runs preprocessing on a small number of subjects to verify the pipeline works before submitting cluster jobs. Check output HDF5 files contain the expected channels and shapes.

---

## Step 4 — Preprocess EDF → HDF5

**Script:** `scripts/preprocess_signals.py`  
**Config:** `configs/preprocessing_params.yaml` (or `preprocessing_params_full.yaml`)  
**Supporting configs:** `configs/channel_definitions.yaml`, `configs/modality_groups.yaml`  
**Output:** `/scratch/boshra95/psg/{dataset}/derived/hdf5_signals/{subject_id}.h5`

### What it does

For each subject's EDF file:
1. Load EDF header using MNE-Python
2. Map dataset-specific channel names → standardised names (via `channel_definitions.yaml`)
3. Assign standardised channels to SleepFM modality groups (BAS/RESP/EKG/EMG) (via `modality_groups.yaml`)
4. Apply the channel limit strategy to decide which channels to save (via `preprocessing_params.yaml`)
5. For each selected channel: FIR bandpass filter → resample to 128 Hz → z-score normalise
6. Save to HDF5: float16, gzip compression level 4, chunk_size=38400 (5 min)
7. Also extract and save annotation files (sleep stage arrays as .npy)

### Channel selection strategy — TWO VERSIONS EXIST

| Config file | Strategy | BAS | RESP | EKG | EMG | Total max | Output path |
|---|---|---|---|---|---|---|---|
| `preprocessing_params.yaml` | **fast** (v1/v2/v3 runs) | 4 | 1 | 1 | 2 | ~8 | `/scratch/boshra95/psg/` |
| `preprocessing_params_full.yaml` | **sleepfm_full** | 10 | 7 | 2 | 4 | 23 | `/scratch/boshra95/psg_full/` |

**[QUESTION for paper]** Which strategy was used for the current v3 results? The CHANNEL_EXPANSION_GUIDE.md says "fast" was used in v1/v2/v3 runs (7–8 channels). The `preprocessing_params.yaml` has the `strategy:` key commented out, which would default to "sleepfm_full" — but the guide clearly states fast was the actual choice. For the paper Methods, we need to state the exact channel counts that reached SleepFM.

### Priority order for channel selection

Within each modality group, channels are selected in priority order (defined in `modality_groups.yaml` and re-stated in `phase0_v3_config.yaml`):

| Modality | Priority order (first N selected, where N = strategy limit) |
|---|---|
| BAS (EEG+EOG) | C3-M2, C4-M1, LOC, ROC, O1-M2, O2-M1, F3-M2, F4-M1, A1, A2 |
| RESP | Airflow, Thor, ABD, SpO2, HR, Snore, RespRate |
| EKG | EKG, ECG-L, ECG-R |
| EMG | CHIN, LLEG, RLEG, EMG |

With the **fast** strategy (BAS=4): the 4 channels saved are C3-M2, C4-M1, LOC, ROC — exactly the two central EEG leads and both EOG channels, which are the minimum required for sleep staging.

### Typical channels actually available per dataset

Not every dataset has all priority channels. The HDF5 will contain however many of the priority channels exist in the EDF:

| Dataset | Typical BAS (fast: up to 4) | RESP (fast: 1) | EKG (fast: 1) | EMG (fast: 1-2) |
|---|---|---|---|---|
| STAGES | C3-M2, C4-M1, LOC, ROC (4) | Airflow | EKG | CHIN, LLEG |
| SHHS | Generic EEG names → often only 1-2 BAS | Airflow | EKG | EMG (generic) |
| APPLES | C3-M2, C4-M1, LOC, ROC (4) | Airflow | EKG | CHIN or EMG |
| MrOS | C3-M2, C4-M1, LOC, ROC (4) | Airflow | EKG | CHIN, LLEG |

Note: SHHS uses generic EEG channel names (e.g., "EEG", "EEG sec") rather than electrode-specific names, which limits the number of distinguishable BAS channels.

### Channel name harmonization — KEY ARCHITECTURE NOTE

Our pipeline performs a two-level harmonization:

**Level 1 (our code):** `configs/channel_definitions.yaml` maps thousands of dataset-specific raw channel name variants to standardised canonical names. Example:
- "EEG C3-A2", "C3M2", "C3_M2", "C3:A2", "EEG_C3-A2" → all mapped to **C3-M2**
- "EOG_LOC-A2", "E1", "LEOG", "L-EOG" → all mapped to **LOC**

This harmonization uses case-insensitive matching with a priority lookup (defined in `channel_priority` within `channel_definitions.yaml`).

**Level 2 (our code):** `configs/modality_groups.yaml` maps standardised channel names to SleepFM's 4 modality groups. Example:
- C3-M2, C4-M1, LOC, ROC, O1-M2, O2-M1 → **BAS**
- EKG, ECG-L → **EKG**

**We do NOT use SleepFM's `channel_groups.json`** (from `sleepfm-clinical/sleepfm/configs/`). That file is SleepFM's internal lookup used during their original pre-training on non-NSRR data. Our HDF5 files already use our standardised names; we bypass SleepFM's name lookup entirely and pass pre-built tensors with explicit channel masks.

### Signal processing per channel

After channel selection, each retained channel goes through:
1. **FIR bandpass filter** via `mne.filter.filter_data(method='fir', fir_design='firwin')` — linear-phase, zero-phase, FFT-based convolution for overnight signals
   - EEG/EOG: 0.3–35 Hz
   - EKG/ECG: 0.5–45 Hz
   - EMG: 10–100 Hz
   - RESP: 0.05–2.0 Hz
2. **Resample to 128 Hz** via `scipy.signal.resample_poly` (integer ratios) or `np.interp` (non-integer)
3. **Z-score normalisation** over full recording; flat channels (std=0) use mean-centre only
4. **NaN/Inf guard**: NaN→0, Inf→clipped to ±10

### HDF5 output format

- One `.h5` file per subject
- One HDF5 dataset per standardised channel name (e.g., key `"C3-M2"` contains a `(n_samples,)` float16 array)
- Metadata attributes: `sampling_rate`, `duration_seconds`, `num_channels`, `original_sfreq`, `normalization_stats`, `channel_names`
- Compression: gzip level 4, chunk_size = 38,400 samples (5 minutes at 128 Hz)

### Annotation extraction (alongside signal preprocessing)

Sleep stage annotations are extracted simultaneously and saved as `.npy` arrays:
- Output: `{dataset}/derived/annotations/{subject_id}_stages.npy`
- Format: int8 array, one label per 30-second epoch
- Values: W=0, N1=1, N2=2, N3=3, N4=3 (merged to N3), REM=5
- Unknown/artefact epochs: -1

Note: REM is stored as **5** at this stage. The `_remap_stages()` function in `context_window_dataset.py` later remaps REM from 5→4.

---

## Step 5 — Extract Task Targets

**Scripts:** `scripts/extract_targets_stages.py`, `extract_targets_shhs.py`, `extract_targets_apples.py`, `extract_targets_mros.py`  
**Configs:** `configs/target_extraction.yaml` (v1), `configs/target_extraction_v2.yaml` (v2, current)

Reads per-dataset CSV files and extracts task-specific labels (AHI, ESS, BDI, PHQ-9, etc.) per subject visit. Applies thresholds to create binary/multiclass labels. Outputs per-dataset CSVs:

- `/scratch/boshra95/psg/unified/targets/{dataset}_targets.csv`

**Config evolution:** `target_extraction.yaml` (v1) had bugs (wrong subject ID column for STAGES, missing columns). `target_extraction_v2.yaml` has all fixes applied. Use v2 for all current and future runs.

Key fixes in v2:
- STAGES: `subject_id_column: subject_code` (was `nsrrid`)
- STAGES insomnia: `isi_score` column in main CSV (no external XLSX needed)
- STAGES sleepiness: `ess_0900` confirmed available
- STAGES AHI: from `STAGESPSGKeySRBDVariables2020-08-29 Deidentified.xlsx`, column `ahi`, subject ID `s_code`

---

## Step 6 — Build Master Targets

**Script:** `scripts/create_master_targets.py`  
**Output:** `/scratch/boshra95/psg/unified/targets_v2/master_targets.parquet`

Merges per-dataset target CSVs into one unified parquet with all subjects × all tasks. Applies multiclass-to-binary mappings where needed (e.g., 4-class AHI → binary apnea). This is the single authoritative label source for all downstream training.

---

## Step 7 — Build Task Subject Lists

**Script:** `scripts/create_task_subject_lists.py`  
**Output:** `/scratch/boshra95/psg/unified/targets_v2/task_subjects/{task}_subjects.csv`

Generates one CSV per task containing only subjects with valid labels for that task. These CSVs are read by `ContextWindowDataset` at training time to build the subject × label index.

---

## Step 8 — Extract SleepFM Embeddings

**Script:** `scripts/extract_sleepfm_embeddings.py`  
**Config:** `configs/phase0_v3_config.yaml` (current) or `configs/phase0_v3_full_config.yaml` (channel expansion)  
**Output:** `/scratch/boshra95/psg/unified/embeddings/sleepfm_5sec/{dataset}/{subject_id}.npy`

### What it does

For each subject's HDF5 file:
1. Read the list of available channel keys from the HDF5
2. For each SleepFM modality group (BAS, RESP, EKG, EMG):
   - Walk through the priority list (`data.channel_priority` from config)
   - Collect whichever of those channels exist in the HDF5 (in order)
   - Zero-pad to `data.max_channels` slots; set `mask=True` for padded slots
3. Process in 5-minute chunks (38,400 samples at 128 Hz), batch_size=16 chunks
4. Run each modality through the frozen SleepFM SetTransformer
5. Extract `patch_embeddings` (second return value): shape `(B, 60, 128)` per modality
6. Concatenate all chunks and save as `[T, 4, 128]` float16 .npy file

### Channel limits used at embedding extraction vs preprocessing

The embedding extractor reads `data.max_channels` from the config, which sets:
- BAS: up to 10, RESP: up to 7, EKG: up to 2, EMG: up to 4 (from `phase0_v3_config.yaml`)

But the ACTUAL number of channels reaching SleepFM is limited by what was saved in the HDF5 during preprocessing. If preprocessing used "fast" (BAS=4), the HDF5 has only 4 BAS channels; the embedding extractor will find 4, zero-pad 6 more, and feed 10 slots to SleepFM with 6 masked. The model receives:
- **Effective real channels**: however many were in HDF5 (fast: 4 BAS, 1 RESP, 1 EKG, 1-2 EMG)
- **Zero-padded (masked) slots**: the remaining capacity up to max_channels

### Output format

- Shape: `[T, 4, 128]`, dtype float16
- T = total 5-second patches (= recording_length_seconds / 5)
- Axis 1 order: BAS=0, RESP=1, EKG=2, EMG=3
- ~5.6 MB per subject; ~22 GB total across all subjects (fast channel run)

### What SleepFM's pre-training used vs what we feed it

SleepFM was pre-trained on PSG data using `channel_groups.json` — a flat list of all channel names they encountered, without priority ordering. Their dataset's channels had different names than ours (e.g., they had "EOG(L)"/"EOG(R)" where we have "LOC"/"ROC").

**We bypass their name lookup entirely.** Our HDF5 already uses standardised names. The embedding extractor manually assembles the `(B, C_max, chunk_size)` tensor per modality using our priority order and passes it directly to SleepFM with the channel mask. SleepFM's attention-pooling layer then learns spatial aggregation across whichever real channels are present.

---

## Step 9 — Test Dataset Loading

**Script:** `scripts/test_context_window_dataset.py`

Verifies that `ContextWindowDataset` correctly loads embeddings, applies the cohort consistency filter, builds the training/val/test index, and yields batches of the expected shape. Run before submitting training jobs.

---

## Step 10 — Train Models

**Script:** `scripts/train_context_sweep.py`  
**Config:** `configs/phase0_v3_config.yaml`  
**Registry:** `experiments/v2_registry.yaml`  
**Commands:** `scripts/gen_commands.py`

Trains lightweight sequence heads (LSTM, Transformer, MeanPool) on top of the pre-extracted frozen embeddings at each context length. See `docs/EXPERIMENTS_GUIDE.md` for full details.

---

## Channel Configuration Architecture Summary

```
configs/channel_definitions.yaml
  → maps raw EDF channel names to standardised labels
  → case-insensitive; priority order per channel
  → 1,000+ raw variants → ~40 standardised labels

configs/modality_groups.yaml
  → maps standardised labels to SleepFM modalities
  → BAS: [C3-M2, C4-M1, LOC, ROC, O1-M2, O2-M1, F3-M2, F4-M1, A1, A2, ...]
  → RESP: [Airflow, Thor, ABD, SpO2, HR, Snore, RespRate]
  → EKG: [EKG, ECG-L, ECG-R]
  → EMG: [CHIN, LLEG, RLEG, EMG, ...]

configs/preprocessing_params.yaml
  → controls HOW MANY channels per modality saved to HDF5
  → strategy "fast": BAS=4, RESP=1, EKG=1, EMG=2  (v1/v2/v3 runs)
  → strategy "sleepfm_full": BAS=10, RESP=7, EKG=2, EMG=4  (channel expansion run)

configs/phase0_v3_config.yaml (data section)
  → channel_priority: which HDF5 keys to pick per modality (priority order)
  → max_channels: BAS=10, RESP=7, EKG=2, EMG=4  (upper limit from HDF5)
  → Actual effective channels = min(what's in HDF5, max_channels)
```

---

## Two Parallel Data Versions

| Version | Preprocessing | HDF5 path | Embeddings path | Results path | Effective channels |
|---|---|---|---|---|---|
| **v3 (current)** | fast (7-8 ch) | `/scratch/boshra95/psg/` | `psg/unified/embeddings/sleepfm_5sec/` | `psg/unified/results/phase0_v3/` | BAS≤4, RESP≤1, EKG≤1, EMG≤2 |
| **v3_full (channel expansion)** | sleepfm_full (up to 23 ch) | `/scratch/boshra95/psg_full/` | `psg_full/unified/embeddings/sleepfm_5sec/` | `psg_full/unified/results/phase0_v3_full/` | BAS≤10, RESP≤7, EKG≤2, EMG≤4 |

The v3_full experiment is planned as an ablation to test whether richer channel inputs improve performance. All architectures (head type, hidden_dim, num_layers) are identical between v3 and v3_full so that any AUROC difference is attributable solely to the channel set.

---

## Cluster Commands Reference

```bash
# Step 1: Channel inventory
python scripts/extract_nsrr_channels.py --datasets stages shhs apples mros

# Step 2: Metadata
python scripts/extract_metadata.py --datasets stages shhs apples mros

# Steps 5-7: Targets
python scripts/extract_targets_stages.py --config configs/target_extraction_v2.yaml
python scripts/extract_targets_shhs.py   --config configs/target_extraction_v2.yaml
python scripts/extract_targets_apples.py --config configs/target_extraction_v2.yaml
python scripts/extract_targets_mros.py   --config configs/target_extraction_v2.yaml
python scripts/create_master_targets.py
python scripts/create_task_subject_lists.py

# Step 4: Preprocessing (cluster — see jobs/)
sbatch jobs/preprocess_signals_parallel.sh stages --config configs/preprocessing_params.yaml
# OR for full channel run:
sbatch jobs/preprocess_signals_parallel.sh stages --config configs/preprocessing_params_full.yaml

# Step 8: Embedding extraction (cluster)
sbatch --export=ALL,CONFIG=configs/phase0_v3_config.yaml jobs/extract_embeddings_gpu.sh

# Step 9: Validate
python scripts/test_context_window_dataset.py

# Step 10: Train (via gen_commands.py)
python scripts/gen_commands.py train sex_binary_lstm | bash
```
