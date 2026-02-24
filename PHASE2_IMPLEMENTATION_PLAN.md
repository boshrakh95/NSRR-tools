# Phase 2: Data Preprocessing Pipeline - Implementation Plan

**Status:** Planning Phase  
**Last Updated:** February 24, 2026  
**Purpose:** Flexible preprocessing system supporting all 4 research ideas (AMTA+CSL, Early-Exit, Safe TTA, MoE Routing)

---

## Table of Contents
1. [Overview & Design Philosophy](#overview--design-philosophy)
2. [Answered Questions](#answered-questions)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Storage Structure](#storage-structure)
5. [Shared Preprocessing Steps](#shared-preprocessing-steps)
6. [Idea-Specific Preprocessing](#idea-specific-preprocessing)
7. [Execution Timeline](#execution-timeline)
8. [Configuration System](#configuration-system)

---

## Overview & Design Philosophy

### Goals
- **Compatibility:** Preprocessing compatible with SleepFM backbone (and other backbones)
- **Flexibility:** Variable-length context, multiple modalities, missing data handling
- **Efficiency:** Cache preprocessed signals in HDF5 for fast training iterations
- **Modularity:** Shared preprocessing + idea-specific augmentations
- **Reproducibility:** All parameters in YAML configs, tracked in metadata

### Two-Stage Approach
1. **Phase 2 (NOW):** Raw EDF → Preprocessed HDF5 (filtered, resampled, normalized signals)
2. **Phase 3 (LATER):** HDF5 signals → SleepFM embeddings (when ready to train)

---

## Answered Questions

### Q1: SleepFM embeddings now or later?
**A:** Preprocess raw signals now → HDF5. Extract SleepFM embeddings later (before training) when idea is chosen.

### Q2: Filter specs and sampling rates?
**A:** Use SleepFM paper specifications from their original preprocessing code/yaml (NOT stages-specific code). Make all parameters configurable in YAML.
- **TODO:** Extract SleepFM preprocessing parameters from `sleepfm-clinical` repo (original datasets, not STAGES)
- If anything missing: user will find it in paper text

### Q3: Storage paths on cluster?
**A:** Need to determine:
- Raw data: `/scratch/boshra95/nsrr_downloads/` (unzipped) vs `/scratch/boshra95/psg/nsrr/*/raw_tar/` (zipped)
- Processed output: `$PROJECT/psg/{dataset}/derived/` (from .tex proposal)
- Cache: `$SCRATCH/psg_cache/`

### Q4: Data source - zipped vs unzipped?
**A:** To be decided based on:
- Login node debugging: probably unzipped (already accessible)
- Batch processing: need to benchmark or test both
- **DECISION NEEDED:** User to confirm data source before Step 2 implementation

### Q5: Start with one dataset?
**A:** Yes, validate end-to-end with STAGES first, then replicate for SHHS/APPLES/MrOS

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA (4 Datasets)                         │
│  /scratch/boshra95/nsrr_downloads/{stages,shhs,apples,mros}/    │
│  - EDF files + annotations (CSV/XML/.annot)                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: Unified Metadata Builder                    │
│  Output: unified_metadata.parquet                                │
│  - Subject ID, dataset, available channels, labels, splits       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 2-4: Signal Preprocessing Pipeline                  │
│  Per-subject HDF5:                                               │
│  /{modality}/signals, /annotations/sleep_stages,                │
│  /availability/{modality}, /metadata                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 5: Dataset Splits                              │
│  splits/in_domain_{dataset}.csv                                  │
│  splits/cross_dataset_train_{A}_test_{B}.csv                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 6-9: Idea-Specific Data Loaders                     │
│  Variable-length sampling, augmentations, stress tests           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│        [LATER] STEP 10: SleepFM Embedding Extraction            │
│  Optional: Cache embeddings for faster training                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Storage Structure

### Raw Data Location (INPUT)
```
/scratch/boshra95/nsrr_downloads/
├── stages/
│   ├── original/STAGES PSGs/{SITE}/{subject_id}.edf
│   └── datasets/{harmonized,main}.csv
├── shhs/
│   ├── polysomnography/edfs/{subject_id}.edf
│   └── polysomnography/annotations-events-nsrr/{visit}/{subject_id}-nsrr.xml
├── apples/
│   ├── polysomnography/{subject_id}.edf + .annot
│   └── datasets/{harmonized,main}.csv
└── mros/
    ├── polysomnography/edfs/{subject_id}.edf
    ├── polysomnography/annotations-events-nsrr/{visit}/{subject_id}-nsrr.xml
    └── datasets/{harmonized,main}.csv
```

### Processed Data Location (OUTPUT)
```
$PROJECT/psg/                            # Persistent storage
├── stages/
│   └── derived/
│       ├── hdf5_signals/{subject_id}.h5
│       ├── metadata/processing_log.csv
│       └── validation/quality_reports/
├── shhs/derived/...
├── apples/derived/...
├── mros/derived/...
└── unified/
    ├── metadata/
    │   ├── unified_metadata.parquet
    │   └── channel_mappings.json
    └── splits/
        ├── in_domain_stages.csv
        ├── in_domain_shhs.csv
        ├── cross_dataset_train_stages_test_shhs.csv
        └── ...

$SCRATCH/psg_cache/                      # Temporary fast storage
├── embedding_cache/                     # [Later] SleepFM embeddings
└── temp_processing/                     # Intermediate files
```

### HDF5 Structure (Per Subject-Night)
```
subject_123.h5
├── /EEG
│   ├── signals          [n_channels, n_timepoints] float32
│   ├── channels         [n_channels] string (channel names)
│   ├── fs               scalar int (sampling rate)
│   └── quality_flags    [n_channels] bool (valid channels)
├── /EOG
│   ├── signals, channels, fs, quality_flags
├── /ECG
│   ├── signals, channels, fs, quality_flags
├── /EMG
│   ├── signals, channels, fs, quality_flags
├── /RESP
│   ├── signals, channels, fs, quality_flags
├── /annotations
│   ├── sleep_stages     [n_epochs] int (30s epochs: 0=Wake, 1=REM, 2=N1, 3=N2, 4=N3)
│   ├── epoch_times      [n_epochs] float (start time in seconds)
│   └── night_labels     attributes (AHI, cognitive scores, etc.)
├── /availability
│   ├── EEG              [n_windows] bool (30s or 2min windows)
│   ├── EOG              [n_windows] bool
│   ├── ECG              [n_windows] bool
│   ├── EMG              [n_windows] bool
│   └── RESP             [n_windows] bool
└── /metadata
    ├── subject_id       attribute: string
    ├── dataset          attribute: string
    ├── duration_sec     attribute: float
    ├── start_time       attribute: string (ISO format)
    ├── processing_date  attribute: string
    ├── preprocessing_params  attribute: JSON string (filters, rates, etc.)
    └── original_edf_path  attribute: string
```

---

## Shared Preprocessing Steps

All 4 research ideas require these core steps.

### **STEP 1: Unified Metadata Builder** ✓ (Partially Complete)

**Purpose:** Single source of truth for all subjects across datasets

**Inputs:**
- Dataset adapters (already working: STAGES, SHHS, APPLES, MrOS)
- Metadata CSVs (harmonized + main per dataset)
- Annotation files (various formats)

**Outputs:**
- `unified/metadata/unified_metadata.parquet`

**Schema:**
```python
{
    'subject_id': str,          # e.g., 'stages_4100080'
    'dataset': str,             # 'stages', 'shhs', 'apples', 'mros'
    'original_id': str,         # Original dataset ID
    'visit': int,               # Visit number (1 or 2 for SHHS/MrOS)
    'edf_path': str,            # Absolute path to EDF
    'annotation_path': str,     # Absolute path to annotation file
    'duration_sec': float,      # Recording duration
    'start_time': str,          # ISO timestamp
    
    # Available channels (from EDF)
    'available_channels': list, # All channel names in EDF
    'n_channels': int,
    
    # Modality mapping
    'eeg_channels': list,       # Mapped to EEG modality
    'eog_channels': list,
    'ecg_channels': list,
    'emg_channels': list,
    'resp_channels': list,
    'unknown_channels': list,   # Unmapped channels
    
    # Sampling rates (from EDF)
    'sampling_rates': dict,     # {channel: rate}
    'eeg_fs': float,            # Modal EEG sampling rate
    'ecg_fs': float,
    # ... etc
    
    # Label availability
    'has_staging': bool,
    'n_epochs': int,            # Number of 30s epochs with labels
    'has_ahi': bool,
    'ahi_value': float,
    'has_cognitive_labels': bool,
    'cognitive_scores': dict,   # Dataset-specific cognitive measures
    
    # Demographics (from harmonized metadata)
    'nsrr_age': float,
    'nsrr_sex': str,
    'nsrr_race': str,
    'nsrr_bmi': float,
    
    # Dataset splits
    'split': str,               # 'train', 'val', 'test'
    'fold': int,                # For cross-validation
    'site': str,                # For domain shift experiments
    
    # Processing status
    'preprocessed': bool,       # HDF5 exists
    'hdf5_path': str,          # Path to processed HDF5
    'processing_date': str,
    'quality_passed': bool,     # Passed QC checks
}
```

**Files:**
- Expand existing `src/nsrr_tools/core/metadata_builder.py`
- Config: `configs/metadata_config.yaml`

**Implementation Notes:**
- Use existing dataset adapters (already working!)
- Add channel availability scanning (read EDF headers with MNE)
- Add modality mapping using ChannelMapper
- Merge with night-level labels from adapters

---

### **STEP 2: Signal Preprocessing Pipeline**

**Purpose:** Extract, filter, resample, normalize signals from EDF files

**Processing Steps:**
1. **Load EDF** (using MNE)
2. **Channel mapping** (standardize names, group by modality)
3. **Quality checks** (flatline, clipping, saturation detection)
4. **Filtering** (bandpass per modality - **PARAMS FROM SLEEPFM**)
5. **Resampling** (target rate per modality - **PARAMS FROM SLEEPFM**)
6. **Normalization** (robust z-score per night: median/IQR)
7. **Save to HDF5** (structured format above)

**Preprocessing Parameters (TO BE EXTRACTED FROM SLEEPFM CODE):**

```yaml
# configs/preprocessing_config.yaml
preprocessing:
  # Target sampling rates per modality
  sampling_rates:
    EEG: ???      # TODO: Extract from SleepFM
    EOG: ???
    ECG: ???
    EMG: ???
    RESP: ???
  
  # Bandpass filters per modality (low_freq, high_freq)
  filters:
    EEG:
      low: ???    # TODO: e.g., 0.3 Hz
      high: ???   # TODO: e.g., 35 Hz
      order: 5
      type: 'butter'
    EOG:
      low: ???
      high: ???
      order: 5
    ECG:
      low: ???
      high: ???
      order: 5
    EMG:
      low: ???
      high: ???
      order: 5
    RESP:
      low: ???
      high: ???
      order: 5
  
  # Normalization method
  normalization:
    method: 'robust'  # 'robust' (median/IQR) or 'standard' (mean/std)
    per_channel: true
    per_night: true
  
  # Quality checks
  quality:
    flatline_threshold: 0.01  # RMS threshold for flatline detection
    clipping_threshold: 0.95  # Percentile for saturation detection
    min_valid_duration_sec: 1800  # Minimum 30 minutes of valid data
```

**Files to Create:**
- `src/nsrr_tools/processing/signal_processor.py` (main processing class)
- `src/nsrr_tools/processing/filters.py` (filtering utilities)
- `src/nsrr_tools/processing/quality_checks.py` (QC functions)
- `src/nsrr_tools/processing/normalization.py` (normalization methods)
- `scripts/run_preprocessing.py` (batch processing script)
- `configs/preprocessing_config.yaml` (parameters)

**Parallelization:**
- Process subjects in parallel (SLURM array jobs)
- Each job processes N subjects

---

### **STEP 3: Annotation Processing**

**Purpose:** Parse and align annotations with preprocessed signals

**Inputs:**
- Annotation files (CSV/XML/.annot per dataset)
- Preprocessed HDF5 files (for timestamp alignment)

**Processing:**
1. **Sleep staging** (30s epochs)
   - Parse dataset-specific format using existing adapters
   - Map to standard labels: 0=Wake, 1=REM, 2=N1, 3=N2, 4=N3, -1=Unknown
   - Align to signal timeline (handle start time offsets)
   - Store as integer array in HDF5 `/annotations/sleep_stages`

2. **Night-level labels**
   - Extract from adapter metadata (already available)
   - AHI, cognitive scores, demographics
   - Store as HDF5 attributes in `/annotations/night_labels`

**Files:**
- `src/nsrr_tools/processing/annotation_processor.py`
- Reuse existing adapter methods: `parse_annotations()`

---

### **STEP 4: Modality Availability Masks**

**Purpose:** Track which modalities/channels are valid per time window

**Window Sizes:**
- 30s windows (aligned with epochs)
- Optional: 2min chunks (for long-context experiments)

**Mask Generation:**
1. For each modality, for each window:
   - Check if channels exist in that time range
   - Check if signals pass quality checks (not flatline/clipped)
   - Set mask = True if valid, False otherwise

2. Store in HDF5 `/availability/{modality}`

**Files:**
- Part of `signal_processor.py` (generate during preprocessing)

---

### **STEP 5: Dataset Splits**

**Purpose:** Create reproducible train/val/test splits

**Two Split Strategies:**

1. **In-Domain Splits** (within each dataset)
   - Subject-wise split (no subject in multiple sets)
   - Stratified by labels (balanced AHI ranges, staging distribution)
   - 70% train, 15% val, 15% test
   - Optional: K-fold cross-validation splits

2. **Cross-Dataset Splits** (for domain shift evaluation)
   - Train: all subjects from dataset A
   - Test: all subjects from dataset B
   - Combinations: STAGES→SHHS, SHHS→STAGES, etc.

**Outputs:**
```
unified/splits/
├── in_domain_stages_fold0.csv       [subject_id, split]
├── in_domain_shhs_fold0.csv
├── cross_dataset_train_stages_test_shhs.csv
├── cross_dataset_train_shhs_test_apples.csv
└── ...
```

**Files:**
- `scripts/create_splits.py`
- Config: `configs/splits_config.yaml`

---

## Idea-Specific Preprocessing

These steps are only needed for specific research ideas.

### **Ideas 1 & 2: Variable-Length Context Support**

**STEP 6a: Variable-Length DataLoader**

**Purpose:** PyTorch Dataset that samples variable-length context windows

**Capabilities:**
- **Staging task:** Anchor at epoch `t`, return past context of length `L` minutes
- **Night-level task:** Sample windows of length `L` from night regions (start/middle/end/full)
- **Paired views (CSL):** Return (short_view, long_view) for same anchor
- **Length menu:** Support predefined lengths [2, 5, 10, 20, 40, 80 minutes]

**Files:**
- `src/nsrr_tools/data/variable_length_dataset.py`
- `src/nsrr_tools/data/samplers.py` (epoch/window sampling strategies)

---

### **Idea 2: Early-Exit Checkpoint Support**

**STEP 6b: Fixed Checkpoint Indexing**

**Purpose:** Ensure longest context is always available for multi-exit training

**Implementation:**
- During sampling, verify anchor has at least `L_max` minutes of context
- Filter out anchors too close to recording start
- Precompute valid anchor indices per subject

**Files:**
- Part of `variable_length_dataset.py`

---

### **Idea 3: Stress-Test Perturbations**

**STEP 7: Augmentation Transforms**

**Purpose:** Simulate robustness challenges at training/test time

**Transforms:**

1. **Missing Modalities**
   - Drop entire modality families (e.g., remove all ECG channels)
   - Probability per modality

2. **Channel Dropout (Montage Variation)**
   - Randomly drop subset of EEG channels
   - Simulate different montage configurations

3. **Intermittent Gaps**
   - Mask contiguous 10-30 minute segments
   - Set signals to zero, update availability masks

4. **Artifact Injection**
   - Gaussian noise bursts
   - Amplitude scaling (simulate gain differences)
   - Baseline drift (low-frequency trends)

5. **Sampling Mismatches**
   - Slight resampling (e.g., 99Hz instead of 100Hz)
   - Different filter cutoffs

**Files:**
- `src/nsrr_tools/data/augmentations.py`
- `src/nsrr_tools/data/stress_test_dataset.py` (wrapper that applies transforms)

**Config:**
```yaml
augmentation:
  missing_modality:
    enabled: true
    drop_prob:
      EEG: 0.1
      ECG: 0.2
      RESP: 0.3
  
  channel_dropout:
    enabled: true
    eeg_keep_prob: 0.7  # Keep 70% of EEG channels
  
  intermittent_gaps:
    enabled: true
    gap_prob: 0.2
    gap_duration_range: [600, 1800]  # 10-30 minutes
  
  artifact_injection:
    enabled: true
    noise_prob: 0.1
    noise_snr_db: 10
```

---

### **STEP 8: Prototype Computation (for TTA)**

**Purpose:** Compute class prototypes from training set for test-time adaptation gates

**Process:**
1. After training source model
2. Extract embeddings for all training samples
3. Compute mean embedding per class
4. Save prototypes for Gate 2 (margin gate)

**Files:**
- `scripts/compute_prototypes.py`
- Output: `prototypes/{model_name}_prototypes.npy`

---

### **Idea 4: Modality-Aware MoE Training**

**STEP 9: Missingness Augmentation Schedule**

**Purpose:** Train MoE router to handle different modality regimes

**Augmentation Strategy:**
- Sample missingness patterns during training
- Ensure router sees EEG-only, ECG-only, mixed, and full modality examples
- Track expert usage statistics

**Files:**
- Part of `augmentations.py`
- Training code will use augmentation config

---

## Execution Timeline

### **Phase 2a: Core Infrastructure (Weeks 1-2)** ← START HERE

**Goal:** End-to-end preprocessing working for STAGES

1. ✅ Step 1: Expand unified metadata builder (reuse adapters)
2. 🔄 Step 2: Signal preprocessing pipeline
   - Extract SleepFM preprocessing parameters first
   - Implement filtering, resampling, normalization
   - Test on 5 STAGES subjects
3. 🔄 Step 3: Annotation processing (reuse adapter methods)
4. 🔄 Step 4: Availability mask generation
5. 🔄 Step 5: Create splits
6. 🔄 Test end-to-end: metadata → HDF5 → splits

**Validation:** Check 5 processed HDF5 files manually

### **Phase 2b: Scale to All Datasets (Week 3)**

1. Replicate Steps 2-4 for SHHS, APPLES, MrOS
2. Run batch processing on cluster (SLURM array jobs)
3. Quality control checks on all datasets

### **Phase 2c: Variable-Length Support (Week 4)**

1. Step 6a: Implement variable-length dataloader
2. Step 6b: Checkpoint indexing for early-exit
3. Test with dummy model

### **Phase 2d: Augmentations (Week 5)**

1. Step 7: Stress-test transforms
2. Step 9: Missingness augmentation
3. Test augmentation pipeline

### **Phase 2e: Optional - SleepFM Embeddings (Week 6+)**

1. Step 10: Extract SleepFM embeddings
2. Cache in same HDF5 structure
3. Benchmark training speed improvement

---

## Configuration System

All parameters configurable via YAML files:

```
configs/
├── channel_definitions.yaml        # ✅ Already complete
├── paths.yaml                      # ✅ Already complete
├── metadata_config.yaml            # Dataset-specific settings
├── preprocessing_config.yaml       # Filters, sampling rates, normalization
├── augmentation_config.yaml        # Stress-test transforms
├── splits_config.yaml              # Split strategies
└── sleepfm_extraction_config.yaml  # [Later] Embedding extraction
```

**Configuration Loading:**
```python
from nsrr_tools.utils.config import Config

config = Config()
preproc_params = config.preprocessing
filter_params = config.preprocessing['filters']['EEG']
```

---

## Next Steps & Questions

### **Immediate Actions (Before Implementation):**

1. **Extract SleepFM Preprocessing Parameters**
   - Find SleepFM original preprocessing code (not STAGES-specific)
   - Extract: sampling rates, filter specs per modality
   - Document in `preprocessing_config.yaml`

2. **Confirm Data Source**
   - Test read speed: unzipped (`/scratch/boshra95/nsrr_downloads/`) vs zipped (`/scratch/boshra95/psg/nsrr/*/raw_tar/`)
   - Decision: Which to use for batch processing?

3. **Confirm Output Paths**
   - `$PROJECT/psg/` → actual path on cluster?
   - `$SCRATCH/psg_cache/` → actual path?

4. **Initial Scope**
   - Start with STAGES only for Phase 2a?
   - Or implement all 4 datasets in parallel?

### **User Approval Needed:**

- [ ] Overall plan structure acceptable?
- [ ] HDF5 schema design acceptable?
- [ ] Storage structure makes sense for cluster?
- [ ] Ready to proceed with Step 1 (metadata expansion)?

---

## References

- **Proposal**: `proposal for objective3.tex` (4 research ideas)
- **Data Prep**: `data_preparation.tex` (shared pipeline design)
- **SleepFM Code**: `sleepfm-clinical/` (preprocessing reference)
- **CogPSGFormer**: `CogPSGFormerPP/` (STAGES preprocessing example)
- **Current Progress**: Phase 1 complete (adapters, channel config)
