# NSRR Data Preprocessing Pipeline - Implementation Plan

**Date:** February 20, 2026  
**Purpose:** Unified preprocessing pipeline for NSRR datasets (STAGES, SHHS, APPLES, MrOS) supporting all 4 research directions  
**Output:** Standardized HDF5 signals + metadata ready for SleepFM embeddings

---

## Executive Summary

This pipeline implements the "Mutual Steps" from `data_preparation.tex` to create a unified, dataset-agnostic preprocessing framework that:

1. **Handles heterogeneity** across 4 NSRR datasets (different channels, annotations, file formats)
2. **Supports variable channels** per subject with modality-aware masking
3. **Produces SleepFM-compatible outputs** (HDF5 @ 128 Hz, 4 modality groups)
4. **Enables all 4 research directions** (context learning, early-exit, TTA, MoE)
5. **Maximizes reusability** (build once, use for all experiments)

---

## Storage Layout (S0)

Following the document structure on Compute Canada:

```
$SCRATCH/psg/nsrr/
├── stages/
│   ├── raw_tar/              # Original .tar.zst archives
│   ├── original/             # Extracted EDF files + annotations
│   └── datasets/             # Metadata CSVs
├── shhs/
│   ├── raw_tar/
│   ├── original/
│   └── datasets/
├── apples/...
└── mros/...

$SCRATCH/psg/
├── unified/
│   ├── metadata/
│   │   ├── unified_metadata.parquet    # S1: Master metadata table
│   │   ├── channel_inventory.csv       # All channels found per dataset
│   │   └── splits/
│   │       ├── in_domain_splits.json   # Train/val/test within dataset
│   │       └── cross_dataset_splits.json
│   └── cache/                           # Temporary processing caches
└── {dataset}/
    └── derived/
        ├── hdf5_signals/                # S2: Preprocessed continuous signals
        │   └── {subject_id}.h5
        ├── annotations/                 # Sleep stages, events
        │   ├── {subject_id}_stages.npy
        │   └── {subject_id}_events.json
        ├── masks/                       # S3: Modality & validity masks
        │   └── {subject_id}_masks.npz
        └── embeddings_sleepfm/          # S5: (Optional, later)
            └── {subject_id}_emb.h5
```

---

## Unified Metadata Table (S1)

**Format:** Parquet (fast, columnar, supports complex types)

**Schema:**
```python
{
    # Identification
    'dataset': str,              # 'stages', 'shhs1', 'shhs2', 'apples', 'mros'
    'subject_id': str,           # Original subject ID
    'night_id': str,             # For multi-night studies
    'unified_id': str,           # {dataset}_{subject_id}_{night_id}
    
    # File paths
    'edf_path': str,             # Absolute path to EDF
    'annotation_path': str,      # Path to annotation file (XML/CSV/etc)
    'hdf5_path': str,            # Path to output HDF5
    
    # Recording info
    'recording_date': datetime,
    'recording_start_time': time,
    'duration_hours': float,
    'duration_seconds': int,
    
    # Channel availability (JSON strings)
    'available_channels': str,   # List of all channel names in EDF
    'channels_by_modality': str, # {'EEG': ['C3-M2', ...], 'ECG': ['EKG'], ...}
    'channel_count': int,
    'sampling_rates': str,       # Per-channel or per-modality
    
    # Modality flags
    'has_eeg': bool,
    'has_eog': bool,
    'has_ecg': bool,
    'has_emg': bool,
    'has_resp': bool,
    'num_eeg_channels': int,
    'num_eog_channels': int,
    'num_ecg_channels': int,
    'num_emg_channels': int,
    'num_resp_channels': int,
    
    # Annotation availability
    'has_sleep_staging': bool,
    'has_respiratory_events': bool,
    'has_arousals': bool,
    'annotation_format': str,    # 'xml', 'csv', 'edf+', etc.
    
    # Phenotype/labels (dataset-specific, nullable)
    'age': float,
    'sex': str,
    'bmi': float,
    'ahi': float,                # Apnea-Hypopnea Index
    'cognitive_scores': str,     # JSON with score names/values
    'diagnosis': str,
    'site': str,                 # For cross-site evaluation
    
    # Data splits
    'split': str,                # 'train', 'val', 'test'
    'fold': int,                 # For k-fold CV
    'split_scheme': str,         # 'in_domain', 'cross_dataset_source', 'cross_dataset_target'
    
    # Processing flags
    'processed': bool,
    'processing_date': datetime,
    'has_embeddings': bool,
    'qc_pass': bool,             # Quality control flag
    'qc_notes': str
}
```

---

## Modality Definitions

Based on SleepFM's `channel_groups.json` and your CogPSGFormerPP experience:

### **BAS (Brain Activity Signals)**
- **EEG**: C3, C4, F3, F4, O1, O2, Fz, Cz, etc. (with references: -M1, -M2, -A1, -A2, -Cz)
- **EOG**: LOC, ROC, E1, E2, EOG(L), EOG(R)
- SleepFM treats both as "BAS" modality

### **EKG (Cardiac)**
- ECG, EKG, ECG1, ECG2, ECGI, ECGII, etc.

### **RESP (Respiratory)**
- **Airflow**: Nasal, NasalP, Flow, Airflow, Thermistor
- **Effort**: THOR, Thorax, Chest, ABD, ABDM, Abdomen
- **Saturation**: SpO2, SaO2, O2
- **Other**: Snore, PPG, Pulse

### **EMG (Muscle)**
- **Chin**: CHIN, Chin1, Chin2
- **Legs**: LLEG, RLEG, LEG(L), LEG(R), Leg1, Leg2
- **Arms**: LArm, RArm

---

## Channel Mapping Strategy

### 1. **Channel Detection with Alternatives**

Reuse CogPSGFormerPP's robust detection logic:

```python
# Example for C3-M2 EEG channel
channel_alternatives = {
    'C3-M2': ['C3-M2', 'C3M2', 'C3:M2', 'EEG C3-M2', 'C3-A2', 'C3A2', 'C3:A2'],
    'C4-M1': ['C4-M1', 'C4M1', 'C4:M1', 'EEG C4-M1', 'C4-A1', 'C4A1', 'C4:A1'],
    'LOC': ['LOC', 'LEOG', 'EOG LOC-A2', 'E1', 'E1-M2', 'E1M2'],
    'ROC': ['ROC', 'REOG', 'EOG ROC-A1', 'E2', 'E2-M2', 'E2M2'],
    'EKG': ['EKG', 'ECG', 'ECG1', 'ECGI', 'ECG II'],
    'Flow': ['Flow', 'Airflow', 'AIRFLOW', 'Nasal', 'NasalP'],
    'Thor': ['Thor', 'THOR', 'Thorax', 'Chest', 'RIP Thorax'],
    'ABD': ['ABD', 'ABDM', 'Abdomen', 'Abdo', 'RIP Abdomen'],
    'CHIN': ['CHIN', 'Chin', 'Chin1', 'CHIN1'],
    'LLEG': ['LLEG', 'LEG(L)', 'L Leg', 'LLeg', 'Leg L'],
    'RLEG': ['RLEG', 'LEG(R)', 'R Leg', 'RLeg', 'Leg R'],
}
```

### 2. **Standardized Output Names**

Map all dataset-specific names to SleepFM-compatible names:

```python
sleepfm_naming = {
    # EEG - keep montage notation
    'C3-M2': 'C3-M2',
    'C4-M1': 'C4-M1',
    'O1-M2': 'O1-M2',
    'O2-M1': 'O2-M1',
    'F3-M2': 'F3-M2',
    'F4-M1': 'F4-M1',
    
    # EOG - use SleepFM convention
    'LOC': 'EOG(L)',
    'ROC': 'EOG(R)',
    
    # ECG - simple name
    'EKG': 'EKG',
    
    # Respiratory - standardized
    'Flow': 'Flow',
    'Thor': 'Thor',
    'ABD': 'ABD',
    
    # EMG - standardized
    'CHIN': 'CHIN',
    'LLEG': 'LLEG',
    'RLEG': 'RLEG',
}
```

---

## Signal Preprocessing (S2)

### Per-Modality Processing Parameters

```python
preprocessing_params = {
    'EEG': {
        'filter_low': 0.3,    # Hz
        'filter_high': 35.0,   # Hz
        'target_sr': 128,      # Hz (SleepFM requirement)
        'filter_type': 'bandpass',
        'filter_order': 4,
    },
    'EOG': {
        'filter_low': 0.3,
        'filter_high': 35.0,
        'target_sr': 128,
        'filter_type': 'bandpass',
        'filter_order': 4,
    },
    'ECG': {
        'filter_low': 0.5,
        'filter_high': 45.0,
        'target_sr': 128,
        'filter_type': 'bandpass',
        'filter_order': 4,
    },
    'EMG': {
        'filter_low': 10.0,
        'filter_high': 100.0,
        'target_sr': 128,
        'filter_type': 'bandpass',
        'filter_order': 4,
    },
    'RESP': {
        'filter_low': 0.05,
        'filter_high': 2.0,
        'target_sr': 128,
        'filter_type': 'bandpass',
        'filter_order': 4,
    },
}
```

### Processing Pipeline

```
For each subject:
1. Load EDF → detect available channels
2. Map to modality families
3. For each channel:
   a. Bandpass filter (modality-specific)
   b. Resample to 128 Hz
   c. Z-score normalize (per-night)
   d. Save normalization stats
4. Save continuous HDF5 (float16, gzip)
5. Generate validity masks
```

### HDF5 Output Structure

```python
# Per-subject HDF5 file
{subject_id}.h5:
    ├── C3-M2: [N_samples] float16
    ├── C4-M1: [N_samples] float16
    ├── O1-M2: [N_samples] float16
    ├── O2-M1: [N_samples] float16
    ├── EOG(L): [N_samples] float16
    ├── EOG(R): [N_samples] float16
    ├── EKG: [N_samples] float16
    ├── Flow: [N_samples] float16
    ├── Thor: [N_samples] float16
    ├── ABD: [N_samples] float16
    ├── CHIN: [N_samples] float16
    ├── LLEG: [N_samples] float16
    ├── RLEG: [N_samples] float16
    └── metadata: {
        'duration': seconds,
        'sampling_rate': 128,
        'normalization_stats': {...},
        'original_channels': [...],
    }
```

---

## Windowing & Masks (S3)

### Canonical Time Units

```python
EPOCH_SECONDS = 30        # For sleep staging
CHUNK_SECONDS = 300       # 5 minutes for SleepFM embeddings
SAMPLING_RATE = 128       # Hz

EPOCH_SAMPLES = 30 * 128 = 3840
CHUNK_SAMPLES = 300 * 128 = 38400
```

### Mask Structure

Per-subject NPZ file containing:

```python
masks.npz:
    # Modality-level masks
    'modality_mask': [N_epochs, 5] bool
        # Dimensions: [time, modality]
        # modality order: [EEG, EOG, ECG, EMG, RESP]
        # True = available, False = missing
    
    # Channel-level masks (optional)
    'channel_mask': [N_epochs, max_channels] bool
        # For fine-grained channel availability
    
    # Validity masks (artifact detection)
    'validity_mask': [N_epochs] bool
        # True = valid, False = artifact/missing
    
    # Metadata
    'epoch_duration': 30,  # seconds
    'num_epochs': N_epochs,
    'modality_names': ['EEG', 'EOG', 'ECG', 'EMG', 'RESP']
```

---

## Dataset-Specific Adapters

### Base Adapter Interface

```python
class BaseNSRRAdapter(ABC):
    """Abstract base class for dataset-specific adapters."""
    
    @abstractmethod
    def find_edf_files(self) -> List[Tuple[str, Path]]:
        """Return [(subject_id, edf_path), ...]"""
        pass
    
    @abstractmethod
    def find_annotation_files(self, subject_id: str) -> Optional[Path]:
        """Return annotation file path for subject."""
        pass
    
    @abstractmethod
    def parse_annotations(self, annotation_path: Path) -> Dict:
        """Parse sleep stages and events."""
        pass
    
    @abstractmethod
    def get_channel_mapping(self) -> Dict[str, List[str]]:
        """Return channel alternatives for this dataset."""
        pass
    
    @abstractmethod
    def extract_phenotype(self, subject_id: str) -> Dict:
        """Extract demographic/clinical data from metadata."""
        pass
```

### STAGES Adapter

```python
class STAGESAdapter(BaseNSRRAdapter):
    """Adapter for STAGES dataset."""
    
    def find_edf_files(self):
        # STAGES structure: original/*/usable/*.edf
        edf_pattern = self.raw_dir / "original" / "*" / "usable" / "*.edf"
        return self._parse_edf_paths(edf_pattern)
    
    def find_annotation_files(self, subject_id):
        # STAGES: annotations inside EDF as EDF+ or separate CSVs
        # Check both locations
        pass
    
    def parse_annotations(self, annotation_path):
        # Parse STAGES-specific format
        pass
    
    def get_channel_mapping(self):
        # STAGES has: E1M2, E2M2, C3M2, C4M1, O1M2, O2M1, EKG, etc.
        return stages_channel_alternatives
```

### SHHS Adapter

```python
class SHHSAdapter(BaseNSRRAdapter):
    """Adapter for SHHS dataset (visit 1 & 2)."""
    
    def find_edf_files(self):
        # SHHS structure: shhs1/polysomnography/edfs/shhs1/*.edf
        #                 shhs2/polysomnography/edfs/shhs2/*.edf
        pass
    
    def find_annotation_files(self, subject_id):
        # SHHS: separate XML files with sleep stages
        # shhs1/polysomnography/annotations-events-nsrr/shhs1/*.xml
        pass
    
    def parse_annotations(self, annotation_path):
        # Parse NSRR XML format
        pass
```

---

## Implementation Phases

### **Phase 1: Infrastructure & Metadata** (Week 1)

**Goal:** Build the foundation - adapters, metadata builder, channel mapper

**Deliverables:**
1. ✅ IMPLEMENTATION_PLAN.md (this document)
2. Core modules:
   - `src/nsrr_tools/core/metadata_builder.py`
   - `src/nsrr_tools/core/channel_mapper.py`
   - `src/nsrr_tools/core/modality_detector.py`
3. Dataset adapters:
   - `src/nsrr_tools/datasets/base_adapter.py`
   - `src/nsrr_tools/datasets/stages_adapter.py`
   - `src/nsrr_tools/datasets/shhs_adapter.py` (basic)
4. Configuration:
   - `configs/channel_definitions.yaml`
   - `configs/modality_groups.yaml`
   - `configs/preprocessing_params.yaml`
5. Scripts:
   - `scripts/01_build_metadata.py` - Generate unified_metadata.parquet
   - `scripts/02_analyze_channels.py` - Channel inventory report

**Success Criteria:**
- Unified metadata table created with 1000+ subjects from STAGES
- Channel inventory shows modality coverage
- No hardcoded paths (all configurable)

---

### **Phase 2: Signal Preprocessing** (Week 2)

**Goal:** Process raw EDFs → standardized HDF5

**Deliverables:**
1. Signal processing module:
   - `src/nsrr_tools/core/signal_processor.py`
2. Scripts:
   - `scripts/03_preprocess_signals.py` - Main processing script
   - `scripts/04_generate_masks.py` - Create modality masks
3. Validation:
   - `scripts/05_validate_outputs.py` - QC checks

**Success Criteria:**
- Process 100 subjects end-to-end
- HDF5 files validated (correct shape, dtype, no NaN/Inf)
- Masks aligned with signals

---

### **Phase 3: PyTorch Dataset API** (Week 3)

**Goal:** Efficient data loading for training

**Deliverables:**
1. Dataset classes:
   - `src/nsrr_tools/data/unified_dataset.py`
   - `src/nsrr_tools/data/variable_length_sampler.py`
2. Utilities:
   - `src/nsrr_tools/data/augmentation.py` - Stress-test transforms
   - `src/nsrr_tools/data/split_utils.py` - Train/val/test splitting

**Success Criteria:**
- Can load variable-length contexts (2-80 min)
- Modality masking works correctly
- Fast loading (<1s per sample)

---

### **Phase 4: Additional Datasets** (Week 4)

**Goal:** Extend to SHHS, APPLES, MrOS

**Deliverables:**
1. Remaining adapters:
   - `src/nsrr_tools/datasets/apples_adapter.py`
   - `src/nsrr_tools/datasets/mros_adapter.py`
   - Complete `shhs_adapter.py`
2. Cross-dataset splits
3. Full documentation

**Success Criteria:**
- All 4 datasets in unified metadata
- Cross-dataset evaluation ready

---

## Key Design Principles

### 1. **No Hardcoding**
- All paths, parameters, channel names in config files
- Easy to modify without code changes

### 2. **Modular & Extensible**
- Dataset-specific logic isolated in adapters
- Core processing logic reusable

### 3. **Compute Canada Optimized**
- SLURM job array support
- Checkpoint/resume capability
- Parallel processing per subject

### 4. **Memory Efficient**
- Process one subject at a time
- Stream writing to HDF5
- Use float16 where possible

### 5. **Quality Control**
- Validation at each step
- QC flags in metadata
- Detailed logging

---

## Configuration Examples

### `configs/channel_definitions.yaml`

```yaml
# Channel name alternatives for robust detection
channel_alternatives:
  C3-M2:
    - C3-M2
    - C3M2
    - C3:M2
    - EEG C3-M2
    - C3-A2
    - C3A2
  C4-M1:
    - C4-M1
    - C4M1
    - C4:M1
    - EEG C4-M1
    - C4-A1
    - C4A1
  # ... more channels
  
# Mapping to SleepFM-compatible names
sleepfm_naming:
  C3-M2: C3-M2
  C4-M1: C4-M1
  LOC: EOG(L)
  ROC: EOG(R)
  EKG: EKG
  # ... more mappings
```

### `configs/modality_groups.yaml`

```yaml
# Modality definitions
modalities:
  EEG:
    description: "Electroencephalography"
    channels:
      - C3-M2
      - C4-M1
      - O1-M2
      - O2-M1
      - F3-M2
      - F4-M1
    processing_params: eeg
  
  EOG:
    description: "Electrooculography"
    channels:
      - EOG(L)
      - EOG(R)
    processing_params: eog
  
  # ... more modalities
```

### `configs/preprocessing_params.yaml`

```yaml
# Signal processing parameters
processing:
  eeg:
    filter_low: 0.3
    filter_high: 35.0
    target_sr: 128
    filter_type: bandpass
    filter_order: 4
    normalization: zscore
  
  ecg:
    filter_low: 0.5
    filter_high: 45.0
    target_sr: 128
    filter_type: bandpass
    filter_order: 4
    normalization: zscore
  
  # ... more modalities

# HDF5 settings
hdf5:
  dtype: float16
  compression: gzip
  compression_level: 4
  chunk_size: 38400  # 5 minutes @ 128 Hz

# Masking
masking:
  epoch_seconds: 30
  chunk_seconds: 300
  min_valid_ratio: 0.5  # For QC
```

---

## Dataset-Specific Notes

### STAGES
- **Structure:** Organized by clinic sites
- **Annotations:** EDF+ or separate CSVs
- **Channels:** High variability in naming (E1M2, C3M2, etc.)
- **Metadata:** Single CSV with demographics, cognitive scores
- **Special:** Sleep stage cropping already done in CogPSGFormerPP

### SHHS
- **Structure:** Visit 1 and 2 as separate cohorts
- **Annotations:** NSRR XML format (separate tar archives)
- **Channels:** More standardized naming
- **Metadata:** Multiple CSVs (CVD, harmonized, events)
- **Special:** Large cohort (~5800 subjects), excellent annotations

### APPLES
- **Structure:** Simpler flat structure
- **Annotations:** Check format
- **Metadata:** To be investigated

### MrOS
- **Structure:** To be investigated
- **Annotations:** To be investigated
- **Metadata:** To be investigated

---

## Questions to Resolve

1. **STAGES sleep crop:** Use CogPSGFormerPP's crop info or reprocess?
2. **Annotation formats:** Confirm APPLES/MrOS annotation formats
3. **Extract archives:** Extract all upfront or on-the-fly?
4. **Compute resources:** How many cores/memory for parallel processing?
5. **SHHS visit 1 vs 2:** Process separately or together?

---

## Success Metrics

- **Coverage:** All subjects with minimum required modalities processed
- **Quality:** <1% failed QC checks
- **Performance:** Process 1 subject in <5 minutes
- **Storage:** <100 MB per subject (HDF5 compressed)
- **Accuracy:** Embeddings match expected distributions

---

## Next Steps After This Plan

1. Create directory structure in NSRR-tools
2. Implement core modules (channel_mapper, modality_detector)
3. Implement STAGES adapter (start with 1 subject)
4. Build unified metadata for STAGES
5. Process 10 subjects end-to-end for validation
6. Scale to full STAGES dataset
7. Extend to other datasets

---

## References

- `data_preparation.tex` - Research plan document
- `proposal for objective3.tex` - Research objectives
- CogPSGFormerPP preprocessing code
- SleepFM `channel_groups.json`
- NSRR dataset documentation
