# Classification Targets Analysis for NSRR Datasets

**Date:** March 10, 2026  
**Purpose:** Comprehensive analysis of available classification tasks across all 4 NSRR datasets  
**Context:** Supporting multiple research directions (context learning, early-exit, TTA, MoE) from `proposal for objective3.tex`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Dataset-Specific Task Analysis](#dataset-specific-task-analysis)
3. [Clinical Thresholds & Evidence](#clinical-thresholds--evidence)
4. [Task Recommendations by Tier](#task-recommendations-by-tier)
5. [Storage Architecture](#storage-architecture)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Data Availability Matrix](#data-availability-matrix)

---

## Executive Summary

### Available Datasets & Status
- ✅ **STAGES**: 1,513 preprocessed subjects (HDF5 + sleep stages)
- ✅ **SHHS**: 8,444 preprocessed subjects (HDF5 signals)
- ✅ **APPLES**: 1,104 preprocessed subjects (HDF5 signals)
- ✅ **MrOS**: 2,907 preprocessed subjects (HDF5 signals)
- **Total**: ~13,968 subjects with PSG signals ready

### Top 3 Recommended Tasks (Phase 1)
1. **Sleep Apnea (apnea_binary)** - ALL 4 datasets (~13,900 subjects)
2. **Depression (depression_binary)** - STAGES + APPLES (~2,600 subjects)
3. **Sleepiness (sleepiness_binary)** - APPLES (~1,100 subjects)

### Key Advantages
- ✅ Well-validated clinical instruments
- ✅ Objective and reproducible thresholds
- ✅ Balanced class prevalence (15-40% positive)
- ✅ Data already in CSV files (no external dependencies)
- ✅ Strong PSG signal for all tasks

---

## Dataset-Specific Task Analysis

### 1. STAGES Dataset (1,513 subjects)

**Available CSV Files:**
- ✅ `stages-harmonized-dataset-0.3.0.csv`
- ✅ `stages-dataset-0.3.0.csv`
- ⚠️ Additional XLSX files needed for some tasks (ASQ ISI, post-sleep questionnaire)

| Task ID | Task Name | Source Column | Clinical Instrument | Threshold | CSV Location | Difficulty | Priority |
|---------|-----------|--------------|-------------------|-----------|-------------|------------|----------|
| `apnea_binary` | Sleep Apnea | AHI | Polysomnography | **≥15** events/hr | ⚠️ XML annotations only | MEDIUM | ⭐⭐⭐ |
| `depression_binary` | Depression | `phq_1000` | PHQ-9 | **≥10** | stages-dataset CSV | EASY | ⭐⭐⭐⭐⭐ |
| `anxiety_binary` | Anxiety | `gad_0800` | GAD-7 | **≥10** | stages-dataset CSV | EASY | ⭐⭐⭐⭐⭐ |
| `fatigue_binary` | Fatigue | `fss_1000` | FSS | **≥36** (total) | stages-dataset CSV | EASY | ⭐⭐⭐ |
| `insomnia_binary` | Insomnia | `isi_score` | ISI | **≥15** | ⚠️ External XLSX | MEDIUM | ⭐⭐ |
| `rested_morning` | Morning Restedness | `isq_0500` | ISQ | Subjective map | ⚠️ External XLSX | HARD | ⭐ |

**STAGES Strengths:**
- ✅ Excellent mental health instruments (PHQ-9, GAD-7, FSS)
- ✅ Clean, well-curated dataset from sleep clinics
- ✅ Younger, more diverse population than other cohorts

**STAGES Challenges:**
- ⚠️ AHI not in CSV (must extract from XML annotations)
- ⚠️ Some questionnaires in separate XLSX files
- ⚠️ Subject ID mismatch between EDF filenames and CSVs (known issue)

---

### 2. APPLES Dataset (1,104 subjects)

**Available CSV Files:**
- ✅ `apples-harmonized-dataset-0.1.0.csv`
- ✅ `apples-dataset-0.1.0.csv`

| Task ID | Task Name | Source Column | Clinical Instrument | Threshold | CSV Location | Difficulty | Priority |
|---------|-----------|--------------|-------------------|-----------|-------------|------------|----------|
| `apnea_binary` | Sleep Apnea | `nsrr_ahi_chicago1999` | Polysomnography | **≥15** events/hr | harmonized CSV | EASY | ⭐⭐⭐⭐⭐ |
| `depression_binary` | Depression | `bditotalscore` | BDI-II | **≥14** (mild) or **≥20** (mod) | main CSV | EASY | ⭐⭐⭐⭐⭐ |
| `sleepiness_binary` | Excessive Sleepiness | `esstotalscoreqc` | ESS | **≥11** | main CSV | EASY | ⭐⭐⭐⭐⭐ |
| `cognition_regression` | Cognition (MMSE) | `mmsetotalscore` | MMSE | Continuous (24-30) | main CSV | MEDIUM | ⭐⭐ |

**APPLES Strengths:**
- ✅✅✅ **EASIEST DATASET** - All targets in CSV, no external files
- ✅ AHI already computed and harmonized
- ✅ Multiple well-validated psychiatric/sleep instruments
- ✅ Visit structure well-documented (BL=1, DX=3)

**APPLES Challenges:**
- ⚠️ MMSE likely has ceiling effect (subjects mostly normal range 24-30)
- ℹ️ Smaller sample size compared to SHHS

**Visit Strategy:**
- Use **Visit 3 (DX)** for PSG-matched labels (apnea, sleepiness)
- Use **Visit 1 (BL)** for baseline questionnaires (depression, cognition)

---

### 3. SHHS Dataset (8,444 subjects)

**Available CSV Files:**
- ✅ `shhs-harmonized-dataset-0.21.0.csv`
- ✅ `shhs1-dataset-0.21.0.csv`
- ✅ `shhs2-dataset-0.21.0.csv`
- ✅ `shhs-cvd-summary-dataset-0.21.0.csv`
- ✅ `shhs-cvd-events-dataset-0.21.0.csv`

| Task ID | Task Name | Source Column | Clinical Instrument | Threshold | CSV Location | Difficulty | Priority |
|---------|-----------|--------------|-------------------|-----------|-------------|------------|----------|
| `apnea_binary` | Sleep Apnea | `rdi3p` | Polysomnography (RDI≥3%) | **≥15** events/hr | shhs1/2 CSV | EASY | ⭐⭐⭐⭐⭐ |
| `cvd_binary` | Cardiovascular Disease | `any_cvd` | Incident CVD events | **== 1** | cvd-summary CSV | MEDIUM | ⭐⭐⭐⭐ |
| `rested_morning` | Morning Restedness | `rest10` (V1) or `ms204c` (V2) | Self-report (1-5 scale) | **≥4** good, **≤2** poor | shhs1/2 CSV | MEDIUM | ⭐⭐ |

**SHHS Strengths:**
- ✅✅✅ **LARGEST DATASET** - 8,444 subjects for robust training
- ✅ Longitudinal follow-up with incident CVD outcomes
- ✅ Well-characterized cohort with extensive phenotyping
- ✅ RDI3p is gold-standard sleep apnea metric

**SHHS Challenges:**
- ⚠️ Limited psychiatric instruments (no PHQ-9, GAD-7, BDI)
- ⚠️ CVD requires special handling (time-to-event, incident vs prevalent)
- ⚠️ Older cohort (cardiovascular focus)

**SHHS Special Considerations:**
- **Visit Structure**: Visit 1 (baseline), Visit 2 (~5 years later)
- **CVD Task**: Consider as binary classification initially, can extend to survival analysis later
- **Rested Morning**: Different questions per visit, need visit-aware mapping

---

### 4. MrOS Dataset (2,907 subjects)

**Available CSV Files:**
- ✅ `mros-visit1-harmonized-0.6.0.csv`
- ✅ `mros-visit1-dataset-0.6.0.csv`
- ✅ `mros-visit2-dataset-0.6.0.csv`
- ✅ `mros-visit1-hrv-summary-0.5.0.csv` (optional)

| Task ID | Task Name | Source Column | Clinical Instrument | Threshold | CSV Location | Difficulty | Priority |
|---------|-----------|--------------|-------------------|-----------|-------------|------------|----------|
| `apnea_binary` | Sleep Apnea | `nsrr_ahi_hp3r_aasm15` | Polysomnography | **≥15** events/hr | harmonized CSV | EASY | ⭐⭐⭐⭐⭐ |
| `insomnia_binary` | Insomnia | `slisiscr` | ISI | **≥15** | visit1/2 CSV | EASY | ⭐⭐⭐⭐ |
| `rested_morning` | Morning Restedness | `poxqual3` | Self-report (1-5 scale) | **≥4** good, **≤2** poor | visit1/2 CSV | MEDIUM | ⭐⭐ |

**MrOS Strengths:**
- ✅ Good sample size (2,907 subjects)
- ✅ ISI available (complements STAGES insomnia data)
- ✅ All-male cohort (useful for sex-specific analyses)
- ✅ Visit structure similar to SHHS

**MrOS Challenges:**
- ⚠️ Limited psychiatric instruments
- ⚠️ Older male-only cohort (generalizability concerns)
- ℹ️ Consider pooling with STAGES for insomnia task

---

## Clinical Thresholds & Evidence

### Sleep Apnea (AHI - Apnea-Hypopnea Index)

**Instrument:** Polysomnography (PSG)  
**Metric:** Events per hour of sleep  
**Validated Thresholds:**
- **<5:** Normal
- **5-14:** Mild OSA
- **≥15:** Moderate OSA ⭐ **RECOMMENDED THRESHOLD**
- **≥30:** Severe OSA

**Clinical Rationale:**
- AHI ≥15 is widely accepted as clinically significant OSA requiring treatment
- Strong association with cardiovascular outcomes, daytime sleepiness, cognitive impairment
- Objective measurement with high inter-rater reliability
- Expected prevalence: ~30-40% in sleep clinic populations, ~20-30% in community cohorts

**References:**
- AASM International Classification of Sleep Disorders, 3rd edition (ICSD-3)
- Berry et al., AASM Scoring Manual Version 2.6 (2020)

---

### Depression Screening

#### PHQ-9 (Patient Health Questionnaire-9) - STAGES

**Instrument:** 9-item self-report questionnaire  
**Score Range:** 0-27  
**Validated Thresholds:**
- **0-4:** Minimal depression
- **5-9:** Mild depression
- **≥10:** Moderate depression ⭐ **RECOMMENDED THRESHOLD**
- **≥15:** Moderately severe depression
- **≥20:** Severe depression

**Clinical Rationale:**
- PHQ-9 ≥10 has 88% sensitivity and 88% specificity for major depression
- Widely used in primary care and research
- Score of 10 commonly used as cut-point for clinical intervention
- Expected prevalence: ~15-25% in sleep disorder populations

**References:**
- Kroenke et al., J Gen Intern Med 2001;16(9):606-613
- Manea et al., CMAJ 2012;184(3):E191-196

#### BDI-II (Beck Depression Inventory-II) - APPLES

**Instrument:** 21-item self-report questionnaire  
**Score Range:** 0-63  
**Validated Thresholds:**
- **0-13:** Minimal depression
- **≥14:** Mild depression ⭐ **RECOMMENDED THRESHOLD (conservative)**
- **≥20:** Moderate depression ⭐ **ALTERNATIVE THRESHOLD**
- **≥29:** Severe depression

**Clinical Rationale:**
- BDI-II ≥14 provides good sensitivity for any depressive disorder
- BDI-II ≥20 is more specific for moderate-severe depression
- **Recommendation:** Start with ≥14 for higher sensitivity, can threshold at ≥20 if needed
- Expected prevalence: 20-30% at ≥14, ~10-15% at ≥20

**References:**
- Beck et al., Psychological Assessment 1996;8(3):290-294
- Wang & Gorenstein, Braz J Psychiatry 2013;35(4):416-431

---

### Anxiety Screening

#### GAD-7 (Generalized Anxiety Disorder-7) - STAGES

**Instrument:** 7-item self-report questionnaire  
**Score Range:** 0-21  
**Validated Thresholds:**
- **0-4:** Minimal anxiety
- **5-9:** Mild anxiety
- **≥10:** Moderate anxiety ⭐ **RECOMMENDED THRESHOLD**
- **≥15:** Severe anxiety

**Clinical Rationale:**
- GAD-7 ≥10 has 89% sensitivity and 82% specificity for GAD
- Also valid for panic disorder, social anxiety, PTSD
- Score of 10 is standard clinical cut-point
- Expected prevalence: ~15-20% in general medical settings

**References:**
- Spitzer et al., Arch Intern Med 2006;166(10):1092-1097
- Plummer et al., Ann Intern Med 2016;165(10):HO2-HO3

---

### Sleepiness

#### ESS (Epworth Sleepiness Scale) - APPLES, SHHS-V2, MrOS

**Instrument:** 8-item self-report questionnaire  
**Score Range:** 0-24  
**Validated Thresholds:**
- **0-10:** Normal daytime sleepiness
- **≥11:** Excessive daytime sleepiness ⭐ **RECOMMENDED THRESHOLD**
- **≥16:** High excessive daytime sleepiness

**Clinical Rationale:**
- ESS ≥11 is standard threshold for excessive daytime sleepiness (EDS)
- Strong correlation with MSLT (Multiple Sleep Latency Test)
- Predicts treatment response and functional impairment
- Expected prevalence: ~25-35% in sleep clinic populations

**References:**
- Johns, Sleep 1991;14(6):540-545
- Kendzerska et al., CMAJ Open 2014;2(1):E11-17

---

### Insomnia

#### ISI (Insomnia Severity Index) - STAGES, MrOS

**Instrument:** 7-item self-report questionnaire  
**Score Range:** 0-28  
**Validated Thresholds:**
- **0-7:** No clinically significant insomnia
- **8-14:** Subthreshold (mild) insomnia
- **≥15:** Moderate insomnia ⭐ **RECOMMENDED THRESHOLD**
- **≥22:** Severe insomnia

**Clinical Rationale:**
- ISI ≥15 indicates clinically significant insomnia requiring intervention
- Strong psychometric properties (Cronbach's α = 0.90-0.91)
- Sensitive to treatment effects
- Expected prevalence: ~20-30% at ≥15 in community samples

**References:**
- Bastien et al., Sleep 2001;24(5):583-587
- Morin et al., Sleep 2011;34(5):601-608

---

### Fatigue

#### FSS (Fatigue Severity Scale) - STAGES

**Instrument:** 9-item self-report questionnaire  
**Score Range:** 9-63 (total) or 1-7 (mean)  
**Validated Thresholds:**
- **Mean ≥4.0** or **Total ≥36:** Clinically significant fatigue ⭐ **RECOMMENDED**
- Higher scores in neurological conditions (MS: mean 4.8-5.1)
- Lower scores in healthy controls (mean 2.3-3.0)

**Clinical Rationale:**
- FSS mean ≥4.0 distinguishes fatigued from non-fatigued individuals
- Widely used in sleep disorders, neurological conditions
- Good reliability (Cronbach's α = 0.89)
- Expected prevalence: ~25-40% in sleep disorder populations

**References:**
- Krupp et al., Arch Neurol 1989;46(10):1121-1123
- Valko et al., J Sleep Res 2008;17(2):217-220

---

### Cardiovascular Disease (CVD) - SHHS

**Data Type:** Incident cardiovascular events  
**Source:** Adjudicated medical records and death certificates  
**Outcomes Included:**
- Myocardial infarction (MI)
- Stroke
- Heart failure
- Cardiovascular death

**Threshold:** `any_cvd == 1` (incident event occurred)

**Clinical Rationale:**
- Gold-standard prospective outcome data
- Strong association with sleep apnea
- Clinically relevant for cardiovascular risk stratification
- Expected prevalence: ~10-20% incident events over follow-up period

**Special Considerations:**
- Time-to-event data available (can use for survival analysis later)
- Start with binary classification (event occurred: yes/no)
- Consider baseline CVD exclusion for pure prediction task

---

### Cognition (MMSE) - APPLES

**Instrument:** Mini-Mental State Examination  
**Score Range:** 0-30  
**Thresholds:**
- **24-30:** Normal cognition
- **19-23:** Mild cognitive impairment
- **10-18:** Moderate impairment
- **<10:** Severe impairment

**Clinical Rationale for APPLES:**
- ⚠️ **NOT RECOMMENDED for binary classification** - Most APPLES subjects score 24-30 (normal)
- Better suited for regression or longitudinal change detection
- Limited variance in this relatively healthy cohort
- Consider only if interested in subtle cognitive differences within normal range

---

## Task Recommendations by Tier

### 🥇 Tier 1: Start Immediately (Highest Success Probability)

#### 1. Sleep Apnea (apnea_binary) - ALL 4 DATASETS
**Total Subjects:** ~13,900  
**Datasets:** STAGES (1,513), SHHS (8,444), APPLES (1,104), MrOS (2,907)  
**Threshold:** AHI ≥15 events/hour

**Why Start Here:**
- ✅ **Objective measurement** - PSG-derived, high reliability
- ✅ **Strong PSG signal** - Direct relationship to respiratory events during sleep
- ✅ **Balanced prevalence** - ~30-40% positive class
- ✅ **All datasets** - Enables cross-dataset transfer learning
- ✅ **Clinical importance** - Major health outcome, treatment decisions
- ✅ **Research precedent** - Many papers demonstrate PSG can predict apnea

**Data Extraction:**
- **APPLES:** `nsrr_ahi_chicago1999` in harmonized CSV ✅ EASIEST
- **SHHS:** `rdi3p` in shhs1/shhs2 CSVs ✅ EASY
- **MrOS:** `nsrr_ahi_hp3r_aasm15` in harmonized CSV ✅ EASY
- **STAGES:** Extract from XML annotations ⚠️ MEDIUM effort

**Expected Performance:** Cohen's κ > 0.6, AUROC > 0.85 (based on literature)

---

#### 2. Depression (depression_binary) - STAGES + APPLES
**Total Subjects:** ~2,600 (STAGES: 1,513, APPLES: 1,104)  
**Instruments:** PHQ-9 (STAGES), BDI-II (APPLES)  
**Thresholds:** PHQ-9 ≥10, BDI-II ≥14

**Why Start Here:**
- ✅ **Validated instruments** - Gold-standard screening tools
- ✅ **Direct CSV extraction** - No external files needed
- ✅ **Clinical relevance** - Depression highly comorbid with sleep disorders
- ✅ **Research interest** - Growing literature on sleep-depression links
- ✅ **Balanced prevalence** - ~15-25% positive class
- ✅ **Two datasets** - Can test cross-instrument generalization

**Research Questions:**
- Can PSG predict depression screening scores?
- Do PHQ-9 and BDI-II predictions transfer across instruments?
- Which PSG features matter most (REM sleep, sleep fragmentation)?

**Expected Performance:** AUROC > 0.65-0.75 (moderate but valuable)

---

#### 3. Sleepiness (sleepiness_binary) - APPLES
**Total Subjects:** ~1,100  
**Instrument:** ESS (Epworth Sleepiness Scale)  
**Threshold:** ESS ≥11

**Why Start Here:**
- ✅ **Direct sleep outcome** - ESS measures daytime sleepiness from poor sleep
- ✅ **Strong PSG signal expected** - Sleep fragmentation, hypoxia, arousal index
- ✅ **Single dataset (APPLES)** - Start simple, extend to SHHS-V2/MrOS later
- ✅ **CSV available** - `esstotalscoreqc` in apples-dataset CSV
- ✅ **Clinical utility** - Predicts driving accidents, QoL, treatment response

**Research Questions:**
- Can PSG predict subjective sleepiness?
- Is AHI alone sufficient or do we need microarchitecture features?

**Expected Performance:** AUROC > 0.70 (sleep-related outcome)

---

### 🥈 Tier 2: Add After Initial Success

#### 4. Anxiety (anxiety_binary) - STAGES only
**Total Subjects:** 1,513  
**Instrument:** GAD-7  
**Threshold:** ≥10

**Why Include:**
- ✅ Similar to depression - test psychiatric screening from PSG
- ✅ CSV available in STAGES dataset
- ✅ Anxiety-sleep bidirectional relationship

**Why Tier 2:**
- ⚠️ Only one dataset (limited sample)
- ⚠️ Less studied than depression in sleep literature
- ⚠️ May have lower signal-to-noise than depression

---

#### 5. Insomnia (insomnia_binary) - STAGES + MrOS
**Total Subjects:** ~4,420 (STAGES: 1,513, MrOS: 2,907)  
**Instrument:** ISI  
**Threshold:** ≥15

**Why Include:**
- ✅ Core sleep disorder - high clinical relevance
- ✅ Two datasets for validation
- ✅ Interesting research question (can objective PSG predict subjective insomnia?)

**Why Tier 2:**
- ⚠️ STAGES requires external XLSX file (not in main CSV)
- ⚠️ Paradoxical insomnia - PSG may look normal despite complaints
- ⚠️ Weaker PSG signal expected (literature shows mixed results)

---

#### 6. CVD (cvd_binary) - SHHS only
**Total Subjects:** 8,444  
**Outcome:** Incident cardiovascular disease

**Why Include:**
- ✅ HUGE sample size (8,444 subjects)
- ✅ Gold-standard outcome (adjudicated events)
- ✅ Major clinical importance
- ✅ Strong apnea-CVD link in literature

**Why Tier 2:**
- ⚠️ Requires special handling (time-to-event data)
- ⚠️ Low prevalence (~10-20%) may need rebalancing
- ⚠️ Confounding by age, comorbidities
- ⚠️ Consider after establishing baseline with apnea task

**Implementation Notes:**
- Start with binary classification (ignoring time-to-event)
- Later: Add survival analysis (Cox model, time-dependent features)
- Baseline CVD exclusion may be needed

---

### 🥉 Tier 3: Consider Later (More Complex)

#### 7. Fatigue (fatigue_binary) - STAGES only
**Why Later:**
- Limited to single dataset
- FSS less commonly used than PHQ-9/GAD-7
- Overlap with depression/sleepiness constructs

#### 8. Rested Morning - STAGES, SHHS, MrOS
**Why Later:**
- Subjective single-item measures
- Visit-specific questions (complex mapping)
- Lower clinical validation than standardized instruments
- May be more noise than signal

#### 9. Cognition (MMSE regression) - APPLES
**Why Later:**
- Ceiling effect in healthy cohort (most score 24-30)
- Better suited for longitudinal decline detection
- Limited variance for cross-sectional prediction

---

## Storage Architecture

### Recommended: Unified Targets Table (Option 1)

**Location:** `/scratch/boshra95/psg/unified/metadata/unified_targets.parquet`

#### Schema Design

```python
unified_targets_schema = {
    # Identifiers
    'unified_id': str,              # {dataset}_{subject_id} - Primary key
    'dataset': str,                 # 'stages', 'shhs', 'apples', 'mros'
    'subject_id': str,              # Original subject ID
    'visit': int,                   # Visit number (1, 2, etc.; 0 if single-visit)
    
    # Binary Classification Targets (0=negative, 1=positive, -1=missing/NA)
    'apnea_binary': int,            # Sleep apnea (AHI ≥15)
    'depression_binary': int,       # Depression (PHQ-9 ≥10 or BDI-II ≥14)
    'sleepiness_binary': int,       # Excessive sleepiness (ESS ≥11)
    'anxiety_binary': int,          # Anxiety (GAD-7 ≥10)
    'insomnia_binary': int,         # Insomnia (ISI ≥15)
    'cvd_binary': int,              # Incident CVD (any_cvd == 1)
    'fatigue_binary': int,          # Fatigue (FSS ≥36)
    'rested_morning': int,          # Morning restedness (0=poor, 1=good, -1=missing)
    
    # Continuous Scores (NaN if missing) - For regression or custom thresholding
    'ahi_score': float,             # Apnea-Hypopnea Index
    'phq9_score': float,            # PHQ-9 total (0-27)
    'bdi_score': float,             # BDI-II total (0-63)
    'gad7_score': float,            # GAD-7 total (0-21)
    'ess_score': float,             # ESS total (0-24)
    'isi_score': float,             # ISI total (0-28)
    'fss_score': float,             # FSS mean (1-7)
    'mmse_score': float,            # MMSE total (0-30)
    
    # Metadata
    'target_source_file': str,      # Which CSV file data came from
    'extraction_date': datetime,    # When targets were extracted
    'phq9_available': bool,         # Flags for data availability
    'bdi_available': bool,
    'ahi_available': bool,
    # ... (one flag per score)
    
    # Quality Control
    'any_target_available': bool,   # At least one target is not missing
    'num_targets_available': int,   # Count of non-missing targets
}
```

#### Directory Structure

```
/scratch/boshra95/psg/
├── unified/
│   ├── metadata/
│   │   ├── unified_metadata.parquet          # ✅ Existing - demographics, channels, paths
│   │   ├── unified_targets.parquet           # 🆕 NEW - All classification targets
│   │   └── target_extraction.log             # 🆕 NEW - Extraction logs
│   └── targets/                              # 🆕 NEW - Per-task CSV exports (optional)
│       ├── apnea_binary.csv                  # Task-specific files for quick filtering
│       ├── depression_binary.csv
│       ├── sleepiness_binary.csv
│       └── README.md                         # Documentation
└── {dataset}/
    └── derived/
        ├── hdf5_signals/                      # ✅ Existing - Preprocessed signals @ 128 Hz
        ├── annotations/                       # ✅ Existing - Sleep stages (30s epochs)
        └── metadata_cache.parquet             # ✅ Existing - Dataset-specific cache
```

---

### Usage Examples

#### Loading Targets with Metadata

```python
import pandas as pd
from pathlib import Path

# Load unified metadata (demographics, channels, file paths)
metadata = pd.read_parquet('/scratch/boshra95/psg/unified/metadata/unified_metadata.parquet')

# Load targets
targets = pd.read_parquet('/scratch/boshra95/psg/unified/metadata/unified_targets.parquet')

# Join on unified_id
data = metadata.merge(targets, on='unified_id', how='left')

# Filter to subjects with apnea labels
apnea_subjects = data[data['apnea_binary'] >= 0]  # 0 or 1, exclude -1 (missing)

# Multi-task subset (apnea + depression)
multitask = data[
    (data['apnea_binary'] >= 0) & 
    (data['depression_binary'] >= 0)
]

print(f"Apnea task: {len(apnea_subjects)} subjects")
print(f"  Datasets: {apnea_subjects['dataset'].value_counts()}")
print(f"  Class distribution: {apnea_subjects['apnea_binary'].value_counts()}")
```

#### PyTorch Dataset Example

```python
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class NSRRMultiTaskDataset(Dataset):
    def __init__(self, metadata_df, targets_df, tasks=['apnea_binary']):
        """
        Args:
            metadata_df: Loaded unified_metadata.parquet
            targets_df: Loaded unified_targets.parquet
            tasks: List of task names to include
        """
        # Join metadata and targets
        self.data = metadata_df.merge(targets_df, on='unified_id', how='inner')
        
        # Filter to subjects with at least one task available
        mask = pd.Series(True, index=self.data.index)
        for task in tasks:
            mask &= (self.data[task] >= 0)  # Exclude -1 (missing)
        self.data = self.data[mask].reset_index(drop=True)
        
        self.tasks = tasks
        print(f"Dataset initialized: {len(self.data)} subjects, {len(tasks)} tasks")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load HDF5 signal
        hdf5_path = row['hdf5_path']
        with h5py.File(hdf5_path, 'r') as f:
            # Load all available channels (example: C3-M2, C4-M1, etc.)
            signals = {}
            for ch_name in f.keys():
                if ch_name != 'metadata':
                    signals[ch_name] = f[ch_name][:]
        
        # Load sleep stages
        stages_path = row['annotation_path']
        stages = np.load(stages_path) if Path(stages_path).exists() else None
        
        # Load targets
        targets = {task: row[task] for task in self.tasks}
        
        return {
            'signals': signals,
            'stages': stages,
            'targets': targets,
            'unified_id': row['unified_id'],
            'dataset': row['dataset']
        }
```

---

### Alternative: Per-Subject JSON Files (Option 2)

**Location:** `{dataset}/derived/targets/{subject_id}_targets.json`

**Pros:**
- Flexible schema per dataset
- Easy to add new targets incrementally
- No need to reload large parquet files

**Cons:**
- Slower for large-scale multi-task training
- Harder to get global statistics (need to iterate all files)
- More disk space (many small files vs one parquet)

**Example JSON:**
```json
{
  "subject_id": "BOGN00004",
  "dataset": "stages",
  "extraction_date": "2026-03-10",
  "targets": {
    "apnea_binary": 1,
    "depression_binary": 0,
    "anxiety_binary": 0,
    "fatigue_binary": -1
  },
  "scores": {
    "ahi": 18.3,
    "phq9": 6,
    "gad7": 4,
    "fss": null
  },
  "source_files": {
    "phq9": "stages-dataset-0.3.0.csv",
    "gad7": "stages-dataset-0.3.0.csv"
  }
}
```

**Recommendation:** Use Option 1 (unified parquet) for main workflow, optionally export Option 2 for debugging/inspection.

---

## Implementation Roadmap

### Phase 1: Core Implementation (Week 1)

#### Step 1: Data Extraction Module
**File:** `scripts/extract_targets.py`

**Features:**
- Load CSV files per dataset
- Extract continuous scores (AHI, PHQ-9, BDI-II, etc.)
- Apply clinical thresholds to create binary labels
- Handle missing values consistently (-1 for binary, NaN for continuous)
- Join with unified_metadata on subject_id

**Priority Order:**
1. ✅ APPLES (easiest - all in CSV)
2. ✅ SHHS (easy - standard CSVs)
3. ✅ MrOS (easy - standard CSVs)
4. ⚠️ STAGES (medium - need to handle external files for some tasks)

**Output:** `unified_targets.parquet`

---

#### Step 2: Validation & QC
**File:** `scripts/validate_targets.py`

**Checks:**
- Count subjects per task per dataset
- Class distribution (expect 15-40% positive for most tasks)
- Missing value patterns
- Score distributions (histograms, outliers)
- Cross-task correlations (e.g., apnea vs sleepiness)

**Output:** 
- `logs/target_validation_report.txt`
- `output/target_distributions.png` (visualizations)

---

#### Step 3: PyTorch Dataset Integration
**File:** `src/nsrr_tools/datasets/multitask_dataset.py`

**Features:**
- Load signals (HDF5) + stages + targets efficiently
- Support single-task and multi-task training
- Handle variable-length signals (windowing)
- Dataset-aware augmentation

---

### Phase 2: Extended Tasks (Week 2)

#### Step 4: Add Tier 2 Tasks
- Anxiety (STAGES)
- Insomnia (STAGES + MrOS)
- CVD (SHHS)

#### Step 5: Handle External Files
- Extract STAGES ISI from XLSX
- Extract STAGES AHI from XML annotations
- Validate cross-task consistency

---

### Phase 3: Advanced Features (Week 3+)

#### Step 6: Multi-Visit Handling
- SHHS Visit 1 vs Visit 2
- MrOS Visit 1 vs Visit 2
- Longitudinal analysis support

#### Step 7: Auxiliary PSG Variables
- Sleep efficiency
- Arousal index
- REM%, N3%
- TST

#### Step 8: Cross-Dataset Harmonization
- Test PHQ-9 vs BDI-II equivalence
- Create harmonized depression score
- Document instrument differences

---

## Data Availability Matrix

### Summary Table

| Task | STAGES | SHHS | APPLES | MrOS | Total N | Extraction Difficulty |
|------|--------|------|--------|------|---------|---------------------|
| **apnea_binary** | ⚠️ 1,513 | ✅ 8,444 | ✅ 1,104 | ✅ 2,907 | ~13,900 | EASY→MEDIUM |
| **depression_binary** | ✅ 1,513 | ❌ | ✅ 1,104 | ❌ | ~2,600 | EASY |
| **sleepiness_binary** | ❌ | 🟡 V2 only | ✅ 1,104 | 🟡 Available | ~1,100+ | EASY |
| **anxiety_binary** | ✅ 1,513 | ❌ | ❌ | ❌ | 1,513 | EASY |
| **insomnia_binary** | ⚠️ 1,513 | ❌ | ❌ | ✅ 2,907 | ~4,400 | MEDIUM |
| **cvd_binary** | ❌ | ✅ 8,444 | ❌ | ❌ | 8,444 | MEDIUM |
| **fatigue_binary** | ✅ 1,513 | ❌ | ❌ | ❌ | 1,513 | EASY |
| **rested_morning** | ⚠️ 1,513 | 🟡 8,444 | ❌ | 🟡 2,907 | Variable | HARD |

**Legend:**
- ✅ Available in main CSV files (EASY extraction)
- ⚠️ Requires external XLSX file (MEDIUM extraction)
- 🟡 Available but complex (visit-specific, subjective mapping)
- ❌ Not available in dataset

---

### Detailed Column Mapping

#### STAGES
```yaml
Data Source: stages-dataset-0.3.0.csv
Columns:
  - phq_1000: PHQ-9 total score → depression_binary (≥10)
  - gad_0800: GAD-7 total score → anxiety_binary (≥10)
  - fss_1000: FSS total score → fatigue_binary (≥36)
  - ess_0900: ESS total score → sleepiness_binary (≥11) [if available]

External Files Needed:
  - STAGES ASQ ISI to DIET 20200513 Final deidentified.xlsx
    → isi_score: ISI total → insomnia_binary (≥15)
  - STAGESPSGKeySRBDVariables.xlsx
    → ahi: Apnea-Hypopnea Index → apnea_binary (≥15)
  - STAGES post sleep questionnaire 2020-09-06 deidentified.xlsx
    → isq_0500 or compared_usual_feel_upon_awakening → rested_morning
```

#### APPLES
```yaml
Data Source: apples-harmonized-dataset-0.1.0.csv, apples-dataset-0.1.0.csv
Columns:
  - nsrr_ahi_chicago1999: AHI → apnea_binary (≥15)
  - bditotalscore: BDI-II total → depression_binary (≥14 or ≥20)
  - esstotalscoreqc: ESS total → sleepiness_binary (≥11)
  - mmsetotalscore: MMSE total → cognition_regression (continuous)

Visit Strategy:
  - Use visitn == 3 (DX visit) for PSG-matched variables (AHI, ESS)
  - Use visitn == 1 (BL visit) for baseline questionnaires (BDI, MMSE)
```

#### SHHS
```yaml
Data Source: shhs1-dataset-0.21.0.csv, shhs2-dataset-0.21.0.csv, 
             shhs-cvd-summary-dataset-0.21.0.csv
Columns:
  - rdi3p: RDI (≥3% desaturation rule) → apnea_binary (≥15)
  - any_cvd: Incident CVD event → cvd_binary (==1)
  - rest10 (Visit 1): Morning restedness (1-5) → rested_morning (≥4 good, ≤2 poor)
  - ms204c (Visit 2): Morning restfulness (1-5) → rested_morning (≥4 good, ≤2 poor)

Visit Strategy:
  - Prefer Visit 1 (baseline) for most tasks
  - CVD is incident outcome (use baseline PSG to predict later CVD)
```

#### MrOS
```yaml
Data Source: mros-visit1-harmonized-0.6.0.csv, 
             mros-visit1-dataset-0.6.0.csv, 
             mros-visit2-dataset-0.6.0.csv
Columns:
  - nsrr_ahi_hp3r_aasm15: AHI (AASM 2015 criteria) → apnea_binary (≥15)
  - slisiscr: ISI total score → insomnia_binary (≥15)
  - poxqual3: Morning sleep quality (1-5) → rested_morning (≥4 good, ≤2 poor)
  - epepwort: ESS total → sleepiness_binary (≥11)

Visit Strategy:
  - Most variables available in both Visit 1 and Visit 2
  - Prefer Visit 1 (larger sample) unless longitudinal analysis
```

---

## Next Steps & Decision Points

### Immediate Decisions Needed

1. **Task Selection for Phase 1:**
   - Confirm Tier 1 tasks: apnea_binary, depression_binary, sleepiness_binary?
   - Add any Tier 2 tasks to Phase 1?

2. **Storage Preference:**
   - Confirm Option 1 (unified parquet table) is acceptable?
   - Need per-subject JSON files (Option 2) as well?

3. **STAGES AHI Extraction:**
   - Extract from XML annotations now (medium effort)?
   - Skip STAGES apnea initially, focus on APPLES/SHHS/MrOS (easier)?

4. **Threshold Choices:**
   - BDI-II: Use ≥14 (sensitive) or ≥20 (specific)?
   - Any custom thresholds based on your research goals?

5. **Multi-Visit Strategy:**
   - For SHHS/MrOS: Use Visit 1 only, or create separate samples per visit?
   - Longitudinal analysis later, or treat visits as independent samples?

---

### Implementation Checklist

- [ ] Create `scripts/extract_targets.py` - Main extraction script
- [ ] Start with APPLES (easiest dataset)
- [ ] Extract SHHS targets (large sample)
- [ ] Extract MrOS targets
- [ ] Extract STAGES targets (handle external files)
- [ ] Generate unified_targets.parquet
- [ ] Create validation report (class distributions, missing patterns)
- [ ] Create PyTorch dataset loader
- [ ] Test multi-task training with sleep staging + apnea
- [ ] Document extraction process and column mappings
- [ ] Add unit tests for target extraction
- [ ] Create visualization scripts (target distributions, correlations)

---

## References & Resources

### Clinical Guidelines
1. **Sleep Apnea:** AASM International Classification of Sleep Disorders (ICSD-3)
2. **Depression:** DSM-5 criteria, PHQ-9 validation studies
3. **Anxiety:** DSM-5 criteria, GAD-7 validation studies
4. **Insomnia:** ICSD-3, ISI validation studies

### NSRR Documentation
- NSRR Data Dictionary: https://sleepdata.org/datasets/
- STAGES: https://sleepdata.org/datasets/stages
- SHHS: https://sleepdata.org/datasets/shhs
- APPLES: https://sleepdata.org/datasets/apples
- MrOS: https://sleepdata.org/datasets/mros

### Key Papers
- **Sleep Apnea Prediction from PSG:** Multiple studies showing AUROC > 0.85
- **Depression-Sleep Links:** Palagini et al., 2013; Baglioni et al., 2011
- **Sleepiness Prediction:** Literature mixed, moderate predictability
- **CVD-Sleep Apnea:** Somers et al., Circulation 2008

---

## Appendix: Column Name Examples

### PHQ-9 Items (STAGES)
```
phq_0100 through phq_0900: Individual items (0-3 each)
phq_1000: Total score (sum of 9 items, 0-27)
```

### GAD-7 Items (STAGES)
```
gad_0100 through gad_0700: Individual items (0-3 each)
gad_0800: Total score (sum of 7 items, 0-21)
```

### FSS Items (STAGES)
```
fss_0100 through fss_0900: Individual items (1-7 each)
fss_1000: Total score (sum of 9 items, 9-63)
fss_mean: Mean score (1-7)
```

### ESS Items (APPLES)
```
Individual items in apples-dataset-0.1.0.csv
esstotalscoreqc: Total score with quality control (0-24)
```

### BDI-II Items (APPLES)
```
bditotalscore: Total score (0-63)
Individual items likely not provided in CSV
```

---

**Document Version:** 1.0  
**Last Updated:** March 10, 2026  
**Author:** NSRR Preprocessing Pipeline Team  
**Status:** Ready for user review and implementation
