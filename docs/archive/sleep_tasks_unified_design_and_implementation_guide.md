# Sleep Tasks Unified Design and Implementation Guide (Detailed)

## Purpose
This document consolidates the full analysis across SHHS, APPLES, and MrOS, including:
- What each column actually represents (from dictionaries)
- Why certain tasks are strong or weak
- Exact label definitions and thresholds
- Which datasets can be merged (and why / why not)
- Concrete config instructions for implementation

This is intended for a coding agent to modify configs, data loaders, and training scripts **without ambiguity**.

---

# 1. CORE PRINCIPLES (CRITICAL)

1. **Do NOT merge datasets unless labels are truly comparable**
   - Same scale
   - Same clinical meaning
   - Same distribution assumptions

2. Prefer:
   - PSG-derived labels (AHI, sleep efficiency)
   - Validated scales (ESS, ISI, PSQI)

3. Avoid early use of:
   - Single-question subjective labels (restedness)
   - Weak mental-state proxies

4. For subjective tasks:
   - Use **binary classification** (reduces noise)

5. For imbalanced tasks:
   - Use AUROC + AUPRC

---

# 2. FULL DATASET ANALYSIS (MERGED INSIGHTS)

## 2.1 SHHS

### Strong variables
- `nsrr_ahi_hp3r_aasm15` → gold-standard AHI
- `ess_s1`, `ess_s2` → ESS (validated)

### Medium
- `any_cvd` → subject-level outcome (but not clean timing unless survival)

### Weak
- `rest10`, `ms204c` → subjective restedness

### Key insight
SHHS is **excellent for physiology (AHI)** and **OK for ESS**, but weak for subjective sleep quality.

---

## 2.2 APPLES

### Strong
- `nsrr_ahi_chicago1999` → AHI (slightly different definition but usable)
- `esstotalscoreqc` → ESS
- PSG metrics:
  - `sleepeffpsg`
  - `wasoqcpsg`

### Very strong (unique)
- Cognitive:
  - `MeanRT`, `LapsesRTge500`

### Medium
- `bditotalscore` → depression (validated but noisy across bins)

### Weak
- subjective SHQ questions (fatigue, unrested)

### Key insight
APPLES is the **best dataset for behavioral + cognitive outcomes**, not just PSG.

---

## 2.3 MrOS

### Strong
- `nsrr_ahi_hp3r_aasm15` (visit 1)
- `epepwort` → ESS
- `slisiscr` → ISI (confirmed)
- `pqpsqi` → PSQI (confirmed)

### Medium
- `poxqual3` → subjective quality

### Weak / risky
- `cvchd` → history, not incident

### Key insight
MrOS is the **best dataset for validated subjective sleep scales (ISI, PSQI)**.

---

# 3. FINAL TASK DEFINITIONS (WITH EXPLANATION)

## 3.1 TASK: OSA (AHI-based)

### Why this task is strong
- Direct physiological signal
- Measured from PSG
- Consistent across datasets

### Definition
- Binary: AHI >= 15
- Multiclass: <5 / 5–15 / 15–30 / ≥30

### Datasets (MERGE = YES)
- SHHS: nsrr_ahi_hp3r_aasm15
- MrOS: nsrr_ahi_hp3r_aasm15 (visit 1 only)
- APPLES: nsrr_ahi_chicago1999 (visit 3)

### Important note
Chicago vs AASM definitions differ slightly → acceptable for classification

---

## 3.2 TASK: Sleepiness (ESS)

### Why this task is strong
- Same scale across datasets
- Validated clinical instrument
- Directly related to sleep physiology

### Definition
- Binary (main): ESS ≥ 11
- Optional: 3-class

### Datasets (MERGE = YES)
- SHHS: ess_s1, ess_s2
- MrOS: epepwort
- APPLES: esstotalscoreqc

---

## 3.3 TASK: Insomnia (ISI)

### Why strong
- Validated clinical scale
- Stronger than raw insomnia flags

### Definition
- Binary: ISI ≥ 15

### Datasets (MERGE = NO)
- MrOS: slisiscr (visit 2 only)
- STAGES: isi_score

### Why not merge
- different cohorts
- distribution mismatch

---

## 3.4 TASK: Sleep Quality (PSQI)

### Why strong
- Gold-standard sleep quality scale

### Definition
- Binary: PSQI > 5

### Dataset
- MrOS only: pqpsqi

---

## 3.5 TASK: PSG Sleep Efficiency

### Why strong
- Objective measure
- Direct PSG output

### Definition
- Binary: < 85%

### Datasets (PARTIAL MERGE)
- SHHS: nsrr_ttleffsp_f1
- APPLES: sleepeffpsg
- MrOS: poslpeff

### Note
Definitions may vary → merging optional

---

## 3.6 TASK: Depression (Extreme Groups)

### Why weak
- subjective
- influenced by many non-sleep factors

### Fix
Use extreme groups only:
- ≤9 vs ≥20

### Datasets (NO MERGE)
- APPLES: bditotalscore
- STAGES: phq_1000

---

## 3.7 TASK: Restedness

### Why weak
- single question
- noisy

### Definition
- ≥4 vs ≤3

### Datasets (NO MERGE)
- SHHS: rest10, ms204c
- MrOS: poxqual3
- APPLES: RefreshInAMHP

---

## 3.8 TASK: Cardiovascular Disease

### Option A: Classification
- MrOS: cvchd
- SHHS: any_cvd
- NO MERGE

### Option B: Survival (BEST)
- SHHS only
- uses event dates + censoring

### Why survival is better
- uses time information
- handles censoring

---

## 3.9 TASK: Cognitive Performance (APPLES)

### Why strong
- objective
- directly affected by sleep

### Variables
- MeanRT
- LapsesRTge500

---

# 4. MERGING DECISIONS (CRITICAL)

| Task | Merge? | Reason |
|------|--------|--------|
| AHI | YES | physiology consistent |
| ESS | YES | same scale |
| ISI | NO | dataset-specific |
| PSQI | NO | only MrOS |
| Sleep efficiency | PARTIAL | definition differences |
| Depression | NO | different scales |
| Restedness | NO | different questions |
| CVD | NO | different definitions |
| Survival | SHHS only | needs time data |

---

# 5. CONFIG CHANGES REQUIRED

## MUST FIX

1. MrOS insomnia
```yaml
insomnia_binary:
  enabled: true
  source: mros-visit2-dataset-0.6.0.csv
  column: slisiscr
```

2. MrOS apnea
```yaml
keep_both_visits: false
```

3. Sleepiness
```yaml
threshold: 11
```

---

## ADD

```yaml
osa_binary
psqi_binary
sleep_efficiency_binary
```

---

# 6. PRIORITY EXPERIMENT PLAN

## Phase 1 (START HERE)
- AHI (merged)
- ESS (merged)

## Phase 2
- sleep efficiency
- ISI

## Phase 3
- PSQI
- cognitive (APPLES)

## Phase 4
- depression (extreme)
- restedness

## Phase 5
- survival (SHHS)

---

# 7. FINAL CONCLUSIONS

1. Only AHI and ESS should be merged.
2. Most subjective tasks must stay dataset-specific.
3. Strongest signals:
   - PSG metrics
   - ESS / ISI / PSQI
4. Weakest signals:
   - restedness
   - general questionnaire items
5. Survival analysis only valid for SHHS CVD.

---

# 8. IMPLEMENTATION NOTES

- Align labels with correct visit
- Do NOT mix visits incorrectly
- Prefer binary tasks first
- Use AUROC/AUPRC for imbalance
- Use Macro-F1 for multiclass

---

END

