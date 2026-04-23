# V2 Task Definitions — Implementation Notes

> Branch: `v2-task-definitions`  
> Created: 2026-04-22  
> Companion document: `v2_task_audit_and_plan.md`

---

## What Changed

V2 adds 9 new tasks without touching the v1 pipeline outputs.  
A separate config `configs/target_extraction_v2.yaml` writes to `targets_v2/`.  
All new task blocks in the adapters are guarded by `enabled: true/false` so running with v1 config leaves behaviour unchanged.

### New tasks summary

| Task | Type | Datasets | Source column(s) | Threshold |
|------|------|----------|-----------------|-----------|
| `sleep_efficiency_binary` | binary | APPLES, SHHS, MrOS | `nsrr_ttleffsp_f1` (APPLES/SHHS harmonized); `poslpeff` (MrOS main) | < 85 % → 1 |
| `psqi_binary` | binary | MrOS | `pqpsqi` (main v1+v2) | > 5 → 1 |
| `depression_extreme_binary` | binary | APPLES, STAGES | BDI (APPLES): ≤9=0, ≥20=1; PHQ-9 (STAGES): ≤4=0, ≥15=1; middle dropped | extreme-group |
| `osa_severity_apples` | 4-class | APPLES | `osaseveritypostqc` (main v1, string parse) | 0=Non-rand, 1=Mild, 2=Moderate, 3=Severe |
| `osa_binary_apples_postqc` | binary | APPLES | derived from `osa_severity_apples` | Non-rand+Mild (0,1) → 0, Moderate+Severe (2,3) → 1 |
| `sex_binary` | binary | APPLES, SHHS, STAGES | `nsrr_sex` (harmonized); female=1, male=0 | N/A |
| `age_regression` | regression (float) | APPLES, SHHS, MrOS, STAGES | `nsrr_age` (harmonized) | N/A |
| `bmi_regression` | regression (float) | APPLES, SHHS, MrOS, STAGES | `nsrr_bmi` (harmonized) | N/A |
| `age_class` | 3-class | APPLES, SHHS, MrOS, STAGES | derived from `age_value` | <50=0, 50–64=1, ≥65=2 |
| `bmi_binary` | binary | APPLES, SHHS, MrOS, STAGES | derived from `bmi_value` | <30=0, ≥30=1 (WHO obesity) |

**MrOS excluded from `sex_binary`** — all-male cohort, zero variance.  
**SHHS `age_regression`** — 67 subjects with `nsrr_age_gt89 == "yes"` set to NaN (censored).  
**MrOS `age_class`** — cohort is 65+, so expect all subjects in class 2; still included for completeness.

---

## Files Modified

| File | Change |
|------|--------|
| `configs/target_extraction_v2.yaml` | **NEW** — full standalone config pointing to `targets_v2/` |
| `scripts/extract_targets_shhs.py` | Added v2 block after CVD merge: sleep_efficiency, sex, age_regression, bmi_regression, age_class, bmi_binary |
| `scripts/extract_targets_apples.py` | Added v2 block after original merge: sleep_efficiency, osa_severity, depression_extreme, sex, age_regression, bmi_regression, age_class, bmi_binary |
| `scripts/extract_targets_mros.py` | Added v2 block after rested merge: sleep_efficiency, psqi, age_regression, bmi_regression, age_class, bmi_binary |
| `scripts/extract_targets_stages.py` | Added v2 block after fatigue: depression_extreme, sex/age_regression/bmi_regression/age_class/bmi_binary (loads harmonized CSV) |
| `scripts/create_master_targets.py` | Extended `_MASTER_COLUMNS` (incl. `bmi_binary`); pass-through for new binary/score/regression cols; `_REGRESSION_VALUE_COLUMNS` guard; updated `build_master()` and `log_statistics()` |
| `scripts/create_task_subject_lists.py` | Added `bmi_binary` to `BINARY_TASKS`; `osa_severity_apples` and `age_class` to `MULTICLASS_TASKS`; new `REGRESSION_TASKS` dict; `build_regression_task_list()` function; regression processing loop |
| `/home/boshra95/.vscode/launch.json` | Added 7 v2 debug configs (prefix `🆕 V2`) |

---

## Commands to Run (on cluster, in order)

Run each script via the VSCode debug configs prefixed `🆕 V2`, **or** from the terminal:

```bash
cd /home/boshra95/NSRR-tools
V2_CFG=configs/target_extraction_v2.yaml

# Step 1 — Per-dataset adapters (can run in any order; independent)
python scripts/extract_targets_apples.py --config $V2_CFG
python scripts/extract_targets_shhs.py   --config $V2_CFG
python scripts/extract_targets_mros.py   --config $V2_CFG
python scripts/extract_targets_stages.py --config $V2_CFG

# Step 2 — Merge into master parquet
python scripts/create_master_targets.py  --config $V2_CFG

# Step 3 — Per-task subject CSVs
python scripts/create_task_subject_lists.py --config $V2_CFG
```

Output lands in `/home/boshra95/scratch/psg/unified/targets_v2/`.

### v2-only subject lists (quick sanity check)

```bash
python scripts/create_task_subject_lists.py \
  --config configs/target_extraction_v2.yaml \
  --tasks sex_binary sleep_efficiency_binary psqi_binary \
          depression_extreme_binary osa_severity_apples osa_binary_apples_postqc \
          age_regression bmi_regression age_class bmi_binary
```

---

## Key Design Decisions

### Versioning
- V2 config writes to `targets_v2/`; v1 outputs in `targets/` are untouched on this branch.
- All adapters can be run with either config. New task blocks are skipped when not present in config.

### APPLES ID equivalence
- `appleid` (main file) == `nsrrid` (harmonized file); outer merge is correct.
- Demographics (sex, age, bmi) read from harmonized v1 and merged on `nsrrid` → `subject_id`.

### APPLES/SHHS/STAGES demographics
- Only populated at visit 1 in harmonized files.
- APPLES & STAGES: single-visit output, so subject_id is unsuffixed — straightforward left-join.
- SHHS: applied per visit (both v1/v2 rows read from harmonized, then merged on `subject_id` + `visit`).
- MrOS: harmonized v1 demographics broadcast to both `_v1` and `_v2` subject rows.

### SHHS age censoring
- `nsrr_age_gt89 == "yes"` → age set to NaN before writing `age_value`. These subjects contribute to classification bins in training code but are excluded from age_regression labels.

### `osaseveritypostqc` string parsing
- String format: `"N) description"` — extract integer prefix via `int(str(val).split(')')[0].strip())`.
- 0 = Non-randomized (included as class 0 per Q2 decision).
- **Two tasks produced from the same source column:**
  - `osa_severity_apples` (4-class) — stored in per-dataset CSV only; handled via `MULTICLASS_TASKS` → `osa_severity_apples_subjects.csv`.
  - `osa_binary_apples_postqc` (binary) — Non-rand+Mild (classes 0,1) → 0, Moderate+Severe (classes 2,3) → 1; stored in per-dataset CSV AND master parquet; handled via `BINARY_TASKS` → `osa_binary_apples_postqc_subjects.csv`.
- Binary computed in the adapter inside the `osa_severity_apples` block via `_osa_to_binary()`; both columns merged together in a single `_merge_left_on_appleid` call.

### Regression storage
- `age_value` and `bmi_value` stored as string floats in per-dataset CSVs (`''` = missing).
- Converted to `float` in master via `_score_to_float()`.
- `_REGRESSION_VALUE_COLUMNS` set prevents them from receiving the binary `-1` sentinel.
- Subject list label column is float; `num_classes=0` in summary TSV signals regression.

### Depression extreme groups
- APPLES: BDI ≤9 → 0, BDI ≥20 → 1; BDI 10–19 dropped (label = `''`).
- STAGES: PHQ-9 ≤4 → 0, PHQ-9 ≥15 → 1; PHQ-9 5–14 dropped.
- Middle-group subjects appear in master with `depression_extreme_binary = -1` (MISSING_BINARY).

### Age and BMI classification
- `age_class` (3-class) thresholds `[50, 65]` follow clinical convention: <50=young, 50–64=middle, ≥65=older.
- `bmi_binary` threshold 30.0 follows WHO obesity definition: <30=0 (non-obese), ≥30=1 (obese).
- Both derived from `targets['age_value']` / `targets['bmi_value']` strings already merged by the regression blocks. Each adapter guards with `if 'age_value' not in targets.columns` to enforce ordering.
- `age_class` stored only in per-dataset CSVs (via `MULTICLASS_TASKS`), **not** in master parquet.
- `bmi_binary` stored in per-dataset CSVs AND master parquet (via `BINARY_TASKS`); pass-through with `-1` sentinel for missing.
- SHHS censored subjects (age set to `''` by age_regression) get `''` for age_class too (~67 subjects).
- MrOS cohort is 65+ — expect all subjects in class 2; logged as a warning.

### MrOS insomnia (unchanged)
- `insomnia_binary` remains `enabled: false` in v2 config (Q1 decision: only 3% positive, no clinical signal).

### MrOS sex (excluded)
- All-male cohort — no `sex_binary` block added to MrOS.

---

## Per-dataset v2 output columns

### APPLES (`apples_targets.csv`)
```
subject_id, dataset, visit,
apnea_class, ahi_score,
depression_class, bdi_score,
sleepiness_class, ess_score,
sleep_efficiency_binary, eff_score,
osa_severity_apples,
osa_binary_apples_postqc,
depression_extreme_binary,
sex_binary,
age_value, age_class,
bmi_value, bmi_binary
```

### SHHS (`shhs_targets.csv`)
```
subject_id, dataset, visit,
apnea_class, ahi_score,
sleepiness_class, ess_score,
cvd_binary,
sleep_efficiency_binary, eff_score,
sex_binary,
age_value, age_class,
bmi_value, bmi_binary
```

### MrOS (`mros_targets.csv`)
```
subject_id, dataset, visit,
apnea_class, ahi_score,
sleepiness_class, ess_score,
insomnia_binary, isi_score,
cvd_binary,
rested_morning, rested_score,
sleep_efficiency_binary, eff_score,
psqi_binary, psqi_score,
age_value, age_class,
bmi_value, bmi_binary
```

### STAGES (`stages_targets.csv`)
```
subject_id, dataset, visit,
apnea_class, ahi_score,
depression_binary, phq9_score,
sleepiness_binary, ess_score,
anxiety_binary, gad7_score,
insomnia_binary, isi_score,
[fatigue_binary, fss_score — if enabled],
depression_extreme_binary,
sex_binary,
age_value, age_class,
bmi_value, bmi_binary
```

---

## Master Target Statistics (first full v2 run — 2026-04-23)

Produced by `create_master_targets.py` with `configs/target_extraction_v2.yaml`.

**Total records: 19,228** — SHHS: 10,115 | MrOS: 5,822 | STAGES: 1,775 | APPLES: 1,516

### Binary tasks

| Column | N valid | Positive | Negative | Missing |
|--------|---------|----------|----------|---------|
| `apnea_binary` | 14,097 | 6,888 (48.9%) | 7,209 (51.1%) | 5,131 |
| `depression_binary` | 2,794 | 750 (26.8%) | 2,044 (73.2%) | 16,434 |
| `sleepiness_binary` | 16,431 | 4,888 (29.7%) | 11,543 (70.3%) | 2,797 |
| `anxiety_binary` | 1,698 | 341 (20.1%) | 1,357 (79.9%) | 17,530 |
| `insomnia_binary` | 1,710 | 761 (44.5%) | 949 (55.5%) | 17,518 |
| `fatigue_binary` | 0 | — | — | 19,228 (disabled) |
| `cvd_binary` | 13,045 | 3,266 (25.0%) | 9,779 (75.0%) | 6,183 |
| `rested_morning` | 3,934 | 1,702 (43.3%) | 2,232 (56.7%) | 15,294 |
| `sex_binary` | 13,163 | 6,756 (51.3%) | 6,407 (48.7%) | 6,065 |
| `sleep_efficiency_binary` | 13,615 | 8,285 (60.9%) | 5,330 (39.1%) | 5,613 |
| `psqi_binary` | 3,933 | 1,727 (43.9%) | 2,206 (56.1%) | 15,295 |
| `depression_extreme_binary` | 1,761 | 234 (13.3%) | 1,527 (86.7%) | 17,467 |
| `bmi_binary` | 15,532 | 5,098 (32.8%) | 10,434 (67.2%) | 3,696 |

### Regression tasks

| Column | N valid | Mean | Std | Min | Max | Missing |
|--------|---------|------|-----|-----|-----|---------|
| `age_value` | 6,190 | 61.3 | 18.0 | 13.0 | 90.0 | 13,038 |
| `bmi_value` | 6,184 | 29.6 | 7.0 | 11.9 | 79.5 | 13,044 |

> Note: age/bmi N (~6,190) reflects APPLES+SHHS+STAGES visit-1 subjects + MrOS visit-1 only (no broadcasting to visit 2 after the 2026-04-23 fix). MrOS visit-2 rows have empty age/bmi. `sex_binary` missing=6,065 reflects MrOS subjects (all-male, excluded by design).

---

## Validation Checklist (run after each adapter)

- [ ] APPLES: `osa_severity_apples` values ∈ {0,1,2,3,''}; no parse errors in log
- [ ] APPLES: `sex_binary` counts match expected ~50/50 split
- [ ] APPLES: `age_value` N ≈ 250 (visit 1 subjects); no -9 values
- [ ] SHHS: `sleep_efficiency_binary` N ≈ 5k v1 + 2.5k v2; ~20–30% positive
- [ ] SHHS: `age_value` 67 censored ages per log line
- [ ] MrOS: `psqi_binary` N ≈ 2k total (v1+v2); ~50–60% positive expected
- [ ] MrOS: `age_value` same N as `bmi_value` (from harmonized v1, broadcast to v2)
- [ ] STAGES: harmonized file loaded OK; `sex_binary` non-empty
- [ ] Master: `age_value` and `bmi_value` are float columns (not int)
- [ ] Task lists: `age_regression_subjects.csv` and `bmi_regression_subjects.csv` present
- [ ] Task lists: `osa_severity_apples_subjects.csv` has 4 distinct label values
- [ ] Task lists: `osa_binary_apples_postqc_subjects.csv` present; labels ∈ {0, 1}
- [ ] Task lists: `age_class_subjects.csv` present; labels ∈ {0, 1, 2} across all 4 datasets (MrOS likely all class 2)
- [ ] Task lists: `bmi_binary_subjects.csv` present; labels ∈ {0, 1} across all 4 datasets
