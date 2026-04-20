# V2 Task Audit and Implementation Plan
*April 2026 — based on `sleep_tasks_unified_design_and_implementation_guide(new).md`*

This document records the full findings from the task audit: verified column names, current implementation state vs. guide recommendations, per-task plans, and open questions. It is the reference document before any v2 code changes.

---

## 1. Column Verification

All column names mentioned in the design guide were verified against actual data files and the nocturn repo YAML configs.

| Column | Dataset | File | Confirmed | Notes |
|--------|---------|------|-----------|-------|
| `nsrr_ahi_hp3r_aasm15` | SHHS, MrOS | harmonized files | ✅ | SHHS: both visits; MrOS: visit 1 only |
| `nsrr_ahi_chicago1999` | APPLES | harmonized | ✅ | Visit 3 (DX visit) |
| `ess_s1` / `ess_s2` | SHHS | shhs1 / shhs2 main | ✅ | NOT in harmonized file |
| `epepwort` | MrOS | main (both visits) | ✅ | |
| `esstotalscoreqc` | APPLES | main | ✅ | Visit 3 |
| `ess_0900` | STAGES | main | ✅ | |
| `isi_score` | STAGES | main | ✅ | |
| `slisiscr` | MrOS | **visit 2 main only** | ✅ | **NOT in visit 1** |
| `slisicat` | MrOS | visit 2 | ✅ | Categorized ISI — not needed |
| `pqpsqi` | MrOS | visit 1 + visit 2 main | ✅ | Both visits have it |
| `poslpeff` | MrOS | visit 1 + visit 2 main | ✅ | Sleep efficiency % |
| `nsrr_ttleffsp_f1` | SHHS | harmonized | ✅ | Sleep efficiency % |
| `nsrr_ttleffsp_f1` | APPLES | harmonized | ✅ | **Identical values** to `sleepeffpsg` |
| `sleepeffpsg` | APPLES | main | ✅ | Same as `nsrr_ttleffsp_f1` above |
| `wasoqcpsg` | APPLES | main | ✅ | WASO — not planned for now |
| `osaseveritypostqc` | APPLES | main | ✅ | **String values** — see section 3.10 |
| `meanrt` / `medianrt` | APPLES | main | ✅ | PVT — visits 3, 5, 7 |
| `lapsesrtge500` | APPLES | main | ✅ | PVT lapses — visits 3, 5, 7 |
| `bditotalscore` | APPLES | main | ✅ | BDI-II depression |
| `phq_1000` | STAGES | main | ✅ | PHQ-9 depression |
| `gad_0800` | STAGES | main | ✅ | GAD-7 anxiety |
| `cvchd` | MrOS | main (both visits) | ✅ | CVD history |
| `any_cvd` | SHHS | cvd-summary file | ✅ | |
| `poxqual3` | MrOS | main (both visits) | ✅ | Rested morning 1–5 scale |
| `rest10` / `ms204c` | SHHS | shhs1 / shhs2 main | ✅ | Rested morning |
| `refreshinamhp` | APPLES | main | ✅ | Note: lowercase in actual CSV |

### Key APPLES notes
- Subject ID differs between files: `appleid` in main CSV, `nsrrid` in harmonized CSV. The current adapters handle this with a merge/rename step.
- `sleepeffpsg` (main) and `nsrr_ttleffsp_f1` (harmonized) are **identical** (same mean 78.37, same std 13.03, same min/max at visit 3). Use `nsrr_ttleffsp_f1` from harmonized for consistency.
- `osaseveritypostqc` is a **string** column with categories like `"0) Non-rand"`, `"1) mild"`, `"2) moderate"`, `"3) severe"`. Only available at visit 1 (baseline), not visit 3.

---

## 2. Current Master Targets State

Master parquet: `/scratch/boshra95/psg/unified/targets/master_targets.parquet`

| Field | Value |
|-------|-------|
| Shape | 19,228 rows × 19 columns |
| Datasets | SHHS: 10,115 · MrOS: 5,822 · STAGES: 1,775 · APPLES: 1,516 |

### Task subject counts (non-missing binary labels)

| Task | N labeled | Note |
|------|-----------|------|
| `apnea_binary` | 14,097 | SHHS + MrOS v1 + APPLES + STAGES |
| `sleepiness_binary` | 16,431 | All 4 datasets |
| `depression_binary` | 2,794 | APPLES + STAGES |
| `anxiety_binary` | 1,698 | STAGES only |
| `insomnia_binary` | 1,710 | STAGES only (MrOS disabled) |
| `cvd_binary` | 13,045 | SHHS + MrOS |
| `rested_morning` | 3,934 | SHHS + MrOS |
| `fatigue_binary` | 0 | Disabled |

### How multiclass → binary conversion works (in `create_master_targets.py`)

| Source column | Rule | Effective threshold |
|--------------|------|---------------------|
| `apnea_class` | class >= 2 | AHI >= 15 |
| `sleepiness_class` | class >= 1 | ESS >= 11 |
| `depression_class` | class >= 1 | BDI >= 11 (too loose — see Task 6) |

---

## 3. Per-Task Analysis and Plan

### 3.1 `apnea_binary` — **No changes needed**

- **Current threshold**: AHI >= 15 ✅ (matches guide)
- **Current AUROC**: ~0.80 (transformer, 120m)
- **MrOS visit handling**: Visit 2 rows already get `apnea_binary=-1` in master (no AHI available, code explicitly creates empty rows). Functionally correct — visit 2 MrOS is excluded from apnea training. The guide says "visit 1 only"; this is already effectively the case.
- **Action**: None.

---

### 3.2 `sleepiness_binary` — **No changes needed**

- **Current threshold**: ESS >= 11 ✅ (matches guide)
  - SHHS/MrOS/APPLES: via multiclass `class >= 1` (class 0 = ESS 0–10, class 1 = ESS 11–15)
  - STAGES: direct `threshold_override: 11.0`
- **Merge**: Yes, all 4 datasets — correct per guide
- **Current AUROC**: ~0.62 — not an implementation problem. Guide explicitly notes subjective scale noise.
- **Action**: None.

---

### 3.3 `insomnia_binary` — **Open question (see Section 4)**

- **Current state**: STAGES only, ISI >= 15, ~1710 subjects, 44.6% positive rate
- **Guide says**: Enable MrOS visit 2 (`slisiscr >= 15`), no merge
- **Problem found**: MrOS visit 2 ISI data has only **31 positive cases out of 1,022** (3.0% positive rate) at threshold >= 15. MrOS is a community cohort of older men who underreport insomnia.
- **Options**:
  - (a) Skip MrOS insomnia — 31 positives is not enough to train
  - (b) Lower threshold to ISI >= 10 for MrOS (subthreshold + moderate) — would give higher positive rate
  - (c) Keep threshold >= 15, accept extreme imbalance with heavy class weights
- **Action**: Pending answer to open question in Section 4.

---

### 3.4 `psqi_binary` — **New task, add**

- **Guide**: MrOS only, PSQI > 5
- **Data verified**:
  - MrOS visit 1: 1,281 / 2,910 positive (44.0%) — `pqpsqi`
  - MrOS visit 2: 446 / 1,023 positive (43.6%) — `pqpsqi`
  - Both visits well-balanced; use both as separate visit records
- **Implementation required**:
  1. `extract_targets_mros.py`: add PSQI extraction from `pqpsqi` (both visits)
  2. `create_master_targets.py`: add `psqi_binary` and `psqi_score` columns to schema
  3. `create_task_subject_lists.py`: generate `psqi_binary_subjects.csv`
- **No SHHS/APPLES/STAGES equivalent confirmed** — MrOS only
- **Action**: Implement in v2 pipeline (Tier 2).

---

### 3.5 `sleep_efficiency_binary` — **New task, add**

- **Guide**: SHHS + APPLES + MrOS, threshold < 85%, cautious merge
- **Data verified**:

| Dataset | Column | File | N valid | % positive (< 85%) |
|---------|--------|------|---------|---------------------|
| SHHS | `nsrr_ttleffsp_f1` | harmonized | 8,455 | 53.7% |
| APPLES | `nsrr_ttleffsp_f1` | harmonized (= `sleepeffpsg`) | 1,223 | 65.3% |
| MrOS v1 | `poslpeff` | visit 1 main | 2,911 | 74.1% |
| MrOS v2 | `poslpeff` | visit 2 main | ~1,026 | similar |

- **Note on heterogeneity**: Positive rate varies 54%–74% across datasets. This is likely real (cohort differences, scoring differences). Class weights handle per-dataset imbalance. Guide says "cautious merge — test per-dataset first then merged."
- **STAGES**: No sleep efficiency column available (confirmed in nocturn stages.yaml: "Sleep efficiency not present").
- **Implementation required**:
  1. `extract_targets_shhs.py`: add `nsrr_ttleffsp_f1 < 85` from harmonized file
  2. `extract_targets_apples.py`: add `nsrr_ttleffsp_f1 < 85` from harmonized (visit 3)
  3. `extract_targets_mros.py`: add `poslpeff < 85` (both visits)
  4. `create_master_targets.py`: add `sleep_efficiency_binary` column
  5. `create_task_subject_lists.py`: generate `sleep_efficiency_binary_subjects.csv`
- **Action**: Implement in v2 pipeline (Tier 2).

---

### 3.6 `depression_binary` → `depression_extreme_binary` — **New task variant**

- **Current state**: BDI >= 11 for APPLES (via `depression_class >= 1`), PHQ >= 10 for STAGES. Both include mild/borderline cases — noisy labels.
  - Current APPLES binary=1: 247 subjects, mean BDI = 14, min = 10, max = 29
- **Guide says**: Extreme groups only, drop middle range, no merge.
  - APPLES: BDI ≤ 9 (class 0) vs BDI >= 20 (class 1), drop BDI 10–19
  - STAGES: PHQ ≤ 4 (class 0) vs PHQ >= 15 (class 1), drop PHQ 5–14
- **Implementation**: Add as **new task name** `depression_extreme_binary`. Keep existing `depression_binary` in master_targets for backwards compatibility (existing Phase 0 results unaffected).
- **Expected impact**: Much smaller dataset (middle range dropped) but cleaner class separation. BDI >= 20 in APPLES will likely yield ~50–100 positive cases — small but unambiguous.
- **Action**: Implement in v2 pipeline (Tier 3).

---

### 3.7 `rested_morning` — **No changes needed**

- **Current threshold**: >= 4 = 1, <= 3 = 0 ✅ (matches guide)
- **Columns**: `poxqual3` (MrOS), `rest10`/`ms204c` (SHHS), `refreshinamhp` (APPLES) — no merge
- **Guide confirms**: Exploratory only; do not use as flagship metric.
- **Action**: None.

---

### 3.8 `anxiety_binary` — **No changes needed**

- **Current threshold**: GAD-7 >= 10 (STAGES only)
- **Guide**: No new columns or datasets suggested. Acknowledged as weak/noisy.
- **Action**: None.

---

### 3.9 `cvd_binary` — **No immediate changes**

- **Current state**: Merged SHHS (`any_cvd`) + MrOS (`cvchd`) — different definitions.
- **Guide says**: Don't merge. SHHS survival (Cox PH with event dates from cvd-summary) is the gold standard.
- **For now**: Keep existing `cvd_binary` unchanged. The survival task requires a different head architecture and is deferred to Phase 5.
- **Future**: Could add `cvd_shhs_binary` and `cvd_mros_binary` as separate tasks.
- **Action**: None for v2 classification pipeline.

---

### 3.10 `osaseveritypostqc` (APPLES) — **Deferred to Tier 3**

- **Column**: `osaseveritypostqc` in APPLES main file
- **Values**: String categories — `"0) Non-rand"` (414), `"1) mild"` (151), `"2) moderate"` (344), `"3) severe"` (607)
- **Available at**: Visit 1 (baseline) only — not visit 3
- **Issue**: `"0) Non-rand"` is an ambiguous category (likely subjects not randomized into CPAP arm or outside AHI criteria). Needs a decision on whether to exclude, use as class 0, or combine with mild.
- **Implementation note**: Requires string parsing (`"1) mild"` → class 1, etc.) rather than a numeric threshold.
- **Action**: Defer. Needs threshold/class definition discussion before implementation.

---

### 3.11 PVT Cognitive Tasks (APPLES) — **Deferred to Tier 4**

- **Columns**: `meanrt`, `medianrt`, `lapsesrtge500` in APPLES main
- **Available at**: Visits 3, 5, 7 (visit 3 = PSG visit, most aligned with embeddings)
- **N at visit 3**: ~1,199 subjects with PVT data
- **Formulations**:
  - Regression: `meanrt` (no threshold needed)
  - Binary: `lapsesrtge500 >= X` — no standard clinical threshold; would need data-driven cutoff (e.g., top quartile or >= 5 lapses)
- **Action**: Defer. Needs threshold definition. Implement after Tier 2 tasks validated.

---

## 4. Open Questions

### Q1 — MrOS Insomnia threshold
MrOS visit 2 has only 31 positive cases at ISI >= 15 (3%). The task is not trainable at this imbalance.

**Options**:
- (a) Skip MrOS insomnia entirely — keep STAGES only
- (b) Lower threshold for MrOS to ISI >= 10 (subthreshold + moderate insomnia)
- (c) Keep >= 15, accept imbalance (heavy class weighting)

### Q2 — `osaseveritypostqc` class definition
How to handle `"0) Non-rand"` in APPLES `osaseveritypostqc`:
- (a) Exclude "Non-rand" subjects → 3-class or binary on remaining
- (b) Map "Non-rand" as class 0 (non-OSA) → 4-class task
- (c) Binary: "severe" vs everything else

### Q3 — `sleep_efficiency_binary` merge strategy
Guide says "cautious — start per-dataset, then test merged." Given the 54%–74% positive rate range across SHHS/APPLES/MrOS, should the initial v2 experiment:
- (a) Merge all three (with class weights per dataset or globally)
- (b) Run per-dataset separately (3 separate tasks)
- (c) Start merged as baseline, then check per-dataset ablation

### Q4 — PVT lapses threshold
For `pvt_lapses_binary` (APPLES):
- (a) Data-driven median split
- (b) Fixed threshold (>= 5 lapses = impaired) — used in some literature
- (c) Regression only (`pvt_meanrt_regression`) — no binary needed

---

## 5. Implementation Order

### Tier 2 (implement first, highest signal quality)
1. `sleep_efficiency_binary` — SHHS + APPLES + MrOS
2. `psqi_binary` — MrOS only
3. `insomnia_binary` MrOS — **pending Q1**

### Tier 3 (implement after Tier 2 results)
4. `depression_extreme_binary` — APPLES (BDI ≤9 vs ≥20) + STAGES (PHQ ≤4 vs ≥15)
5. `osaseveritypostqc` — **pending Q2**

### Tier 4 (deferred)
6. PVT cognitive tasks — **pending Q4**
7. SHHS CVD survival (requires new head architecture — out of scope for Phase 0)

---

## 6. What Does NOT Need to Change

The following tasks are already correctly implemented per the design guide:
- `apnea_binary` (threshold, columns, merge logic)
- `sleepiness_binary` (threshold ESS >= 11, merge across all 4 datasets)
- `rested_morning` (threshold, per-dataset)
- `anxiety_binary` (threshold, STAGES only)
- `insomnia_binary` STAGES (threshold ISI >= 15, correct)

Poor results on sleepiness/anxiety/restedness are **not caused by implementation errors** — they reflect inherent label noise in subjective questionnaire items. The design guide confirms this.

---

## 7. Data Versioning Plan

All v2 outputs use new paths so nothing overwrites existing Phase 0 data.

| File | Current | V2 |
|------|---------|-----|
| Per-dataset CSVs | `targets/{dataset}_targets.csv` | `targets/{dataset}_targets_v2.csv` |
| Master parquet | `targets/master_targets.parquet` | `targets/master_targets_v2.parquet` |
| Task subject lists | `targets/task_subjects/` | `targets/task_subjects_v2/` |
| Training results | `results/phase0/{task}_{head}/` | `results/phase0/{task}_{head}/` (same — new tasks have new names) |

Use a git feature branch: `feature/v2-tasks`

---

*End of audit document.*
