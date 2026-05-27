# Experiment Rerun Checklist

*Written 2026-05-25. Protocol audit of all v3/v2/legacy runs.*
*Updated 2026-05-25: Archive/rename step completed — results are ready for clean reruns.*

---

## What was archived (2026-05-25)

All archiving was done in `$BASE = /scratch/boshra95/psg/unified/results/phase0_v3/`.

**Training context dirs** (renamed so training script won't skip, gen_commands won't show "already trained"):
- `bmi_binary_lstm/context_120m` → `context_120m_old_accum2`
- `psqi_binary_lstm/context_120m` → `context_120m_old_accum2`
- `psqi_binary_lstm/context_240m` → `context_240m_old_accum2`
- `sleep_efficiency_binary_lstm/context_120m` → `context_120m_old_accum4`
- `sleep_efficiency_binary_lstm/context_240m` → `context_240m_old_accum8`
- `sleep_efficiency_binary_transformer/context_80m` → `context_80m_old_accum2`
- `sleep_efficiency_binary_transformer/context_120m` → `context_120m_old_accum4`
- `sleep_efficiency_binary_transformer/context_240m` → `context_240m_old_accum8`
- `depression_extreme_binary_lstm/context_30s` → `context_30s_failed`
- `depression_extreme_binary_lstm/context_10m` → `context_10m_failed`
- `osa_severity_apples_lstm/context_10m` → `context_10m_cuda_error`

**Experiment-level `summary.csv`** → `summary_old.csv` for: bmi_binary_lstm, psqi_binary_lstm, sleep_efficiency_binary_lstm, sleep_efficiency_binary_transformer
*(After reruns, summary.csv will only accumulate the newly-trained rows. This is OK — the authoritative data for downstream analysis is per-context `metrics.json`, read by `collect_results_v2.py` directly.)*

**Inference context dirs** (same naming, in `inference/`):
- Same 8 context dirs as above renamed to `_old_accum{N}` in `inference/{exp}/`

**Inference analysis files** → `_old` suffix for: window_analysis.md, heatmap_df_test.csv, window_analysis_test.csv in affected `inference/{exp}/` dirs.

**Figures** → moved to `figures/_old_accum/` for affected experiments (bmi_binary_lstm, psqi_binary_lstm, sleep_efficiency_binary_lstm, sleep_efficiency_binary_transformer).

**Collected CSV** → `collected/` renamed to `collected_old/`. Regenerate with `gen_commands.py collect` after ALL reruns are done.

---

## Why reruns are needed

Two protocol issues exist in older runs:
1. **Grad accumulation**: `accum_steps > 1` (micro-batch 16/8/4 at long contexts) — effective batch=32 but gradient statistics differ
2. **Math attention instead of Flash**: `src_key_padding_mask=key_mask_f` was passed (now `None` when mask is all-False) — v2 logs show this; Flash attention only in v3 code

Target protocol: `batch_size=32, accum_steps=1` for ALL contexts.

---

## Group A — v3 runs with wrong accum (specific contexts only)

These experiments exist in `results/phase0_v3/` and have results, but specific contexts used old protocol. **Rerun those contexts only; good contexts are already correct.**

| Experiment | Bad contexts | Bad protocol | Old AUROC | Status | New AUROC |
|---|---|---|---|---|---|
| `bmi_binary_lstm` | 120m | micro=16, accum=2 | 0.754 | ✅ **DONE** (2026-05-25) | **0.729** |
| `psqi_binary_lstm` | 120m | micro=16, accum=2 | 0.520 | ✅ **DONE** (2026-05-25) | **0.524** |
| `psqi_binary_lstm` | 240m | micro=16, accum=2 | 0.527 | ✅ **DONE** (2026-05-25) | **0.522** |
| `sleep_efficiency_binary_lstm` | 120m | micro=8, accum=4 | 0.717 | ⬜ **TODO** | — |
| `sleep_efficiency_binary_lstm` | 240m | micro=4, accum=8 | 0.770 | ⬜ **TODO** | — |
| `sleep_efficiency_binary_transformer` | 80m | micro=16, accum=2 | 0.721 | ✅ **DONE** (2026-05-26) | **0.722** |
| `sleep_efficiency_binary_transformer` | 120m | micro=8, accum=4 | 0.747 | ✅ **DONE** (2026-05-26) | **0.747** |
| `sleep_efficiency_binary_transformer` | 240m | micro=4, accum=8 | 0.797 | ✅ **DONE** (2026-05-26) | **0.800** |

**Remaining: 2 context reruns (sleep_efficiency_binary_lstm 120m + 240m)**

### Commands

```bash
# ⬜ Still needed:
python3 scripts/gen_commands.py train sleep_efficiency_binary_lstm --context 120m 240m
```

> **Note**: `sleep_efficiency_binary_transformer_verify32` dir has a valid 240m run (accum=1) — safe to ignore once the main transformer directory is rerun (now done).

---

## Group B — v3 runs with missing contexts

These experiments have SOME contexts done correctly in v3 but are missing specific contexts.

| Experiment | Context | Status | New AUROC | Notes |
|---|---|---|---|---|
| `depression_extreme_binary_lstm` | 30s | ✅ **DONE** (2026-05-25) | **0.750** | Previously failed |
| `depression_extreme_binary_lstm` | 10m | ✅ **DONE** (2026-05-25) | **0.767** | Previously failed |
| `depression_extreme_binary_lstm` | 40m | ✅ **DONE** (2026-05-25) | **0.756** | Was never run |
| `depression_extreme_binary_lstm` | 240m | ✅ **DONE** (2026-05-25) | **0.737** | Added to registry (6 ctx total) |
| `osa_binary_apples_postqc_lstm` | 30s | ✅ **DONE** (2026-05-25) | **0.664** | Was never run |
| `osa_binary_apples_postqc_lstm` | 10m | ✅ **DONE** (2026-05-25) | **0.703** | Was never run |
| `osa_severity_apples_lstm` | all 5 | ⬜ **TODO** | — | No jobs submitted yet |

**Remaining: osa_severity_apples_lstm (all 5 contexts)**

> **Note on depression_extreme_binary_lstm**: Now all 6 contexts complete (registry updated to include 240m).
> 80m (AUROC=0.742) and 120m (AUROC=0.750) were already correct; 30s/10m/40m/240m added in this batch.
> All use batch=32, accum=1, APPLES+STAGES (n_train=5340).

### Commands

```bash
# ⬜ Still needed:
python3 scripts/gen_commands.py train osa_severity_apples_lstm
```

---

## Group C — Missing v3 runs (were in v2 with old code, or mean_pool heads never run)

All `phase0_v2` results used old training code (old mask, variable accum). V3 results dir has
nothing for these. Also includes mean_pool heads that were never run for existing Tier 1 tasks.

| Experiment | Status | Notes |
|---|---|---|
| `sex_binary_lstm` | ✅ **DONE** 6/6 (2026-05-26) | All batch=32, accum=1 ✅ |
| `sex_binary_transformer` | ✅ **DONE** 6/6 (2026-05-26) | All batch=32, accum=1 ✅ |
| `sex_binary_mean_pool` | ⬜ **TODO** | Never run |
| `age_class_lstm` | ✅ **DONE** 6/6 (2026-05-26) | All batch=32, accum=1 ✅ |
| `age_class_transformer` | 🔄 **RUNNING** 5/6 (job 41521400) | 240m still training; epoch ~28/40, best val AUROC=0.895; auto-requeue from job 41470727 |
| `age_class_mean_pool` | ⬜ **TODO** | Never run |
| `bmi_binary_mean_pool` | ⬜ **TODO** | Never run |
| `sleep_efficiency_binary_mean_pool` | ⬜ **TODO** | Never run |

**Remaining: 4 experiments pending + 1 currently training**

### Commands

```bash
# ⬜ Still needed:
python3 scripts/gen_commands.py train sex_binary_mean_pool
python3 scripts/gen_commands.py train age_class_mean_pool         # after job 41521400 finishes
python3 scripts/gen_commands.py train bmi_binary_mean_pool
python3 scripts/gen_commands.py train sleep_efficiency_binary_mean_pool
```

---

## Group D — Legacy tasks, fresh v3 runs (in registry, pending)

These tasks existed in phase0 with old non-overlapping-window protocol, or were newly added.
All are registered and pending — run fresh in v3.

### Tier 1 legacy (high priority)

| Experiment | N-ctx | Datasets | Notes |
|---|---|---|---|
| `apnea_binary_lstm` | 6 | apples,shhs,mros,stages | Tier 1 anchor; all 3 heads planned |
| `apnea_binary_transformer` | 6 | apples,shhs,mros,stages | |
| `apnea_binary_mean_pool` | 6 | apples,shhs,mros,stages | |
| `sleep_staging_lstm` | 6 | shhs,mros,stages,apples | seq2seq; metric = Cohen's κ + per-stage F1 |
| `sleep_staging_transformer` | 6 | shhs,mros,stages,apples | |
| `sleep_staging_mean_pool` | 6 | shhs,mros,stages,apples | MeanPool loses positional info → expected lower κ |

### Tier 2 legacy

| Experiment | N-ctx | Datasets | Notes |
|---|---|---|---|
| `cvd_binary_lstm` | 6 | shhs,mros | SHHS=any_CVD composite, MrOS=CHD only — note definition mismatch in paper |
| `sleepiness_binary_lstm` | 6 | apples,shhs,mros,stages | Phase0 AUROC 0.59–0.61; borderline useful |

**Total: 48 context runs (8 experiments × 6 contexts)**

### Commands

```bash
# Tier 1 legacy — submit first
python3 scripts/gen_commands.py train apnea_binary_lstm
python3 scripts/gen_commands.py train apnea_binary_transformer
python3 scripts/gen_commands.py train apnea_binary_mean_pool
python3 scripts/gen_commands.py train sleep_staging_lstm
python3 scripts/gen_commands.py train sleep_staging_transformer
python3 scripts/gen_commands.py train sleep_staging_mean_pool

# Tier 2
python3 scripts/gen_commands.py train cvd_binary_lstm
python3 scripts/gen_commands.py train sleepiness_binary_lstm
```

---

## Group E — Deferred (submit last, only if time permits)

| Experiment | N-ctx | Datasets | Phase0 AUROC | Notes |
|---|---|---|---|---|
| `insomnia_binary_lstm` | 5 | stages | 0.56–0.60 | Borderline signal |
| `rested_morning_lstm` | 5 | mros | ~0.54 | Near chance; included for completeness only |
| `anxiety_binary_lstm` | 5 | stages | 0.56–0.58 | Near chance |

**Total: 15 context runs**

### Commands

```bash
python3 scripts/gen_commands.py train insomnia_binary_lstm
python3 scripts/gen_commands.py train rested_morning_lstm
python3 scripts/gen_commands.py train anxiety_binary_lstm
```

---

## Summary counts

*Updated 2026-05-26 to reflect completed runs.*

| Group | Description | Total | Done | Remaining |
|---|---|---|---|---|
| A | v3 wrong protocol, partial rerun | 8 | ✅ 6 | ⬜ 2 (sleep_eff_lstm 120m+240m) |
| B | v3 missing contexts | 10 | ✅ 6 | ⬜ 1 exp (osa_severity all 5 ctx) |
| C | v2→v3 reruns + missing mean_pool heads | 48 | ✅ 24 + 🔄 1 | ⬜ 23 (4 mean_pools + age_class_transformer 240m finishing) |
| D | Legacy Tier 1+2, fresh v3 runs | 48 | ⬜ 0 | ⬜ 48 |
| **Priority subtotal** | | **114** | **~37** | **~77** |
| E | Deferred | 15 | ⬜ 0 | ⬜ 15 |
| **Grand total** | | **129** | | |

---

## What to submit next (as of 2026-05-26)

**Immediate (1–2 jobs):**
1. `sleep_efficiency_binary_lstm --context 120m 240m` — last 2 Group A contexts
2. `osa_severity_apples_lstm` — 5 jobs, completes Group B entirely

**After those finish (Group C remainder):**
3. `sex_binary_mean_pool` / `age_class_mean_pool` / `bmi_binary_mean_pool` / `sleep_efficiency_binary_mean_pool`
   — 4 × 6 = 24 jobs; wait for job 41521400 (age_class_transformer 240m) to finish first

**Then Group D (major batch):**
4. apnea_binary (3 heads × 6 ctx = 18 jobs) — Tier 1, high priority
5. sleep_staging (3 heads × 6 ctx = 18 jobs) — Tier 1, seq2seq
6. cvd_binary_lstm + sleepiness_binary_lstm (12 jobs) — Tier 2

**Deferred (Group E):** insomnia, rested_morning, anxiety — only if time permits.

---

## After training: inference and collection

After each group's training is complete, run inference to generate parquets:

```bash
python3 scripts/gen_commands.py infer <exp_id>
```

Then run analysis per experiment:

```bash
python3 scripts/gen_commands.py analyze <exp_id>
```

**After ALL reruns and inferences are complete**, regenerate the collected CSV from scratch:

```bash
# collected_old/ is the archived version; a fresh collected/ will be created
python3 scripts/gen_commands.py collect
```

### Note on experiment-level summary.csv

The training script (`train_context_sweep.py`) appends one row to `summary.csv` per trained
context. After partial reruns (Groups A/B), the new `summary.csv` will only contain the
newly-trained context rows (good contexts are skipped and not re-appended).

**This does not affect downstream analysis**, because `collect_results_v2.py` reads from
per-context `metrics.json` files directly — not from `summary.csv`. The collected CSV and
all figures will be correct.

If you need a complete `summary.csv` for a quick sanity check, run:

```bash
python3 -c "
import json, csv, pathlib
exp_dir = pathlib.Path('/scratch/boshra95/psg/unified/results/phase0_v3/<exp_id>')
rows = []
for ctx_dir in sorted(exp_dir.glob('context_[0-9]*[ms]')):
    mf = ctx_dir / 'metrics.json'
    if mf.exists():
        with open(mf) as f: m = json.load(f)
        row = {k: v for k, v in m.items() if not isinstance(v, dict)}
        for split in ('val', 'test'):
            for k, v in m.get(split, {}).items():
                row[f'{split}_{k}'] = v
        rows.append(row)
if rows:
    with open(exp_dir / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'Written {len(rows)} rows')
"
```
Replace `<exp_id>` with the experiment name.

---

## Threshold tuning (post-all-inference)

After all inference runs are done, apply post-hoc threshold tuning for binary tasks.
See `docs/POSTHOC_THRESHOLD_TUNING.md` for the implementation plan.
Priority tasks: `bmi_binary` (+0.015 confirmed), `osa_binary_apples_postqc` (est. +0.06–0.09).
Requires saving `val_windows.parquet` during inference — **not yet implemented**.
