# Experiment Rerun Checklist

*Created 2026-05-25. Last updated 2026-05-27.*

---

## ⚠️ 2026-06-27 update: v3_abl (modality ablation) also needs a full rerun

Not part of this doc's original scope (predates the ablation work), flagging here too since this
is the first place anyone would check for "what needs rerunning." All 25 `phase0_v3_abl`
experiments were trained with the wrong architecture (`hidden_dim: 256, num_layers: 2` — the
sleep-staging arch — instead of `128, 1`, the seq2label arch used everywhere else). Nothing here
needs touching; see `docs/REMAINING_TRAINING_CHECKLIST.md` § Later: re-running v3_abl analysis
after the architecture fix for the full archive-then-rerun pipeline, and
`docs/SOTA_COMPARISON_AND_ABLATIONS.md` §A.7 item 7 for the bug writeup.

---

## What was archived (2026-05-25)

Directories renamed in `$BASE = /scratch/boshra95/psg/unified/results/phase0_v3/` so that
gen_commands no longer marks old-protocol contexts as trained and training script can cleanly
overwrite them. Old results preserved under `_old_accum{N}` / `_failed` / `_cuda_error` suffixes.
See archive details at the bottom of this file.

---

## Training status — all experiments

*Updated 2026-05-27. Running jobs counted as done.*

### Group A — v3 wrong-protocol reruns (specific contexts)

| Experiment | Context | Old AUROC | Status | New AUROC |
|---|---|---|---|---|
| `bmi_binary_lstm` | 120m | 0.754 | ✅ DONE | **0.729** |
| `psqi_binary_lstm` | 120m | 0.520 | ✅ DONE | **0.524** |
| `psqi_binary_lstm` | 240m | 0.527 | ✅ DONE | **0.522** |
| `sleep_efficiency_binary_lstm` | 120m | 0.717 | ✅ DONE | — |
| `sleep_efficiency_binary_lstm` | 240m | 0.770 | ✅ DONE | — |
| `sleep_efficiency_binary_transformer` | 80m | 0.721 | ✅ DONE | **0.722** |
| `sleep_efficiency_binary_transformer` | 120m | 0.747 | ✅ DONE | **0.747** |
| `sleep_efficiency_binary_transformer` | 240m | 0.797 | ✅ DONE | **0.800** |

**Group A: 8/8 complete ✅**

### Group B — v3 missing contexts

| Experiment | Context | Status | New AUROC | Notes |
|---|---|---|---|---|
| `depression_extreme_binary_lstm` | 30s | ✅ DONE | **0.750** | Previously failed |
| `depression_extreme_binary_lstm` | 10m | ✅ DONE | **0.767** | Previously failed |
| `depression_extreme_binary_lstm` | 40m | ✅ DONE | **0.756** | Was never run |
| `depression_extreme_binary_lstm` | 240m | ✅ DONE | **0.737** | Registry now has 6 ctx |
| `osa_binary_apples_postqc_lstm` | 30s | ✅ DONE | **0.664** | Was never run |
| `osa_binary_apples_postqc_lstm` | 10m | ✅ DONE | **0.703** | Was never run |
| `osa_severity_apples_lstm` | all 5 | ⬜ TODO | — | Not yet submitted |

> `depression_extreme_binary_lstm`: all 6 contexts now done (80m=0.742, 120m=0.750 were
> already correct). All use batch=32, accum=1, APPLES+STAGES (n_train=5340).

**Group B: 6/7 items complete. Remaining: `osa_severity_apples_lstm` (5 contexts)**

```bash
python3 scripts/gen_commands.py train osa_severity_apples_lstm
```

### Group C — v2→v3 reruns + missing mean_pool heads

| Experiment | Status | Notes |
|---|---|---|
| `sex_binary_lstm` | ✅ DONE 6/6 | batch=32, accum=1 ✅ |
| `sex_binary_transformer` | ✅ DONE 6/6 | batch=32, accum=1 ✅ |
| `sex_binary_mean_pool` | ⬜ TODO | |
| `age_class_lstm` | ✅ DONE 6/6 | batch=32, accum=1 ✅ |
| `age_class_transformer` | ✅ DONE 6/6 | batch=32, accum=1 ✅ |
| `age_class_mean_pool` | ⬜ TODO | |
| `bmi_binary_mean_pool` | ⬜ TODO | |
| `sleep_efficiency_binary_mean_pool` | ⬜ TODO | |

**Group C: 4/8 complete. Remaining: 4 mean_pool experiments (24 contexts)**

```bash
python3 scripts/gen_commands.py train sex_binary_mean_pool
python3 scripts/gen_commands.py train age_class_mean_pool
python3 scripts/gen_commands.py train bmi_binary_mean_pool
python3 scripts/gen_commands.py train sleep_efficiency_binary_mean_pool
```

### Group D — Legacy Tier 1+2, fresh v3 runs

| Experiment | Tier | Status | Notes |
|---|---|---|---|
| `apnea_binary_lstm` | 1 | ✅ DONE 6/6 | batch=32, accum=1 ✅ |
| `apnea_binary_transformer` | 1 | ✅ DONE 6/6 | batch=32, accum=1 ✅ |
| `apnea_binary_mean_pool` | 1 | ⬜ TODO | |
| `sleep_staging_lstm` | 1 | 🔄 RUNNING 6/6 | Jobs 41660569/41663169/41663173/41663174/41663175/41663176 |
| `sleep_staging_transformer` | 1 | ⬜ TODO | |
| `sleep_staging_mean_pool` | 1 | ⬜ TODO | |
| `cvd_binary_lstm` | 2 | ✅ DONE 6/6 | batch=32, accum=1 ✅ |
| `cvd_binary_transformer` | 2 | ✅ DONE 6/6 | batch=32, accum=1 ✅ (added to registry) |
| `sleepiness_binary_lstm` | 2 | 🔄 RUNNING 6/6 | Jobs 41660528/29/30, 41663275/76/77 |

**Group D: 4/9 complete, 2 currently running. Remaining: 3 (mean_pool + staging transformer)**

```bash
# After sleep_staging_lstm finishes:
python3 scripts/gen_commands.py train sleep_staging_transformer
python3 scripts/gen_commands.py train sleep_staging_mean_pool
python3 scripts/gen_commands.py train apnea_binary_mean_pool
```

### Group E — Deferred

| Experiment | Status | Notes |
|---|---|---|
| `insomnia_binary_lstm` | ⬜ TODO | Phase0 AUROC 0.56–0.60, borderline |
| `rested_morning_lstm` | ⬜ TODO | Phase0 AUROC ~0.54, near chance |
| `anxiety_binary_lstm` | ⬜ TODO | Phase0 AUROC 0.56–0.58, near chance |

---

## Training summary (as of 2026-05-27)

| Group | Description | Total ctx | Done | Remaining |
|---|---|---|---|---|
| A | v3 wrong-protocol reruns | 8 | ✅ 8 | — |
| B | v3 missing contexts | 10 | ✅ 6 | ⬜ 5 (osa_severity) |
| C | v2→v3 + missing mean_pools | 48 | ✅ 24 | ⬜ 24 |
| D | Legacy Tier 1+2 | 54 | ✅ 24 + 🔄 12 | ⬜ 18 |
| **Priority subtotal** | | **120** | **~64** | **~47** |
| E | Deferred | 15 | — | ⬜ 15 |

---

## Inference & Analysis Status

### Already analyzed (v3, clean)

| Experiment | Inference | Analysis | Notes |
|---|---|---|---|
| `bmi_binary_transformer` | ✅ 6/6 | ✅ analyzed | Only experiment fully analyzed end-to-end in v3 |

### Have partial inference — need top-up only (script auto-skips done contexts)

The inference script skips contexts where `inference/{exp}/context_{ctx}/test_windows.parquet`
already exists. For these experiments, run `gen_commands.py infer` once and the new rerun
contexts will be filled in automatically.

| Experiment | Already inferred | Missing (new reruns) |
|---|---|---|
| `bmi_binary_lstm` | 30s, 10m, 40m, 80m, 240m | 120m |
| `psqi_binary_lstm` | 30s, 10m, 40m, 80m | 120m, 240m |
| `sleep_efficiency_binary_lstm` | 30s, 10m, 40m, 80m | 120m, 240m |
| `sleep_efficiency_binary_transformer` | 30s, 10m, 40m | 80m, 120m, 240m |

### Need full inference (no parquets at all)

| Experiment | Contexts | Datasets |
|---|---|---|
| `sex_binary_lstm` | 6 | apples, shhs |
| `sex_binary_transformer` | 6 | apples, shhs |
| `age_class_lstm` | 6 | apples, shhs, mros |
| `age_class_transformer` | 6 | apples, shhs, mros |
| `depression_extreme_binary_lstm` | 6 | apples, stages |
| `osa_binary_apples_postqc_lstm` | 5 | apples |
| `apnea_binary_lstm` | 6 | apples, shhs, mros, stages |
| `apnea_binary_transformer` | 6 | apples, shhs, mros, stages |
| `cvd_binary_lstm` | 6 | shhs, mros |
| `cvd_binary_transformer` | 6 | shhs, mros |
| `sleep_staging_lstm` | 6 | shhs, mros, stages, apples — after training finishes |
| `sleepiness_binary_lstm` | 6 | apples, shhs, mros, stages — after training finishes |

### Inference commands — submit all at once

```bash
# Top-up inference (fills missing contexts, skips already-done):
python3 scripts/gen_commands.py infer bmi_binary_lstm
python3 scripts/gen_commands.py infer psqi_binary_lstm
python3 scripts/gen_commands.py infer sleep_efficiency_binary_lstm
python3 scripts/gen_commands.py infer sleep_efficiency_binary_transformer

# Full inference (no parquets yet):
python3 scripts/gen_commands.py infer sex_binary_lstm
python3 scripts/gen_commands.py infer sex_binary_transformer
python3 scripts/gen_commands.py infer age_class_lstm
python3 scripts/gen_commands.py infer age_class_transformer
python3 scripts/gen_commands.py infer depression_extreme_binary_lstm
python3 scripts/gen_commands.py infer osa_binary_apples_postqc_lstm
python3 scripts/gen_commands.py infer apnea_binary_lstm
python3 scripts/gen_commands.py infer apnea_binary_transformer
python3 scripts/gen_commands.py infer cvd_binary_lstm
python3 scripts/gen_commands.py infer cvd_binary_transformer

# After sleep_staging_lstm and sleepiness_binary_lstm training jobs finish:
python3 scripts/gen_commands.py infer sleep_staging_lstm
python3 scripts/gen_commands.py infer sleepiness_binary_lstm
```

---

## Post-inference: Analysis & Plotting pipeline

Run after ALL inference parquets are in place. Steps must be done in order.

### Step 1 — Window analysis per experiment

Generates `window_analysis.md`, `window_analysis_test.csv`, `heatmap_df_test.csv` per experiment.

```bash
# Run for each experiment that has full inference:
for exp in bmi_binary_lstm psqi_binary_lstm \
           sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer \
           sex_binary_lstm sex_binary_transformer \
           age_class_lstm age_class_transformer \
           depression_extreme_binary_lstm osa_binary_apples_postqc_lstm \
           apnea_binary_lstm apnea_binary_transformer \
           cvd_binary_lstm cvd_binary_transformer \
           sleep_staging_lstm sleepiness_binary_lstm; do
  python3 scripts/gen_commands.py analyze $exp
done
```

> `bmi_binary_transformer` is already analyzed — skip or re-run to refresh with any new collect.

### Step 2 — Collect all results into CSVs

Reads per-context `metrics.json` and `window_analysis_{split}.csv` from all experiments and
produces `results/collected/{channel}/analysis.csv` and `training.csv`. These are the inputs
for all downstream plots.

```bash
# Fast-channel → results/collected/phase0_v3/
python3 scripts/collect_results_v2.py --force

# Full-channel → results/collected/phase0_v3_full/
python3 scripts/collect_results_v2.py \
  --results-dir /scratch/boshra95/psg_full/unified/results/phase0_v3_full \
  --force
```

Use `--force` any time you've re-run `analyze` (e.g. after adding bootstrap CIs) so the
updated values overwrite the stale keys in the CSV.

> `collected_old/` is an archived stale version from before earlier reruns — safe to delete
> once the channel-specific directories are generated and verified.

### Step 3 — Iso-compute plots (context-length saturation)

```bash
python3 scripts/gen_commands.py scaling-laws     # §1 saturation curves per task
python3 scripts/gen_commands.py saturation       # head comparison (lstm/transformer/mean_pool)
python3 scripts/gen_commands.py iso-plots        # 7 iso-compute figures
```

### Step 4 — Per-experiment analysis plots

```bash
python3 scripts/gen_commands.py build-heatmap <exp_id>   # context × K heatmap
python3 scripts/gen_commands.py calibration              # reliability + ECE (§2)
python3 scripts/gen_commands.py window-position          # position profiles (§4)
python3 scripts/gen_commands.py subject-consistency      # variance + hard subjects (§5)
python3 scripts/gen_commands.py task-comparison          # sensitivity across tasks (§6)
python3 scripts/gen_commands.py cohort-saturation        # per-dataset breakdown
python3 scripts/gen_commands.py precision-recall         # PR curves (imbalanced tasks)
python3 scripts/gen_commands.py subject-kstar            # min windows to correct prediction
```

### Step 5 — Post-hoc threshold tuning (imbalanced tasks only) ✅ COMPLETED 2026-05-30

Val inference and threshold tuning run for all 14 binary experiments.
Results in `inference/{exp_id}/threshold_tuning.csv`.

**Use tuned BA for paper:** `osa_binary_apples_postqc_lstm` (best +0.22!),
`depression_extreme_binary_lstm` (+0.065 at 80m — surprise), `bmi_binary_transformer` (+0.013 avg),
`bmi_binary_lstm` (+0.006 avg), `sex_binary_lstm` (+0.009 avg), `sleepiness_binary_*` (+0.006 avg).

**Keep t=0.5:** `cvd_binary_*` (tuning hurts — val set too small), `apnea_binary_*`,
`sleep_efficiency_binary_*`, `sex_binary_transformer` (all near zero or negative).

See `docs/POSTHOC_THRESHOLD_TUNING.md` for full numbers, surprises, and final paper wording.

---

## Remaining training to-do list (2026-05-27)

```
IMMEDIATE:
  [ ] osa_severity_apples_lstm             (Group B, 5 ctx)
  [ ] sex_binary_mean_pool                 (Group C, 6 ctx)
  [ ] age_class_mean_pool                  (Group C, 6 ctx)
  [ ] bmi_binary_mean_pool                 (Group C, 6 ctx)
  [ ] sleep_efficiency_binary_mean_pool    (Group C, 6 ctx)
  [ ] apnea_binary_mean_pool               (Group D Tier 1, 6 ctx)
  [ ] sleep_staging_transformer            (Group D Tier 1, 6 ctx — after lstm finishes)
  [ ] sleep_staging_mean_pool              (Group D Tier 1, 6 ctx)

CURRENTLY RUNNING:
  [~] sleep_staging_lstm                   (Group D Tier 1, 6/6 running)
  [~] sleepiness_binary_lstm               (Group D Tier 2, 6/6 running)

DEFERRED (only if time permits):
  [ ] insomnia_binary_lstm                 (5 ctx)
  [ ] rested_morning_lstm                  (5 ctx)
  [ ] anxiety_binary_lstm                  (5 ctx)
```

---

## Archive notes (2026-05-25)

Renamed in `results/phase0_v3/` (training) and `results/phase0_v3/inference/`:
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

Also archived: `summary.csv` → `summary_old.csv` for 4 affected experiments;
inference analysis files → `_old` suffix; figures → `figures/_old_accum/`;
`collected/` → `collected_old/`.

**Note on experiment-level `summary.csv`:** The training script appends one row per trained
context. After partial reruns, `summary.csv` only contains the new rows. This is fine —
`collect_results_v2.py` reads per-context `metrics.json` directly, not `summary.csv`.
To rebuild a complete `summary.csv` manually:

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
