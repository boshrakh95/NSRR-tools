# Remaining Training Checklist — Transformer + Mean-Pool Backfill

*Created 2026-06-22.*

---

## Context: kept tasks for the paper

Only **8 tasks** are kept for the paper (the rest had poor/near-chance phase0 AUROC and are
dropped):

```
sex_binary, age_class, bmi_binary, sleep_efficiency_binary, apnea_binary,
depression_extreme_binary, osa_binary_apples_postqc, sleep_staging
```

**`sleep_staging` is excluded from this checklist** — it has its own arch/config (256/2,
`val_kappa` monitor) and incomplete training state; handled separately.

This checklist covers the **remaining 7 seq2label tasks**, both **fast-channel (v3)** and
**full-channel (v3_full)**, for the heads that haven't been fully run yet:

- **Transformer** — only missing for `depression_extreme_binary` and `osa_binary_apples_postqc`
  (you already have transformer for sex/age/bmi/sleep_efficiency/apnea).
- **Mean-pool** — missing for **all 7** tasks, both channels.

---

## What was modified in the registries (done already)

`experiments/v2_registry.yaml` (fast) and `experiments/v2_full_registry.yaml` (full) **did not
already have** `_transformer`/`_mean_pool` entries for `depression_extreme_binary` and
`osa_binary_apples_postqc` (only `_lstm` existed for those two — they were originally Tier-2
"lstm only" tasks). Added 4 new entries to **each** registry (8 total):

```
depression_extreme_binary_transformer
depression_extreme_binary_mean_pool
osa_binary_apples_postqc_transformer
osa_binary_apples_postqc_mean_pool
```

All new entries mirror the existing `_lstm` entry for the same task (same `datasets`,
`contexts: [30s, 10m, 40m, 80m, 120m, 240m]`, `lr: 1.0e-4`, `tier: 2`, `n_size: small`) — only
`head:` differs.

**Mean-pool entries for the other 5 tasks (sex/age/bmi/sleep_efficiency/apnea) already existed**
in both registries (added previously, never trained) — no registry change was needed for those.

**No other file needs to change.** This is the answer to "what do I need to modify per round":
switching between fast and full channel is **purely the `--registry` flag** to
`gen_commands.py`. The registry file itself embeds the correct `config:`, `results_dir:`,
`inference_dir:`, and `logs_dir:` for its channel — nothing in `jobs/train_context_sweep_gpu.sh`,
`configs/phase0_v3_config.yaml`, or `configs/phase0_v3_full_config.yaml` needs editing for a
normal (non-ablation) training run. (The one place that *did* need a real fix was
`configs/phase0_v3_abl_config.yaml`'s model architecture — that's a separate, already-identified
issue, out of scope here.)

```bash
REG_FAST="experiments/v2_registry.yaml"        # default — no --registry flag needed
REG_FULL="experiments/v2_full_registry.yaml"   # pass --registry experiments/v2_full_registry.yaml
```

---

## Job count

| Channel | Head | Tasks | Contexts each | Jobs |
|---|---|---|---|---|
| Fast (v3) | transformer | depression_extreme_binary, osa_binary_apples_postqc | 6 | 12 |
| Full (v3_full) | transformer | depression_extreme_binary, osa_binary_apples_postqc | 6 | 12 |
| Fast (v3) | mean_pool | sex, age, bmi, sleep_efficiency, apnea, depression, osa (7 tasks) | 6 | 42 |
| Full (v3_full) | mean_pool | same 7 tasks | 6 | 42 |
| **Total** | | | | **108 training jobs** |

Each "job" here is one SLURM submission per (task, head, context) — `gen_commands.py train
<exp_id>` submits all 6 contexts for one exp_id at once.

---

## Step 1 — Training

Working directory: `cd /home/boshra95/NSRR-tools`

### 1a. Transformer — fast channel (depression, osa)

```bash
python scripts/gen_commands.py train depression_extreme_binary_transformer | bash
python scripts/gen_commands.py train osa_binary_apples_postqc_transformer  | bash
```

### 1b. Transformer — full channel (depression, osa)

```bash
REG="--registry experiments/v2_full_registry.yaml"

python scripts/gen_commands.py $REG train depression_extreme_binary_transformer | bash
python scripts/gen_commands.py $REG train osa_binary_apples_postqc_transformer  | bash
```

### 1c. Mean-pool — fast channel (all 7 tasks)

```bash
python scripts/gen_commands.py train sex_binary_mean_pool                | bash
python scripts/gen_commands.py train age_class_mean_pool                 | bash
python scripts/gen_commands.py train bmi_binary_mean_pool                | bash
python scripts/gen_commands.py train sleep_efficiency_binary_mean_pool   | bash
python scripts/gen_commands.py train apnea_binary_mean_pool              | bash
python scripts/gen_commands.py train depression_extreme_binary_mean_pool | bash
python scripts/gen_commands.py train osa_binary_apples_postqc_mean_pool  | bash
```

### 1d. Mean-pool — full channel (all 7 tasks)

```bash
REG="--registry experiments/v2_full_registry.yaml"

python scripts/gen_commands.py $REG train sex_binary_mean_pool                | bash
python scripts/gen_commands.py $REG train age_class_mean_pool                 | bash
python scripts/gen_commands.py $REG train bmi_binary_mean_pool                | bash
python scripts/gen_commands.py $REG train sleep_efficiency_binary_mean_pool   | bash
python scripts/gen_commands.py $REG train apnea_binary_mean_pool              | bash
python scripts/gen_commands.py $REG train depression_extreme_binary_mean_pool | bash
python scripts/gen_commands.py $REG train osa_binary_apples_postqc_mean_pool  | bash
```

All 108 jobs are independent — safe to submit all four blocks at once.

**Check status at any time:**
```bash
python scripts/gen_commands.py status                                          # fast
python scripts/gen_commands.py --registry experiments/v2_full_registry.yaml status  # full
```

**Check job history (after submitting):**
```bash
python scripts/gen_commands.py runs depression_extreme_binary_transformer
python scripts/gen_commands.py --registry experiments/v2_full_registry.yaml runs sex_binary_mean_pool
```

---

## Step 2 — Inference (after each training block finishes)

One GPU job per experiment; auto-discovers trained contexts.

### Fast channel
```bash
python scripts/gen_commands.py infer depression_extreme_binary_transformer | bash
python scripts/gen_commands.py infer osa_binary_apples_postqc_transformer  | bash
python scripts/gen_commands.py infer sex_binary_mean_pool                | bash
python scripts/gen_commands.py infer age_class_mean_pool                 | bash
python scripts/gen_commands.py infer bmi_binary_mean_pool                | bash
python scripts/gen_commands.py infer sleep_efficiency_binary_mean_pool   | bash
python scripts/gen_commands.py infer apnea_binary_mean_pool              | bash
python scripts/gen_commands.py infer depression_extreme_binary_mean_pool | bash
python scripts/gen_commands.py infer osa_binary_apples_postqc_mean_pool  | bash
```

### Full channel
```bash
REG="--registry experiments/v2_full_registry.yaml"

python scripts/gen_commands.py $REG infer depression_extreme_binary_transformer | bash
python scripts/gen_commands.py $REG infer osa_binary_apples_postqc_transformer  | bash
python scripts/gen_commands.py $REG infer sex_binary_mean_pool                | bash
python scripts/gen_commands.py $REG infer age_class_mean_pool                 | bash
python scripts/gen_commands.py $REG infer bmi_binary_mean_pool                | bash
python scripts/gen_commands.py $REG infer sleep_efficiency_binary_mean_pool   | bash
python scripts/gen_commands.py $REG infer apnea_binary_mean_pool              | bash
python scripts/gen_commands.py $REG infer depression_extreme_binary_mean_pool | bash
python scripts/gen_commands.py $REG infer osa_binary_apples_postqc_mean_pool  | bash
```

---

## Step 3 — Window analysis (per experiment, local, no GPU)

This is also where the **repo-mirroring** added recently kicks in automatically:
`analyze_windows.py` is invoked by `gen_commands.py` with `--repo-out` already pointing at
`results/inference/{phase0_v3|phase0_v3_full}/{exp_id}/`, so `window_analysis.md` and
`window_analysis_{split}.csv` land in both scratch and the git repo with no extra flags.

```bash
source /home/boshra95/sleepfm_env/bin/activate
```

### Fast channel
```bash
for exp in depression_extreme_binary_transformer osa_binary_apples_postqc_transformer \
           sex_binary_mean_pool age_class_mean_pool bmi_binary_mean_pool \
           sleep_efficiency_binary_mean_pool apnea_binary_mean_pool \
           depression_extreme_binary_mean_pool osa_binary_apples_postqc_mean_pool; do
    python scripts/gen_commands.py analyze $exp --plot | bash
done
```

### Full channel
```bash
REG="--registry experiments/v2_full_registry.yaml"

for exp in depression_extreme_binary_transformer osa_binary_apples_postqc_transformer \
           sex_binary_mean_pool age_class_mean_pool bmi_binary_mean_pool \
           sleep_efficiency_binary_mean_pool apnea_binary_mean_pool \
           depression_extreme_binary_mean_pool osa_binary_apples_postqc_mean_pool; do
    python scripts/gen_commands.py $REG analyze $exp --plot | bash
done
```

> `--plot` also auto-mirrors PNGs to `results/figures/{round}/{exp_id}/` in the repo (see
> `repo_sync.py`). PDFs stay scratch-only (gitignored under `results/figures/`).

---

## Step 4 — Collect into CSVs (repo + scratch)

```bash
python scripts/gen_commands.py collect | bash
python scripts/gen_commands.py --registry experiments/v2_full_registry.yaml collect | bash
```

Writes/updates:
```
results/collected/phase0_v3/{training,analysis}.csv
results/collected/phase0_v3_full/{training,analysis}.csv
```
(both scratch and the git-tracked repo copy, automatically).

---

## Step 5 — Saturation curves & head comparison (now meaningful with mean_pool added)

Once mean_pool is trained for a task, the 3-head saturation comparison (LSTM vs Transformer vs
MeanPool — your H4 hypothesis) becomes complete for that task.

```bash
# Fast channel
python scripts/gen_commands.py saturation sex_binary --heads lstm transformer mean_pool | bash
python scripts/gen_commands.py saturation age_class --heads lstm transformer mean_pool | bash
python scripts/gen_commands.py saturation bmi_binary --heads lstm transformer mean_pool | bash
python scripts/gen_commands.py saturation sleep_efficiency_binary --heads lstm transformer mean_pool | bash
python scripts/gen_commands.py saturation apnea_binary --heads lstm transformer mean_pool | bash
python scripts/gen_commands.py saturation depression_extreme_binary --heads lstm transformer mean_pool | bash
python scripts/gen_commands.py saturation osa_binary_apples_postqc --heads lstm transformer mean_pool | bash

# Full channel
REG="--registry experiments/v2_full_registry.yaml"
python scripts/gen_commands.py $REG saturation sex_binary --heads lstm transformer mean_pool | bash
python scripts/gen_commands.py $REG saturation age_class --heads lstm transformer mean_pool | bash
python scripts/gen_commands.py $REG saturation bmi_binary --heads lstm transformer mean_pool | bash
python scripts/gen_commands.py $REG saturation sleep_efficiency_binary --heads lstm transformer mean_pool | bash
python scripts/gen_commands.py $REG saturation apnea_binary --heads lstm transformer mean_pool | bash
python scripts/gen_commands.py $REG saturation depression_extreme_binary --heads lstm transformer mean_pool | bash
python scripts/gen_commands.py $REG saturation osa_binary_apples_postqc --heads lstm transformer mean_pool | bash
```

Figures land in `results/figures/{phase0_v3|phase0_v3_full}/saturation/` (repo) and the matching
scratch `figures/saturation/` dir.

---

## Step 6 — Paper tables (Tables 1, 2, 4, 5, 10)

These were previously only generated for fast channel (`*_fast.csv`). Once all 7 tasks have all
3 heads in both channels, regenerate with the correct `--channel` tag:

```bash
# Fast (refresh — now includes mean_pool for all 7 tasks)
python scripts/make_table1_peak_auroc.py --channel fast
python scripts/make_table5_heads.py --channel fast --heads lstm transformer mean_pool

# Full (never generated before — backfill)
python scripts/make_table1_peak_auroc.py --collected-dir results/collected/phase0_v3_full --channel full
python scripts/make_table5_heads.py --collected-dir results/collected/phase0_v3_full --channel full --heads lstm transformer mean_pool
```

(Tables 2/4/10 follow the same `--collected-dir`/`--channel` pattern — see
`docs/EXPERIMENTS_GUIDE.md` § Experiment Registry and Command Generator for the full flag list.)

---

## Notes

- **Why depression/osa were "lstm only" until now:** they were originally Tier-2 small-N tasks,
  run with LSTM first to validate the pipeline before committing GPU time to all 3 heads.
- **batch_size in mean_pool entries:** set to 32 to match the lstm/transformer entries for the
  same task, but this is informational only — `gen_commands.py` always uses
  `gradient_accumulation.context_micro_batch[context]` (= 32 at every context, for every
  registry) unless an experiment opts into `batch_mode: memory_bounded`. None of the new entries
  do, so the effective batch is always 32 regardless of what's written here.
- **Architecture consistency:** all new entries use the registry's default `model:` section from
  `configs/phase0_v3_config.yaml` / `configs/phase0_v3_full_config.yaml` (hidden_dim=128,
  num_layers=1) — the correct, matching architecture for seq2label tasks in both channels. (This
  is the same default that `phase0_v3_abl_config.yaml` incorrectly deviates from — see the
  separate ablation-architecture fix, not part of this checklist.)
