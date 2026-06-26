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

---

## Full Analysis / Plot / Table Refresh — v3 + v3_full (2026-06-26)

Training and inference for all 21×2 experiments below are confirmed complete (verified via
`gen_commands.py list` — all show `6/6 trained, 6/6 inferred`). This section covers the **full
post-inference pipeline**: `analyze --k-dense --bootstrap 1000 --plot` → `collect` → all
figures/tables, run **once, cleanly, for everything that belongs in the paper**.

### Scope — paper-kept tasks only

```
sex_binary, age_class, bmi_binary, sleep_efficiency_binary, apnea_binary,
depression_extreme_binary, osa_binary_apples_postqc
```
× `lstm`, `transformer`, `mean_pool` = **21 experiments per channel, 42 total**.

**`sleep_staging` is excluded** (training still pending — its analysis pipeline differs anyway:
κ-based, not AUROC-based, no `window_analysis.md` in the same format).

**Explicitly dropped from the paper** (still sitting in old `analysis.csv`/`training.csv` rows
from earlier exploration, and in some already-generated table/figure files — see "Why no folder
renaming" below for how this is handled): `cvd_binary`, `sleepiness_binary`, `psqi_binary`,
`osa_severity_apples`, `insomnia_binary`, `rested_morning`, `anxiety_binary`.

### Backup plan — preserve the current files before refreshing (do this first)

You want to **keep the current version of every file** and only then generate new ones, with
zero risk of the new run silently overwriting something you wanted to keep. None of the scripts
in this pipeline have a "write to a new location" flag (`analyze_windows.py`,
`collect_results_v2.py`, and the plot scripts all write to a fixed path derived from
`task`/`head`/`results_dir`) — so the only reliable way to guarantee nothing is lost is to **copy
the current files to a backup location before running anything**, then proceed with the refresh
exactly as planned. This mirrors the project's own existing convention for this situation (see
"Archive notes" at the bottom of this file from 2026-05-25 — old context dirs were preserved
under `_old_accum{N}`/`_failed` suffixes before being overwritten by reruns).

Two different mechanisms, because the repo is git-tracked and scratch is not:

**1. Repo side — git already does this losslessly. Commit + tag, don't copy files.**
```bash
cd /home/boshra95/NSRR-tools
git add -A && git commit -m "Snapshot before full v3/v3_full analysis refresh (2026-06-26)"
git tag pre-refresh-2026-06-26
```
Every CSV/figure/table currently in `results/` is now recoverable forever via
`git show pre-refresh-2026-06-26:<path>` or `git checkout pre-refresh-2026-06-26 -- <path>`, even
after the refresh overwrites the working copies. This is strictly better than a manual file copy
(no duplication, full history, easy diffing) — use it instead of `cp`-ing repo files.

**2. Scratch side — no git, so copy the small files that are about to be overwritten.**
Deliberately **excludes parquets/checkpoints** (large, unaffected, not what you're trying to
preserve) — only the files this refresh will actually touch:

```bash
TS=20260626

# ── Fast channel ──────────────────────────────────────────────────────────────
BASE=/scratch/boshra95/psg/unified/results/phase0_v3

# window_analysis*.{csv,md} + threshold_tuning.csv, per experiment (small, no parquets)
rsync -a --prune-empty-dirs \
  --exclude='_*/' \
  --include='*/' --include='window_analysis*.csv' --include='window_analysis.md' \
  --include='threshold_tuning.csv' --exclude='*' \
  "$BASE/inference/" "$BASE/inference_backup_$TS/"

# figures/ — no parquets here, safe to copy wholesale
cp -r "$BASE/figures" "$BASE/figures_backup_$TS"

# collected/ — just the two CSVs, not predictions/*.parquet
mkdir -p "$BASE/collected_backup_$TS"
cp "$BASE/collected/training.csv" "$BASE/collected/analysis.csv" "$BASE/collected_backup_$TS/" 2>/dev/null

# ── Full channel ──────────────────────────────────────────────────────────────
BASE_FULL=/scratch/boshra95/psg_full/unified/results/phase0_v3_full

rsync -a --prune-empty-dirs \
  --exclude='_*/' \
  --include='*/' --include='window_analysis*.csv' --include='window_analysis.md' \
  --include='threshold_tuning.csv' --exclude='*' \
  "$BASE_FULL/inference/" "$BASE_FULL/inference_backup_$TS/"

cp -r "$BASE_FULL/figures" "$BASE_FULL/figures_backup_$TS"

mkdir -p "$BASE_FULL/collected_backup_$TS"
cp "$BASE_FULL/collected/training.csv" "$BASE_FULL/collected/analysis.csv" "$BASE_FULL/collected_backup_$TS/" 2>/dev/null
```

Once you've confirmed the new refresh looks good, the `*_backup_$TS` scratch directories and the
`pre-refresh-2026-06-26` git tag are safe to delete — they're a safety net, not meant to be kept
forever. (Scratch is typically purged/quota-limited; don't leave large backups sitting there
indefinitely.)

**Why "pollution from dropped tasks" is a separate, smaller issue — not solved by backups:**
`analysis.csv`/`training.csv` will keep accumulating rows for dropped tasks (cvd_binary,
sleepiness_binary, etc.) regardless of backups, since `collect_results_v2.py` scans the whole
`results_dir`. That's fine to leave as historical record — the actual fix is to **always pass an
explicit task list** to multi-task plots/tables (Tables 1/2/4/5/10, `task-comparison`) rather than
relying on "all tasks found in analysis.csv." `run_analysis.sh` already does this for you
automatically (it derives the task list from the experiment IDs you pass it) — Steps 1, 3, and 4
below follow the same explicit-list discipline.

### Step 0 — Verify training/inference (sanity check, already confirmed above)

```bash
python scripts/gen_commands.py list
python scripts/gen_commands.py --registry experiments/v2_full_registry.yaml list
```
All 21 rows per channel should show `analyzed` or `inferred (6/6)` — none should show
`none`/partial counts.

### Step 1 — Full analysis + plot pipeline (the efficient way: `run_analysis.sh`)

This single call per channel runs all 13 steps from `EXPERIMENTS_GUIDE.md` § End-to-End Run
Playbooks: `analyze --k-dense --bootstrap 1000 --plot`, `collect`, `build-heatmap`, `iso-plots`,
`saturation`, `scaling-laws`, `calibration`, `window-position`, `subject-consistency`,
`cohort-saturation`, `precision-recall`, `subject-kstar`, `task-comparison` — and because
`task-comparison`/`saturation` derive their task list from the experiment IDs you pass in, **only
the 7 kept tasks ever appear in the multi-task plots**, automatically.

Run in `tmux` — bootstrap (1000 resamples × dense K × 21 experiments) takes hours.

```bash
source /home/boshra95/sleepfm_env/bin/activate
tmux new -s analysis_fast
```

**Fast channel:**
```bash
bash scripts/run_analysis.sh \
  sex_binary_lstm sex_binary_transformer sex_binary_mean_pool \
  age_class_lstm age_class_transformer age_class_mean_pool \
  bmi_binary_lstm bmi_binary_transformer bmi_binary_mean_pool \
  sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer sleep_efficiency_binary_mean_pool \
  apnea_binary_lstm apnea_binary_transformer apnea_binary_mean_pool \
  depression_extreme_binary_lstm depression_extreme_binary_transformer depression_extreme_binary_mean_pool \
  osa_binary_apples_postqc_lstm osa_binary_apples_postqc_transformer osa_binary_apples_postqc_mean_pool \
  --heads lstm transformer mean_pool \
  --bootstrap 1000 \
  2>&1 | tee analysis_fast_full_refresh.log
```

**Full channel** (new tmux pane/session, or sequentially after fast finishes):
```bash
bash scripts/run_analysis.sh \
  sex_binary_lstm sex_binary_transformer sex_binary_mean_pool \
  age_class_lstm age_class_transformer age_class_mean_pool \
  bmi_binary_lstm bmi_binary_transformer bmi_binary_mean_pool \
  sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer sleep_efficiency_binary_mean_pool \
  apnea_binary_lstm apnea_binary_transformer apnea_binary_mean_pool \
  depression_extreme_binary_lstm depression_extreme_binary_transformer depression_extreme_binary_mean_pool \
  osa_binary_apples_postqc_lstm osa_binary_apples_postqc_transformer osa_binary_apples_postqc_mean_pool \
  --registry experiments/v2_full_registry.yaml \
  --heads lstm transformer mean_pool \
  --bootstrap 1000 \
  2>&1 | tee analysis_full_full_refresh.log
```

**Figures/markdown are dual-saved automatically** — `gen_commands.py`'s builders for `analyze`
and all 10 plot subcommands already pass `--repo-out`/`--repo-figures-dir` derived from the
registry's `results_dir` name, so every PNG/`window_analysis.md`/`window_analysis_{split}.csv`
lands in both:
```
scratch:  {results_dir}/figures/...              and  {results_dir}/inference/{exp}/...
repo:     results/figures/{phase0_v3|phase0_v3_full}/...
          results/inference/{phase0_v3|phase0_v3_full}/{exp}/...
```
PDFs stay scratch-only (gitignored under `results/figures/` — PNGs are kept in the repo).

`run_analysis.sh`'s own `collect` step (Step 2 of the 13) runs **without** `--force`, so it will
**not** refresh rows for experiments that were already collected before this run (e.g.
`bmi_binary_transformer`, already fully analyzed previously) — it only adds rows for genuinely
new keys. **Do not rely on this for the final collected CSV** — do the explicit force-recollect
in Step 2 below instead, which is the one that actually picks up the new dense-K/bootstrap values
for every experiment uniformly.

### Step 2 — Collect cleanly, with no duplication (important — read this)

`collect_results_v2.py --force` re-reads every experiment's `metrics.json`/`window_analysis_*.csv`
from disk, but **does not deduplicate against the existing CSV** — concatenating "forced" fresh
rows onto an `analysis.csv` that already contains the *same* keys produces literal duplicate
rows. The safe way to force-refresh without duplicating is to make sure `collect_results_v2.py`
starts from an **empty** `existing` DataFrame, so every row it writes is genuinely new.

**✅ Already done for this refresh (2026-06-26):** rather than deleting the current collected
CSVs, they were **renamed in place** — `training.csv` → `training_old_20260626.csv`,
`analysis.csv` → `analysis_old_20260626.csv` — for both channels, both in scratch and in the
repo. This is equivalent to delete-after-backup but keeps the "old" copy sitting right next to
where the new one will appear, with zero risk window in between (rename is atomic; copy-then-rm
has a brief window where both the backup and the original exist as separate operations).
Verified: `results/collected/phase0_v3/` and `results/collected/phase0_v3_full/` (and their
scratch counterparts) now contain only `*_old_20260626.csv` — no `training.csv`/`analysis.csv` at
the original path, so the next `collect_results_v2.py` run starts clean (no `--force` needed at
all, since `existing` will be empty either way).

```bash
# Fast channel — fresh collect (original path is now empty; no --force needed)
python scripts/collect_results_v2.py
# --repo-out defaults to results/collected/phase0_v3/ automatically — dual-write confirmed.

# Full channel — same
python scripts/collect_results_v2.py \
  --results-dir /scratch/boshra95/psg_full/unified/results/phase0_v3_full
```

This re-scans **all** experiments under each `results_dir` — including the dropped tasks
(cvd_binary, sleepiness_binary, etc.) and the archived `_`-prefixed sleep-staging variants
(correctly skipped — see the archived-dir fix in `collect_results_v2.py`). That's fine: the
dropped-task rows are harmless in `analysis.csv` itself, as long as every downstream multi-task
command filters with `--tasks` (Step 3 does this explicitly; `run_analysis.sh` does it
automatically).

**If repeating this pattern later** (e.g. for the v3_abl rerun, or any future re-collect), the
rename-in-place recipe is:
```bash
TS=$(date +%Y%m%d)
for f in training analysis; do
  mv "$BASE/collected/$f.csv" "$BASE/collected/${f}_old_$TS.csv" 2>/dev/null
  mv "results/collected/<round>/$f.csv" "results/collected/<round>/${f}_old_$TS.csv" 2>/dev/null
done
python scripts/collect_results_v2.py --results-dir "$BASE"
```
The `*_old_$TS.csv` files are safe to delete once you've confirmed the fresh collect looks right
— they're a transition safety net, not meant to be kept forever (and the `pre-refresh` git tag /
scratch backup dirs already preserve the same content anyway).

**Verify no duplicates after rebuilding:**
```bash
python3 -c "
import pandas as pd
for ch, path in [('fast','results/collected/phase0_v3/analysis.csv'),
                 ('full','results/collected/phase0_v3_full/analysis.csv')]:
    df = pd.read_csv(path)
    dupe_key = ['task','head','run_tag','context_length','k','split']
    n_dupes = df.duplicated(subset=dupe_key).sum()
    print(f'{ch}: {len(df)} rows, {n_dupes} duplicate keys (should be 0)')
"
```

### Step 3 — Multi-task paper tables (1, 2, 4, 5, 10), explicitly filtered to kept tasks

`make_tableN_*.py` defaults to "all tasks found in analysis.csv" when `--tasks` is omitted — you
**must** pass the kept-task list explicitly here (unlike `run_analysis.sh`, these scripts don't
auto-derive it). Also pass `--results-dir` to get the scratch copy in addition to the repo
default.

```bash
KEPT_TASKS="sex_binary age_class bmi_binary sleep_efficiency_binary apnea_binary depression_extreme_binary osa_binary_apples_postqc"

# ── Fast channel ──────────────────────────────────────────────────────────────
for table in make_table1_peak_auroc make_table2_lstar make_table10_ci; do
  python scripts/$table.py \
    --collected-dir results/collected/phase0_v3 --channel fast \
    --tasks $KEPT_TASKS --heads lstm transformer mean_pool \
    --results-dir /scratch/boshra95/psg/unified/results/phase0_v3
done
python scripts/make_table4_sensitivity.py \
  --collected-dir results/collected/phase0_v3 --channel fast \
  --tasks $KEPT_TASKS --head lstm \
  --results-dir /scratch/boshra95/psg/unified/results/phase0_v3
python scripts/make_table5_heads.py \
  --collected-dir results/collected/phase0_v3 --channel fast \
  --tasks $KEPT_TASKS --heads lstm transformer mean_pool \
  --results-dir /scratch/boshra95/psg/unified/results/phase0_v3

# ── Full channel ──────────────────────────────────────────────────────────────
for table in make_table1_peak_auroc make_table2_lstar make_table10_ci; do
  python scripts/$table.py \
    --collected-dir results/collected/phase0_v3_full --channel full \
    --tasks $KEPT_TASKS --heads lstm transformer mean_pool \
    --results-dir /scratch/boshra95/psg_full/unified/results/phase0_v3_full
done
python scripts/make_table4_sensitivity.py \
  --collected-dir results/collected/phase0_v3_full --channel full \
  --tasks $KEPT_TASKS --head lstm \
  --results-dir /scratch/boshra95/psg_full/unified/results/phase0_v3_full
python scripts/make_table5_heads.py \
  --collected-dir results/collected/phase0_v3_full --channel full \
  --tasks $KEPT_TASKS --heads lstm transformer mean_pool \
  --results-dir /scratch/boshra95/psg_full/unified/results/phase0_v3_full
```

### Step 4 — Per-experiment tables (3: K-grid, 9: cohort breakdown)

These are per-`exp_id`, so a `for` loop over the 21 kept experiments is the natural shape:

```bash
EXP_IDS="sex_binary_lstm sex_binary_transformer sex_binary_mean_pool \
age_class_lstm age_class_transformer age_class_mean_pool \
bmi_binary_lstm bmi_binary_transformer bmi_binary_mean_pool \
sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer sleep_efficiency_binary_mean_pool \
apnea_binary_lstm apnea_binary_transformer apnea_binary_mean_pool \
depression_extreme_binary_lstm depression_extreme_binary_transformer depression_extreme_binary_mean_pool \
osa_binary_apples_postqc_lstm osa_binary_apples_postqc_transformer osa_binary_apples_postqc_mean_pool"

# Fast channel
for exp in $EXP_IDS; do
  python scripts/make_table3_kgrid.py $exp \
    --collected-dir results/collected/phase0_v3 --channel fast \
    --results-dir /scratch/boshra95/psg/unified/results/phase0_v3
  python scripts/make_table9_cohort.py $exp \
    --collected-dir results/collected/phase0_v3 --channel fast \
    --results-dir /scratch/boshra95/psg/unified/results/phase0_v3
done

# Full channel
for exp in $EXP_IDS; do
  python scripts/make_table3_kgrid.py $exp \
    --collected-dir results/collected/phase0_v3_full --channel full \
    --results-dir /scratch/boshra95/psg_full/unified/results/phase0_v3_full
  python scripts/make_table9_cohort.py $exp \
    --collected-dir results/collected/phase0_v3_full --channel full \
    --results-dir /scratch/boshra95/psg_full/unified/results/phase0_v3_full
done
```

`make_table9_cohort.py` reads per-window parquets directly (not just `analysis.csv`), so it needs
`--results-dir` unconditionally (not optional like the others).

### Step 5 — Threshold tuning (binary tasks only — likely missed item)

Flagging this because you didn't mention it but it's part of the standard pipeline
(`EXPERIMENTS_GUIDE.md` § Post-hoc Decision Threshold Tuning) and **was never run for any of the
new experiments** (new heads for depression/osa, mean_pool for all 7, and **all of full-channel**
has never had threshold tuning at all). `age_class` is multiclass — skip it; the other 6 tasks
are binary.

Needs val-split inference first (cheap, reuses trained checkpoints, no GPU-heavy work):

```bash
BINARY_EXPS="sex_binary_lstm sex_binary_transformer sex_binary_mean_pool \
bmi_binary_lstm bmi_binary_transformer bmi_binary_mean_pool \
sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer sleep_efficiency_binary_mean_pool \
apnea_binary_lstm apnea_binary_transformer apnea_binary_mean_pool \
depression_extreme_binary_lstm depression_extreme_binary_transformer depression_extreme_binary_mean_pool \
osa_binary_apples_postqc_lstm osa_binary_apples_postqc_transformer osa_binary_apples_postqc_mean_pool"

# Fast channel
for exp in $BINARY_EXPS; do
  python scripts/gen_commands.py infer $exp --split val | bash
done
for exp in $BINARY_EXPS; do
  python scripts/gen_commands.py threshold-tuning $exp | bash
done

# Full channel
REG="--registry experiments/v2_full_registry.yaml"
for exp in $BINARY_EXPS; do
  python scripts/gen_commands.py $REG infer $exp --split val | bash
done
for exp in $BINARY_EXPS; do
  python scripts/gen_commands.py $REG threshold-tuning $exp | bash
done
```

`threshold_tuning.csv` is dual-saved automatically (the `--repo-out` flag is already wired into
`cmd_threshold_tuning` in `gen_commands.py`).

### Other things you may have missed (flagging per your request)

- **`results_summary_fast_vs_full.csv`** (`scripts/summarize_results.py --compare`) — the
  fast-vs-full comparison table. Re-run after Step 2 so it reflects the new mean_pool/transformer
  coverage: `python scripts/summarize_results.py --compare --out results/collected/results_summary_fast_vs_full.csv`.
- **`heatmap_df_{split}.csv`** (written by `build-heatmap`, part of `run_analysis.sh` Step 1) is
  intentionally **scratch-only** — it's redundant with `analysis.csv`'s `total_compute_min`
  column, not mirrored to the repo by design.
- **Sleep staging** — once training finishes, it needs its *own* checklist: different metric
  (κ, not AUROC), different `analyze` invocation, no `window_analysis.md` in the same format.
  Not covered here.

---

## Later: re-running `v3_abl` analysis after the architecture fix

`phase0_v3_abl` was trained with the wrong architecture (hidden_dim=256/num_layers=2 — the
sleep-staging arch — instead of 128/1) for all 25 experiments. Once you fix
`configs/phase0_v3_abl_config.yaml` and **retrain all 25 jobs**, the following must be re-run
in order to keep every downstream file correct — listed so nothing gets silently left stale:

1. **Re-infer + re-analyze all 25 experiments** (same pattern as the main ablation playbook in
   `EXPERIMENTS_GUIDE.md` § Modality ablation run, Steps 2–3). Use `--k-dense --bootstrap 1000
   --plot` this time, to match the rigor of the v3/v3_full refresh above (the original ablation
   run used plain `analyze` with no flags).

2. **Force-recollect `phase0_v3_abl`'s analysis.csv — do not just re-run `collect` plain.** This
   is the one case where the duplication trap in Step 2 above is *guaranteed* to bite you if you
   don't clear the existing CSV first: the retrained experiments reuse the **same** experiment
   folder paths (same task/head/run_tag/context), so a normal `collect` (no `--force`) will see
   those keys as "already collected" and **silently keep the old, wrong-architecture numbers**.
   Use the same **rename-in-place** pattern as Step 2 above (commit/tag the repo copy first):
   ```bash
   git add -A && git commit -m "Snapshot before v3_abl re-collect (post arch fix)"
   git tag pre-abl-refresh-$(date +%Y%m%d)
   git push origin pre-abl-refresh-$(date +%Y%m%d)

   TS=$(date +%Y%m%d)
   mv /scratch/boshra95/psg/unified/results/phase0_v3_abl/collected/training.csv \
      /scratch/boshra95/psg/unified/results/phase0_v3_abl/collected/training_old_$TS.csv
   mv /scratch/boshra95/psg/unified/results/phase0_v3_abl/collected/analysis.csv \
      /scratch/boshra95/psg/unified/results/phase0_v3_abl/collected/analysis_old_$TS.csv
   mv results/collected/phase0_v3_abl/training.csv results/collected/phase0_v3_abl/training_old_$TS.csv
   mv results/collected/phase0_v3_abl/analysis.csv  results/collected/phase0_v3_abl/analysis_old_$TS.csv

   python scripts/collect_results_v2.py \
     --results-dir /scratch/boshra95/psg/unified/results/phase0_v3_abl
   ```

3. **Regenerate Table 6**: `python scripts/make_table6_modality.py`. This reads both
   `results/collected/phase0_v3/analysis.csv` (the "Full" baseline — unaffected by the ablation
   fix, no need to touch it) and `results/collected/phase0_v3_abl/analysis.csv` (now corrected).

4. **Rewrite the interpretation in `docs/SOTA_COMPARISON_AND_ABLATIONS.md` §A.6.1** — the current
   text was already flagged as invalid pending this fix; once Table 6 is regenerated, redo the
   per-task interpretation bullets and the "most necessary modality per task" summary table with
   the corrected numbers.

5. **No figures exist yet for `phase0_v3_abl`** (the original ablation `analyze` calls never used
   `--plot`) — if you want per-task ablation figures, generate them fresh after the fix; there's
   nothing stale to clean up there.

6. **Nothing in `phase0_v3`/`phase0_v3_full` is affected** by the ablation fix — their collected
   CSVs, tables, and figures from this refresh remain valid and don't need to be touched again.
