# Results Collection — Reference Guide

This document covers `scripts/collect_results_v2.py`: what it collects, where outputs go, the full column schema of each file, and how each file maps to every analysis plot and table in the paper pipeline.

---

## Table of Contents

1. [What the collector does](#what-the-collector-does)
2. [How to run it](#how-to-run-it)
3. [Cross-cluster sync workflow](#cross-cluster-sync-workflow)
4. [Input paths (what it reads)](#input-paths-what-it-reads)
5. [Output paths (what it writes)](#output-paths-what-it-writes)
6. [File schemas](#file-schemas)
   - [training.csv](#trainingcsv)
   - [analysis.csv](#analysiscsv)
   - [predictions/ parquets](#predictions-parquets)
7. [How to use the files for analysis](#how-to-use-the-files-for-analysis)
   - [Paper performance tables](#paper-performance-tables)
   - [Learning curves](#learning-curves)
   - [Figure 1 — Saturation curve](#figure-1--saturation-curve)
   - [Figure 2 — 2D heatmap](#figure-2--2d-heatmap)
   - [Figure 3 — K-sweep row curves](#figure-3--k-sweep-row-curves)
   - [Iso-compute plots (7 plots)](#iso-compute-plots-7-plots)
   - [Custom aggregations from per-window predictions](#custom-aggregations-from-per-window-predictions)
8. [Design notes](#design-notes)

---

## What the collector does

`scripts/collect_results_v2.py` scans the scratch results directory for all completed training runs and inference outputs and appends new results to three flat files:

| Output | Content | Destination |
|--------|---------|-------------|
| `training.csv` | One row per (task, head, context, epoch) | Repo **and** scratch |
| `analysis.csv` | One row per (task, head, context, K, split) | Repo **and** scratch |
| `predictions/{task}_{head}_{ctx}_{split}.parquet` | One row per (task, head, context, split, subject, window) | Scratch only |

**Append-safe:** the script computes a set of already-present key tuples before scanning and skips any row whose key already exists. It is safe to rerun at any time and from either cluster — it only adds new rows, never overwrites existing ones.

**Multi-class safe:** all probability columns are padded to `prob_class0` … `prob_class4` with `NaN` for unused classes, so binary, 3-class, and 4-class tasks all share one parquet schema.

---

## How to run it

```bash
cd /home/boshra95/NSRR-tools
python scripts/collect_results_v2.py                               # uses phase0_v3 by default
python scripts/collect_results_v2.py --results-dir /scratch/boshra95/psg/unified/results/phase0_v3
python scripts/collect_results_v2.py --exp-ids sex_binary_lstm sex_binary_transformer  # filter to specific experiments
```

The script defaults to `phase0_v3`. Run it from either cluster whenever new training or inference results are available. It is also accessible via `gen_commands.py collect` which prints the full command with correct paths.

Example output:
```
Scanning:    /scratch/boshra95/psg/unified/results/phase0_v3
Repo out:    /home/boshra95/NSRR-tools/results/collected
Scratch out: /scratch/boshra95/psg/unified/results/phase0_v3/collected

Collecting training results...
  → 84 new rows
Collecting window analysis results...
  → 36 new rows
Collecting per-window predictions (scratch only)...
    predictions/age_class_lstm_10m_test.parquet: 12,345 rows
  → 3 new parquet files

To sync across clusters:
  git add results/collected/ && git commit -m 'collect results' && git push
```

---

## Cross-cluster sync workflow

The two flat CSVs are committed to the repo so they accumulate results from both clusters without any file-level sync:

```bash
# After running the collector on either cluster:
git add results/collected/
git commit -m "collect results — $(hostname): $(date +%Y-%m-%d)"
git push

# On the other cluster, pull before running the collector:
git pull
python scripts/collect_results_v2.py   # adds only whatever is new on this cluster
git add results/collected/
git commit -m "collect results — $(hostname): $(date +%Y-%m-%d)"
git push
```

The predictions parquets are scratch-only (too large for git). They do not need to be synced — they can be regenerated on each cluster independently by running the collector after inference.

---

## Input paths (what it reads)

All inputs come from the scratch results directory:

```
/scratch/boshra95/psg/unified/results/phase0_v3/
│
├── {task}_{head}/
│   └── context_{L}/
│       ├── training_curves.csv    ← per-epoch loss and accuracy curves
│       └── metrics.json           ← final metrics and metadata at best epoch
│
└── inference/
    └── {task}_{head}/
        ├── window_analysis_{split}.csv    ← K-sweep metrics (from analyze_windows.py)
        └── context_{L}/
            └── {split}_windows.parquet   ← per-window predictions (from infer_subject_windows.py)
```

The collector requires `training_curves.csv` to be present for a context to be included in `training.csv`. It requires `window_analysis_{split}.csv` for `analysis.csv`, and `{split}_windows.parquet` for the per-window prediction parquets. Missing files for a context are silently skipped.

---

## Output paths (what it writes)

```
/home/boshra95/NSRR-tools/results/collected/    ← in git repo
    training.csv
    analysis.csv

/scratch/boshra95/psg/unified/results/phase0_v3/collected/    ← scratch copy
    training.csv
    analysis.csv
    predictions/
        {task}_{head}_{context}_{split}.parquet
```

Both `training.csv` and `analysis.csv` are written to the repo **and** to scratch. The repo copy is the one that gets synced across clusters via git. The scratch copy is a convenience for scripts running on that cluster that don't want to reference the repo path.

---

## File schemas

### training.csv

**Key columns (uniquely identify a row):** `task`, `head`, `context_length`, `epoch`

**Present on every row:**

| Column | Type | Description |
|--------|------|-------------|
| `task` | str | e.g. `sex_binary`, `age_class` |
| `head` | str | `lstm` / `transformer` / `mean_pool` |
| `context_length` | str | e.g. `30s`, `10m`, `240m` |
| `epoch` | int | Training epoch number (1-indexed) |
| `is_best_epoch` | bool | True for the epoch selected by early stopping |
| `train_loss` | float | Cross-entropy loss on training split |
| `val_loss` | float | Cross-entropy loss on validation split |
| `train_bal_acc` | float | Balanced accuracy on training split |
| `val_bal_acc` | float | Balanced accuracy on validation split |
| `val_auroc` | float | AUROC on validation split (the early-stopping monitor) |
| `num_classes` | int | Number of output classes (2, 3, 4, or 5) |
| `n_train` | int | Number of subjects in training split |
| `n_val` | int | Number of subjects in validation split |
| `n_test` | int | Number of subjects in test split |
| `n_epochs_run` | int | Total epochs run (including resumed jobs) |
| `training_time_min` | float | Total wall-clock training time in minutes |

**Present only on the best epoch row (`is_best_epoch == True`):**

| Column | Type | Description |
|--------|------|-------------|
| `{split}_accuracy` | float | Accuracy on `train`/`val`/`test` split |
| `{split}_balanced_accuracy` | float | Balanced accuracy |
| `{split}_macro_f1` | float | Macro-averaged F1 |
| `{split}_auroc` | float | AUROC (macro OvR for multi-class) |
| `{split}_recall_class{0..4}` | float | Per-class recall; NaN for unused class indices |

All columns exist in the CSV for all rows; non-best rows have NaN in the best-epoch-only columns.

---

### analysis.csv

**Key columns (uniquely identify a row):** `task`, `head`, `context_length`, `k`, `split`

| Column | Type | Description |
|--------|------|-------------|
| `task` | str | Task name |
| `head` | str | Head type |
| `context_length` | str | Context length string (e.g. `10m`) |
| `k` | str | Number of windows per subject at inference (`"1"`, `"5"`, …, `"all"`) |
| `split` | str | `test` or `val` |
| `context_length_min` | float | Context length in minutes (0.5 for `30s`, 10.0 for `10m`, etc.) |
| `total_compute_min` | float | `context_length_min × k` — the iso-compute axis; NaN when `k == "all"` |
| `n_subjects` | int | Number of subjects in this split |
| `n_segments` | int | Total number of windows across all subjects |
| `seg_accuracy` | float | Accuracy at the segment (window) level |
| `seg_balanced_accuracy` | float | Balanced accuracy at the segment level |
| `seg_macro_f1` | float | Macro F1 at the segment level |
| `seg_auroc` | float | AUROC at the segment level |
| `mean_prob_accuracy` | float | Subject-level accuracy: mean softmax → argmax |
| `mean_prob_balanced_accuracy` | float | Subject-level balanced accuracy via mean-prob aggregation |
| `mean_prob_macro_f1` | float | Subject-level macro F1 via mean-prob aggregation |
| `mean_prob_auroc` | float | Subject-level AUROC via mean-prob aggregation |
| `majority_accuracy` | float | Subject-level accuracy via majority vote |
| `majority_balanced_accuracy` | float | Subject-level balanced accuracy via majority vote |
| `majority_macro_f1` | float | Subject-level macro F1 via majority vote |
| `majority_auroc` | float | Subject-level AUROC via majority vote |

The `k` column is stored as a string because it can be `"all"`. When loading for analysis, cast with `pd.to_numeric(df['k'], errors='coerce')` to get floats (NaN for `"all"` rows).

---

### predictions/ parquets

One parquet file per (task, head, context, split), e.g. `sex_binary_lstm_10m_test.parquet`.

| Column | Type | Description |
|--------|------|-------------|
| `task` | str | Task name |
| `head` | str | Head type |
| `context_length` | str | Context length string |
| `split` | str | `test` or `val` |
| `subject_id` | str | Subject identifier |
| `dataset` | str | Source dataset (e.g. `shhs`, `apples`) |
| `window_idx` | int32 | 0-indexed window position for this subject |
| `true_label` | int16 | Ground truth class label |
| `pred_label` | int16 | Argmax prediction |
| `prob_class0` | float32 | Softmax probability for class 0 |
| `prob_class1` | float32 | Softmax probability for class 1 |
| `prob_class2` | float32 | Softmax probability for class 2 (NaN for binary tasks) |
| `prob_class3` | float32 | Softmax probability for class 3 (NaN for binary/3-class) |
| `prob_class4` | float32 | Softmax probability for class 4 (NaN unless 5-class) |

To load all parquets at once (after they've been collected):
```python
import pandas as pd
pred = pd.read_parquet(
    "/scratch/boshra95/psg/unified/results/phase0_v3/collected/predictions/"
)
# pred is a single DataFrame with all tasks, heads, contexts, and splits
```

---

## How to use the files for analysis

### Paper performance tables

Filter `training.csv` to best-epoch rows for the test split:

```python
import pandas as pd

train = pd.read_csv("results/collected/training.csv")
best = train[train["is_best_epoch"]]

# Table: test AUROC by task, head, context_length
table = best.pivot_table(
    index=["task", "head"],
    columns="context_length",
    values="test_auroc",
)
print(table.to_string())
```

Relevant columns for a typical paper table: `test_auroc`, `test_balanced_accuracy`, `test_macro_f1`, `test_recall_class{0..N}`.

---

### Learning curves

Use all rows (not filtered by `is_best_epoch`) to plot loss or accuracy over epochs:

```python
import matplotlib.pyplot as plt

train = pd.read_csv("results/collected/training.csv")
run = train[(train.task == "sex_binary") & (train.head == "lstm") & (train.context_length == "10m")]

plt.plot(run["epoch"], run["val_auroc"], label="val_auroc")
plt.plot(run["epoch"], run["val_bal_acc"], label="val_bal_acc")
plt.axvline(run.loc[run["is_best_epoch"], "epoch"].values[0], linestyle="--", label="best epoch")
plt.legend(); plt.xlabel("Epoch"); plt.show()
```

---

### Figure 1 — Saturation curve

AUROC vs context length per head (answers H1: does performance saturate?). Use `analysis.csv` with `k == "all"` for the most complete picture, or use `training.csv` best-epoch `test_auroc` as a quick alternative.

```python
import pandas as pd, matplotlib.pyplot as plt

analysis = pd.read_csv("results/collected/analysis.csv")
ctx_order = {"30s": 0.5, "10m": 10, "40m": 40, "80m": 80, "120m": 120, "240m": 240}

sat = analysis[(analysis.task == "sex_binary") & (analysis.k == "all") & (analysis.split == "test")]
sat = sat.copy()
sat["ctx_min"] = sat["context_length"].map(ctx_order)
sat = sat.sort_values("ctx_min")

for head, grp in sat.groupby("head"):
    plt.plot(grp["ctx_min"], grp["mean_prob_auroc"], marker="o", label=head)

plt.xscale("log"); plt.xlabel("Context length (min)"); plt.ylabel("Test AUROC (K=all)")
plt.legend(); plt.title("sex_binary — saturation curve"); plt.show()
```

This is what `scripts/plot_saturation.py` does. You can also generate it via:
```bash
python scripts/gen_commands.py saturation sex_binary --heads lstm transformer mean_pool | bash
```

---

### Figure 2 — 2D heatmap

The heatmap (L × K grid, color = AUROC) requires dense K values, generated by the full pipeline:

```bash
# 1. Dense K sweep (25 K values per context)
python scripts/gen_commands.py analyze sex_binary_lstm --k-dense | bash

# 2. Collect into analysis.csv (picks up the new dense K rows)
python scripts/collect_results_v2.py

# 3. Build heatmap-ready DataFrame
python scripts/gen_commands.py build-heatmap sex_binary_lstm | bash

# 4. Plot
python scripts/gen_commands.py iso-plots sex_binary_lstm | bash
```

Or read `analysis.csv` directly for a custom heatmap:

```python
analysis = pd.read_csv("results/collected/analysis.csv")
hm = analysis[
    (analysis.task == "sex_binary") &
    (analysis.head == "lstm") &
    (analysis.split == "test")
].copy()
hm["k_num"] = pd.to_numeric(hm["k"], errors="coerce")
hm = hm.dropna(subset=["k_num", "context_length_min"])

pivot = hm.pivot_table(index="context_length_min", columns="k_num", values="mean_prob_auroc")
import seaborn as sns
sns.heatmap(pivot, annot=False, cmap="viridis")
```

---

### Figure 3 — K-sweep row curves

Each row of the heatmap as a line: AUROC vs K for a fixed context length. Answers H3 (aggregation saturation).

```python
analysis = pd.read_csv("results/collected/analysis.csv")
row = analysis[
    (analysis.task == "sex_binary") & (analysis.head == "lstm") &
    (analysis.context_length == "10m") & (analysis.split == "test")
].copy()
row["k_num"] = pd.to_numeric(row["k"], errors="coerce")
row = row.dropna(subset=["k_num"]).sort_values("k_num")

plt.plot(row["k_num"], row["mean_prob_auroc"], marker="o")
plt.xscale("log"); plt.xlabel("K (windows per subject)"); plt.ylabel("AUROC"); plt.show()
```

Also generated directly by:
```bash
python scripts/gen_commands.py analyze sex_binary_lstm --plot | bash
```

---

### Iso-compute plots (7 plots)

The full iso-compute pipeline (Steps 4a–4c in the experiment guide) uses `analysis.csv` indirectly through `build_heatmap_df.py`. After running the dense K sweep and collecting results, the 7 plots are produced by:

```bash
python scripts/gen_commands.py build-heatmap sex_binary_lstm | bash
python scripts/gen_commands.py iso-plots sex_binary_lstm | bash
```

The `total_compute_min` column in `analysis.csv` is the iso-compute axis (`context_length_min × k`). To manually extract iso-compute diagonals:

```python
analysis = pd.read_csv("results/collected/analysis.csv")
# All (task, head, context, K) combinations where total signal ≈ 80 min
iso80 = analysis[
    (analysis.task == "sex_binary") & (analysis.head == "lstm") &
    (analysis.split == "test") &
    (analysis["total_compute_min"].between(75, 85))
].sort_values("context_length_min")
print(iso80[["context_length", "k", "total_compute_min", "mean_prob_auroc"]])
```

The 7 plots and what they answer:

| Plot | `analysis.csv` usage | Research question |
|------|---------------------|-------------------|
| Heatmap (L × K) | All rows, pivot on context_length_min × k | Overview of H1 + H3 |
| Metric vs K | Rows per context, sort by k | H3: aggregation saturation |
| Metric vs total context | Use `total_compute_min` as x-axis | H2: iso-compute comparison |
| Pareto front | Best AUROC per `total_compute_min` budget | H2: optimal (L, K) strategy |
| Min-cost frontier | Min `total_compute_min` to reach AUROC threshold | Clinical deployment cost |
| Marginal gain | Δ AUROC per additional window | Diminishing returns |
| Double tradeoff | Compare doubling K vs doubling L | H2: direct head-to-head |

---

### Custom aggregations from per-window predictions

The predictions parquets let you recompute any aggregation or metric not covered by `analyze_windows.py`:

```python
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

# Load one parquet
df = pd.read_parquet(
    "/scratch/boshra95/psg/unified/results/phase0_v3/collected/predictions/"
    "sex_binary_lstm_10m_test.parquet"
)

# Custom aggregation: mean prob over first K windows per subject
K = 5
top_k = (df.groupby(["subject_id", "dataset"])
           .head(K)
           .groupby(["subject_id", "dataset"])
           .agg(mean_prob1=("prob_class1", "mean"),
                true_label=("true_label", "first"))
           .reset_index())

auroc = roc_auc_score(top_k["true_label"], top_k["mean_prob1"])
print(f"K={K} mean-prob AUROC: {auroc:.4f}")
```

Use cases for the predictions parquets:
- **Plot A (ROC at iso-compute):** Select K=floor(budget/L) windows per subject, build ROC curve from `prob_class1`.
- **Plot B (recall at fixed precision):** Sweep majority-vote threshold t from 1 to K.
- **Dataset-specific breakdowns:** Filter by `dataset` column to check per-cohort results.
- **Per-subject analysis:** Identify subjects where the model is consistently wrong across windows.

---

## Design notes

**Why two CSVs instead of one big one?**  
Training data (one row per epoch) and analysis data (one row per K value) have different granularities and different key structures. Merging them would require filling thousands of NaN cells. The two CSVs are self-contained: `training.csv` for anything about the model training process, `analysis.csv` for anything about inference and aggregation at test time.

**Why are predictions stored separately (not in analysis.csv)?**  
The full per-window probability vectors are very large (~10,000–100,000 rows per experiment) and are not needed for standard tables or plots. `analysis.csv` already contains the aggregated metrics. The parquets are for custom aggregations and the supplementary ROC/threshold plots.

**Why NaN padding for `prob_class2..4`?**  
Tasks differ in number of classes (2, 3, 4, or 5). Padding all parquets to 5 columns with NaN means all files share one schema and can be concatenated trivially with `pd.read_parquet(directory)`. Filter out NaN prob columns per task before computing metrics.

**Adding a new task:** Run the full pipeline (train → infer → analyze), then run `collect_results_v2.py`. New rows are appended automatically. No schema changes needed.

**Changing a model:** Use a different `run_tag` in the registry so results land in a separate directory. The collector will pick them up as a new `(task, head)` combination because the directory name (and thus the parsed `task`/`head` values) will differ.
