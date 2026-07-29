"""Data loading utilities for paper figures.

All paths are resolved relative to WORKSPACE_ROOT, which should be set at the
top of each notebook:

    from pathlib import Path
    WORKSPACE_ROOT = Path("../../../../../")  # adjust depth if needed
    from utils.data import set_root, load_analysis, ...
    set_root(WORKSPACE_ROOT)
"""

from pathlib import Path
import pandas as pd
import numpy as np

from .style import CONTEXT_TO_MIN

_ROOT: Path | None = None


def set_root(root: Path):
    global _ROOT
    _ROOT = Path(root).resolve()


def _final(experiment: str) -> Path:
    assert _ROOT is not None, "Call set_root() before loading data."
    return _ROOT / "final_results" / experiment


# ── Analysis CSV ──────────────────────────────────────────────────────────────

def load_analysis(experiment: str = "phase0_v3", split: str = "test",
                  k: str = "all") -> pd.DataFrame:
    """Load analysis.csv, optionally filtered to a specific split and k."""
    p = _final(experiment) / "collected" / "analysis.csv"
    df = pd.read_csv(p)
    if "context_length_min" not in df.columns:
        df["context_length_min"] = df["context_length"].map(
            lambda s: CONTEXT_TO_MIN.get(str(s).strip()))
    if split:
        df = df[df["split"] == split]
    if k:
        df = df[df["k"].astype(str) == str(k)]
    return df.reset_index(drop=True)


def load_analysis_all_k(experiment: str = "phase0_v3",
                        split: str = "test") -> pd.DataFrame:
    """Load analysis.csv keeping all K values (needed for waterfall decomposition)."""
    return load_analysis(experiment=experiment, split=split, k=None)


def load_training(experiment: str = "phase0_v3") -> pd.DataFrame:
    p = _final(experiment) / "collected" / "training.csv"
    return pd.read_csv(p)


# ── Heatmap DataFrames ────────────────────────────────────────────────────────

def load_heatmap(experiment: str, task: str, head: str,
                 split: str = "test") -> pd.DataFrame:
    """Load heatmap_df_{split}.csv for a single task/head combo."""
    p = _final(experiment) / "inference" / f"{task}_{head}" / f"heatmap_df_{split}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "context_length_min" not in df.columns and "context_label" in df.columns:
        df["context_length_min"] = df["context_label"].map(
            lambda s: CONTEXT_TO_MIN.get(str(s).strip()))
    return df.sort_values(["context_length_min", "k"]).reset_index(drop=True)


# ── Collected prediction parquets ─────────────────────────────────────────────

def load_parquets(experiment: str, task: str, head: str,
                  split: str = "test") -> dict[str, pd.DataFrame]:
    """Return {context_label: DataFrame} from collected/predictions/.

    Each parquet has columns: task, head, context_length, split, subject_id,
    dataset, true_label, pred_label, prob_class0, prob_class1, window_idx, ...
    """
    pred_dir = _final(experiment) / "collected" / "predictions"
    out = {}
    for ctx_label in CONTEXT_TO_MIN:
        fname = f"{task}_{head}_{ctx_label}_{split}.parquet"
        p = pred_dir / fname
        if p.exists():
            out[ctx_label] = pd.read_parquet(p)
    return out


def load_all_parquets(experiment: str, tasks: list, heads: list,
                      split: str = "test") -> dict:
    """Return nested dict [task][head][ctx] = DataFrame."""
    result = {}
    for task in tasks:
        result[task] = {}
        for head in heads:
            pqs = load_parquets(experiment, task, head, split)
            if pqs:
                result[task][head] = pqs
    return result


# ── Subject aggregation ───────────────────────────────────────────────────────

def subject_mean_probs(df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    """Mean-prob aggregate over ALL windows per (subject_id, dataset).

    Reproduces analyze_windows.py's K='all' convention, generalised to
    arbitrary num_classes (needed for age_class's 3-way AUROC).
    """
    prob_cols = [f"prob_class{c}" for c in range(num_classes)]
    rows = []
    for (sid, dset), grp in df.groupby(["subject_id", "dataset"], sort=False):
        mean_prob = grp[prob_cols].values.astype(np.float64).mean(axis=0)
        rows.append({
            "subject_id": sid, "dataset": dset,
            "true_label": int(grp["true_label"].iloc[0]),
            **{f"mean_prob_class{c}": mean_prob[c] for c in range(num_classes)},
        })
    return pd.DataFrame(rows)


def cohort_auroc_table(experiment: str, task: str, head: str, num_classes: int,
                       split: str = "test", min_subjects: int = 20) -> pd.DataFrame:
    """Per-cohort (+ pooled) AUROC vs context length for one task/head.

    Adds a group-by on the `dataset` column before scoring, on top of the
    paper's own mean-prob K='all' convention, so cohort AUROCs are directly
    comparable to analysis.csv's pooled numbers. Returns one row per
    (context, dataset), with dataset='POOLED' for the recombined estimate.
    Cells with <2 classes or <min_subjects subjects are NaN.
    """
    from sklearn.metrics import roc_auc_score

    def _auroc(sub: pd.DataFrame):
        if sub["true_label"].nunique() < 2 or len(sub) < min_subjects:
            return float("nan")
        y = sub["true_label"].values
        if num_classes == 2:
            return float(roc_auc_score(y, sub["mean_prob_class1"].values))
        P = sub[[f"mean_prob_class{c}" for c in range(num_classes)]].values
        try:
            return float(roc_auc_score(y, P, multi_class="ovr", average="macro"))
        except ValueError:
            return float("nan")

    records = []
    for ctx, df in load_parquets(experiment, task, head, split).items():
        subj = subject_mean_probs(df, num_classes)
        records.append({"task": task, "context": ctx,
                        "context_length_min": CONTEXT_TO_MIN[ctx],
                        "dataset": "POOLED", "n_subjects": len(subj),
                        "auroc": _auroc(subj)})
        for dset, sub in subj.groupby("dataset"):
            records.append({"task": task, "context": ctx,
                            "context_length_min": CONTEXT_TO_MIN[ctx],
                            "dataset": dset, "n_subjects": len(sub),
                            "auroc": _auroc(sub)})
    return pd.DataFrame(records)


def aggregate_windows(df: pd.DataFrame,
                      k: int | None = None,
                      prob_col: str = "prob_class1") -> pd.DataFrame:
    """Mean-pool prob_col over K evenly-spaced windows per subject.

    k=None means all windows.  Returns: subject_id, dataset, true_label,
    mean_prob, n_windows.
    """
    if prob_col not in df.columns:
        return pd.DataFrame()

    if k is not None:
        def _select(g):
            n = len(g)
            if n <= k:
                return g
            idx = np.linspace(0, n - 1, k, dtype=int)
            return g.iloc[idx]
        df = df.groupby("subject_id", group_keys=False).apply(_select)

    return (
        df.groupby(["subject_id", "dataset"])
        .agg(
            mean_prob=(prob_col, "mean"),
            n_windows=(prob_col, "count"),
            true_label=("true_label", "first"),
        )
        .reset_index()
    )
