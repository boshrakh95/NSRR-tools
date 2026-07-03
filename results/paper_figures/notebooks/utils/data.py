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
