"""Data loading helpers for exploratory figures (xfig_* notebooks).

Mirrors the structure of ../paper_figures/notebooks/utils/data.py but is
kept separate so it never pollutes the main pipeline.

Call set_root(WORKSPACE_ROOT) once per notebook before using any loader.
"""

from pathlib import Path
import pandas as pd
import numpy as np

_ROOT: Path | None = None

CONTEXT_TO_MIN = {
    "30s": 0.5, "10m": 10.0, "40m": 40.0,
    "80m": 80.0, "120m": 120.0, "240m": 240.0,
}
CTX_ORDER = list(CONTEXT_TO_MIN.keys())
MIN_TO_CTX = {v: k for k, v in CONTEXT_TO_MIN.items()}


def set_root(root: Path):
    global _ROOT
    _ROOT = Path(root).resolve()


def _final(experiment: str) -> Path:
    assert _ROOT is not None, "Call set_root(WORKSPACE_ROOT) before loading."
    return _ROOT / "final_results" / experiment


# ── analysis.csv ──────────────────────────────────────────────────────────────

def load_analysis(experiment: str = "phase0_v3",
                  split: str = "test",
                  k: str | None = "all") -> pd.DataFrame:
    """Return analysis.csv filtered to split and optionally a single k value."""
    p = _final(experiment) / "collected" / "analysis.csv"
    df = pd.read_csv(p)
    # Ensure context_length_min exists
    if "context_length_min" not in df.columns:
        df["context_length_min"] = df["context_length"].map(
            lambda s: CONTEXT_TO_MIN.get(str(s).strip()))
    # Drop stale split label produced by earlier pipeline versions
    df = df[df["split"] == split]
    # Drop ablation rows (run_tag not NaN) unless caller passes include_abl=True
    if "run_tag" in df.columns:
        df = df[df["run_tag"].isna()].drop(columns=["run_tag"], errors="ignore")
    if k is not None:
        df = df[df["k"].astype(str) == str(k)]
    return df.reset_index(drop=True)


def load_analysis_all_k(experiment: str = "phase0_v3",
                        split: str = "test") -> pd.DataFrame:
    """Load analysis.csv with all K values (for K-sweep analysis)."""
    p = _final(experiment) / "collected" / "analysis.csv"
    df = pd.read_csv(p)
    if "context_length_min" not in df.columns:
        df["context_length_min"] = df["context_length"].map(
            lambda s: CONTEXT_TO_MIN.get(str(s).strip()))
    df = df[df["split"] == split]
    if "run_tag" in df.columns:
        df = df[df["run_tag"].isna()].drop(columns=["run_tag"], errors="ignore")
    # Numeric k where possible; keep 'all' as string
    def _knum(x):
        s = str(x)
        if s == "all":
            return np.inf
        try:
            return float(s)
        except ValueError:
            return np.nan
    df["k_num"] = df["k"].apply(_knum)
    return df.reset_index(drop=True)


# ── heatmap DataFrames ────────────────────────────────────────────────────────

def load_heatmap(experiment: str, task: str, head: str,
                 split: str = "test") -> pd.DataFrame:
    """Load heatmap_df_{split}.csv for a single task/head combo.

    Returns empty DataFrame if file does not exist.
    """
    p = (_final(experiment) / "inference" / f"{task}_{head}"
         / f"heatmap_df_{split}.csv")
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "context_length_min" not in df.columns and "context_label" in df.columns:
        df["context_length_min"] = df["context_label"].map(
            lambda s: CONTEXT_TO_MIN.get(str(s).strip()))
    return df.sort_values(["context_length_min", "k"]).reset_index(drop=True)


# ── parquets ──────────────────────────────────────────────────────────────────

def load_parquets(experiment: str, task: str, head: str,
                  split: str = "test") -> dict[str, pd.DataFrame]:
    """Return {context_label: DataFrame} from collected/predictions/."""
    pred_dir = _final(experiment) / "collected" / "predictions"
    out = {}
    for ctx_label in CTX_ORDER:
        fname = f"{task}_{head}_{ctx_label}_{split}.parquet"
        p = pred_dir / fname
        if p.exists():
            out[ctx_label] = pd.read_parquet(p)
    return out


def load_modality_table(nsrr_tools_root: Path) -> pd.DataFrame:
    """Load table6_modality.csv from results/tables/."""
    p = Path(nsrr_tools_root) / "results" / "tables" / "table6_modality.csv"
    return pd.read_csv(p)


# ── subject-level aggregation ─────────────────────────────────────────────────

def subject_predictions(parquets: dict[str, pd.DataFrame],
                        prob_col: str = "prob_class1",
                        k: int | None = None) -> pd.DataFrame:
    """Aggregate per-window parquets to one row per (subject, context).

    Returns a DataFrame with columns:
      context_label, subject_id, dataset, true_label, mean_prob, std_prob,
      n_windows, pred_correct (mean_prob > 0.5 == true_label)
    """
    rows = []
    for ctx_label, df in parquets.items():
        if prob_col not in df.columns:
            continue
        df = df.copy()
        # Optionally subsample to K windows per subject
        if k is not None:
            def _sel(g):
                n = len(g)
                if n <= k:
                    return g
                idx = np.linspace(0, n - 1, k, dtype=int)
                return g.iloc[idx]
            df = df.groupby("subject_id", group_keys=False).apply(_sel)

        grp = df.groupby(["subject_id", "dataset"])
        agg = grp.agg(
            mean_prob=(prob_col, "mean"),
            std_prob=(prob_col, "std"),
            n_windows=(prob_col, "count"),
            true_label=("true_label", "first"),
        ).reset_index()
        agg["context_label"] = ctx_label
        agg["context_min"] = CONTEXT_TO_MIN[ctx_label]
        agg["pred_correct"] = (agg["mean_prob"] > 0.5) == agg["true_label"].astype(bool)
        rows.append(agg)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def subject_correctness_matrix(parquets: dict[str, pd.DataFrame],
                                contexts: list[str] | None = None,
                                prob_col: str = "prob_class1") -> pd.DataFrame:
    """Return (n_subjects × n_contexts) DataFrame of correctness (0/1).

    Only subjects that appear in ALL requested contexts are kept.
    """
    if contexts is None:
        contexts = CTX_ORDER
    available = [c for c in contexts if c in parquets]

    per_ctx = {}
    for ctx in available:
        df = parquets[ctx]
        if prob_col not in df.columns:
            continue
        agg = (df.groupby("subject_id")
               .agg(mean_prob=(prob_col, "mean"),
                    true_label=("true_label", "first"))
               .reset_index())
        agg["correct"] = (agg["mean_prob"] > 0.5) == agg["true_label"].astype(bool)
        per_ctx[ctx] = agg.set_index("subject_id")["correct"]

    if not per_ctx:
        return pd.DataFrame()

    mat = pd.DataFrame(per_ctx)
    mat = mat.dropna()  # keep only subjects present in all contexts
    return mat
