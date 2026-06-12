"""
table_utils.py — Shared utilities for all make_tableN_*.py scripts.

Provides: task display names, context ordering, analysis.csv loader,
L* computation, markdown/LaTeX output helpers.
"""

from pathlib import Path
import pandas as pd

# ── Canonical task list (display name + default heads) ────────────────────────
SEQ2LABEL_TASKS = [
    ("sex_binary",                "Sex",                     ["lstm", "transformer"]),
    ("age_class",                 "Age group",               ["lstm", "transformer"]),
    ("bmi_binary",                "BMI (obese)",             ["lstm", "transformer"]),
    ("apnea_binary",              "Sleep apnea (AHI≥15)",   ["lstm", "transformer"]),
    ("sleep_efficiency_binary",   "Sleep efficiency",        ["lstm", "transformer"]),
    ("cvd_binary",                "CVD history",             ["lstm", "transformer"]),
    ("sleepiness_binary",         "Daytime sleepiness",      ["lstm", "transformer"]),
    ("depression_extreme_binary", "Depression (extreme)",    ["lstm"]),
    ("osa_binary_apples_postqc",  "OSA (APPLES)",            ["lstm"]),
    ("psqi_binary",               "Sleep quality (PSQI)",    ["lstm"]),
]

TASK_DISPLAY = {t: d for t, d, _ in SEQ2LABEL_TASKS}
TASK_ORDER   = [t for t, _, _ in SEQ2LABEL_TASKS]

# ── Context canonical ordering and minute conversions ─────────────────────────
CTX_ORDER = ["30s", "10m", "40m", "80m", "120m", "240m", "full_night"]

def ctx_to_min(s: str) -> float:
    if s.endswith("s"):   return float(s[:-1]) / 60
    if s.endswith("m"):   return float(s[:-1])
    if s.endswith("h"):   return float(s[:-1]) * 60
    if s == "full_night": return 480.0
    raise ValueError(f"Unknown context format: {s}")

CTX_MIN = {c: ctx_to_min(c) for c in CTX_ORDER}

def sort_contexts(ctxs):
    def _key(c):
        try:
            return ctx_to_min(c)
        except ValueError:
            return float("inf")
    return sorted(ctxs, key=_key)

# ── analysis.csv loader ───────────────────────────────────────────────────────

def load_analysis(collected_dir: Path, split: str = "test") -> pd.DataFrame:
    """Load analysis.csv and return rows for the requested split."""
    path = Path(collected_dir) / "analysis.csv"
    df = pd.read_csv(path)
    df = df[df["split"] == split].copy()
    df["k_num"] = pd.to_numeric(df["k"], errors="coerce")
    return df


def filter_tasks(df: pd.DataFrame, tasks: list | None, heads: list | None) -> pd.DataFrame:
    if tasks:
        df = df[df["task"].isin(tasks)]
    if heads:
        df = df[df["head"].isin(heads)]
    return df

# ── Core analysis helpers ─────────────────────────────────────────────────────

def best_row(df: pd.DataFrame, k_val, metric: str = "mean_prob_auroc"):
    """Return the row with the highest metric at the given k value."""
    if k_val == "all":
        sub = df[df["k"] == "all"]
    else:
        sub = df[df["k_num"] == float(k_val)]
    if sub.empty:
        return None
    return sub.loc[sub[metric].idxmax()]


def compute_lstar(df_task_head: pd.DataFrame, tolerance: float = 0.005,
                  metric: str = "mean_prob_auroc") -> str | None:
    """
    Find L*: the smallest context length where metric(L, k=all) >= max_metric - tolerance.
    Returns the context_length string, or None if insufficient data.
    """
    sub = df_task_head[df_task_head["k"] == "all"].copy()
    if sub.empty:
        return None
    max_val = sub[metric].max()
    threshold = max_val - tolerance
    eligible = sub[sub[metric] >= threshold]
    if eligible.empty:
        return None
    eligible = eligible.copy()
    eligible["_ctx_min"] = eligible["context_length"].map(ctx_to_min)
    return eligible.sort_values("_ctx_min").iloc[0]["context_length"]


def n_subjects(df_task_head: pd.DataFrame) -> int | None:
    """Return subject count (constant across k; read from k=1 rows)."""
    sub = df_task_head[df_task_head["k_num"] == 1]
    if sub.empty:
        return None
    return int(sub["n_subjects"].iloc[0])

# ── Output helpers ────────────────────────────────────────────────────────────

def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a GitHub-flavoured markdown table."""
    cols = list(df.columns)
    sep  = ["-" * max(len(c), 5) for c in cols]
    rows = [cols, sep] + [list(map(str, r)) for r in df.values]
    col_widths = [max(len(rows[i][j]) for i in range(len(rows))) for j in range(len(cols))]

    def fmt_row(row):
        return "| " + " | ".join(v.ljust(col_widths[j]) for j, v in enumerate(row)) + " |"

    lines = [fmt_row(rows[0])]
    lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")
    for row in rows[2:]:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def save_outputs(df: pd.DataFrame, out_dir: Path, stem: str,
                 latex_str: str | None = None, verbose: bool = True,
                 scratch_dir: Path | None = None):
    """Save CSV, markdown, and LaTeX to out_dir (repo). If scratch_dir is given,
    also save CSV, markdown, and LaTeX there (mirroring the repo copy)."""
    for dest, label in [(Path(out_dir), ""), (Path(scratch_dir), " (scratch)") if scratch_dir else (None, None)]:
        if dest is None:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        csv_path = dest / f"{stem}.csv"
        md_path  = dest / f"{stem}.md"
        df.to_csv(csv_path, index=False)
        md_path.write_text(df_to_markdown(df))
        if verbose:
            print(f"  saved CSV  → {csv_path}{label}")
            print(f"  saved MD   → {md_path}{label}")
        if latex_str:
            tex_path = dest / f"{stem}.tex"
            tex_path.write_text(latex_str)
            if verbose:
                print(f"  saved LaTeX→ {tex_path}{label}")


def latex_table(df: pd.DataFrame, caption: str, label: str,
                col_fmt: str | None = None) -> str:
    """Wrap a DataFrame in a LaTeX table environment using booktabs."""
    cols = list(df.columns)
    if col_fmt is None:
        col_fmt = "l" + "r" * (len(cols) - 1)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{label}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def fmt_auroc(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:.{digits}f}"
