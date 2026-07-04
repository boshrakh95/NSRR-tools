"""Panel functions for exploratory figures (xfig_* notebooks).

Naming: xfig_NN_* mirrors the idea numbers in docs/NEW_PLOT_IDEAS.md.

Each function draws into the Axes passed as the first argument.
Layout and saving are handled by the notebook.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist

from .data_explore import CONTEXT_TO_MIN, CTX_ORDER, MIN_TO_CTX

# ── shared constants ──────────────────────────────────────────────────────────

FONT_BASE  = 8
FONT_ANNOT = 6
FONT_LABEL = 7

TASK_LABEL = {
    "sex_binary":                "Sex",
    "age_class":                 "Age",
    "apnea_binary":              "Apnea",
    "bmi_binary":                "BMI",
    "sleep_efficiency_binary":   "Sleep Eff.",
    "depression_extreme_binary": "Depression",
    "osa_binary_apples_postqc":  "OSA (APPLES)",
    "cvd_binary":                "CVD",
}

MAIN_TASKS = ["sex_binary", "bmi_binary", "age_class",
              "sleep_efficiency_binary", "apnea_binary"]

CTX_MIN_ORDER = [CONTEXT_TO_MIN[c] for c in CTX_ORDER]

MODALITY_LABELS = ["No BAS", "No RESP", "No EKG", "Cardio only", "BAS only"]
MODALITY_COLORS = ["#E86A33", "#3A7EBF", "#44A15E", "#C94040", "#7B5EA7"]

HEAD_COLOR = {
    "lstm":        "#3A7EBF",
    "transformer": "#E86A33",
    "mean_pool":   "#44A15E",
}

_VIRIDIS = sns.color_palette("viridis", 6)


def _spine_clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _log_ctx_axis(ax, xs=None):
    ax.set_xscale("log")
    ticks = sorted(xs.tolist() if hasattr(xs, "tolist") else list(xs)) if xs is not None else CTX_MIN_ORDER
    ax.set_xticks(ticks)
    ax.set_xticklabels([MIN_TO_CTX.get(float(x), f"{x:.0f}m") for x in ticks],
                       fontsize=FONT_BASE)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_02 — Clinical Threshold Unlock Map
# ═══════════════════════════════════════════════════════════════════════════════

def threshold_unlock_heatmap(ax, analysis_df: pd.DataFrame,
                              tasks: list[str],
                              head: str = "transformer",
                              thresholds: list[float] | None = None,
                              metric: str = "mean_prob_auroc"):
    """Heatmap: first context length where AUROC crosses each threshold.

    analysis_df: pre-filtered to split='test', k='all', run_tag NaN.
    Rows = AUROC thresholds; Columns = tasks.
    Color = context length needed (darker = shorter = easier); gray = never reached.
    """
    if thresholds is None:
        thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]

    ctx_vals = sorted(CONTEXT_TO_MIN.values())  # [0.5, 10, 40, 80, 120, 240]
    n_ctx = len(ctx_vals)

    # Build matrix: rows=thresholds (hi→lo), cols=tasks
    mat = np.full((len(thresholds), len(tasks)), np.nan)

    for j, task in enumerate(tasks):
        sub = (analysis_df[
                   (analysis_df["task"] == task) &
                   (analysis_df["head"] == head) &
                   analysis_df[metric].notna()
               ]
               .sort_values("context_length_min"))
        for i, thr in enumerate(thresholds):
            hit = sub[sub[metric] >= thr]
            if not hit.empty:
                mat[i, j] = hit["context_length_min"].iloc[0]

    # Reverse thresholds so highest threshold is on top
    mat = mat[::-1]
    thresh_labels = [f"≥{t:.2f}" for t in reversed(thresholds)]

    # Colormap: viridis mapped to ctx_vals (log scale); NaN → gray
    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad(color="#cccccc")

    log_mat = np.where(np.isnan(mat), np.nan, np.log10(np.maximum(mat, 0.1)))
    log_ticks = np.log10(ctx_vals)

    im = ax.imshow(log_mat, aspect="auto", cmap=cmap,
                   vmin=log_ticks[0], vmax=log_ticks[-1])

    # Annotate cells
    for i in range(len(thresholds)):
        for j in range(len(tasks)):
            val = mat[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=FONT_ANNOT, color="#555555")
            else:
                lbl = MIN_TO_CTX.get(float(val), f"{val:.0f}m")
                ax.text(j, i, lbl, ha="center", va="center",
                        fontsize=FONT_ANNOT, color="white" if val >= 40 else "black",
                        fontweight="bold")

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_LABEL.get(t, t) for t in tasks],
                       fontsize=FONT_BASE, rotation=30, ha="right")
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels(thresh_labels, fontsize=FONT_BASE)
    ax.set_xlabel("Task", fontsize=FONT_LABEL)
    ax.set_ylabel("Required AUROC", fontsize=FONT_LABEL)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([MIN_TO_CTX.get(v, f"{v}m") for v in ctx_vals],
                        fontsize=FONT_ANNOT)
    cbar.set_label("Context needed", fontsize=FONT_ANNOT)

    _spine_clean(ax)
    return im


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_04 — Deployment Scenario Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def deployment_scenario_panel(ax, heatmap_df: pd.DataFrame, task: str,
                               budgets_min: list[float] | None = None,
                               targets: list[float] | None = None,
                               metric: str = "auroc"):
    """Grid: rows=target AUROC, cols=budget (total minutes); cell=best achievable AUROC.

    heatmap_df: from load_heatmap(), columns context_length_min, k, auroc.
    For each (budget, target), we find the (L, K) with L*K <= budget that
    maximises AUROC, then colour by whether the target is met.
    """
    if budgets_min is None:
        budgets_min = [30, 60, 120, 240, 480, 960]
    if targets is None:
        targets = [0.70, 0.75, 0.80, 0.85, 0.90]

    if heatmap_df.empty:
        ax.text(0.5, 0.5, f"No heatmap data\nfor {task}",
                ha="center", va="center", transform=ax.transAxes, fontsize=FONT_BASE)
        return

    df = heatmap_df.copy()
    df["k_num"] = pd.to_numeric(df["k"], errors="coerce")
    df = df.dropna(subset=["k_num", metric])
    df["budget"] = df["context_length_min"] * df["k_num"]

    n_rows, n_cols = len(targets), len(budgets_min)
    best_auroc = np.full((n_rows, n_cols), np.nan)
    best_label = [[""]*n_cols for _ in range(n_rows)]

    for ci, budget in enumerate(budgets_min):
        feasible = df[df["budget"] <= budget + 1e-3]
        if feasible.empty:
            continue
        best_row = feasible.loc[feasible[metric].idxmax()]
        auroc_val = best_row[metric]
        ctx_lbl = MIN_TO_CTX.get(float(best_row["context_length_min"]), "?")
        k_val = int(best_row["k_num"])
        for ri, tgt in enumerate(reversed(targets)):
            best_auroc[ri, ci] = auroc_val
            if auroc_val >= tgt:
                best_label[ri][ci] = f"{ctx_lbl}\nK={k_val}"
            else:
                best_label[ri][ci] = f"✗\n{auroc_val:.2f}"

    vmin = min(t - 0.02 for t in targets)
    vmax = max(targets) + 0.02
    cmap = plt.cm.RdYlGn
    im = ax.imshow(best_auroc, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax)

    for ri in range(n_rows):
        for ci in range(n_cols):
            txt = best_label[ri][ci]
            if txt:
                val = best_auroc[ri, ci]
                col = "white" if (not np.isnan(val) and (val > vmin + 0.7*(vmax-vmin))) else "black"
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=FONT_ANNOT, color=col)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"{b:g}m" for b in budgets_min], fontsize=FONT_BASE)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"≥{t:.2f}" for t in reversed(targets)], fontsize=FONT_BASE)
    ax.set_xlabel("Recording budget (total min)", fontsize=FONT_LABEL)
    ax.set_ylabel("Required AUROC", fontsize=FONT_LABEL)

    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label("Best achievable AUROC", fontsize=FONT_ANNOT)
    cbar.ax.tick_params(labelsize=FONT_ANNOT)
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_06 — Modality Radar Chart
# ═══════════════════════════════════════════════════════════════════════════════

def modality_radar_panel(ax, modality_df: pd.DataFrame,
                          tasks: list[str] | None = None):
    """Polar/radar chart: each axis = modality importance (-ΔAUROC) per task.

    modality_df: table6_modality.csv loaded as DataFrame.
    Axes: [No BAS, No RESP, No EKG, Cardio only, BAS only] (inverted Δ = importance).
    One polygon per task.
    """
    if tasks is None:
        tasks = list(modality_df.index) if modality_df.index.dtype == object else None
        if tasks is None:
            tasks = modality_df["Task"].tolist() if "Task" in modality_df.columns else []

    # Column names for Δ values in table6_modality.csv
    delta_cols = ["Δ(No BAS)", "Δ(No RESP)", "Δ(No EKG)",
                  "Δ(Cardio only)", "Δ(BAS only)"]
    spoke_labels = ["No BAS", "No RESP", "No EKG", "Cardio\nonly", "BAS\nonly"]

    n_spokes = len(delta_cols)
    angles = np.linspace(0, 2 * np.pi, n_spokes, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    task_col = "Task" if "Task" in modality_df.columns else modality_df.columns[0]
    palette = sns.color_palette("tab10", len(tasks))

    for task_idx, task in enumerate(tasks):
        # Match task in table (partial match for display names)
        row = modality_df[modality_df[task_col].str.lower().str.contains(
            TASK_LABEL.get(task, task).lower()[:4], na=False)]
        if row.empty:
            # Try exact task key
            row = modality_df[modality_df[task_col] == task]
        if row.empty:
            continue
        row = row.iloc[0]

        try:
            vals = [abs(float(str(row[c]).replace("+", ""))) for c in delta_cols]
        except (KeyError, ValueError):
            continue
        vals += vals[:1]

        color = palette[task_idx]
        ax.plot(angles, vals, color=color, linewidth=1.5,
                label=TASK_LABEL.get(task, task))
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(spoke_labels, fontsize=FONT_BASE)
    ax.set_ylabel("")
    ax.set_yticklabels([])
    ax.grid(True, linestyle="--", alpha=0.4)

    # Radial grid labels
    max_val = 0.12
    for r in [0.03, 0.06, 0.09, 0.12]:
        ax.text(0, r, f"−{r:.2f}", fontsize=FONT_ANNOT, ha="center", va="center",
                color="#888888")
    ax.set_ylim(0, max_val)

    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_08 — Night Fingerprint Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def night_fingerprint_panel(ax, parquets: dict[str, pd.DataFrame],
                             subject_id: str,
                             prob_col: str = "prob_class1",
                             contexts: list[str] | None = None,
                             n_bins: int = 30):
    """Heatmap: rows = context lengths, columns = normalised night position.

    Each cell = mean predicted probability in that position bin for that context.
    """
    if contexts is None:
        contexts = [c for c in CTX_ORDER if c in parquets]

    rows, ctx_labels = [], []
    for ctx in contexts:
        if ctx not in parquets or prob_col not in parquets[ctx].columns:
            continue
        df = parquets[ctx]
        sub = df[df["subject_id"] == subject_id].copy()
        if sub.empty:
            continue
        # Normalise window_idx to [0, 1]
        n_wins = sub["window_idx"].max() + 1
        sub["pos_norm"] = sub["window_idx"] / max(n_wins - 1, 1)
        sub["pos_bin"] = pd.cut(sub["pos_norm"], bins=n_bins,
                                labels=False, include_lowest=True)
        row = sub.groupby("pos_bin")[prob_col].mean().reindex(
            range(n_bins)).values
        rows.append(row)
        ctx_labels.append(ctx)

    if not rows:
        ax.text(0.5, 0.5, f"Subject {subject_id} not found",
                ha="center", va="center", transform=ax.transAxes)
        return

    mat = np.array(rows)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=0.0, vmax=1.0, interpolation="nearest")

    ax.set_yticks(range(len(ctx_labels)))
    ax.set_yticklabels(ctx_labels, fontsize=FONT_BASE)
    ax.set_xlabel("Night position (normalised)", fontsize=FONT_LABEL)
    ax.set_ylabel("Context length", fontsize=FONT_LABEL)
    ax.set_xticks([0, n_bins // 4, n_bins // 2, 3 * n_bins // 4, n_bins - 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=FONT_ANNOT)

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.9)
    cbar.set_label("prob(positive)", fontsize=FONT_ANNOT)
    cbar.ax.tick_params(labelsize=FONT_ANNOT)
    _spine_clean(ax)
    return im


def pick_representative_subjects(parquets: dict[str, pd.DataFrame],
                                  prob_col: str = "prob_class1",
                                  n_per_type: int = 1) -> dict[str, list[str]]:
    """Pick subjects of 4 types: always-correct, always-wrong, pos-sensitive, neg-sensitive.

    Returns dict with keys: 'always_correct', 'always_wrong',
    'context_sensitive_pos', 'context_sensitive_neg'.
    """
    from .data_explore import subject_correctness_matrix
    mat = subject_correctness_matrix(parquets, prob_col=prob_col)
    if mat.empty:
        return {}

    n_ctx = mat.shape[1]
    n_correct = mat.sum(axis=1)

    result = {}
    # Always correct: correct at all contexts
    ac = mat[n_correct == n_ctx].index.tolist()
    result["always_correct"] = ac[:n_per_type]

    # Always wrong: correct at 0 contexts
    aw = mat[n_correct == 0].index.tolist()
    result["always_wrong"] = aw[:n_per_type]

    # Context sensitive: correct at few early contexts only → becomes correct later
    cs = mat[(n_correct >= 2) & (n_correct <= n_ctx - 1)].copy()
    if not cs.empty:
        # Prefer subjects that go from wrong to right (improving with context)
        improving = cs.apply(lambda row: row.values.tolist(), axis=1)
        imp_ids = [idx for idx, vals in improving.items()
                   if vals[0] == 0 and vals[-1] == 1]
        result["context_sensitive_pos"] = (imp_ids or cs.index.tolist())[:n_per_type]

        det_ids = [idx for idx, vals in improving.items()
                   if vals[0] == 1 and vals[-1] == 0]
        result["context_sensitive_neg"] = (det_ids or cs.index.tolist())[:n_per_type]

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_12 — Subject Prediction Stability Grid
# ═══════════════════════════════════════════════════════════════════════════════

def subject_stability_heatmap(ax, parquets: dict[str, pd.DataFrame],
                               prob_col: str = "prob_class1",
                               contexts: list[str] | None = None,
                               max_subjects: int = 200,
                               k_agg: int | None = None):
    """Heatmap: rows=subjects (sorted), cols=contexts, color=mean_prob.

    Subjects are sorted first by true_label then by prediction entropy
    (most stable at bottom, most variable at top within each label group).
    """
    if contexts is None:
        contexts = [c for c in CTX_ORDER if c in parquets]
    available = [c for c in contexts if c in parquets]

    # Build subject × context DataFrame of mean_prob
    per_ctx = {}
    true_labels = {}
    for ctx in available:
        df = parquets[ctx].copy()
        if prob_col not in df.columns:
            continue
        if k_agg is not None:
            def _sel(g):
                n = len(g)
                if n <= k_agg:
                    return g
                idx = np.linspace(0, n - 1, k_agg, dtype=int)
                return g.iloc[idx]
            df = df.groupby("subject_id", group_keys=False).apply(_sel)
        agg = (df.groupby("subject_id")
               .agg(mean_prob=(prob_col, "mean"),
                    true_label=("true_label", "first"))
               .reset_index())
        per_ctx[ctx] = agg.set_index("subject_id")["mean_prob"]
        for _, row in agg.iterrows():
            true_labels[row["subject_id"]] = row["true_label"]

    if not per_ctx:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    mat = pd.DataFrame(per_ctx).dropna()

    # Sort: by true_label then by entropy across contexts
    mat["true_label"] = mat.index.map(lambda s: true_labels.get(s, 0))
    mat["entropy"] = mat[available].apply(
        lambda row: -np.sum([p * np.log(p + 1e-9) + (1-p) * np.log(1-p+1e-9)
                             for p in row.values]), axis=1)
    mat = mat.sort_values(["true_label", "entropy"])

    # Subsample if too many subjects
    if len(mat) > max_subjects:
        neg = mat[mat["true_label"] == 0].iloc[:max_subjects // 2]
        pos = mat[mat["true_label"] == 1].iloc[:max_subjects // 2]
        mat = pd.concat([neg, pos])

    true_lab = mat["true_label"].values
    data = mat[available].values  # (n_subjects, n_contexts)

    im = ax.imshow(data, aspect="auto", cmap="RdBu_r",
                   vmin=0.0, vmax=1.0, interpolation="nearest",
                   origin="lower")

    # Dividing line between neg and pos
    n_neg = int((true_lab == 0).sum())
    if 0 < n_neg < len(true_lab):
        ax.axhline(n_neg - 0.5, color="black", linewidth=1.5, linestyle="--")
        ax.text(-0.5, n_neg / 2, "neg", fontsize=FONT_ANNOT,
                ha="right", va="center", color="#333333")
        ax.text(-0.5, n_neg + (len(true_lab) - n_neg) / 2, "pos",
                fontsize=FONT_ANNOT, ha="right", va="center", color="#333333")

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(available, fontsize=FONT_BASE)
    ax.set_xlabel("Context length", fontsize=FONT_LABEL)
    ax.set_ylabel(f"Test subjects (N={len(data)}, sorted)", fontsize=FONT_LABEL)
    ax.set_yticks([])

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.02, shrink=0.9)
    cbar.set_label("Mean prob(positive)", fontsize=FONT_ANNOT)
    cbar.ax.tick_params(labelsize=FONT_ANNOT)
    _spine_clean(ax)
    return im


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_14 — Task Similarity Clustermap (standalone figure, not an axes panel)
# ═══════════════════════════════════════════════════════════════════════════════

def task_clustermap(analysis_df: pd.DataFrame,
                    tasks: list[str],
                    head: str = "lstm",
                    metric: str = "mean_prob_auroc",
                    figsize: tuple[float, float] = (7.0, 4.0)):
    """Return a seaborn ClusterGrid.

    Rows = tasks, columns = context lengths; values = AUROC.
    Dendrogram clusters tasks with similar saturation curve shapes.
    """
    sub = analysis_df[
        (analysis_df["head"] == head) &
        (analysis_df["task"].isin(tasks)) &
        analysis_df[metric].notna()
    ]

    pivot = (sub.pivot_table(index="task", columns="context_length_min",
                              values=metric, aggfunc="mean")
             .reindex(columns=sorted(CONTEXT_TO_MIN.values())))

    # Rename columns to context labels
    pivot.columns = [MIN_TO_CTX.get(c, f"{c}m") for c in pivot.columns]
    pivot.index = [TASK_LABEL.get(t, t) for t in pivot.index]

    pivot = pivot.dropna(how="all").fillna(pivot.mean())

    g = sns.clustermap(
        pivot,
        cmap="YlOrRd",
        figsize=figsize,
        linewidths=0.4,
        linecolor="#dddddd",
        annot=True,
        fmt=".2f",
        annot_kws={"size": FONT_ANNOT},
        cbar_pos=(0.02, 0.85, 0.03, 0.12),
        row_cluster=True,
        col_cluster=False,   # keep context lengths in temporal order
        yticklabels=True,
        xticklabels=True,
    )
    g.ax_heatmap.set_xlabel("Context length", fontsize=FONT_LABEL)
    g.ax_heatmap.set_ylabel("")
    g.cax.tick_params(labelsize=FONT_ANNOT)
    g.fig.suptitle(f"Task similarity by saturation curve ({head.upper()})",
                   fontsize=FONT_BASE, y=1.01)
    return g


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_19 — Modality Ablation Clustermap (standalone figure)
# ═══════════════════════════════════════════════════════════════════════════════

def ablation_clustermap(modality_df: pd.DataFrame,
                         figsize: tuple[float, float] = (7.0, 4.0)):
    """Diverging clustermap of ΔAUROC values from table6_modality.csv.

    Rows = tasks, columns = ablation conditions.
    Red = harmful removal (large drop), white = neutral, blue = slight improvement.
    """
    # Rename columns if needed
    task_col = "Task" if "Task" in modality_df.columns else modality_df.columns[0]
    delta_map = {
        "Δ(No BAS)":      "No BAS",
        "Δ(No RESP)":     "No RESP",
        "Δ(No EKG)":      "No EKG",
        "Δ(Cardio only)": "Cardio only",
        "Δ(BAS only)":    "BAS only",
    }

    existing_cols = {k: v for k, v in delta_map.items() if k in modality_df.columns}
    plot_df = modality_df[[task_col] + list(existing_cols.keys())].copy()
    plot_df = plot_df.rename(columns={task_col: "Task", **existing_cols})
    plot_df = plot_df.set_index("Task")

    # Convert to float (values may be strings like "+0.010")
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")

    max_abs = plot_df.abs().max().max()

    g = sns.clustermap(
        plot_df,
        cmap="RdBu",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        figsize=figsize,
        linewidths=0.5,
        linecolor="#dddddd",
        annot=True,
        fmt=".3f",
        annot_kws={"size": FONT_BASE},
        cbar_pos=(0.02, 0.80, 0.03, 0.15),
        row_cluster=True,
        col_cluster=True,
        yticklabels=True,
        xticklabels=True,
    )
    g.ax_heatmap.set_xlabel("Ablation condition", fontsize=FONT_LABEL)
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.tick_params(axis="x", labelsize=FONT_BASE)
    g.ax_heatmap.tick_params(axis="y", labelsize=FONT_BASE)
    g.cax.tick_params(labelsize=FONT_ANNOT)
    g.fig.suptitle("Modality importance: ΔAUROC (red = harmful removal)",
                   fontsize=FONT_BASE, y=1.01)
    return g


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_25 — SOTA Comparison Bubble Chart
# ═══════════════════════════════════════════════════════════════════════════════

SOTA_DATA = {
    # (method, task_display, auroc, pretraining_hours, uses_eeg, note)
    "SleepFounder_osa":  ("SleepFounder", "OSA",        0.917, 800_000, False, "cardio-only"),
    "SleepFounder_sex":  ("SleepFounder", "Sex",        0.850, 800_000, False, "cardio-only"),
    "OSF_coronary":      ("OSF",          "CVD",        0.681, 166_500, True,  "linear probe"),
    "SleepMaMi_staging": ("SleepMaMi",   "Staging (acc)", 0.819, 158_028, True, "5-class acc"),
    # Our results (LSTM unless noted)
    "Ours_apnea_lstm":   ("Ours (LSTM)",  "OSA/Apnea",  0.832, 100_000, True,  "fast-ch"),
    "Ours_apnea_tr":     ("Ours (Transf.)","OSA/Apnea", 0.857, 100_000, True,  "fast-ch"),
    "Ours_sex_lstm":     ("Ours (LSTM)",  "Sex",        0.872, 100_000, True,  "fast-ch"),
    "Ours_sex_tr":       ("Ours (Transf.)","Sex",       0.910, 100_000, True,  "fast-ch"),
    "Ours_apnea_full":   ("Ours (full-ch)","OSA/Apnea", 0.874, 100_000, True,  "full-ch"),
    "Ours_sex_full":     ("Ours (full-ch)","Sex",       0.887, 100_000, True,  "full-ch"),
}

METHOD_STYLE = {
    "SleepFounder":    {"color": "#C94040", "marker": "D"},
    "OSF":             {"color": "#7B5EA7", "marker": "^"},
    "SleepMaMi":       {"color": "#44A15E", "marker": "s"},
    "Ours (LSTM)":     {"color": "#3A7EBF", "marker": "o"},
    "Ours (Transf.)":  {"color": "#E86A33", "marker": "o"},
    "Ours (full-ch)":  {"color": "#E86A33", "marker": "*"},
}


def sota_bubble_panel(ax, our_analysis_df: pd.DataFrame | None = None,
                       our_tasks: dict | None = None):
    """Bubble chart: x = pretraining hours (log), y = AUROC.

    Bubble size proportional to pretraining scale. Color/marker by method.
    EEG methods use filled markers; non-EEG methods use open markers.

    our_analysis_df: optional; if provided, extracts our numbers directly.
    our_tasks: dict mapping task_key → (display_name, context) to override hardcoded.
    """
    ax.set_xscale("log")

    seen_methods = set()
    for key, (method, task_disp, auroc, hours, eeg, note) in SOTA_DATA.items():
        style = METHOD_STYLE.get(method, {"color": "gray", "marker": "o"})
        fill = style["color"] if eeg else "none"
        edge = style["color"]
        size = max(40, hours / 2500)   # bubble size by pretraining scale

        sc = ax.scatter(hours, auroc,
                        s=size, c=fill, edgecolors=edge,
                        marker=style["marker"],
                        linewidths=1.5, zorder=3,
                        label=method if method not in seen_methods else "_nolegend_")
        seen_methods.add(method)
        # Annotate
        ax.annotate(f"{task_disp}\n{auroc:.3f}", (hours, auroc),
                    textcoords="offset points", xytext=(4, 2),
                    fontsize=FONT_ANNOT, color=edge, alpha=0.85)

    ax.set_xlabel("Pre-training hours (log scale)", fontsize=FONT_LABEL)
    ax.set_ylabel("AUROC", fontsize=FONT_LABEL)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0.62, 0.96)

    # Vertical lines for training scale
    for hrs, lbl in [(100_000, "SleepFM\n(100k h)"),
                     (800_000, "SleepFounder\n(800k h)")]:
        ax.axvline(hrs, color="#aaaaaa", linestyle=":", linewidth=0.8, zorder=1)
        ax.text(hrs, 0.635, lbl, fontsize=FONT_ANNOT - 1, color="#888888",
                ha="center", va="bottom")

    # Legend for EEG vs no-EEG
    ours_patch = mpatches.Patch(facecolor="#3A7EBF", label="Uses EEG (filled)")
    noeeg_patch = mpatches.Patch(facecolor="none", edgecolor="#C94040",
                                  linewidth=1.5, label="No EEG (open)")
    warning = mpatches.Patch(color="none",
                              label="⚠ Different eval protocols — not directly comparable")
    ax.legend(handles=[ours_patch, noeeg_patch, warning],
              fontsize=FONT_ANNOT, frameon=False, loc="lower right")

    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_28 — Saturation Curves with Bootstrap Significance Markers
# ═══════════════════════════════════════════════════════════════════════════════

def saturation_significance_panel(ax, analysis_df: pd.DataFrame,
                                   task: str,
                                   head: str = "transformer",
                                   metric: str = "mean_prob_auroc",
                                   ci_lo_col: str = "mean_prob_auroc_ci_lo",
                                   ci_hi_col: str = "mean_prob_auroc_ci_hi"):
    """Saturation curve with 95% CI bands and significance markers.

    Significance markers (**/ns) placed between adjacent context pairs
    where CIs do/don't overlap.
    """
    sub = (analysis_df[
               (analysis_df["task"] == task) &
               (analysis_df["head"] == head) &
               analysis_df[metric].notna()
           ]
           .sort_values("context_length_min"))

    if sub.empty:
        ax.text(0.5, 0.5, f"No data: {task}/{head}",
                ha="center", va="center", transform=ax.transAxes)
        return

    xs = sub["context_length_min"].values
    ys = sub[metric].values

    has_ci = (ci_lo_col in sub.columns and ci_hi_col in sub.columns
              and sub[ci_lo_col].notna().any())

    style = {"color": "#E86A33" if head == "transformer" else "#3A7EBF",
             "marker": "o", "linewidth": 1.8, "markersize": 4}

    ax.plot(xs, ys, **{k: v for k, v in style.items()
                       if k in ["color", "linewidth"]},
            marker=style["marker"], markersize=style["markersize"], zorder=3)

    if has_ci:
        los = sub[ci_lo_col].values
        his = sub[ci_hi_col].values
        ax.fill_between(xs, los, his, alpha=0.20, color=style["color"])

        # Significance markers between adjacent pairs
        for i in range(len(xs) - 1):
            lo_i, hi_i = los[i], his[i]
            lo_j, hi_j = los[i + 1], his[i + 1]
            # Non-overlapping CIs → significant
            sig = (lo_j > hi_i) or (lo_i > hi_j)
            marker_x = (xs[i] * xs[i + 1]) ** 0.5   # geometric mean (log mid)
            marker_y = max(his[i] if not np.isnan(his[i]) else ys[i],
                           his[i+1] if not np.isnan(his[i+1]) else ys[i+1]) + 0.004
            txt = "**" if sig else "ns"
            col = "#333333" if sig else "#aaaaaa"
            ax.text(marker_x, marker_y, txt, ha="center", va="bottom",
                    fontsize=FONT_ANNOT, color=col)
    else:
        ax.text(0.5, 0.02, "CI bands pending (run --bootstrap)",
                ha="center", va="bottom", transform=ax.transAxes,
                fontsize=FONT_ANNOT, color="#888888", style="italic")
        # Annotate values instead
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.003, f"{y:.3f}", ha="center", va="bottom",
                    fontsize=FONT_ANNOT - 1, color="#555555")

    _log_ctx_axis(ax, xs)
    ax.set_xlabel("Context length", fontsize=FONT_LABEL)
    ax.set_ylabel("AUROC", fontsize=FONT_LABEL)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    y_lo = max(0.5, ys.min() - 0.05)
    y_hi = min(1.0, ys.max() + 0.04)
    ax.set_ylim(y_lo, y_hi)
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_30 — Waterfall Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def waterfall_panel(ax, analysis_df: pd.DataFrame,
                    task: str,
                    metric: str = "mean_prob_auroc"):
    """Waterfall: decompose AUROC gains into aggregation, context, architecture.

    Steps:
      Start : MeanPool @ 30s, K=1
      +Agg  : MeanPool @ 30s, K=5
      +Ctx  : MeanPool @ 240m, K=5
      +Arch : Transformer @ 240m, K=5
    """
    def _get(head, ctx, k):
        k_str = str(k)
        sub = analysis_df[
            (analysis_df["task"] == task) &
            (analysis_df["head"] == head) &
            (analysis_df["context_length"] == ctx) &
            (analysis_df["k"].astype(str) == k_str) &
            analysis_df[metric].notna()
        ]
        return float(sub[metric].iloc[0]) if not sub.empty else np.nan

    v_start = _get("mean_pool", "30s", 1)
    v_agg   = _get("mean_pool", "30s", 5)
    v_ctx   = _get("mean_pool", "240m", 5)
    v_arch  = _get("transformer", "240m", 5)

    if any(np.isnan(v) for v in [v_start, v_agg, v_ctx, v_arch]):
        # Fall back to K=all for aggregation step
        v_agg_all = _get("mean_pool", "30s", "all")
        if not np.isnan(v_agg_all):
            v_agg = v_agg_all

    # Compute increments
    deltas = {
        "Base\n(MeanPool\n30s K=1)": (0, v_start),
        "+Aggregation\n(K=1→5)":     (v_start, v_agg - v_start),
        "+Context\n(30s→240m)":      (v_agg,   v_ctx  - v_agg),
        "+Architecture\n(→Transf.)": (v_ctx,   v_arch - v_ctx),
        "Final\n(Transf.\n240m K=5)": (0, v_arch),
    }

    is_final = [False, False, False, False, True]

    colors_pos  = "#44A15E"  # green: gain
    colors_neg  = "#C94040"  # red: loss (shouldn't happen here)
    colors_base = "#3A7EBF"
    colors_final = "#E86A33"

    bottoms, heights, colors = [], [], []
    for i, (lbl, (bot, dlt)) in enumerate(deltas.items()):
        if is_final[i]:
            bottoms.append(0)
            heights.append(v_arch)
            colors.append(colors_final)
        else:
            bottoms.append(bot)
            heights.append(abs(dlt))
            if i == 0:
                colors.append(colors_base)
            elif dlt >= 0:
                colors.append(colors_pos)
            else:
                colors.append(colors_neg)

    x = np.arange(len(deltas))
    bars = ax.bar(x, heights, bottom=bottoms, color=colors,
                  edgecolor="white", linewidth=0.5, width=0.55)

    # Annotations on bars
    for i, (bar, (lbl, (bot, dlt))) in enumerate(zip(bars, deltas.items())):
        top = bot + dlt if not is_final[i] else v_arch
        prefix = "+" if (not is_final[i] and i > 0 and dlt > 0) else ""
        val_str = f"{prefix}{dlt:.3f}" if not is_final[i] else f"{v_arch:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                top + 0.004, val_str,
                ha="center", va="bottom", fontsize=FONT_ANNOT,
                fontweight="bold" if is_final[i] else "normal")

    # Connector lines between steps
    for i in range(len(deltas) - 2):
        x_left  = x[i] + 0.275
        x_right = x[i + 1] - 0.275
        b, d = list(deltas.values())[i]
        top = b + d
        ax.plot([x_left, x_right], [top, top],
                color="#888888", linewidth=0.8, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(list(deltas.keys()), fontsize=FONT_BASE)
    ax.set_ylabel("AUROC", fontsize=FONT_LABEL)
    ax.set_ylim(max(0, min(v_start, v_agg, v_ctx, v_arch) - 0.06),
                min(1.0, max(v_start, v_agg, v_ctx, v_arch) + 0.05))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _spine_clean(ax)
