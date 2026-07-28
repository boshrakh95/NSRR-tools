"""Panel functions for paper figures.

Each function takes an Axes object as its first argument and fills it with
content.  All sizing is handled by the notebook; functions just draw.

Conventions
-----------
* All metric values are passed/stored on the 0-1 scale and multiplied ×100
  only for display (axis labels, annotations).
* `analysis_df` is pre-filtered to split='test', k='all' before being passed
  (except where multiple k values are needed).
* `heatmap_df` has columns: context_length_min, context_label, k, auroc, ...
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from .style import (
    HEAD_STYLE, TASK_LABEL, CONTEXT_TO_MIN, MIN_TO_CTX, CTX_ORDER,
    ABL_CONDITIONS, ABL_CONTEXT, FONT_BASE, FONT_ANNOT, FONT_LABEL,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

_CTX_ORDER_MIN = [CONTEXT_TO_MIN[c] for c in CTX_ORDER]


def _palette(n: int):
    return sns.color_palette("viridis", max(n, 2))


def _fmt_ctx(x: float) -> str:
    return MIN_TO_CTX.get(float(x), f"{x:.0f}m")


def _spine_clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _log_ctx_axis(ax, xs=None):
    """Log x-axis with context-length tick labels."""
    ax.set_xscale("log")
    ticks = sorted(xs or _CTX_ORDER_MIN)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt_ctx(x) for x in ticks], fontsize=FONT_BASE)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


# ═══════════════════════════════════════════════════════════════════════════════
# S-Fig 2 — Saturation curves (all heads overlaid)
# ═══════════════════════════════════════════════════════════════════════════════

def saturation_panel(ax, analysis_df: pd.DataFrame, task: str,
                     heads: list[str] = ("lstm", "transformer", "mean_pool"),
                     metric_col: str = "mean_prob_auroc",
                     show_ci: bool = True,
                     show_values: bool = True):
    """AUROC vs context length, one line per head.

    analysis_df must be pre-filtered to split='test', k='all'.
    """
    xs_seen = []
    any_data = False

    for head in heads:
        sub = analysis_df[(analysis_df["task"] == task) &
                          (analysis_df["head"] == head) &
                          analysis_df[metric_col].notna()]
        if sub.empty:
            continue
        sub = sub.sort_values("context_length_min")
        xs = sub["context_length_min"].values
        ys = sub[metric_col].values * 100
        xs_seen.extend(xs.tolist())

        sty = HEAD_STYLE.get(head, {"color": "gray", "marker": "o",
                                    "ls": "-", "label": head})
        ax.plot(xs, ys, color=sty["color"], marker=sty["marker"],
                linestyle=sty["ls"], lw=1.5, ms=5, label=sty["label"])

        if show_values:
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}", (x, y),
                            textcoords="offset points", xytext=(0, 5),
                            fontsize=FONT_ANNOT, ha="center", color=sty["color"])

        # CI bands
        if show_ci:
            ci_lo_col = metric_col + "_ci_lo"
            ci_hi_col = metric_col + "_ci_hi"
            if ci_lo_col in sub.columns and sub[ci_lo_col].notna().any():
                lo = sub[ci_lo_col].values * 100
                hi = sub[ci_hi_col].values * 100
                ax.fill_between(xs, lo, hi, color=sty["color"], alpha=0.12)

        any_data = True

    if not any_data:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    _log_ctx_axis(ax, sorted(set(xs_seen)))
    ax.set_xlabel("Context length", fontsize=FONT_LABEL)
    ax.set_ylabel("AUROC (%)", fontsize=FONT_LABEL)
    ax.grid(True, which="major", alpha=0.25, lw=0.5)
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# S-Fig 1 — K-aggregation (AUROC vs K at fixed context)
# ═══════════════════════════════════════════════════════════════════════════════

def k_agg_panel(ax, analysis_df_full_k: pd.DataFrame, task: str,
                heads: list[str] = ("lstm", "transformer"),
                contexts: list[str] = ("40m", "120m", "240m"),
                show_legend: bool = True):
    """AUROC vs K for selected context lengths.

    analysis_df_full_k: analysis.csv WITHOUT k filter (all k values).
    show_legend: set False when the caller will draw a single shared legend.
    """
    ctx_mins = {c: CONTEXT_TO_MIN[c] for c in contexts if c in CONTEXT_TO_MIN}
    palette = _palette(len(ctx_mins))
    ctx_color = dict(zip(ctx_mins.keys(), palette))

    any_data = False
    for head in heads:
        sty = HEAD_STYLE.get(head, {"ls": "-", "label": head})
        for ctx_label, ctx_min in ctx_mins.items():
            sub = analysis_df_full_k[
                (analysis_df_full_k["task"] == task) &
                (analysis_df_full_k["head"] == head) &
                (analysis_df_full_k["context_length_min"] == ctx_min) &
                analysis_df_full_k["mean_prob_auroc"].notna()
            ].copy()
            if sub.empty:
                continue

            def _k_to_num(k):
                try:
                    return float(k)
                except Exception:
                    return np.nan

            sub["k_num"] = sub["k"].apply(_k_to_num)
            sub = sub.dropna(subset=["k_num"]).sort_values("k_num")
            if sub.empty:
                continue

            label = f"{sty['label']}, {ctx_label}" if len(heads) > 1 else ctx_label
            ax.plot(sub["k_num"].values, sub["mean_prob_auroc"].values * 100,
                    color=ctx_color[ctx_label], ls=sty["ls"], lw=1.4, ms=3,
                    marker="o", label=label, alpha=0.85)
            any_data = True

    if not any_data:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    ax.set_xscale("log")
    ax.set_xlabel("K (windows per subject)", fontsize=FONT_LABEL)
    ax.set_ylabel("AUROC (%)", fontsize=FONT_LABEL)
    if show_legend:
        ax.legend(fontsize=FONT_ANNOT, frameon=False, handlelength=3.5)
    ax.grid(True, which="major", alpha=0.25, lw=0.5)
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Fig 2 — AUROC vs K  (iso-compute, from heatmap_df)
# ═══════════════════════════════════════════════════════════════════════════════

_ISO_BUDGETS = [10, 30, 60, 120, 240, 480]


def _build_ctx_lookup(df: pd.DataFrame, col: str) -> dict:
    lookup = {}
    for ctx_min, grp in df.groupby("context_length_min"):
        grp = grp.dropna(subset=[col]).sort_values("k")
        if grp.empty:
            continue
        lookup[float(ctx_min)] = (grp["k"].values.astype(float),
                                  grp[col].values.astype(float))
    return lookup


def _interp(lookup, ctx_min, k):
    if ctx_min not in lookup:
        return np.nan
    ks, vs = lookup[ctx_min]
    if k < ks[0] or k > ks[-1]:
        return np.nan
    return float(np.interp(k, ks, vs))


def kvsk_panel(ax, heatmap_df: pd.DataFrame, col: str = "auroc",
               budget: float = 480.0, show_iso: bool = True,
               show_legend: bool = True):
    """AUROC vs K (log-x) with iso-compute lines.

    show_legend: set False when the caller will draw a single shared legend.
    """
    if heatmap_df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    contexts = sorted(heatmap_df["context_length_min"].unique())
    palette  = _palette(len(contexts))
    lookup   = _build_ctx_lookup(heatmap_df, col)
    ctx_lbl  = {ctx: heatmap_df[heatmap_df["context_length_min"] == ctx]
                ["context_label"].iloc[0]
                for ctx in contexts if not heatmap_df[
                    heatmap_df["context_length_min"] == ctx].empty}

    for i, ctx in enumerate(contexts):
        if ctx not in lookup:
            continue
        ks, vs = lookup[ctx]
        ax.plot(ks, vs * 100, color=palette[i], lw=1.4, label=ctx_lbl.get(ctx, ""),
                marker="o", ms=2, markevery=max(1, len(ks) // 10))

    if show_iso:
        iso_colors = plt.cm.Greys(np.linspace(0.3, 0.7, len(_ISO_BUDGETS)))
        for ic, cb in enumerate(_ISO_BUDGETS):
            if cb > budget:
                continue
            pts = [(cb / ctx, _interp(lookup, ctx, cb / ctx)) for ctx in contexts]
            pts = [(k, v) for k, v in pts if not np.isnan(v)]
            if len(pts) >= 2:
                pts.sort()
                ks_iso, vs_iso = zip(*pts)
                lt = f"{cb}m" if cb < 60 else f"{cb // 60}h"
                ax.plot(ks_iso, [v * 100 for v in vs_iso],
                        color=iso_colors[ic], lw=1.2, ls="--", alpha=0.75)
                # Label at the LEFT end of the iso line (small K, where AUROC
                # curves are most spread apart and text is readable).
                ax.annotate(lt, (ks_iso[0], vs_iso[0] * 100), fontsize=FONT_ANNOT,
                            fontweight="bold", color=iso_colors[ic],
                            ha="left", va="bottom", xytext=(2, 2),
                            textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("K (windows per subject)", fontsize=FONT_LABEL)
    ax.set_ylabel("AUROC (%)", fontsize=FONT_LABEL)
    if show_legend:
        ax.legend(title="Context", fontsize=FONT_ANNOT, title_fontsize=FONT_ANNOT,
                  frameon=False, loc="lower right")
    ax.grid(True, alpha=0.25, lw=0.5)
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Fig 3 — Heatmap  (iso-compute)
# ═══════════════════════════════════════════════════════════════════════════════

def heatmap_panel(ax, heatmap_df: pd.DataFrame, col: str = "auroc",
                  budget: float = 480.0,
                  show_ylabels: bool = True, show_cbar: bool = True,
                  show_cbar_label: bool = True):
    """2-D context × K heatmap with iso-compute lines.

    show_ylabels  : show y-axis tick labels (context lengths) — False for
                    right-hand panels in a multi-column row.
    show_cbar     : always True; kept for API compatibility.
    show_cbar_label : show the "AUROC (%)" text label on the colour bar —
                    False for left-hand panels so the bar is visible but
                    not labelled (the right neighbour carries the label).
    """
    if heatmap_df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    contexts = sorted(heatmap_df["context_length_min"].unique())
    ctx_lbl  = {ctx: heatmap_df[heatmap_df["context_length_min"] == ctx]
                ["context_label"].iloc[0]
                for ctx in contexts if not heatmap_df[
                    heatmap_df["context_length_min"] == ctx].empty}

    # Keep ~10 log-spaced K values so labels remain readable even at narrow widths.
    target_ks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    all_ks = sorted(heatmap_df["k"].unique())
    sub_ks = [k for k in target_ks if any(abs(k - ak) < 0.6 for ak in all_ks)]
    if not sub_ks:
        sub_ks = sorted(set(round(k) for k in all_ks))[:10]

    matrix = np.full((len(contexts), len(sub_ks)), np.nan)
    for i, ctx in enumerate(contexts):
        grp = (heatmap_df[heatmap_df["context_length_min"] == ctx]
               .dropna(subset=[col]).set_index("k")[col])
        for j, k in enumerate(sub_ks):
            near = grp.index.values
            if len(near) == 0:
                continue
            idx = np.argmin(np.abs(near - k))
            if abs(near[idx] - k) < 1:
                matrix[i, j] = grp.iloc[idx]

    valid = matrix[~np.isnan(matrix)] * 100
    if valid.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    vmin = float(np.floor(valid.min() / 5) * 5)
    vmax = float(np.ceil(valid.max() / 5) * 5)

    cbar_label = "AUROC (%)" if show_cbar_label else ""
    ytick_labels = [ctx_lbl.get(c, str(c)) for c in contexts]
    sns.heatmap(
        matrix * 100, ax=ax,
        cmap=sns.color_palette("YlOrRd", as_cmap=True),
        xticklabels=[str(k) for k in sub_ks],
        yticklabels=ytick_labels if show_ylabels else False,
        cbar=True,
        cbar_kws={"label": cbar_label, "shrink": 0.8},
        linewidths=0.2, linecolor="white",
        mask=np.isnan(matrix), vmin=vmin, vmax=vmax,
        annot=False,
    )
    ax.set_xlabel("K (windows)", fontsize=FONT_LABEL)
    ax.set_ylabel("Context length" if show_ylabels else "", fontsize=FONT_LABEL)
    ax.tick_params(axis="x", labelsize=FONT_ANNOT, rotation=45)
    ax.tick_params(axis="y", labelsize=FONT_ANNOT, rotation=0)

    # Iso-compute lines
    iso_colors = plt.cm.cool(np.linspace(0.2, 0.9, len(_ISO_BUDGETS)))
    for ic, cb in enumerate(_ISO_BUDGETS):
        if cb > budget:
            continue
        xs, ys = [], []
        for i, ctx in enumerate(contexts):
            kn = cb / ctx
            if kn < 1:
                continue
            for jj in range(len(sub_ks) - 1):
                if sub_ks[jj] <= kn <= sub_ks[jj + 1]:
                    frac = (kn - sub_ks[jj]) / (sub_ks[jj + 1] - sub_ks[jj])
                    xs.append(jj + frac + 0.5)
                    ys.append(i + 0.5)
                    break
        if len(xs) >= 2:
            lt = f"{cb}m" if cb < 60 else f"{cb // 60}h"
            ax.plot(xs, ys, color=iso_colors[ic], lw=1.8, ls="--", alpha=0.85)
            ax.annotate(lt, (xs[0], ys[0]), fontsize=FONT_ANNOT,
                        fontweight="bold", color=iso_colors[ic],
                        ha="center", va="bottom", xytext=(0, -10),
                        textcoords="offset points")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Fig 4 — iso-main: metric_vs_total + pareto
# ═══════════════════════════════════════════════════════════════════════════════

def vs_total_panel(ax, heatmap_df: pd.DataFrame, col: str = "auroc"):
    """AUROC vs total compute (context × K) — one line per context length."""
    if heatmap_df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    contexts = sorted(heatmap_df["context_length_min"].unique())
    palette  = _palette(len(contexts))
    lookup   = _build_ctx_lookup(heatmap_df, col)
    ctx_lbl  = {ctx: heatmap_df[heatmap_df["context_length_min"] == ctx]
                ["context_label"].iloc[0]
                for ctx in contexts if not heatmap_df[
                    heatmap_df["context_length_min"] == ctx].empty}

    for i, ctx in enumerate(contexts):
        if ctx not in lookup:
            continue
        ks, vs = lookup[ctx]
        ax.plot(ks * ctx, vs * 100, color=palette[i], lw=1.4,
                label=ctx_lbl.get(ctx, ""), marker="o", ms=2,
                markevery=max(1, len(ks) // 10))

    ax.set_xscale("log")
    ax.set_xlabel("Total context (min) = L × K", fontsize=FONT_LABEL)
    ax.set_ylabel("AUROC (%)", fontsize=FONT_LABEL)
    ax.legend(title="Context L", fontsize=FONT_ANNOT, title_fontsize=FONT_ANNOT,
              frameon=False)
    ax.grid(True, alpha=0.25, lw=0.5)
    _spine_clean(ax)


def pareto_panel(ax, heatmap_df: pd.DataFrame, col: str = "auroc",
                 budget: float = 480.0):
    """Pareto-optimal (L, K) configurations vs total compute budget."""
    if heatmap_df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    contexts = sorted(heatmap_df["context_length_min"].unique())
    palette  = _palette(len(contexts))
    lookup   = _build_ctx_lookup(heatmap_df, col)
    ctx_lbl  = {ctx: heatmap_df[heatmap_df["context_length_min"] == ctx]
                ["context_label"].iloc[0]
                for ctx in contexts if not heatmap_df[
                    heatmap_df["context_length_min"] == ctx].empty}

    budgets_sweep = np.unique(np.concatenate([
        np.arange(0.5, 20, 0.5), np.arange(20, 100, 2),
        np.arange(100, budget + 1, 5),
    ]))

    opt_b, opt_v, opt_c, opt_k = [], [], [], []
    for b in budgets_sweep:
        best_v, best_ctx, best_k_val = -1.0, None, None
        for ctx in contexts:
            if ctx not in lookup:
                continue
            ks, _ = lookup[ctx]
            k_use = min(b / ctx, ks[-1])
            if k_use < 1:
                continue
            v = _interp(lookup, ctx, k_use)
            if not np.isnan(v) and v > best_v:
                best_v, best_ctx, best_k_val = v, ctx, k_use
        if best_ctx is not None:
            opt_b.append(b); opt_v.append(best_v)
            opt_c.append(best_ctx); opt_k.append(best_k_val)

    if not opt_b:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    ctx_to_color = {c: palette[i] for i, c in enumerate(contexts)}
    segments, s = [], 0
    for j in range(1, len(opt_b)):
        if opt_c[j] != opt_c[j - 1]:
            segments.append((s, j)); s = j
    segments.append((s, len(opt_b)))

    labeled = set()
    for s, e in segments:
        ctx = opt_c[s]
        lbl = ctx_lbl.get(ctx, "") if ctx not in labeled else None
        ax.plot(opt_b[s:e], [v * 100 for v in opt_v[s:e]],
                color=ctx_to_color[ctx], lw=2.5, solid_capstyle="round", label=lbl)
        labeled.add(ctx)
        mid = (s + e) // 2
        ax.annotate(f"{ctx_lbl.get(ctx, '')}\nk≈{opt_k[mid]:.0f}",
                    (opt_b[mid], opt_v[mid] * 100), fontsize=FONT_ANNOT,
                    fontweight="bold", color=ctx_to_color[ctx],
                    ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Total compute budget (min)", fontsize=FONT_LABEL)
    ax.set_ylabel("Best AUROC (%)", fontsize=FONT_LABEL)
    ax.legend(title="Optimal L", fontsize=FONT_ANNOT, title_fontsize=FONT_ANNOT,
              frameon=False, loc="lower right")
    ax.grid(True, alpha=0.25, lw=0.5)
    _spine_clean(ax)


def combined_iso_panel(ax, heatmap_df: pd.DataFrame, col: str = "auroc",
                       alpha_bg: float = 0.20, budget: float = 480.0):
    """Overlay of vs-total curves (semi-transparent) + Pareto frontier (opaque).

    Background: all (L, K) configurations as faded lines, one colour per L.
    Foreground: Pareto-optimal frontier, same colour scheme, full opacity,
                annotated with the optimal L label at each segment midpoint.
    No legend drawn — caller rebuilds handles at alpha=1 for a shared legend.
    """
    if heatmap_df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    contexts = sorted(heatmap_df["context_length_min"].unique())
    palette  = _palette(len(contexts))
    lookup   = _build_ctx_lookup(heatmap_df, col)
    ctx_lbl  = {ctx: heatmap_df[heatmap_df["context_length_min"] == ctx]
                ["context_label"].iloc[0]
                for ctx in contexts
                if not heatmap_df[heatmap_df["context_length_min"] == ctx].empty}

    # ── Background: all vs-total lines (faded) ───────────────────────────────
    for i, ctx in enumerate(contexts):
        if ctx not in lookup:
            continue
        ks, vs = lookup[ctx]
        ax.plot(ks * ctx, vs * 100,
                color=palette[i], lw=1.0, alpha=alpha_bg,
                label=ctx_lbl.get(ctx, ""))

    # ── Foreground: Pareto-optimal frontier ──────────────────────────────────
    budgets_sweep = np.unique(np.concatenate([
        np.arange(0.5, 20, 0.5), np.arange(20, 100, 2),
        np.arange(100, budget + 1, 5),
    ]))
    opt_b, opt_v, opt_c, opt_k = [], [], [], []
    for b in budgets_sweep:
        best_v, best_ctx, best_k_val = -1.0, None, None
        for ctx in contexts:
            if ctx not in lookup:
                continue
            ks, _ = lookup[ctx]
            k_use = min(b / ctx, ks[-1])
            if k_use < 1:
                continue
            v = _interp(lookup, ctx, k_use)
            if not np.isnan(v) and v > best_v:
                best_v, best_ctx, best_k_val = v, ctx, k_use
        if best_ctx is not None:
            opt_b.append(b); opt_v.append(best_v)
            opt_c.append(best_ctx); opt_k.append(best_k_val)

    if not opt_b:
        return

    ctx_to_color = {c: palette[i] for i, c in enumerate(contexts)}
    segments, s = [], 0
    for j in range(1, len(opt_b)):
        if opt_c[j] != opt_c[j - 1]:
            segments.append((s, j)); s = j
    segments.append((s, len(opt_b)))

    for s, e in segments:
        ctx = opt_c[s]
        ax.plot(opt_b[s:e], [v * 100 for v in opt_v[s:e]],
                color=ctx_to_color[ctx], lw=2.5, solid_capstyle="round",
                alpha=1.0, zorder=3)
        mid = (s + e) // 2
        ax.annotate(ctx_lbl.get(ctx, ''),
                    (opt_b[mid], opt_v[mid] * 100),
                    fontsize=FONT_ANNOT, fontweight="bold",
                    color=ctx_to_color[ctx],
                    ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Total context (min) = L × K", fontsize=FONT_LABEL)
    ax.set_ylabel("AUROC (%)", fontsize=FONT_LABEL)
    ax.grid(True, alpha=0.25, lw=0.5)
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Fig 5 — Min-cost frontier
# ═══════════════════════════════════════════════════════════════════════════════

def mincost_panel(ax, heatmap_df: pd.DataFrame, col: str = "auroc",
                  budget: float = 480.0):
    """Minimum compute cost to reach each target AUROC level."""
    if heatmap_df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    contexts = sorted(heatmap_df["context_length_min"].unique())
    palette  = _palette(len(contexts))
    lookup   = _build_ctx_lookup(heatmap_df, col)
    ctx_lbl  = {ctx: heatmap_df[heatmap_df["context_length_min"] == ctx]
                ["context_label"].iloc[0]
                for ctx in contexts if not heatmap_df[
                    heatmap_df["context_length_min"] == ctx].empty}

    all_vals = [v for ks, vs in lookup.values() for v in vs if not np.isnan(v)]
    if not all_vals:
        return
    val_min = float(np.floor(min(all_vals) * 20) / 20)
    val_max = float(np.ceil(max(all_vals) * 20) / 20)
    target_vals = np.arange(val_min, val_max + 0.002, 0.005)
    annot_targets = [t for t in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
                     if val_min <= t <= val_max]

    for i, ctx in enumerate(contexts):
        if ctx not in lookup:
            continue
        ks, vals = lookup[ctx]
        costs, tgts = [], []
        for tgt in target_vals:
            idx = np.searchsorted(vals, tgt)
            if idx >= len(ks):
                continue
            costs.append(ctx * ks[idx])
            tgts.append(tgt * 100)
        if tgts:
            ax.plot(tgts, costs, color=palette[i], lw=1.4,
                    label=ctx_lbl.get(ctx, ""))

    for tgt in annot_targets:
        best_cost, best_i = float("inf"), None
        for i, ctx in enumerate(contexts):
            if ctx not in lookup:
                continue
            ks, vals = lookup[ctx]
            idx = np.searchsorted(vals, tgt)
            if idx >= len(ks):
                continue
            cost = ctx * ks[idx]
            if cost < best_cost:
                best_cost, best_i = cost, i
        if best_i is not None and best_cost <= budget:
            ax.plot(tgt * 100, best_cost, "o", color=palette[best_i],
                    ms=5, zorder=5, markeredgecolor="black", markeredgewidth=0.5)

    ax.set_yscale("log")
    ax.set_xlabel(f"Target AUROC (%)", fontsize=FONT_LABEL)
    ax.set_ylabel("Min total compute (min)", fontsize=FONT_LABEL)
    ax.legend(title="Context L", fontsize=FONT_ANNOT, title_fontsize=FONT_ANNOT,
              frameon=False, loc="upper left")
    ax.axhline(y=budget, color="red", ls=":", lw=1.0, alpha=0.6)
    # Use mixed transform: x in axes fraction (always visible), y in data space
    ax.text(0.98, budget, f"{budget:.0f}m",
            transform=ax.get_yaxis_transform(),
            fontsize=FONT_ANNOT, color="red",
            va="bottom", ha="right", clip_on=False)
    ax.grid(True, alpha=0.25, lw=0.5)
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Fig 6 — PR curves (from parquets)
# ═══════════════════════════════════════════════════════════════════════════════

def pr_curves_panel(ax, parquets: dict[str, pd.DataFrame],
                    contexts: list[str] = ("30s", "40m", "120m", "240m"),
                    k: int | None = None,
                    prob_col: str = "prob_class1"):
    """PR curves at each context length (K=all mean-pool).

    parquets: {context_label: DataFrame} from data.load_parquets()
    """
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
    except ImportError:
        ax.text(0.5, 0.5, "sklearn required", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    avail_ctx = [c for c in contexts if c in parquets and not parquets[c].empty]
    if not avail_ctx:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    cmap = cm.get_cmap("viridis_r", max(len(avail_ctx), 2))
    ctx_color = {c: cmap(i / max(len(avail_ctx) - 1, 1))
                 for i, c in enumerate(avail_ctx)}

    for ctx in avail_ctx:
        df = parquets[ctx]
        if prob_col not in df.columns:
            continue
        sub = (df.groupby("subject_id")
               .agg(mean_prob=(prob_col, "mean"), true_label=("true_label", "first"))
               .reset_index())
        if sub.empty or sub["true_label"].nunique() < 2:
            continue
        labels = sub["true_label"].values
        scores = sub["mean_prob"].values
        prec, rec, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        ax.plot(rec, prec, color=ctx_color[ctx], lw=1.4,
                label=f"{ctx} (AP={ap:.2f})")

    ax.set_xlabel("Recall", fontsize=FONT_LABEL)
    ax.set_ylabel("Precision", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_ANNOT, frameon=False)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.25, lw=0.5)
    _spine_clean(ax)


def aucpr_panel(ax, analysis_df: pd.DataFrame, task: str,
                heads: list[str] = ("lstm", "transformer")):
    """AUC-PR vs context length from analysis.csv (mean_prob_auroc proxy)."""
    for head in heads:
        sub = analysis_df[(analysis_df["task"] == task) &
                          (analysis_df["head"] == head) &
                          analysis_df["mean_prob_auroc"].notna()]
        if sub.empty:
            continue
        sub = sub.sort_values("context_length_min")
        sty = HEAD_STYLE.get(head, {"color": "gray", "marker": "o",
                                    "ls": "-", "label": head})
        ax.plot(sub["context_length_min"].values,
                sub["mean_prob_auroc"].values * 100,
                color=sty["color"], marker=sty["marker"],
                ls=sty["ls"], lw=1.4, ms=4, label=sty["label"])

    _log_ctx_axis(ax)
    ax.set_xlabel("Context length", fontsize=FONT_LABEL)
    ax.set_ylabel("AUROC (%)", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_ANNOT, frameon=False)
    ax.grid(True, alpha=0.25, lw=0.5)
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# S-Fig 3 — Compute scaling (1B)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from scipy.optimize import curve_fit as _curve_fit
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _power_law(F, a, b, c):
    return c - a * np.asarray(F, dtype=float) ** (-b)


def _flops_per_step(head: str, seq_len, input_dim, hidden_dim) -> float:
    """Analytical FLOPs per gradient step (forward + backward ≈ 3× forward)."""
    try:
        sl = int(seq_len); id_ = int(input_dim); hd = int(hidden_dim)
    except (TypeError, ValueError):
        return float("nan")
    if sl <= 0:
        return float("nan")
    if head == "lstm":
        return 3.0 * sl * 4 * hd * (id_ + hd)
    elif head == "transformer":
        return 3.0 * sl * (sl * hd + 4 * hd * hd)
    elif "pool" in head or "mean" in head:
        return 3.0 * sl * id_
    return float("nan")


def compute_scaling_panel(ax, training_df: pd.DataFrame, task: str,
                          heads: list[str] = ("lstm", "transformer", "mean_pool"),
                          legend: bool = True):
    """Compute scaling: test AUROC at best epoch vs total training FLOPs.

    Matches plot_scaling_laws.py 1B style:
      - point colour  = context length (viridis_r)
      - point marker  = head architecture
      - FLOPs computed analytically from seq_len × steps_per_epoch × epoch
    """
    CTX_ORDER_LOCAL = ["30s", "10m", "40m", "80m", "120m", "240m"]

    # Filter to is_best_epoch rows, normalising the boolean column
    df = training_df[training_df["task"] == task].copy()
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    for col in ("is_best_epoch", "is_overfit_epoch"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            ).fillna(False)

    best = df[df.get("is_best_epoch", pd.Series([False]*len(df))).values].copy()
    if best.empty:
        ax.text(0.5, 0.5, "no best-epoch rows", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    required = {"seq_len", "steps_per_epoch", "input_dim", "hidden_dim", "test_auroc"}
    if not required.issubset(best.columns):
        ax.text(0.5, 0.5, "missing cols", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    best["total_flops"] = best.apply(
        lambda r: _flops_per_step(r["head"], r["seq_len"],
                                  r["input_dim"], r["hidden_dim"])
                  * r["steps_per_epoch"] * r["epoch"],
        axis=1,
    )
    best = best.dropna(subset=["total_flops", "test_auroc"])
    if best.empty:
        ax.text(0.5, 0.5, "FLOPs NaN", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    # Context colour map (viridis_r, same as original)
    contexts = sorted(best["context_length"].dropna().unique(),
                      key=lambda c: CTX_ORDER_LOCAL.index(c)
                      if c in CTX_ORDER_LOCAL else 99)
    n = max(len(contexts), 2)
    cmap = plt.cm.get_cmap("viridis_r", n)
    ctx_color = {c: cmap(i / (n - 1)) for i, c in enumerate(contexts)}

    for head in heads:
        hsub = best[best["head"] == head]
        if hsub.empty:
            continue
        sty = HEAD_STYLE.get(head, {"color": "gray", "marker": "o", "label": head})
        Fs, aucs = [], []
        for _, row in hsub.iterrows():
            ctx = str(row["context_length"])
            ax.scatter(row["total_flops"], row["test_auroc"] * 100,
                       color=ctx_color.get(ctx, "gray"),
                       marker=sty["marker"], s=70, zorder=5,
                       edgecolors=sty["color"], linewidths=1.5)
            Fs.append(row["total_flops"])
            aucs.append(row["test_auroc"] * 100)

        # Log-linear fit: AUROC = a·log10(FLOPs) + b → straight line on log x-axis
        if len(Fs) >= 2:
            Fs_a  = np.array(Fs,  dtype=float)
            auc_a = np.array(aucs, dtype=float)
            coeffs = np.polyfit(np.log10(Fs_a), auc_a, 1)
            log_grid = np.linspace(np.log10(Fs_a.min()), np.log10(Fs_a.max()), 50)
            ax.plot(10**log_grid, np.polyval(coeffs, log_grid),
                    color=sty["color"], lw=2, ls="--", alpha=0.7,
                    label=f"{sty['label']} fit")

    # Context colour legend entries
    for ctx in contexts:
        ax.scatter([], [], color=ctx_color[ctx], s=40, label=f"L={ctx}")
    # Head marker legend entries (only if no fit lines were drawn)
    for head in heads:
        sty = HEAD_STYLE.get(head, {"color": "gray", "marker": "o", "label": head})
        ax.scatter([], [], color="gray", marker=sty["marker"],
                   s=50, label=sty["label"])

    ax.set_xscale("log")
    ax.set_xlabel("Total training FLOPs", fontsize=FONT_LABEL)
    ax.set_ylabel("Test AUROC (%)", fontsize=FONT_LABEL)
    if legend:
        ax.legend(fontsize=FONT_ANNOT, frameon=False, handlelength=1.5, ncol=2)
    ax.grid(True, alpha=0.25, lw=0.5)
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# S-Fig 4 — Task landscape (6A scatter + 6C L* lollipop)
# ═══════════════════════════════════════════════════════════════════════════════

def task_scatter_panel(ax, analysis_df: pd.DataFrame,
                       tasks: list[str], head: str = "lstm"):
    """Scatter: task difficulty (1-AUROC@30s) vs context sensitivity (ΔAUROC)."""
    sub = analysis_df[(analysis_df["head"] == head) &
                      analysis_df["mean_prob_auroc"].notna()].copy()

    task_colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
    tcolor = dict(zip(tasks, task_colors))

    for task in tasks:
        t_sub = sub[sub["task"] == task].sort_values("context_length_min")
        if len(t_sub) < 2:
            continue
        baseline = t_sub["mean_prob_auroc"].iloc[0]
        best     = t_sub["mean_prob_auroc"].max()
        difficulty   = 1 - baseline
        sensitivity  = best - baseline
        label = TASK_LABEL.get(task, task)
        ax.scatter(difficulty, sensitivity, color=tcolor[task], s=60, zorder=3)
        ax.annotate(label, (difficulty, sensitivity), fontsize=FONT_ANNOT,
                    xytext=(4, 2), textcoords="offset points",
                    color=tcolor[task])

    ax.axhline(0, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax.axvline(0.5, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax.set_xlabel("Task difficulty (1 – AUROC @ 30s)", fontsize=FONT_LABEL)
    ax.set_ylabel("Context sensitivity (ΔAUROC 30s→best)", fontsize=FONT_LABEL)
    ax.grid(True, alpha=0.25, lw=0.5)
    _spine_clean(ax)


def lstar_panel(ax, analysis_df: pd.DataFrame,
                tasks: list[str], head: str = "lstm",
                tol: float = 0.005):
    """Horizontal lollipop: L* (saturation context) per task."""
    sub = analysis_df[(analysis_df["head"] == head) &
                      analysis_df["mean_prob_auroc"].notna()].copy()

    task_colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
    lstar_vals, lstar_labels, lstar_colors = [], [], []

    for i, task in enumerate(tasks):
        t_sub = sub[sub["task"] == task].sort_values("context_length_min")
        if t_sub.empty:
            continue
        best = t_sub["mean_prob_auroc"].max()
        threshold = best - tol
        sat = t_sub[t_sub["mean_prob_auroc"] >= threshold].iloc[0]
        lstar_vals.append(sat["context_length_min"])
        lstar_labels.append(TASK_LABEL.get(task, task))
        lstar_colors.append(task_colors[i])

    y_pos = np.arange(len(lstar_vals))
    ax.barh(y_pos, lstar_vals, color=lstar_colors, height=0.5, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(lstar_labels, fontsize=FONT_BASE)
    ax.set_xscale("log")
    ax.set_xticks(list(CONTEXT_TO_MIN.values()))
    ax.set_xticklabels([_fmt_ctx(x) for x in CONTEXT_TO_MIN.values()],
                       fontsize=FONT_BASE)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Saturation context L* (min)", fontsize=FONT_LABEL)
    ax.grid(True, which="major", alpha=0.25, lw=0.5, axis="x")
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# S-Fig 5 — Channel comparison (fast vs full)
# ═══════════════════════════════════════════════════════════════════════════════

def channel_comparison_panel(ax, fast_analysis: pd.DataFrame,
                              full_analysis: pd.DataFrame,
                              task: str, head: str = "transformer",
                              metric_col: str = "mean_prob_auroc"):
    """Overlay fast-channel (dashed) vs full-channel (solid) saturation."""
    for df, ls, label in [(fast_analysis, "--", "Fast-ch"),
                          (full_analysis, "-",  "Full-ch")]:
        sub = df[(df["task"] == task) & (df["head"] == head) &
                 df[metric_col].notna()].sort_values("context_length_min")
        if sub.empty:
            continue
        xs = sub["context_length_min"].values
        ys = sub[metric_col].values * 100
        ax.plot(xs, ys, color="#E86A33", ls=ls, lw=1.5, ms=5,
                marker="o" if ls == "--" else "s", label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 4), fontsize=FONT_ANNOT, ha="center",
                        color="#E86A33", alpha=0.75)

    # Annotate Δ at 240m
    fast_240 = fast_analysis[(fast_analysis["task"] == task) &
                              (fast_analysis["head"] == head) &
                              (fast_analysis["context_length"] == "240m")][metric_col]
    full_240 = full_analysis[(full_analysis["task"] == task) &
                              (full_analysis["head"] == head) &
                              (full_analysis["context_length"] == "240m")][metric_col]
    if not fast_240.empty and not full_240.empty:
        delta = (float(full_240.iloc[0]) - float(fast_240.iloc[0])) * 100
        sign = "+" if delta >= 0 else ""
        ax.text(0.97, 0.04, f"Δ240m: {sign}{delta:.1f} pp",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FONT_ANNOT, color="gray",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

    _log_ctx_axis(ax)
    ax.set_xlabel("Context length", fontsize=FONT_LABEL)
    ax.set_ylabel("AUROC (%)", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_ANNOT, frameon=False)
    ax.grid(True, which="major", alpha=0.25, lw=0.5)
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# S-Fig 6 — Modality ablation bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def modality_bar_panel(ax, abl_analysis: pd.DataFrame,
                       fast_analysis: pd.DataFrame,
                       full_analysis: pd.DataFrame,
                       task: str, head: str = "lstm",
                       split: str = "test"):
    """Horizontal bars: ΔAUROC per ablation condition vs fast-ch baseline."""
    ctx = ABL_CONTEXT.get(task, "120m")

    def _get(df, run_tag=""):
        mask = (
            (df["task"] == task) & (df["head"] == head) &
            (df["split"] == split) & (df["k"].astype(str) == "all") &
            (df["context_length"].astype(str) == ctx)
        )
        if "run_tag" in df.columns:
            mask &= (df["run_tag"].fillna("") == run_tag)
        sub = df.loc[mask, "mean_prob_auroc"].dropna()
        return float(sub.iloc[0]) if not sub.empty else None

    fast_val = _get(fast_analysis)
    full_val = _get(full_analysis)

    if fast_val is None:
        ax.text(0.5, 0.5, "no baseline", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    y_labels, deltas, colors = [], [], []
    for run_tag, cond_label, color in ABL_CONDITIONS:
        abl_val = _get(abl_analysis, run_tag)
        deltas.append((abl_val - fast_val) if abl_val is not None else np.nan)
        y_labels.append(cond_label)
        colors.append(color)

    y_pos = np.arange(len(y_labels))
    for i, (delta, color) in enumerate(zip(deltas, colors)):
        if not np.isnan(delta):
            ax.barh(i, delta, color=color, height=0.5, edgecolor="white", lw=0.4)
            sign = "+" if delta >= 0 else ""
            ax.text(delta + (0.001 if delta >= 0 else -0.001), i,
                    f"{sign}{delta:.3f}", va="center",
                    ha="left" if delta >= 0 else "right",
                    fontsize=FONT_ANNOT, color=color)

    ax.axvline(0, color="black", lw=0.8, zorder=3)
    if full_val is not None:
        fd = full_val - fast_val
        ax.axvline(fd, color="gray", lw=0.8, ls="--", zorder=2)
        ax.text(fd, len(y_labels) - 0.5, "full-ch", fontsize=FONT_ANNOT,
                color="gray", ha="center", va="bottom", rotation=90)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=FONT_BASE)
    ax.set_xlabel("ΔAUROC from baseline", fontsize=FONT_LABEL)
    ax.invert_yaxis()

    all_deltas = [d for d in deltas if not np.isnan(d)]
    if full_val is not None:
        all_deltas.append(full_val - fast_val)
    if all_deltas:
        xmax = max(0.04, max(abs(d) for d in all_deltas) + 0.015)
        ax.set_xlim(-xmax, xmax)

    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# S-Fig 7 — Aggregate scaling (delta, norm, slope)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_agg_curves(analysis_df: pd.DataFrame, tasks: list[str],
                      heads: list[str]):
    """Compute per-task per-head delta and norm curves."""
    std_ctx = sorted(CONTEXT_TO_MIN.values())
    curve_data = {h: {} for h in heads}

    sub = analysis_df[analysis_df["mean_prob_auroc"].notna()]
    for head in heads:
        for task in tasks:
            t_sub = sub[(sub["task"] == task) & (sub["head"] == head)]
            t_sub = t_sub.sort_values("context_length_min").reset_index(drop=True)
            if len(t_sub) < 2:
                continue
            ctx   = t_sub["context_length_min"].values
            auroc = t_sub["mean_prob_auroc"].values
            delta = auroc - auroc[0]
            total = auroc[-1] - auroc[0]
            norm  = delta / total if total > 0.005 else np.full_like(delta, np.nan)

            valid = (ctx > 0) & ~np.isnan(delta)
            slope = np.nan
            if valid.sum() >= 3:
                coeffs = np.polyfit(np.log2(ctx[valid]), delta[valid], deg=1)
                slope = float(coeffs[0])

            curve_data[head][task] = {
                "ctx": ctx, "auroc": auroc, "delta": delta,
                "norm": norm, "slope": slope,
            }

    agg = {}
    for head in heads:
        td = curve_data[head]
        delta_at = {c: [] for c in std_ctx}
        norm_at  = {c: [] for c in std_ctx}
        for d in td.values():
            for c in std_ctx:
                idx = np.where(np.abs(d["ctx"] - c) < 0.05)[0]
                if idx.size:
                    delta_at[c].append(d["delta"][idx[0]])
                    v = d["norm"][idx[0]]
                    if not np.isnan(v):
                        norm_at[c].append(v)
        valid_ctx = [c for c in std_ctx if delta_at[c]]
        if not valid_ctx:
            continue
        agg[head] = {
            "ctx":        np.array(valid_ctx),
            "mean_delta": np.array([np.mean(delta_at[c]) for c in valid_ctx]),
            "std_delta":  np.array([np.std(delta_at[c], ddof=1) if len(delta_at[c]) > 1
                                    else 0.0 for c in valid_ctx]),
            "mean_norm":  np.array([np.mean(norm_at[c]) if norm_at[c] else np.nan
                                    for c in valid_ctx]),
            "std_norm":   np.array([np.std(norm_at[c], ddof=1) if len(norm_at[c]) > 1
                                    else np.nan for c in valid_ctx]),
        }

    slopes = {}
    for head in heads:
        per_task = {t: d["slope"] for t, d in curve_data[head].items()
                    if not np.isnan(d["slope"])}
        if per_task:
            vals = list(per_task.values())
            slopes[head] = {"per_task": per_task, "mean": np.mean(vals),
                            "std": np.std(vals, ddof=1) if len(vals) > 1 else 0.0}

    return curve_data, agg, slopes


def delta_panel(ax, analysis_df: pd.DataFrame, tasks: list[str],
                heads: list[str] = ("lstm", "transformer", "mean_pool")):
    """ΔAUROC from 30s baseline, mean ± std across tasks."""
    curve_data, agg, _ = _build_agg_curves(analysis_df, tasks, heads)
    for head in heads:
        if head not in agg:
            continue
        sty = HEAD_STYLE.get(head, {"color": "gray", "marker": "o",
                                    "ls": "-", "label": head})
        d = agg[head]
        for td in curve_data[head].values():
            ax.plot(td["ctx"], td["delta"] * 100, color=sty["color"],
                    lw=0.6, alpha=0.18)
        ax.fill_between(d["ctx"],
                        (d["mean_delta"] - d["std_delta"]) * 100,
                        (d["mean_delta"] + d["std_delta"]) * 100,
                        color=sty["color"], alpha=0.15)
        ax.plot(d["ctx"], d["mean_delta"] * 100, color=sty["color"],
                ls=sty["ls"], marker=sty["marker"], lw=1.8, ms=5,
                label=sty["label"])

    _log_ctx_axis(ax)
    ax.axhline(0, color="gray", ls=":", lw=0.7, alpha=0.5)
    ax.set_xlabel("Context length", fontsize=FONT_LABEL)
    ax.set_ylabel("ΔAUROC from 30s (pp)", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_ANNOT, frameon=False)
    ax.grid(True, alpha=0.25, lw=0.5)
    _spine_clean(ax)


def norm_panel(ax, analysis_df: pd.DataFrame, tasks: list[str],
               heads: list[str] = ("lstm", "transformer", "mean_pool")):
    """Normalised gain (0=30s, 100=240m), mean ± std across tasks."""
    curve_data, agg, _ = _build_agg_curves(analysis_df, tasks, heads)
    for head in heads:
        if head not in agg:
            continue
        sty = HEAD_STYLE.get(head, {"color": "gray", "marker": "o",
                                    "ls": "-", "label": head})
        d = agg[head]
        valid = ~np.isnan(d["mean_norm"])
        if not valid.any():
            continue
        mn = d["mean_norm"][valid]
        sn = np.where(np.isnan(d["std_norm"][valid]), 0.0, d["std_norm"][valid])
        ax.fill_between(d["ctx"][valid], (mn - sn) * 100, (mn + sn) * 100,
                        color=sty["color"], alpha=0.15)
        ax.plot(d["ctx"][valid], mn * 100, color=sty["color"],
                ls=sty["ls"], marker=sty["marker"], lw=1.8, ms=5,
                label=sty["label"])

    _log_ctx_axis(ax)
    ax.axhline(0,   color="gray", ls=":", lw=0.7, alpha=0.5)
    ax.axhline(100, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax.set_ylim(-20, 130)
    ax.set_xlabel("Context length", fontsize=FONT_LABEL)
    ax.set_ylabel("Normalised gain (%)\n(0 = 30s, 100 = 240m)", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_ANNOT, frameon=False)
    ax.grid(True, alpha=0.25, lw=0.5)
    _spine_clean(ax)


def slope_bar_panel(ax, analysis_df: pd.DataFrame, tasks: list[str],
                    heads: list[str] = ("lstm", "transformer", "mean_pool")):
    """Bar chart: ΔAUROC per log₂ doubling of context (OLS slope)."""
    _, _, slopes = _build_agg_curves(analysis_df, tasks, heads)
    ordered = [h for h in heads if h in slopes]
    x_pos  = np.arange(len(ordered))
    colors = [HEAD_STYLE.get(h, {"color": "gray"})["color"] for h in ordered]
    labels = [HEAD_STYLE.get(h, {"label": h})["label"] for h in ordered]
    means  = [slopes[h]["mean"] * 100 for h in ordered]
    stds   = [slopes[h]["std"]  * 100 for h in ordered]

    for xi, head in zip(x_pos, ordered):
        for sv in slopes[head]["per_task"].values():
            ax.scatter(xi, sv * 100, color=colors[xi], alpha=0.35, s=15, zorder=3)

    bars = ax.bar(x_pos, means, color=colors, edgecolor="white",
                  width=0.45, alpha=0.85, zorder=2)
    ax.errorbar(x_pos, means, yerr=stds, fmt="none",
                color="black", capsize=3, lw=1.2, zorder=4)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=FONT_ANNOT)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=FONT_BASE)
    ax.set_ylabel("ΔAUROC per log₂ doubling (pp)", fontsize=FONT_LABEL)
    ax.axhline(0, color="gray", ls=":", lw=0.7)
    ax.grid(True, alpha=0.25, lw=0.5, axis="y")
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# S-Fig 8 — Variance violins (within-subject prediction std)
# ═══════════════════════════════════════════════════════════════════════════════

def variance_violin_panel(ax, parquets: dict[str, pd.DataFrame],
                          contexts: list[str] = ("30s", "40m", "120m", "240m"),
                          prob_col: str = "prob_class1"):
    """Violin plots of within-subject prob std, by correct/incorrect classification."""
    avail = [c for c in contexts if c in parquets and not parquets[c].empty]
    if not avail:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    records = []
    for ctx in avail:
        df = parquets[ctx]
        if prob_col not in df.columns:
            continue
        stats = (
            df.groupby("subject_id")
            .agg(
                mean_prob=(prob_col, "mean"),
                std_prob=(prob_col, "std"),
                true_label=("true_label", "first"),
            )
            .reset_index()
        )
        stats["correct"] = ((stats["mean_prob"] >= 0.5) == stats["true_label"].astype(bool))
        stats["context"] = ctx
        records.append(stats)

    if not records:
        return

    combined = pd.concat(records, ignore_index=True)
    combined["std_prob"] = combined["std_prob"].fillna(0)
    combined["status"] = combined["correct"].map({True: "Correct", False: "Incorrect"})

    palette = {"Correct": "#3A7EBF", "Incorrect": "#E86A33"}
    x_pos = {ctx: i for i, ctx in enumerate(avail)}

    for status, color in palette.items():
        sub = combined[combined["status"] == status]
        data_by_ctx = [sub[sub["context"] == ctx]["std_prob"].dropna().values
                       for ctx in avail]
        parts = ax.violinplot(data_by_ctx, positions=list(x_pos.values()),
                              showmedians=True, widths=0.35)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color(color)
        parts["cbars"].set_color(color)
        parts["cmins"].set_color(color)
        parts["cmaxes"].set_color(color)
        ax.plot([], [], color=color, lw=3, label=status)

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels(avail, fontsize=FONT_BASE)
    ax.set_xlabel("Context length", fontsize=FONT_LABEL)
    ax.set_ylabel("Within-subject std(prob)", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_ANNOT, frameon=False)
    ax.grid(True, alpha=0.25, lw=0.5, axis="y")
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# S-Fig 10 — Reliability / calibration
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_ece(y_true, prob_pred, n_bins=10):
    """ECE + calibration curve points. Matches plot_calibration.py."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece, mids, fracs, weights = 0.0, [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob_pred >= lo) & (prob_pred < hi)
        n = mask.sum()
        if n < 2:
            continue
        conf = prob_pred[mask].mean()
        acc  = y_true[mask].mean()
        w    = n / len(y_true)
        ece += w * abs(conf - acc)
        mids.append(conf); fracs.append(acc); weights.append(w)
    return ece, np.array(mids), np.array(fracs)


def reliability_panel(ax, parquets: dict[str, pd.DataFrame],
                      contexts: list[str] | None = None,
                      k: int = 5,
                      n_bins: int = 10,
                      prob_col: str = "prob_class1"):
    """Reliability diagram overlaying multiple context lengths.

    Matches plot_calibration.py 2A style:
      - bar chart + line + dots per context
      - perfect-calibration diagonal (dashed)
      - ECE annotated below each context's label in the legend
      - colours from viridis_r by context length

    contexts: list of context labels to show (default: 3 representatives).
    k       : windows to aggregate per subject (default 5, matching paper).
    """
    avail = [c for c in (contexts or list(CONTEXT_TO_MIN.keys()))
             if c in parquets and not parquets[c].empty]
    if not avail:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    # 3 representative contexts if not specified explicitly
    if contexts is None:
        show = [avail[0]]
        if len(avail) > 2:
            show.append(avail[len(avail) // 2])
        show.append(avail[-1])
        avail = list(dict.fromkeys(show))

    n = max(len(avail), 2)
    cmap = plt.cm.get_cmap("viridis_r", n)
    ctx_color = {c: cmap(i / (n - 1)) for i, c in enumerate(avail)}

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, zorder=0)

    for ctx in avail:
        df = parquets[ctx]
        if prob_col not in df.columns:
            continue

        # Aggregate K evenly-spaced windows per subject
        rows = []
        for sid, grp in df.sort_values("window_idx").groupby("subject_id"):
            if k == "all" or len(grp) <= k:
                rows.append(grp)
            else:
                idx = np.linspace(0, len(grp) - 1, int(k), dtype=int)
                rows.append(grp.iloc[idx])
        selected = pd.concat(rows, ignore_index=True) if rows else df
        sub = (selected.groupby("subject_id")
               .agg(mean_prob=(prob_col, "mean"),
                    true_label=("true_label", "first"))
               .reset_index())

        if sub.empty or sub["true_label"].nunique() < 2:
            continue

        ece, mids, fracs = _compute_ece(
            sub["true_label"].values.astype(float),
            sub["mean_prob"].values, n_bins=n_bins)

        color = ctx_color[ctx]
        ctx_min = CONTEXT_TO_MIN.get(ctx, "?")
        lbl = f"{ctx}  ECE={ece:.3f}"

        if len(mids):
            ax.bar(mids, fracs, width=0.07, alpha=0.25, color=color,
                   edgecolor="none")
            ax.plot(mids, fracs, "o-", color=color, lw=1.4,
                    ms=4, label=lbl, zorder=3)

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_xlabel("Mean predicted probability", fontsize=FONT_LABEL)
    ax.set_ylabel("Fraction of positives", fontsize=FONT_LABEL)
    ax.legend(fontsize=FONT_ANNOT, frameon=False, handlelength=2.5,
              loc="lower right")
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# S-Fig 11 — Hard subjects (fraction correct across contexts)
# ═══════════════════════════════════════════════════════════════════════════════

def hard_subjects_panel(ax, parquets: dict[str, pd.DataFrame],
                        contexts_ordered: list[str] | None = None,
                        prob_col: str = "prob_class1"):
    """Bar: fraction of subjects correct at 0, 1, 2, … all context lengths."""
    avail_contexts = contexts_ordered or [c for c in CTX_ORDER if c in parquets]
    avail_contexts = [c for c in avail_contexts if c in parquets and not parquets[c].empty]
    if not avail_contexts:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    subject_correct = {}
    for ctx in avail_contexts:
        df = parquets[ctx]
        if prob_col not in df.columns:
            continue
        sub = (df.groupby("subject_id")
               .agg(mean_prob=(prob_col, "mean"), true_label=("true_label", "first"))
               .reset_index())
        for _, row in sub.iterrows():
            sid = row["subject_id"]
            correct = int((row["mean_prob"] >= 0.5) == bool(row["true_label"]))
            subject_correct.setdefault(sid, 0)
            subject_correct[sid] += correct

    counts = np.bincount(list(subject_correct.values()),
                         minlength=len(avail_contexts) + 1)
    n_total = sum(counts)
    fracs   = counts / n_total if n_total > 0 else counts

    x_pos = np.arange(len(avail_contexts) + 1)
    ax.bar(x_pos, fracs, color="#3A7EBF", edgecolor="white", lw=0.4, alpha=0.85)
    for x, f in zip(x_pos, fracs):
        if f > 0.01:
            ax.text(x, f + 0.005, f"{f:.0%}", ha="center", va="bottom",
                    fontsize=FONT_ANNOT)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i) for i in x_pos], fontsize=FONT_BASE)
    ax.set_xlabel("# context lengths correctly predicted", fontsize=FONT_LABEL)
    ax.set_ylabel("Fraction of subjects", fontsize=FONT_LABEL)
    ax.set_ylim(0, min(1.0, fracs.max() * 1.25))
    ax.grid(True, alpha=0.25, lw=0.5, axis="y")
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# S-Fig 12 — Window position profiles
# ═══════════════════════════════════════════════════════════════════════════════

def position_panel(ax, parquets: dict[str, pd.DataFrame],
                   contexts: list[str] = ("30s", "120m", "240m"),
                   n_bins: int = 20,
                   prob_col: str = "prob_class1"):
    """Mean predicted probability vs normalised window position."""
    avail = [c for c in contexts if c in parquets and not parquets[c].empty]
    if not avail:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_BASE, color="gray")
        return

    ctx_colors = {c: _palette(len(avail))[i] for i, c in enumerate(avail)}
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for ctx in avail:
        df = parquets[ctx]
        if prob_col not in df.columns or "window_idx" not in df.columns:
            continue
        df = df.copy()
        df["norm_pos"] = (df.groupby("subject_id")["window_idx"]
                          .transform(lambda x: x / max(x.max(), 1)))
        df["pos_bin"] = pd.cut(df["norm_pos"], bins=bins, labels=False,
                               include_lowest=True)
        binned = (df.groupby("pos_bin")[prob_col]
                  .mean().reindex(range(n_bins)).values)
        valid = ~np.isnan(binned)
        ax.plot(bin_centers[valid], binned[valid] * 100,
                color=ctx_colors[ctx], lw=1.4, label=ctx, marker="o",
                ms=3, markevery=3)

    ax.set_xlabel("Normalised window position (0=start, 1=end)", fontsize=FONT_LABEL)
    ax.set_ylabel("Mean predicted prob (%)", fontsize=FONT_LABEL)
    ax.set_xlim([0, 1])
    ax.legend(title="Context", fontsize=FONT_ANNOT, title_fontsize=FONT_ANNOT,
              frameon=False)
    ax.grid(True, alpha=0.25, lw=0.5)
    _spine_clean(ax)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Fig 6 — Waterfall gain decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def waterfall_panel(ax, analysis_df: pd.DataFrame,
                    task: str,
                    metric: str = "mean_prob_auroc",
                    show_xlabels: bool = True):
    """Waterfall: decompose AUROC gain into aggregation, context, architecture.

    Steps (left → right):
      Base     : MeanPool @ 30s, K=1      (minimal baseline)
      +Agg     : MeanPool @ 30s, K=all    (inference aggregation gain)
      +Context : MeanPool @ 240m, K=all   (training context gain)
      +Arch    : Transformer @ 240m, K=all (architecture gain)
      Final    : Transformer @ 240m, K=all (stacked total)

    analysis_df must include all K values — load with load_analysis_all_k().
    """
    def _get(head, ctx, k):
        sub = analysis_df[
            (analysis_df["task"] == task) &
            (analysis_df["head"] == head) &
            (analysis_df["context_length"] == ctx) &
            (analysis_df["k"].astype(str) == str(k)) &
            analysis_df[metric].notna()
        ]
        return float(sub[metric].iloc[0]) if not sub.empty else np.nan

    v_start = _get("mean_pool",    "30s",  1)
    v_agg   = _get("mean_pool",    "30s",  "all")
    v_ctx   = _get("mean_pool",    "240m", "all")
    v_arch  = _get("transformer",  "240m", "all")

    steps = {
        "Base\n(MPool,30s,K=1)":   (0,       v_start),
        "+Aggregation\n(K=1→Kmax)": (v_start, v_agg  - v_start),
        "+Context\n(30s→240m)":    (v_agg,   v_ctx  - v_agg),
        "+Architecture\n(→Transf.)": (v_ctx,   v_arch - v_ctx),
        "Final\n(Transf.,240m)":   (0,       v_arch),
    }
    is_final = [False, False, False, False, True]

    _C_BASE  = "#3A7EBF"
    _C_POS   = "#44A15E"
    _C_NEG   = "#C94040"
    _C_FINAL = "#E86A33"

    bottoms, heights, colors = [], [], []
    for i, (bot, dlt) in enumerate(steps.values()):
        if is_final[i]:
            bottoms.append(0);         heights.append(v_arch); colors.append(_C_FINAL)
        else:
            bottoms.append(bot);       heights.append(abs(dlt))
            colors.append(_C_BASE if i == 0 else (_C_POS if dlt >= 0 else _C_NEG))

    x = np.arange(len(steps))
    bars = ax.bar(x, heights, bottom=bottoms, color=colors,
                  edgecolor="white", linewidth=0.5, width=0.55)

    for i, (bar, (lbl, (bot, dlt))) in enumerate(zip(bars, steps.items())):
        top = v_arch if is_final[i] else (bot + dlt)
        prefix = "+" if (not is_final[i] and i > 0 and dlt > 0) else ""
        val_str = f"{v_arch:.2f}" if is_final[i] else f"{prefix}{dlt:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                top + 0.004, val_str,
                ha="center", va="bottom", rotation=90,
                fontsize=FONT_BASE+2,
                fontweight="bold" if is_final[i] else "normal")

    for i in range(len(steps) - 2):
        bot, dlt = list(steps.values())[i]
        ax.plot([x[i] + 0.275, x[i + 1] - 0.275], [bot + dlt, bot + dlt],
                color="#888888", linewidth=0.8, linestyle=":")

    ax.set_xticks(x)
    if show_xlabels:
        ax.set_xticklabels(list(steps.keys()), fontsize=FONT_BASE+2,
                           rotation=60, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels([])
    ax.set_ylabel("AUROC", fontsize=FONT_BASE+2)
    ax.tick_params(axis="y", labelsize=FONT_BASE+2)
    ax.set_ylim(max(0, min(v_start, v_agg, v_ctx, v_arch) - 0.06),
                min(1.0, max(v_start, v_agg, v_ctx, v_arch) + 0.05))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _spine_clean(ax)
