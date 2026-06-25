#!/usr/bin/env python3
"""
plot_iso_compute.py — Step 3 of iso-compute analysis pipeline.

Produces 7 iso-compute plots from a heatmap DataFrame (heatmap_df_{split}.csv),
adapted from mock-compute-optimal-tradeoffs-plots-main/ for real experimental data.

Plots produced (one .png + .pdf each):
  1. heatmap_{metric}           — 2D grid: context (Y) × K (X), iso-compute lines
  2. metric_vs_k_{metric}       — per-context AUROC vs K on log-x axis
  3. metric_vs_total_{metric}   — per-context AUROC vs total compute on log-x axis
  4. pareto_front_{metric}      — Pareto-optimal (L, K) at each compute budget
  5. min_cost_frontier_{metric} — cheapest way to reach each target metric value
  6. marginal_gain_{metric}     — marginal gain per additional window (log-log)
  7. double_tradeoff_{metric}   — gain from doubling K vs switching to 2× longer context

Prerequisites:
  Run analyze_windows.py --k-dense  (Step 1)
  Run build_heatmap_df.py           (Step 2)

Usage:
  python scripts/plot_iso_compute.py \\
      --task sex_binary --head lstm \\
      --results-dir /scratch/boshra95/psg/unified/results/phase0_v2 \\
      --split test \\
      --metric auroc balanced_accuracy \\
      --budget 480
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from repo_sync import configure_repo_figures, save_figure


# ── Config ────────────────────────────────────────────────────────────────────

CONTEXT_ORDER = ["30s", "10m", "40m", "80m", "120m", "240m"]
ISO_BUDGETS   = [10, 30, 60, 120, 240, 480]   # minutes


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sort_key(label: str) -> float:
    try:
        return CONTEXT_ORDER.index(label)
    except ValueError:
        return 9999


def _metric_label(col: str) -> str:
    return {
        "auroc":              "AUROC (%)",
        "balanced_accuracy":  "Balanced Accuracy (%)",
        "f1":                 "F1 (%)",
        "seg_auroc":          "Seg AUROC (%)",
        "majority_auroc":     "Majority AUROC (%)",
    }.get(col, col.replace("_", " ").title() + " (%)")


def _load_df(inf_dir: Path, split: str) -> pd.DataFrame:
    path = inf_dir / f"heatmap_df_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Not found: {path}\n"
            "Run build_heatmap_df.py first."
        )
    df = pd.read_csv(path)
    df = df.sort_values(["context_length_min", "k"]).reset_index(drop=True)
    return df


def _build_ctx_lookup(df: pd.DataFrame, col: str) -> dict:
    """Return {ctx_min: (ks_sorted, vals_sorted)} for linear interpolation."""
    lookup = {}
    for ctx_min, grp in df.groupby("context_length_min"):
        grp = grp.dropna(subset=[col]).sort_values("k")
        if grp.empty:
            continue
        lookup[float(ctx_min)] = (
            grp["k"].values.astype(float),
            grp[col].values.astype(float),
        )
    return lookup


def _interp(lookup: dict, ctx_min: float, k: float) -> float:
    """Interpolate metric at (ctx_min, k). Returns NaN if outside data range."""
    if ctx_min not in lookup:
        return float("nan")
    ks, vals = lookup[ctx_min]
    if k < ks[0] or k > ks[-1]:
        return float("nan")
    return float(np.interp(k, ks, vals))


def _palette(n: int):
    return sns.color_palette("viridis", max(n, 2))


def _save(fig, out_dir: Path, stem: str):
    save_figure(fig, out_dir, stem)
    plt.close(fig)


# ── Plot 1: 2D Heatmap ────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, col: str, task: str, head: str,
                 out_dir: Path, budget: float):
    contexts    = sorted(df["context_length_min"].unique())
    ctx_labels  = {ctx: df[df.context_length_min == ctx]["context_label"].iloc[0]
                   for ctx in contexts}

    # Subsampled K axis — prefer these values for a readable x-axis
    target_ks = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50,
                 60, 80, 100, 120, 160, 200, 250, 320, 400, 500]
    all_ks = sorted(df["k"].unique())
    sub_ks = [k for k in target_ks if any(abs(k - ak) < 0.6 for ak in all_ks)]
    if not sub_ks:
        sub_ks = sorted(set(round(k) for k in all_ks))[:30]

    matrix = np.full((len(contexts), len(sub_ks)), np.nan)
    for i, ctx in enumerate(contexts):
        grp = (df[df.context_length_min == ctx]
               .dropna(subset=[col])
               .set_index("k")[col])
        for j, k in enumerate(sub_ks):
            near = grp.index.values
            if len(near) == 0:
                continue
            idx = np.argmin(np.abs(near - k))
            if abs(near[idx] - k) < 1:
                matrix[i, j] = grp.iloc[idx]

    valid = matrix[~np.isnan(matrix)] * 100
    if valid.size == 0:
        print(f"  [heatmap] No data for {col} — skipping.")
        return
    vmin = float(np.floor(valid.min() / 5) * 5)
    vmax = float(np.ceil(valid.max() / 5) * 5)

    fig, ax = plt.subplots(figsize=(max(14, len(sub_ks) * 0.45),
                                    max(5, len(contexts) * 0.9 + 2)))
    sns.heatmap(
        matrix * 100, ax=ax,
        cmap=sns.color_palette("YlOrRd", as_cmap=True),
        xticklabels=[str(k) for k in sub_ks],
        yticklabels=[ctx_labels[c] for c in contexts],
        cbar_kws={"label": _metric_label(col)},
        linewidths=0.3, linecolor="white",
        mask=np.isnan(matrix), vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel("k (windows per subject)", fontsize=12)
    ax.set_ylabel("Context Length", fontsize=12)
    ax.set_title(f"{task} · {head}  —  Iso-Compute Heatmap: {_metric_label(col)}",
                 fontsize=13)

    # Iso-compute lines
    iso_colors = plt.cm.cool(np.linspace(0.2, 0.9, len(ISO_BUDGETS)))
    for ic, cb in enumerate(ISO_BUDGETS):
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
            ax.plot(xs, ys, color=iso_colors[ic], linewidth=2.5,
                    linestyle="--", alpha=0.85)
            ax.annotate(lt, (xs[0], ys[0]), fontsize=9, fontweight="bold",
                        color=iso_colors[ic], ha="center", va="bottom",
                        xytext=(0, -14), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec=iso_colors[ic], alpha=0.8))
    plt.tight_layout()
    _save(fig, out_dir, f"heatmap_{col}")


# ── Plot 2: Metric vs K ───────────────────────────────────────────────────────

def plot_vs_k(df: pd.DataFrame, col: str, task: str, head: str, out_dir: Path):
    contexts   = sorted(df["context_length_min"].unique())
    palette    = _palette(len(contexts))
    lookup     = _build_ctx_lookup(df, col)
    ctx_labels = {ctx: df[df.context_length_min == ctx]["context_label"].iloc[0]
                  for ctx in contexts}

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, ctx in enumerate(contexts):
        if ctx not in lookup:
            continue
        ks, vals = lookup[ctx]
        ax.plot(ks, vals * 100, color=palette[i], linewidth=2,
                label=ctx_labels[ctx], marker="o", markersize=4,
                markevery=max(1, len(ks) // 12))

    iso_colors = plt.cm.Greys(np.linspace(0.3, 0.75, len(ISO_BUDGETS)))
    for ic, cb in enumerate(ISO_BUDGETS):
        pts = [(cb / ctx, _interp(lookup, ctx, cb / ctx)) for ctx in contexts]
        pts = [(k, v) for k, v in pts if not np.isnan(v)]
        if len(pts) >= 2:
            pts.sort()
            ks_iso, vs_iso = zip(*pts)
            lt = f"{cb}m" if cb < 60 else f"{cb // 60}h"
            ax.plot(ks_iso, [v * 100 for v in vs_iso],
                    color=iso_colors[ic], linewidth=2, linestyle="--", alpha=0.8)
            ax.annotate(lt, (ks_iso[-1], vs_iso[-1] * 100), fontsize=9,
                        fontweight="bold", color=iso_colors[ic],
                        ha="left", va="bottom", xytext=(4, 2),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec=iso_colors[ic], alpha=0.8))

    ax.set_xscale("log")
    ax.set_xlabel("k (windows per subject)", fontsize=12)
    ax.set_ylabel(_metric_label(col), fontsize=12)
    ax.set_title(f"{task} · {head}  —  {_metric_label(col)} vs k", fontsize=13)
    ax.legend(title="Context Length", fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, f"metric_vs_k_{col}")


# ── Plot 3: Metric vs Total Context ──────────────────────────────────────────

def plot_vs_total(df: pd.DataFrame, col: str, task: str, head: str, out_dir: Path):
    contexts   = sorted(df["context_length_min"].unique())
    palette    = _palette(len(contexts))
    lookup     = _build_ctx_lookup(df, col)
    ctx_labels = {ctx: df[df.context_length_min == ctx]["context_label"].iloc[0]
                  for ctx in contexts}

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, ctx in enumerate(contexts):
        if ctx not in lookup:
            continue
        ks, vals = lookup[ctx]
        ax.plot(ks * ctx, vals * 100, color=palette[i], linewidth=2,
                label=ctx_labels[ctx], marker="o", markersize=4,
                markevery=max(1, len(ks) // 12))

    ax.set_xscale("log")
    ax.set_xlabel("Total context (minutes) = context_length × k", fontsize=12)
    ax.set_ylabel(_metric_label(col), fontsize=12)
    ax.set_title(f"{task} · {head}  —  {_metric_label(col)} vs Total Context",
                 fontsize=13)
    ax.legend(title="Context Length", fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, f"metric_vs_total_{col}")


# ── Plot 4: Pareto Front ──────────────────────────────────────────────────────

def plot_pareto(df: pd.DataFrame, col: str, task: str, head: str,
                out_dir: Path, budget: float):
    contexts   = sorted(df["context_length_min"].unique())
    palette    = _palette(len(contexts))
    lookup     = _build_ctx_lookup(df, col)
    ctx_labels = {ctx: df[df.context_length_min == ctx]["context_label"].iloc[0]
                  for ctx in contexts}

    budgets_sweep = np.unique(np.concatenate([
        np.arange(0.5, 20, 0.5),
        np.arange(20, 100, 2),
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
        print(f"  [pareto] No data — skipping.")
        return

    ctx_to_color = {c: palette[i] for i, c in enumerate(contexts)}
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, ctx in enumerate(contexts):
        if ctx not in lookup:
            continue
        ks, vals = lookup[ctx]
        ax.plot(ks * ctx, vals * 100, color=palette[i], linewidth=1, alpha=0.2)

    segments, s = [], 0
    for j in range(1, len(opt_b)):
        if opt_c[j] != opt_c[j - 1]:
            segments.append((s, j)); s = j
    segments.append((s, len(opt_b)))

    labeled = set()
    for s, e in segments:
        ctx = opt_c[s]
        lbl = ctx_labels[ctx] if ctx not in labeled else None
        ax.plot(opt_b[s:e], [v * 100 for v in opt_v[s:e]],
                color=ctx_to_color[ctx], linewidth=3.5,
                solid_capstyle="round", label=lbl)
        labeled.add(ctx)
        mid = (s + e) // 2
        ax.annotate(f"{ctx_labels[ctx]}\nk≈{opt_k[mid]:.0f}",
                    (opt_b[mid], opt_v[mid] * 100),
                    fontsize=9, fontweight="bold", color=ctx_to_color[ctx],
                    ha="center", va="bottom", xytext=(0, 8),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=ctx_to_color[ctx], alpha=0.85))

    for s, e in segments:
        if s > 0:
            ax.plot(opt_b[s], opt_v[s] * 100, "o",
                    color=ctx_to_color[opt_c[s]], markersize=7, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Total compute budget (minutes)", fontsize=12)
    ax.set_ylabel(f"Best achievable {_metric_label(col)}", fontsize=12)
    ax.set_title(f"{task} · {head}  —  Pareto Front: {_metric_label(col)}",
                 fontsize=13)
    ax.legend(title="Optimal context", fontsize=9, title_fontsize=10,
              loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, f"pareto_front_{col}")


# ── Plot 5: Min-Cost Frontier ─────────────────────────────────────────────────

def plot_min_cost(df: pd.DataFrame, col: str, task: str, head: str,
                  out_dir: Path, budget: float):
    contexts   = sorted(df["context_length_min"].unique())
    palette    = _palette(len(contexts))
    lookup     = _build_ctx_lookup(df, col)
    ctx_labels = {ctx: df[df.context_length_min == ctx]["context_label"].iloc[0]
                  for ctx in contexts}

    all_vals = [v for ks, vs in lookup.values() for v in vs if not np.isnan(v)]
    if not all_vals:
        return
    val_min = float(np.floor(min(all_vals) * 20) / 20)
    val_max = float(np.ceil(max(all_vals) * 20) / 20)
    target_vals = np.arange(val_min, val_max + 0.002, 0.005)
    annot_targets = [t for t in
                     [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
                     if val_min <= t <= val_max]

    fig, ax = plt.subplots(figsize=(14, 7))

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
            ax.plot(tgts, costs, color=palette[i], linewidth=2,
                    label=ctx_labels[ctx])

    for tgt in annot_targets:
        best_cost, best_k, best_i = float("inf"), None, None
        for i, ctx in enumerate(contexts):
            if ctx not in lookup:
                continue
            ks, vals = lookup[ctx]
            idx = np.searchsorted(vals, tgt)
            if idx >= len(ks):
                continue
            cost = ctx * ks[idx]
            if cost < best_cost:
                best_cost, best_k, best_i = cost, ks[idx], i
        if best_i is not None and best_cost <= budget:
            ax.plot(tgt * 100, best_cost, "o", color=palette[best_i],
                    markersize=8, zorder=5,
                    markeredgecolor="black", markeredgewidth=0.8)
            ax.annotate(
                f"k={best_k:.0f}\n{ctx_labels[contexts[best_i]]}",
                (tgt * 100, best_cost), fontsize=8, fontweight="bold",
                color=palette[best_i], ha="center", va="top",
                xytext=(0, -12), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=palette[best_i], alpha=0.85))

    ax.set_yscale("log")
    ax.set_xlabel(f"Target {_metric_label(col)}", fontsize=12)
    ax.set_ylabel("Minimum total compute (minutes)", fontsize=12)
    ax.set_title(f"{task} · {head}  —  Cheapest Way to Reach Target "
                 f"{_metric_label(col)}", fontsize=13)
    ax.legend(title="Context Length", fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=budget, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.annotate(f"{budget:.0f}m budget", (target_vals[0] * 100, budget),
                fontsize=9, color="red", va="bottom",
                xytext=(5, 3), textcoords="offset points")
    for cb in [10, 30, 60, 120, 240]:
        lt = f"{cb}m" if cb < 60 else f"{cb // 60}h"
        ax.axhline(y=cb, color="grey", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.annotate(lt, (target_vals[-1] * 100, cb), fontsize=8, color="grey",
                    ha="right", va="bottom")
    plt.tight_layout()
    _save(fig, out_dir, f"min_cost_frontier_{col}")


# ── Plot 6: Marginal Gain ─────────────────────────────────────────────────────

def plot_marginal(df: pd.DataFrame, col: str, task: str, head: str,
                  out_dir: Path):
    contexts   = sorted(df["context_length_min"].unique())
    palette    = _palette(len(contexts))
    lookup     = _build_ctx_lookup(df, col)
    ctx_labels = {ctx: df[df.context_length_min == ctx]["context_label"].iloc[0]
                  for ctx in contexts}

    fig, ax = plt.subplots(figsize=(12, 6))
    any_data = False

    for i, ctx in enumerate(contexts):
        if ctx not in lookup:
            continue
        ks, vals = lookup[ctx]
        if len(ks) < 2:
            continue
        dk = np.diff(ks)
        dv = np.diff(vals) / np.where(dk > 0, dk, np.nan)
        k_mid = (ks[:-1] + ks[1:]) / 2
        mask = (~np.isnan(dv)) & (dv > 0)
        if mask.sum() < 1:
            continue
        ax.plot(k_mid[mask], dv[mask] * 100, color=palette[i], linewidth=1.5,
                label=ctx_labels[ctx], alpha=0.85)
        any_data = True

    if not any_data:
        plt.close()
        print(f"  [marginal] No positive marginals — skipping.")
        return

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("k (windows per subject)", fontsize=12)
    ax.set_ylabel(f"Marginal {_metric_label(col)} gain per vote", fontsize=12)
    ax.set_title(f"{task} · {head}  —  Diminishing Returns: Marginal Gain per Vote",
                 fontsize=13)
    ax.legend(title="Context Length", fontsize=9, title_fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir, f"marginal_gain_{col}")


# ── Plot 7: Double Tradeoff ───────────────────────────────────────────────────

def plot_double(df: pd.DataFrame, col: str, task: str, head: str,
                out_dir: Path):
    contexts   = sorted(df["context_length_min"].unique())
    lookup     = _build_ctx_lookup(df, col)
    ctx_labels = {ctx: df[df.context_length_min == ctx]["context_label"].iloc[0]
                  for ctx in contexts}

    # Pair each context with the nearest one ≥ 1.5× longer
    ctx_pairs = []
    for ctx in contexts[:-1]:
        candidates = [c for c in contexts if c >= ctx * 1.5]
        if candidates:
            ctx2 = min(candidates)
            ctx_pairs.append((ctx, ctx2))

    if not ctx_pairs:
        print("  [double] Not enough context lengths — skipping.")
        return

    n     = len(ctx_pairs)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for idx, (ctx, ctx2) in enumerate(ctx_pairs):
        ax = axes_flat[idx]
        if ctx not in lookup or ctx2 not in lookup:
            ax.set_visible(False); continue

        ks1 = lookup[ctx][0]
        ks2 = lookup[ctx2][0]

        budgets = np.unique(np.concatenate([
            np.arange(ctx, 20, max(ctx, 0.5)),
            np.arange(20, 100, 5),
            np.arange(100, 600, 10),
        ]))

        bs, gk_list, gc_list = [], [], []
        for b in budgets:
            k = b / ctx
            if k < 1 or k > ks1[-1]:
                continue
            k2 = 2 * k
            if k2 > ks1[-1]:
                continue
            k_ctx2 = b / ctx2
            if k_ctx2 < 1 or k_ctx2 > ks2[-1]:
                continue

            v0   = _interp(lookup, ctx,  k)
            v_dk = _interp(lookup, ctx,  k2)
            v_dc = _interp(lookup, ctx2, k_ctx2)
            if any(np.isnan(x) for x in [v0, v_dk, v_dc]):
                continue

            bs.append(b)
            gk_list.append((v_dk - v0) * 100)
            gc_list.append((v_dc - v0) * 100)

        if bs:
            ax.plot(bs, gk_list, color="steelblue", linewidth=2, label="Double k")
            ax.plot(bs, gc_list, color="coral",     linewidth=2,
                    label=f"Switch to {ctx_labels[ctx2]}")
            ax.axhline(y=0, color="grey", linewidth=0.5)
            ax.fill_between(bs, gk_list, gc_list,
                            where=[a > b for a, b in zip(gk_list, gc_list)],
                            alpha=0.15, color="steelblue")
            ax.fill_between(bs, gk_list, gc_list,
                            where=[b > a for a, b in zip(gk_list, gc_list)],
                            alpha=0.15, color="coral")

        ax.set_xscale("log")
        ax.set_title(f"From {ctx_labels[ctx]}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlabel("Current compute budget (min)", fontsize=9)

    for idx in range(len(ctx_pairs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.supylabel(f"{_metric_label(col)} gain from doubling", fontsize=12, x=0.02)
    fig.suptitle(f"{task} · {head}  —  Double Context vs Double k?",
                 fontsize=13, y=0.98)
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    _save(fig, out_dir, f"double_tradeoff_{col}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Produce 7 iso-compute plots from a heatmap DataFrame."
    )
    parser.add_argument("--task",    required=True)
    parser.add_argument("--head",    required=True)
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0_v2"),
                        dest="results_dir")
    parser.add_argument("--split",   default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--run-tag", default="", dest="run_tag")
    parser.add_argument("--metric",  nargs="+",
                        default=["auroc", "balanced_accuracy"],
                        help="Metric column(s) in heatmap_df to plot "
                             "(default: auroc balanced_accuracy)")
    parser.add_argument("--budget",  type=float, default=480.0,
                        help="Max compute budget in minutes for Pareto / min-cost "
                             "plots (default: 480)")
    parser.add_argument("--repo-figures-dir", type=Path, default=None,
                        dest="repo_figures_dir",
                        help="Also mirror PNGs into this repo dir (e.g. "
                             "results/figures/phase0_v3). Default: no repo mirror.")
    args = parser.parse_args()

    configure_repo_figures(args.results_dir, args.repo_figures_dir)
    exp_id  = (f"{args.task}_{args.head}"
               + (f"_{args.run_tag}" if args.run_tag else ""))
    inf_dir = args.results_dir / "inference" / exp_id

    df = _load_df(inf_dir, args.split)
    print(f"Loaded heatmap_df: {len(df)} rows, "
          f"{df['context_length_min'].nunique()} contexts, "
          f"{df['k'].nunique()} K values")

    for col in args.metric:
        if col not in df.columns:
            print(f"  [skip] '{col}' not in DataFrame "
                  f"— run build_heatmap_df.py with --metrics {col}")
            continue
        out_dir = (args.results_dir / "figures" / exp_id
                   / f"{col}_{args.split}")
        print(f"\n── {col.upper()}  →  {out_dir}")
        plot_heatmap(df, col, args.task, args.head, out_dir, args.budget)
        plot_vs_k(df, col, args.task, args.head, out_dir)
        plot_vs_total(df, col, args.task, args.head, out_dir)
        plot_pareto(df, col, args.task, args.head, out_dir, args.budget)
        plot_min_cost(df, col, args.task, args.head, out_dir, args.budget)
        plot_marginal(df, col, args.task, args.head, out_dir)
        plot_double(df, col, args.task, args.head, out_dir)
        print(f"  All 7 plots → {out_dir}")


if __name__ == "__main__":
    main()
