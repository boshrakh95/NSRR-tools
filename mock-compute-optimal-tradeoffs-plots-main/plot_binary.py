#!/usr/bin/env python3
"""Binary classification analysis: Context Length vs Majority Voting."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom, norm
from functools import lru_cache

# ============================================================
# Configuration
# ============================================================
context_lengths_min = [0.5, 5, 10, 20, 40, 80, 120, 180, 240]
context_labels = ["30s", "5m", "10m", "20m", "40m", "80m", "120m", "180m", "240m"]
budget_min = 16 * 60  # 960 minutes
prevalence = 0.5

max_ks = [int(budget_min // c) for c in context_lengths_min]
all_ks = sorted(set(k for mk in max_ks for k in range(1, mk + 1)))

target_ks_display = sorted(set(
    [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 60, 80, 96,
     120, 160, 192, 240, 320, 384, 480, 640, 768, 960, 1280, 1920]
))
sub_ks = [k for k in target_ks_display if k in all_ks]

budgets_sweep = np.unique(np.concatenate([
    np.arange(1, 20, 1), np.arange(20, 100, 5),
    np.arange(100, budget_min + 1, 10),
]))

palette = sns.color_palette("viridis", len(context_lengths_min))

# ============================================================
# Base classifier (single prediction, no voting)
# ============================================================
def base_tpr(ctx_min):
    return 0.51 + 0.22 * np.log1p(ctx_min) / np.log1p(240)

def base_fpr(ctx_min):
    return 0.48 - 0.28 * np.log1p(ctx_min) / np.log1p(240)


# ============================================================
# Majority voting metric computation
# ============================================================
@lru_cache(maxsize=None)
def _tpr_maj(k, ctx_min):
    return float(1 - binom.cdf(k // 2, k, base_tpr(ctx_min)))

@lru_cache(maxsize=None)
def _fpr_maj(k, ctx_min):
    return float(1 - binom.cdf(k // 2, k, base_fpr(ctx_min)))

def m_accuracy(ctx, k):
    t, f = _tpr_maj(k, ctx), _fpr_maj(k, ctx)
    return (t * prevalence + (1 - f) * (1 - prevalence)) * 100

def m_auroc(ctx, k):
    tpr, fpr = base_tpr(ctx), base_fpr(ctx)
    d = tpr - fpr
    var = tpr * (1 - tpr) + fpr * (1 - fpr)
    return norm.cdf(np.sqrt(k) * d / np.sqrt(var)) * 100 if var > 0 else 100.0

def m_precision(ctx, k):
    t, f = _tpr_maj(k, ctx), _fpr_maj(k, ctx)
    tp, fp = t * prevalence, f * (1 - prevalence)
    return (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0

def m_recall(ctx, k):
    return _tpr_maj(k, ctx) * 100

def m_f1(ctx, k):
    p, r = m_precision(ctx, k), m_recall(ctx, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

METRICS = {
    "accuracy":  {"func": m_accuracy,  "label": "Accuracy (%)",  "vmin": 50, "vmax": 100,
                  "annot": [55, 60, 70, 80, 90, 95, 99]},
    "auroc":     {"func": m_auroc,     "label": "AUROC (%)",     "vmin": 50, "vmax": 100,
                  "annot": [55, 60, 70, 80, 90, 95, 99]},
    "f1":        {"func": m_f1,        "label": "F1 Score (%)",  "vmin": 0,  "vmax": 100,
                  "annot": [30, 50, 60, 70, 80, 90, 95]},
    "precision": {"func": m_precision, "label": "Precision (%)", "vmin": 50, "vmax": 100,
                  "annot": [55, 60, 70, 80, 90, 95, 99]},
    "recall":    {"func": m_recall,    "label": "Recall (%)",    "vmin": 0,  "vmax": 100,
                  "annot": [55, 60, 70, 80, 90, 95]},
}


# ============================================================
# Plot 1: Iso-compute heatmap
# ============================================================
def plot_heatmap(mf, mc, outdir):
    matrix = np.full((len(context_lengths_min), len(sub_ks)), np.nan)
    for i, ctx in enumerate(context_lengths_min):
        for j, k in enumerate(sub_ks):
            if ctx * k <= budget_min:
                matrix[i, j] = mf(ctx, k)

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.heatmap(matrix, ax=ax, cmap=sns.color_palette("YlOrRd", as_cmap=True),
                xticklabels=[str(k) for k in sub_ks], yticklabels=context_labels,
                cbar_kws={"label": mc["label"]}, linewidths=0.5, linecolor="white",
                mask=np.isnan(matrix), vmin=mc["vmin"], vmax=mc["vmax"])
    ax.set_xlabel("k (number of majority votes)", fontsize=13)
    ax.set_ylabel("Context Length", fontsize=13)
    ax.set_title(f"Iso-Compute Heatmap: {mc['label']} (Budget = 16h)", fontsize=15)

    iso_computes = [10, 30, 60, 120, 240, 480]
    iso_colors = plt.cm.cool(np.linspace(0.2, 0.9, len(iso_computes)))
    for ic, cb in enumerate(iso_computes):
        xs, ys = [], []
        for i, ctx in enumerate(context_lengths_min):
            kn = cb / ctx
            if kn < 1 or kn > max_ks[i]:
                continue
            for jj in range(len(sub_ks) - 1):
                if sub_ks[jj] <= kn <= sub_ks[jj + 1]:
                    frac = (kn - sub_ks[jj]) / (sub_ks[jj + 1] - sub_ks[jj])
                    xs.append(jj + frac + 0.5); ys.append(i + 0.5)
                    break
        if len(xs) >= 2:
            lt = f"{cb}m" if cb < 60 else f"{cb // 60}h"
            ax.plot(xs, ys, color=iso_colors[ic], linewidth=2.5, linestyle="--", alpha=0.85)
            ax.annotate(lt, (xs[0], ys[0]), fontsize=10, fontweight="bold",
                        color=iso_colors[ic], ha="center", va="bottom",
                        xytext=(0, -14), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec=iso_colors[ic], alpha=0.8))
    plt.tight_layout()
    plt.savefig(f"{outdir}/heatmap.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{outdir}/heatmap.pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Plot 2: Metric vs k
# ============================================================
def plot_vs_k(mf, mc, outdir):
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (ctx, label) in enumerate(zip(context_lengths_min, context_labels)):
        ks_valid = [k for k in all_ks if ctx * k <= budget_min]
        vals = [mf(ctx, k) for k in ks_valid]
        ax.plot(ks_valid, vals, color=palette[i], linewidth=2, label=label,
                marker="o", markersize=3, markevery=max(1, len(ks_valid) // 15))

    # Iso-compute lines
    iso_computes = [10, 30, 60, 120, 240, 480]
    iso_colors = plt.cm.Greys(np.linspace(0.3, 0.7, len(iso_computes)))
    for ic, cb in enumerate(iso_computes):
        ks_iso, vals_iso = [], []
        for ctx in context_lengths_min:
            ka = cb / ctx
            if ka < 1 or ka > budget_min / ctx:
                continue
            ks_iso.append(ka); vals_iso.append(mf(ctx, int(round(ka))))
        if len(ks_iso) >= 2:
            order = np.argsort(ks_iso)
            ks_iso = [ks_iso[o] for o in order]
            vals_iso = [vals_iso[o] for o in order]
            lt = f"{cb}m" if cb < 60 else f"{cb // 60}h"
            ax.plot(ks_iso, vals_iso, color=iso_colors[ic], linewidth=2,
                    linestyle="--", alpha=0.8)
            ax.annotate(lt, (ks_iso[-1], vals_iso[-1]), fontsize=9,
                        fontweight="bold", color=iso_colors[ic],
                        ha="left", va="bottom", xytext=(4, 2),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec=iso_colors[ic], alpha=0.8))

    ax.set_xscale("log")
    ax.set_xlabel("k (number of majority votes)", fontsize=13)
    ax.set_ylabel(mc["label"], fontsize=13)
    ax.set_title(f"{mc['label']} vs k by Context Length", fontsize=15)
    ax.legend(title="Context Length", fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/vs_k.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{outdir}/vs_k.pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Plot 3: Metric vs total context
# ============================================================
def plot_vs_total(mf, mc, outdir):
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (ctx, label) in enumerate(zip(context_lengths_min, context_labels)):
        ks_valid = [k for k in all_ks if ctx * k <= budget_min]
        total = [ctx * k for k in ks_valid]
        vals = [mf(ctx, k) for k in ks_valid]
        ax.plot(total, vals, color=palette[i], linewidth=2, label=label,
                marker="o", markersize=3, markevery=max(1, len(ks_valid) // 15))
    ax.set_xscale("log")
    ax.set_xlabel("Total Context (minutes) = context_length x k", fontsize=13)
    ax.set_ylabel(mc["label"], fontsize=13)
    ax.set_title(f"{mc['label']} vs Total Context", fontsize=15)
    ax.legend(title="Context Length", fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/vs_total_context.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{outdir}/vs_total_context.pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Plot 4: Optimal tradeoff (Pareto front)
# ============================================================
def plot_pareto(mf, mc, outdir):
    opt_b, opt_v, opt_c, opt_k = [], [], [], []
    for b in budgets_sweep:
        best_v, best_ctx, best_k = -1, None, None
        for ctx in context_lengths_min:
            km = int(b // ctx)
            if km < 1:
                continue
            v = mf(ctx, km)
            if v > best_v:
                best_v, best_ctx, best_k = v, ctx, km
        if best_ctx is not None:
            opt_b.append(b); opt_v.append(best_v)
            opt_c.append(best_ctx); opt_k.append(best_k)

    fig, ax = plt.subplots(figsize=(13, 6))
    ctx_to_color = {c: palette[i] for i, c in enumerate(context_lengths_min)}
    ctx_to_label = dict(zip(context_lengths_min, context_labels))

    # Background
    for i, (ctx, label) in enumerate(zip(context_lengths_min, context_labels)):
        bs, vs = [], []
        for b in budgets_sweep:
            km = int(b // ctx)
            if km < 1: continue
            bs.append(b); vs.append(mf(ctx, km))
        ax.plot(bs, vs, color=palette[i], linewidth=1, alpha=0.25)

    # Segments
    segments, s = [], 0
    for j in range(1, len(opt_b)):
        if opt_c[j] != opt_c[j - 1]:
            segments.append((s, j)); s = j
    segments.append((s, len(opt_b)))

    labeled = set()
    for s, e in segments:
        ctx = opt_c[s]
        lbl = ctx_to_label[ctx] if ctx not in labeled else None
        ax.plot(opt_b[s:e], opt_v[s:e], color=ctx_to_color[ctx],
                linewidth=3.5, solid_capstyle="round", label=lbl)
        labeled.add(ctx)
        mid = (s + e) // 2
        ax.annotate(f"{ctx_to_label[ctx]}\nmaj@{opt_k[mid]}",
                    (opt_b[mid], opt_v[mid]), fontsize=9, fontweight="bold",
                    color=ctx_to_color[ctx], ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=ctx_to_color[ctx], alpha=0.85))

    for s, e in segments:
        if s > 0:
            ax.plot(opt_b[s], opt_v[s], "o", color=ctx_to_color[opt_c[s]],
                    markersize=7, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Total Compute Budget (minutes)", fontsize=13)
    ax.set_ylabel(f"Best {mc['label']}", fontsize=13)
    ax.set_title(f"Optimal Tradeoff: {mc['label']} (Pareto Front)", fontsize=15)
    ax.legend(title="Optimal Context", fontsize=10, title_fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/pareto.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{outdir}/pareto.pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Plot 5: Min-cost frontier
# ============================================================
def plot_min_cost(mf, mc, outdir):
    fig, ax = plt.subplots(figsize=(14, 7))
    target_vals = np.arange(mc["vmin"], mc["vmax"] + 0.5, 0.5)

    for i, (ctx, label) in enumerate(zip(context_lengths_min, context_labels)):
        costs, targets = [], []
        for t in target_vals:
            for k in range(1, int(budget_min // ctx) + 1):
                if mf(ctx, k) >= t:
                    costs.append(ctx * k); targets.append(t)
                    break
        if targets:
            ax.plot(targets, costs, color=palette[i], linewidth=2, label=label)

    for target in mc["annot"]:
        best_cost, best_k, best_i = float("inf"), None, None
        for i, ctx in enumerate(context_lengths_min):
            for k in range(1, int(budget_min // ctx) + 1):
                if mf(ctx, k) >= target:
                    cost = ctx * k
                    if cost < best_cost:
                        best_cost, best_k, best_i = cost, k, i
                    break
        if best_i is not None and best_cost <= budget_min:
            ax.plot(target, best_cost, "o", color=palette[best_i], markersize=8,
                    zorder=5, markeredgecolor="black", markeredgewidth=0.8)
            ax.annotate(f"k={best_k}\n{context_labels[best_i]}",
                        (target, best_cost), fontsize=8, fontweight="bold",
                        color=palette[best_i], ha="center", va="top",
                        xytext=(0, -12), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec=palette[best_i], alpha=0.85))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"Target {mc['label']}", fontsize=13)
    ax.set_ylabel("Minimum Total Compute (minutes)", fontsize=13)
    ax.set_title(f"Cheapest Way to Reach Target {mc['label']}", fontsize=15)
    ax.legend(title="Context Length", fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=budget_min, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.annotate("16h budget", (target_vals[0], budget_min), fontsize=10,
                color="red", va="bottom", xytext=(5, 3), textcoords="offset points")
    for cb in [10, 30, 60, 120, 240, 480]:
        lt = f"{cb}m" if cb < 60 else f"{cb // 60}h"
        ax.axhline(y=cb, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.annotate(lt, (target_vals[-1], cb), fontsize=8, color="grey",
                    ha="right", va="bottom")
    plt.tight_layout()
    plt.savefig(f"{outdir}/min_cost.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{outdir}/min_cost.pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Plot 6: Marginal gain
# ============================================================
def plot_marginal(mf, mc, outdir):
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (ctx, label) in enumerate(zip(context_lengths_min, context_labels)):
        km = int(budget_min // ctx)
        if km < 2: continue
        ks = np.arange(1, km + 1)
        vals = np.array([mf(ctx, k) for k in ks])
        marginal = np.diff(vals)
        # Filter to positive marginals for log scale
        mask = marginal > 0
        if mask.any():
            ax.plot(ks[1:][mask], marginal[mask], color=palette[i],
                    linewidth=1.5, label=label, alpha=0.8)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("k (number of majority votes)", fontsize=13)
    ax.set_ylabel(f"Marginal {mc['label']} Gain per Vote", fontsize=13)
    ax.set_title(f"Diminishing Returns: Marginal {mc['label']} Gain", fontsize=15)
    ax.legend(title="Context Length", fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.01, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.annotate("0.01% threshold", (2, 0.01), fontsize=10, color="red",
                va="bottom", xytext=(5, 3), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(f"{outdir}/marginal.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{outdir}/marginal.pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Plot 7: Double context vs double k
# ============================================================
def plot_double(mf, mc, outdir):
    ctx_pairs = []
    for i, ctx in enumerate(context_lengths_min[:-1]):
        doubled = [c for c in context_lengths_min if c >= 2 * ctx * 0.8]
        if doubled:
            ctx_pairs.append((i, ctx,
                              context_lengths_min.index(doubled[0]), doubled[0]))

    n = len(ctx_pairs)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(16, 9),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, (i1, ctx, i2, ctx2) in enumerate(ctx_pairs):
        ax = axes[idx]
        bs, gk, gc = [], [], []
        for b in budgets_sweep:
            k = int(b // ctx)
            if k < 1: continue
            k2 = 2 * k
            if ctx * k2 > budget_min: continue
            if ctx2 * k > budget_min: continue
            bs.append(b)
            gk.append(mf(ctx, k2) - mf(ctx, k))
            gc.append(mf(ctx2, k) - mf(ctx, k))

        if bs:
            ax.plot(bs, gk, color="steelblue", linewidth=2, label="Double k")
            ax.plot(bs, gc, color="coral", linewidth=2,
                    label=f"Double ctx ({context_labels[i2]})")
            ax.axhline(y=0, color="grey", linewidth=0.5)
            ax.fill_between(bs, gk, gc,
                            where=[a > b for a, b in zip(gk, gc)],
                            alpha=0.15, color="steelblue")
            ax.fill_between(bs, gk, gc,
                            where=[b > a for a, b in zip(gk, gc)],
                            alpha=0.15, color="coral")
            ax.set_xscale("log")
            ax.set_title(f"From {context_labels[i1]}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    for idx in range(len(ctx_pairs), len(axes)):
        axes[idx].set_visible(False)

    fig.supxlabel("Current Compute Budget (minutes)", fontsize=13, y=0.02)
    fig.supylabel(f"{mc['label']} Gain from Doubling", fontsize=13, x=0.02)
    fig.suptitle(f"Double Context vs Double k ({mc['label']})", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0.03, 0.04, 1, 0.96])
    plt.savefig(f"{outdir}/double_tradeoff.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{outdir}/double_tradeoff.pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# ADDITIONAL PLOT A: ROC curves at iso-compute budgets
# ============================================================
def plot_roc_iso_compute():
    """At equal total compute, which (ctx, k) gives better discrimination?"""
    iso_budgets = [10, 60, 240, 960]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, cb in enumerate(iso_budgets):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        for i, (ctx, label) in enumerate(zip(context_lengths_min, context_labels)):
            k = int(cb // ctx)
            if k < 1: continue
            tpr_base = base_tpr(ctx)
            fpr_base = base_fpr(ctx)
            # Full ROC: vary threshold from 0 to k+1
            thresholds = np.arange(0, k + 2)
            tprs = np.ones(len(thresholds))
            fprs = np.ones(len(thresholds))
            for ti, t in enumerate(thresholds):
                if t == 0:
                    tprs[ti] = 1.0; fprs[ti] = 1.0
                else:
                    tprs[ti] = 1 - binom.cdf(t - 1, k, tpr_base)
                    fprs[ti] = 1 - binom.cdf(t - 1, k, fpr_base)
            ax.plot(fprs, tprs, color=palette[i], linewidth=2,
                    label=f"{label} (k={k})", alpha=0.8)

        lt = f"{cb}m" if cb < 60 else f"{cb // 60}h"
        ax.set_title(f"Budget = {lt}", fontsize=12, fontweight="bold")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    fig.suptitle("ROC Curves at Iso-Compute Budgets", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("img/binary/roc_iso_compute.png", dpi=200, bbox_inches="tight")
    plt.savefig("img/binary/roc_iso_compute.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved roc_iso_compute")


# ============================================================
# ADDITIONAL PLOT B: Recall at fixed precision frontier
# ============================================================
def plot_recall_at_fixed_precision():
    """In precision-constrained settings, how to allocate compute?"""
    prec_targets = [0.80, 0.90, 0.95]
    fig, axes = plt.subplots(1, len(prec_targets), figsize=(18, 6), sharey=True)

    for pidx, prec_target in enumerate(prec_targets):
        ax = axes[pidx]
        for i, (ctx, label) in enumerate(zip(context_lengths_min, context_labels)):
            budgets_plot, recalls = [], []
            km = int(budget_min // ctx)
            for k in range(1, km + 1):
                tpr_base = base_tpr(ctx)
                fpr_base = base_fpr(ctx)
                # Find threshold where precision >= target
                best_recall = 0
                for t in range(1, k + 2):
                    tpr_t = float(1 - binom.cdf(t - 1, k, tpr_base))
                    fpr_t = float(1 - binom.cdf(t - 1, k, fpr_base))
                    if tpr_t + fpr_t == 0:
                        continue
                    prec_t = tpr_t / (tpr_t + fpr_t)  # balanced
                    if prec_t >= prec_target and tpr_t > best_recall:
                        best_recall = tpr_t
                if best_recall > 0:
                    budgets_plot.append(ctx * k)
                    recalls.append(best_recall * 100)
            if budgets_plot:
                ax.plot(budgets_plot, recalls, color=palette[i], linewidth=1.5,
                        label=label, alpha=0.8)

        ax.set_xscale("log")
        ax.set_xlabel("Total Compute (minutes)", fontsize=11)
        ax.set_ylabel("Best Recall (%)" if pidx == 0 else "", fontsize=11)
        ax.set_title(f"Precision >= {prec_target * 100:.0f}%",
                     fontsize=12, fontweight="bold")
        ax.legend(title="Context", fontsize=7, title_fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Recall at Fixed Precision: Compute Allocation",
                 fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("img/binary/recall_at_precision.png", dpi=200, bbox_inches="tight")
    plt.savefig("img/binary/recall_at_precision.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved recall_at_precision")


# ============================================================
# ADDITIONAL PLOT C: Metric-optimal configuration comparison
# ============================================================
def plot_metric_comparison():
    """Do different metrics agree on optimal compute allocation?"""
    metric_names = list(METRICS.keys())
    n_metrics = len(metric_names)

    # For each budget, find optimal ctx for each metric
    budgets_fine = np.unique(np.concatenate([
        np.arange(1, 20, 1), np.arange(20, 100, 2),
        np.arange(100, budget_min + 1, 5),
    ]))

    # Build matrix: (n_metrics, n_budgets) -> ctx_index
    opt_ctx_matrix = np.full((n_metrics, len(budgets_fine)), np.nan)
    for mi, mn in enumerate(metric_names):
        mf = METRICS[mn]["func"]
        for bi, b in enumerate(budgets_fine):
            best_v, best_ci = -1, 0
            for ci, ctx in enumerate(context_lengths_min):
                km = int(b // ctx)
                if km < 1: continue
                v = mf(ctx, km)
                if v > best_v:
                    best_v, best_ci = v, ci
            opt_ctx_matrix[mi, bi] = best_ci

    # Side-by-side Pareto fronts
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for mi, mn in enumerate(metric_names):
        ax = axes[mi]
        mf = METRICS[mn]["func"]
        mc = METRICS[mn]

        # Background
        for ci, (ctx, label) in enumerate(zip(context_lengths_min, context_labels)):
            bs, vs = [], []
            for b in budgets_fine:
                km = int(b // ctx)
                if km < 1: continue
                bs.append(b); vs.append(mf(ctx, km))
            ax.plot(bs, vs, color=palette[ci], linewidth=0.8, alpha=0.2)

        # Pareto front
        opt_b, opt_v, opt_c = [], [], []
        for bi, b in enumerate(budgets_fine):
            ci = int(opt_ctx_matrix[mi, bi])
            ctx = context_lengths_min[ci]
            km = int(b // ctx)
            opt_b.append(b); opt_v.append(mf(ctx, km)); opt_c.append(ci)

        # Colored segments
        segs, ss = [], 0
        for j in range(1, len(opt_b)):
            if opt_c[j] != opt_c[j - 1]:
                segs.append((ss, j)); ss = j
        segs.append((ss, len(opt_b)))

        for s, e in segs:
            ax.plot(opt_b[s:e], opt_v[s:e], color=palette[opt_c[s]],
                    linewidth=3, solid_capstyle="round")

        ax.set_xscale("log")
        ax.set_title(mn.upper(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Budget (min)" if mi >= 2 else "")
        ax.set_ylabel(mc["label"] if mi % 3 == 0 else "")
        ax.grid(True, alpha=0.3)

    # Legend in last subplot
    axes[5].set_visible(True)
    axes[5].axis("off")
    for ci, label in enumerate(context_labels):
        axes[5].plot([], [], color=palette[ci], linewidth=3, label=label)
    axes[5].legend(title="Optimal Context", fontsize=10, title_fontsize=11,
                   loc="center")

    fig.suptitle("Which Metric Picks Which Context Length?", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("img/binary/metric_comparison.png", dpi=200, bbox_inches="tight")
    plt.savefig("img/binary/metric_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved metric_comparison")


# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    PLOT_FUNCS = [
        ("heatmap", plot_heatmap),
        ("vs_k", plot_vs_k),
        ("vs_total", plot_vs_total),
        ("pareto", plot_pareto),
        ("min_cost", plot_min_cost),
        ("marginal", plot_marginal),
        ("double", plot_double),
    ]

    for mname, mc in METRICS.items():
        outdir = f"img/binary/{mname}"
        os.makedirs(outdir, exist_ok=True)
        print(f"Generating plots for {mname}...")
        for pname, pfunc in PLOT_FUNCS:
            pfunc(mc["func"], mc, outdir)
            print(f"  {pname} done")

    print("\nGenerating additional binary-specific plots...")
    plot_roc_iso_compute()
    plot_recall_at_fixed_precision()
    plot_metric_comparison()

    print("\nAll done!")
