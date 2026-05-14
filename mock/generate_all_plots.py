#!/usr/bin/env python3
"""
mock/generate_all_plots.py

Generate synthetic mock plots for all analyses in docs/ANALYSIS_IDEAS.md.
All data is fabricated with realistic-looking curves — no real PSG data needed.

Run with:
    /home/boshra95/sleepfm_env/bin/python mock/generate_all_plots.py

Outputs: mock/img/<section>/<plot_name>.png
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.stats import binom, norm
from scipy.optimize import curve_fit
from sklearn.metrics import precision_recall_curve, average_precision_score

RNG = np.random.default_rng(42)
OUT = Path(__file__).parent / "img"

# ── Shared style ──────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                      "savefig.dpi": 150, "axes.spines.top": False,
                      "axes.spines.right": False})

# ── Experimental grid ─────────────────────────────────────────────────────────

CTX_MIN   = [0.5, 10.0, 40.0, 80.0, 120.0, 240.0]
CTX_LABEL = ["30s", "10m", "40m", "80m", "120m", "240m"]
CTX_COLOR = sns.color_palette("viridis_r", len(CTX_MIN))

K_SPARSE = [1, 5, 10, 20, 50, 100]
K_DENSE  = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100]

HEADS = {
    "lstm":       {"color": "#1f77b4", "marker": "o", "ceiling_scale": 1.00, "k_sat": 10},
    "transformer":{"color": "#ff7f0e", "marker": "s", "ceiling_scale": 0.97, "k_sat": 12},
    "mean_pool":  {"color": "#2ca02c", "marker": "^", "ceiling_scale": 0.88, "k_sat":  5},
}

TASKS = {
    "sex_binary":          {"base": 0.73, "ceil": 0.94, "L_star": 8,   "label": "Sex (binary)",        "color": "#e41a1c", "type": "demographics"},
    "bmi_binary":          {"base": 0.65, "ceil": 0.88, "L_star": 30,  "label": "BMI (binary)",        "color": "#377eb8", "type": "demographics"},
    "sleep_eff_binary":    {"base": 0.63, "ceil": 0.87, "L_star": 55,  "label": "Sleep Eff (binary)",  "color": "#4daf4a", "type": "sleep"},
    "age_class":           {"base": 0.66, "ceil": 0.84, "L_star": 75,  "label": "Age (3-class)",       "color": "#984ea3", "type": "demographics"},
    "osa_binary":          {"base": 0.62, "ceil": 0.86, "L_star": 50,  "label": "OSA (binary)",        "color": "#ff7f00", "type": "comorbidity"},
    "psqi_binary":         {"base": 0.60, "ceil": 0.82, "L_star": 85,  "label": "PSQI (binary)",       "color": "#a65628", "type": "sleep"},
    "depression_binary":   {"base": 0.66, "ceil": 0.83, "L_star": 35,  "label": "Depression (binary)", "color": "#f781bf", "type": "comorbidity"},
}

DATASETS = {"apples": (1.00, 0.00), "shhs": (0.97, 0.015), "mros": (0.93, -0.025)}


# ── Core synthetic metric functions ───────────────────────────────────────────

def auroc(ctx, k, task, head="lstm", noise=0.0):
    """Synthetic AUROC as a function of context length (min) and K."""
    tp = TASKS[task]
    hp = HEADS[head]
    ceil = tp["ceil"] * hp["ceiling_scale"]
    ctx_gain = (ceil - tp["base"]) * (1 - np.exp(-ctx / tp["L_star"]))
    auroc_k1 = tp["base"] + ctx_gain
    k_gain = (1 - auroc_k1) * 0.45 * (1 - np.exp(-np.asarray(k) / hp["k_sat"]))
    val = auroc_k1 + k_gain
    if noise > 0:
        val = val + RNG.normal(0, noise, np.shape(val))
    return np.clip(val, 0.5, 0.995)


def balanced_acc(ctx, k, task, head="lstm", noise=0.0):
    """Balanced accuracy ≈ AUROC − 0.05 for synthetic purposes."""
    return np.clip(auroc(ctx, k, task, head, noise) - 0.05 + noise * 0.5, 0.45, 0.99)


# ─────────────────────────────────────────────────────────────────────────────
# §1  OVERFITTING CURVES AND COMPUTE SCALING LAWS
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_curves(ctx, n_epochs=120):
    """Return (epochs, train_loss, val_loss, epoch_star)."""
    ep = np.arange(1, n_epochs + 1)
    # Longer context → slower convergence, lower optimal val loss, later optimal epoch
    conv_rate   = 0.10 + 0.04 / (ctx + 1)
    opt_val     = 0.55 - 0.18 * np.log1p(ctx) / np.log1p(240)
    epoch_star  = int(12 + 14 * np.log1p(ctx))
    overfit_k   = 1.5e-4 * np.exp(-ctx / 30)   # shorter ctx overfits faster

    train_loss  = opt_val - 0.07 + (0.8 - opt_val + 0.07) * np.exp(-conv_rate * ep)
    val_loss    = opt_val + 0.35 * np.exp(-conv_rate * ep) \
                  + overfit_k * np.maximum(0, ep - epoch_star) ** 1.8
    return ep, train_loss, val_loss, epoch_star


def plot_overfitting_curves():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=True)
    fig.suptitle("Overfitting Curves — Train vs Val Loss per Context Length\n"
                 "(Synthetic: training continues past early stopping)", fontsize=13)

    for ax, ctx, lbl in zip(axes.flat, CTX_MIN, CTX_LABEL):
        ep, tr, vl, es = synthetic_curves(ctx)
        ax.plot(ep, tr, color="#1f77b4", lw=2, label="Train loss")
        ax.plot(ep, vl, color="#d62728", lw=2, label="Val loss")
        ax.axvline(es, color="gray", ls="--", lw=1.5, label=f"Best epoch ({es})")
        ax.fill_between(ep, vl, tr, where=vl > tr, alpha=0.12,
                        color="red", label="Generalisation gap")
        ax.set_title(f"Context = {lbl}", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xlim(1, len(ep))
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "overfitting" / "1A_overfitting_curves.png")
    plt.close(fig)
    print("  1A done")


def plot_scaling_law():
    # FLOPs per step (rough analytical estimates): LSTM ∝ seq_len, Transformer ∝ seq_len²
    seq_len   = {c: max(1, int(c * 2)) for c in CTX_MIN}  # 30s→1, 10m→20, 240m→480
    base_lstm = 75e6  # FLOPs/step at seq_len=1
    base_tr   = 3e6   # Transformer head scaling base
    n_steps   = 2031  # batches/epoch (13k subjects × 5 windows / 32)

    fig, ax = plt.subplots(figsize=(9, 6))
    power_law = lambda F, a, b, c: c - a * F ** (-b)

    all_F, all_A, all_head = [], [], []
    for head, hp in HEADS.items():
        Fs, As = [], []
        for ctx, lbl, col in zip(CTX_MIN, CTX_LABEL, CTX_COLOR):
            sl = seq_len[ctx]
            if head == "lstm":
                flops_step = base_lstm * sl
            elif head == "transformer":
                flops_step = base_tr * sl ** 2
            else:
                flops_step = 1e5 * sl

            # Optimal epoch for this context
            _, _, _, es = synthetic_curves(ctx)
            total_F = flops_step * n_steps * es

            auc = auroc(ctx, k=100, task="sex_binary", head=head,
                        noise=0.005)
            ax.scatter(total_F, auc, color=col, marker=hp["marker"],
                       s=90, zorder=5,
                       label=f"{head} {lbl}" if head == "lstm" else "_nolegend_")
            Fs.append(total_F); As.append(auc)
            all_F.append(total_F); all_A.append(auc); all_head.append(head)

        # Fit power law for this head
        Fs_a = np.array(Fs); As_a = np.array(As)
        try:
            p0 = [0.1, 0.15, 0.95]
            popt, _ = curve_fit(power_law, Fs_a / Fs_a.min(),
                                As_a, p0=p0, maxfev=5000)
            F_grid = np.linspace(Fs_a.min(), Fs_a.max(), 200)
            ax.plot(F_grid, power_law(F_grid / Fs_a.min(), *popt),
                    color=hp["color"], lw=2, ls="--", alpha=0.7)
        except Exception:
            pass

    # Context color legend
    for ctx, lbl, col in zip(CTX_MIN, CTX_LABEL, CTX_COLOR):
        ax.scatter([], [], color=col, s=60, label=lbl)
    # Head marker legend
    for head, hp in HEADS.items():
        ax.scatter([], [], color="gray", marker=hp["marker"], s=70, label=head)

    ax.set_xscale("log")
    ax.set_xlabel("Total Training FLOPs (log scale)")
    ax.set_ylabel("Best Val AUROC")
    ax.set_title("Compute Scaling Law\n"
                 "Color = context length · Marker = head · Dashed = power-law fit")
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "overfitting" / "1B_scaling_law.png")
    plt.close(fig)
    print("  1B done")


def plot_optimal_epoch_vs_context():
    fig, ax = plt.subplots(figsize=(8, 4))
    epoch_stars = [synthetic_curves(c)[3] for c in CTX_MIN]
    bars = ax.bar(CTX_LABEL, epoch_stars, color=CTX_COLOR, edgecolor="white", width=0.6)
    for bar, es in zip(bars, epoch_stars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(es), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Optimal Epoch (val-loss minimum)")
    ax.set_title("Optimal Epoch vs Context Length\n"
                 "Longer context needs more epochs to converge")
    fig.tight_layout()
    fig.savefig(OUT / "overfitting" / "1C_optimal_epoch_vs_context.png")
    plt.close(fig)
    print("  1C done")


# ─────────────────────────────────────────────────────────────────────────────
# §2  MODEL CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_probs(ctx, n=800, prevalence=0.5, k=5):
    """
    Generate synthetic (y_true, prob_pred) pairs.
    Short context → overconfident (miscalibrated).
    Long context + large K → well-calibrated.
    """
    n_pos = int(n * prevalence)
    y = np.array([1] * n_pos + [0] * (n - n_pos))

    # Base signal strength increases with context
    signal = 0.6 + 0.35 * (1 - np.exp(-ctx / 60))
    k_boost = 0.15 * (1 - np.exp(-k / 10))
    total_signal = min(signal + k_boost, 0.95)

    pos_mean = 0.5 + 0.5 * total_signal
    neg_mean = 0.5 - 0.5 * total_signal

    # Miscalibration: short context → overconfident (push probs toward 0/1)
    overconf = max(0, 1.0 - ctx / 40)

    probs = np.empty(n)
    probs[:n_pos] = np.clip(RNG.beta(
        max(0.5, pos_mean * (3 + overconf * 5)),
        max(0.5, (1 - pos_mean) * (3 + overconf * 5)),
        n_pos), 0.01, 0.99)
    probs[n_pos:] = np.clip(RNG.beta(
        max(0.5, neg_mean * (3 + overconf * 5) * 0.6),
        max(0.5, (1 - neg_mean) * (3 + overconf * 5) * 1.4),
        n - n_pos), 0.01, 0.99)
    return y, probs


def plot_reliability_diagrams():
    show_ctx = [(0.5, "30s"), (40.0, "40m"), (240.0, "240m")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    fig.suptitle("Reliability Diagrams (Calibration) — sex_binary, K=5\n"
                 "Diagonal = perfect calibration", fontsize=12)

    for ax, (ctx, lbl) in zip(axes, show_ctx):
        y, p = _synthetic_probs(ctx, k=5)
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_mids, frac_pos, counts = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() > 5:
                bin_mids.append(p[mask].mean())
                frac_pos.append(y[mask].mean())
                counts.append(mask.sum())
        bin_mids = np.array(bin_mids)
        frac_pos = np.array(frac_pos)

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect")
        ax.bar(bin_mids, frac_pos, width=0.08, alpha=0.4, color="#1f77b4",
               label="Fraction positive")
        ax.plot(bin_mids, frac_pos, "o-", color="#1f77b4", lw=2)
        ax.fill_between([0, 1], [0, 1], [0, 1], alpha=0.05, color="gray")

        ece = np.sum(np.array(counts) / sum(counts) * np.abs(frac_pos - bin_mids))
        ax.set_title(f"Context = {lbl}\nECE = {ece:.3f}", fontsize=11)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "calibration" / "2A_reliability_diagrams.png")
    plt.close(fig)
    print("  2A done")


def _ece(ctx, k):
    y, p = _synthetic_probs(ctx, k=k)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() < 3:
            continue
        ece += mask.sum() / len(y) * abs(p[mask].mean() - y[mask].mean())
    return ece


def plot_ece_vs_context():
    fig, ax = plt.subplots(figsize=(8, 5))
    for head, hp in HEADS.items():
        eces = []
        for ctx in CTX_MIN:
            # Head quality modulates calibration: better heads → lower ECE
            scale = {"lstm": 1.0, "transformer": 1.05, "mean_pool": 1.25}[head]
            eces.append(_ece(ctx, k=10) * scale)
        ax.plot(CTX_MIN, eces, color=hp["color"], marker=hp["marker"],
                lw=2, ms=8, label=head)
    ax.set_xscale("log")
    ax.set_xticks(CTX_MIN)
    ax.set_xticklabels(CTX_LABEL)
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("ECE vs Context Length\nLower is better — longer context → better calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "calibration" / "2B_ece_vs_context.png")
    plt.close(fig)
    print("  2B done")


def plot_ece_vs_k():
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
    for ctx, lbl, col in zip(CTX_MIN, CTX_LABEL, CTX_COLOR):
        eces = [_ece(ctx, k) for k in ks]
        ax.plot(ks, eces, color=col, lw=2, marker="o", ms=5, label=lbl)
    ax.set_xlabel("K (windows per subject at inference)")
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_title("ECE vs K\nAggregating more windows improves calibration")
    ax.legend(title="Context", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "calibration" / "2C_ece_vs_k.png")
    plt.close(fig)
    print("  2C done")


# ─────────────────────────────────────────────────────────────────────────────
# §3  BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_auroc(ctx, head, n_subj=400, n_boot=500):
    """Simulate bootstrap CI on AUROC for a given (ctx, head)."""
    center = auroc(ctx, k=100, task="sex_binary", head=head)
    # CI width decreases with sqrt(N) and shrinks with larger AUROC (less variance near ceiling)
    width = (0.04 + 0.02 * (1 - center)) / np.sqrt(n_subj / 400)
    lo = center - 1.96 * width
    hi = center + 1.96 * width
    return center, max(lo, 0.5), min(hi, 0.995)


def plot_saturation_with_ci():
    fig, ax = plt.subplots(figsize=(9, 6))
    for head, hp in HEADS.items():
        centers, los, his = [], [], []
        for ctx in CTX_MIN:
            c, lo, hi = _bootstrap_auroc(ctx, head)
            centers.append(c); los.append(lo); his.append(hi)
        centers = np.array(centers); los = np.array(los); his = np.array(his)
        ax.plot(CTX_MIN, centers, color=hp["color"], marker=hp["marker"],
                lw=2.5, ms=9, label=head, zorder=5)
        ax.fill_between(CTX_MIN, los, his, color=hp["color"], alpha=0.15)

    ax.set_xscale("log")
    ax.set_xticks(CTX_MIN)
    ax.set_xticklabels(CTX_LABEL)
    ax.set_xlabel("Training Context Length")
    ax.set_ylabel("Test AUROC (K=all)")
    ax.set_title("Saturation Curve with 95% Bootstrap Confidence Intervals\nsex_binary task")
    ax.legend(title="Head")
    ax.set_ylim(0.68, 0.99)
    fig.tight_layout()
    fig.savefig(OUT / "bootstrap" / "3A_saturation_with_ci.png")
    plt.close(fig)
    print("  3A done")


# ─────────────────────────────────────────────────────────────────────────────
# §4  WINDOW POSITION AND TEMPORAL STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_window_position_profiles():
    n_pos = 200  # position bins (0=start, 1=end of night)
    pos = np.linspace(0, 1, n_pos)
    # Synthetic: sleep efficiency signal peaks at late-night; sex signal is flat
    show = [("sex_binary", "flat"), ("sleep_eff_binary", "late")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle("Mean Predicted Probability vs Normalized Window Position\n"
                 "(0 = start of recording, 1 = end of night)", fontsize=12)

    for ax, (task, pattern) in zip(axes, show):
        tp = TASKS[task]
        for ctx, lbl, col in zip(
            [0.5, 40.0, 240.0], ["30s", "40m", "240m"],
            [CTX_COLOR[0], CTX_COLOR[2], CTX_COLOR[5]]
        ):
            signal = 0.55 + 0.35 * (1 - np.exp(-ctx / tp["L_star"]))
            if pattern == "late":
                # Late-night signal: rising sigmoid
                pos_curve = signal * (0.5 + 0.5 * (pos - 0.5) / (0.3 + np.abs(pos - 0.5)))
                neg_curve = (1 - signal) * (0.4 + 0.1 * pos)
            else:
                # Flat pattern: position doesn't matter
                pos_curve = np.full(n_pos, signal * 0.85)
                neg_curve = np.full(n_pos, (1 - signal) * 0.35)

            noise = RNG.normal(0, 0.01, n_pos)
            ax.plot(pos, np.clip(pos_curve + noise, 0, 1), color=col, lw=2,
                    label=f"Positive (L={lbl})")
            ax.plot(pos, np.clip(neg_curve + noise, 0, 1), color=col, lw=2,
                    ls="--", alpha=0.6)

        ax.set_title(f"{tp['label']}\n(solid=positive, dashed=negative subjects)")
        ax.set_xlabel("Normalized window position")
        ax.set_ylabel("Mean prob_class1")
        ax.set_ylim(0, 1)
        ax.axvline(0.5, color="gray", ls=":", lw=1, alpha=0.5, label="Midpoint")

    # Shared legend for context lengths
    handles = [mpatches.Patch(color=CTX_COLOR[i], label=l)
               for i, l in enumerate(["30s", "", "40m", "", "", "240m"]) if l]
    axes[1].legend(handles=handles, title="Context", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "window_position" / "4A_window_position_profiles.png")
    plt.close(fig)
    print("  4A done")


# ─────────────────────────────────────────────────────────────────────────────
# §5  PER-SUBJECT CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

def _subject_variance(ctx, n_subjects=400):
    """
    Generate per-subject (std of prob_class1 across windows, correct_at_kall).
    Longer context → lower within-subject variance (more stable windows).
    """
    # Variance of per-window predictions decreases with context length
    sigma = 0.28 * np.exp(-ctx / 50) + 0.04
    std_per_subj = RNG.exponential(sigma, n_subjects)
    # Correctness: roughly matches AUROC
    auc = auroc(ctx, k=100, task="sex_binary")
    correct = RNG.random(n_subjects) < auc
    return std_per_subj, correct


def plot_subject_variance_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    fig.suptitle("Within-Subject Prediction Variance (std of prob_class1 across windows)\n"
                 "Correct vs incorrect subjects", fontsize=12)
    for ax, (ctx, lbl) in zip(axes, [(0.5, "30s"), (40.0, "40m"), (240.0, "240m")]):
        std_all, correct = _subject_variance(ctx)
        ax.violinplot([std_all[correct], std_all[~correct]],
                      positions=[0, 1], showmedians=True)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Correct", "Incorrect"])
        ax.set_title(f"Context = {lbl}")
        ax.set_ylabel("Within-subject std(prob)")
        ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(OUT / "subject_consistency" / "5A_variance_distribution.png")
    plt.close(fig)
    print("  5A done")


def plot_variance_vs_k():
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
    for ctx, lbl, col in zip(CTX_MIN, CTX_LABEL, CTX_COLOR):
        # Variance of aggregated prediction ≈ sigma / sqrt(K)
        sigma_base = 0.28 * np.exp(-ctx / 50) + 0.04
        variance = [sigma_base / np.sqrt(k) for k in ks]
        ax.plot(ks, variance, color=col, lw=2, marker="o", ms=5, label=lbl)
    ax.set_xlabel("K (windows aggregated)")
    ax.set_ylabel("Expected prediction variance (std)")
    ax.set_title("Prediction Variance vs K\n"
                 "Aggregation reduces uncertainty; longer context starts lower")
    ax.legend(title="Context", fontsize=8, title_fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "subject_consistency" / "5B_variance_vs_k.png")
    plt.close(fig)
    print("  5B done")


def plot_hard_subject_analysis():
    """Fraction of subjects correctly predicted by 0,1,...,6 context lengths."""
    n_ctx = len(CTX_MIN)
    n_subj = 500

    # Each subject has a 'difficulty': harder subjects need longer context
    difficulty = RNG.beta(2, 3, n_subj)  # most subjects are moderate
    correct_counts = np.zeros(n_subj, dtype=int)
    for ctx in CTX_MIN:
        auc = auroc(ctx, k=100, task="sex_binary")
        p_correct = auc * (1 - difficulty * 0.5)  # harder subjects less likely correct
        correct_counts += (RNG.random(n_subj) < p_correct).astype(int)

    bins = np.arange(n_ctx + 2) - 0.5
    fig, ax = plt.subplots(figsize=(8, 5))
    counts, edges = np.histogram(correct_counts, bins=bins)
    bar_cols = [CTX_COLOR[min(i, len(CTX_COLOR)-1)] for i in range(n_ctx + 1)]
    ax.bar(range(n_ctx + 1), counts / n_subj * 100, color=bar_cols,
           edgecolor="white", width=0.7)
    ax.set_xticks(range(n_ctx + 1))
    ax.set_xticklabels([f"{i}/{n_ctx}" for i in range(n_ctx + 1)])
    ax.set_xlabel("Context lengths at which subject was correctly predicted")
    ax.set_ylabel("Fraction of subjects (%)")
    ax.set_title("Hard Subject Analysis\n"
                 "Subjects correctly predicted at 0 lengths = irreducible hard cases")
    fig.tight_layout()
    fig.savefig(OUT / "subject_consistency" / "5C_hard_subject_analysis.png")
    plt.close(fig)
    print("  5C done")


# ─────────────────────────────────────────────────────────────────────────────
# §6  TASK × CONTEXT SENSITIVITY MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def _task_sensitivity():
    rows = {}
    for task, tp in TASKS.items():
        auc_30s = auroc(0.5, k=100, task=task, head="lstm")
        auc_best = auroc(240.0, k=100, task=task, head="lstm")
        rows[task] = {
            "baseline": auc_30s,
            "best": auc_best,
            "sensitivity": auc_best - auc_30s,
            "difficulty": 1 - auc_30s,
            "L_star": tp["L_star"],
            "label": tp["label"],
            "color": tp["color"],
            "type": tp["type"],
        }
    return rows


def plot_task_sensitivity_scatter():
    rows = _task_sensitivity()
    fig, ax = plt.subplots(figsize=(9, 6))
    for task, r in rows.items():
        ax.scatter(r["difficulty"], r["sensitivity"], color=r["color"],
                   s=180, zorder=5, edgecolors="white", lw=1.5)
        ax.annotate(r["label"], (r["difficulty"], r["sensitivity"]),
                    textcoords="offset points", xytext=(7, 3), fontsize=8.5)
    ax.axhline(0.08, color="gray", ls="--", lw=1, alpha=0.5, label="Sensitivity threshold (0.08)")
    ax.set_xlabel("Baseline difficulty  (1 − AUROC at 30s context)")
    ax.set_ylabel("Context sensitivity  (ΔAUROC from 30s → best context)")
    ax.set_title("Task × Context Sensitivity\n"
                 "Upper-right: hard tasks that benefit most from longer context")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "task_sensitivity" / "6A_task_sensitivity_scatter.png")
    plt.close(fig)
    print("  6A done")


def plot_task_sensitivity_bars():
    rows = _task_sensitivity()
    tasks_sorted = sorted(rows, key=lambda t: rows[t]["sensitivity"])
    cols_by_ctx = sns.color_palette("viridis_r", len(CTX_MIN))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tasks_sorted))
    width = 0.13
    for j, (ctx, lbl, col) in enumerate(zip(CTX_MIN, CTX_LABEL, cols_by_ctx)):
        aucs = [auroc(ctx, k=100, task=t, head="lstm") for t in tasks_sorted]
        ax.bar(x + (j - 2.5) * width, aucs, width,
               color=col, label=lbl, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([rows[t]["label"] for t in tasks_sorted], rotation=20, ha="right")
    ax.set_ylabel("Test AUROC (LSTM, K=all)")
    ax.set_title("AUROC by Context Length per Task — Sorted by Context Sensitivity\n"
                 "Rightmost tasks benefit most from longer context")
    ax.legend(title="Context", fontsize=8, title_fontsize=9, ncol=3)
    ax.set_ylim(0.55, 0.98)
    fig.tight_layout()
    fig.savefig(OUT / "task_sensitivity" / "6B_task_sensitivity_bars.png")
    plt.close(fig)
    print("  6B done")


def plot_lstar_per_task():
    rows = _task_sensitivity()
    tasks_sorted = sorted(rows, key=lambda t: rows[t]["L_star"])
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(tasks_sorted))
    l_stars = [rows[t]["L_star"] for t in tasks_sorted]
    colors  = [rows[t]["color"]  for t in tasks_sorted]
    ax.scatter(l_stars, y, color=colors, s=160, zorder=5, edgecolors="white", lw=1.5)
    ax.hlines(y, 0, l_stars, colors=colors, lw=2, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([rows[t]["label"] for t in tasks_sorted])
    ax.set_xlabel("L* — Context length at which performance saturates (minutes)")
    ax.set_title("Task-Specific Saturation Context Length (L*)\n"
                 "Minimum context to reach near-ceiling AUROC")
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 40, 80, 120])
    ax.set_xticklabels(["1m", "5m", "10m", "40m", "80m", "120m"])
    fig.tight_layout()
    fig.savefig(OUT / "task_sensitivity" / "6C_lstar_per_task.png")
    plt.close(fig)
    print("  6C done")


# ─────────────────────────────────────────────────────────────────────────────
# §7  COHORT-STRATIFIED SATURATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_cohort_saturation():
    ds_styles = {"apples": ("-",  "o"), "shhs": ("--", "s"), "mros": (":",  "^")}
    ds_colors = {"apples": "#1f77b4", "shhs": "#ff7f0e", "mros": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Cohort-Stratified Saturation Curves (LSTM, K=all)\n"
                 "Robustness across APPLES, SHHS, and MrOS", fontsize=12)

    for ax, task in zip(axes, ["sex_binary", "sleep_eff_binary"]):
        for ds, (ls, mk) in ds_styles.items():
            scale, bias = DATASETS[ds]
            aucs = [auroc(c, k=100, task=task) * scale + bias for c in CTX_MIN]
            aucs = np.clip(aucs, 0.5, 0.99)
            ax.plot(CTX_MIN, aucs, color=ds_colors[ds], ls=ls, marker=mk,
                    lw=2, ms=8, label=ds.upper())
        ax.set_xscale("log")
        ax.set_xticks(CTX_MIN)
        ax.set_xticklabels(CTX_LABEL)
        ax.set_title(TASKS[task]["label"])
        ax.set_xlabel("Context Length")
        ax.set_ylabel("Test AUROC")
        ax.legend(title="Dataset")
    fig.tight_layout()
    fig.savefig(OUT / "cohort" / "7A_cohort_saturation.png")
    plt.close(fig)
    print("  7A done")


# ─────────────────────────────────────────────────────────────────────────────
# §8  PRECISION-RECALL AND CLINICAL OPERATING POINTS
# ─────────────────────────────────────────────────────────────────────────────

def _pr_curve_data(ctx, prevalence=0.35, n=1000, k=50):
    """Generate synthetic (precision, recall) curve for a given context length."""
    y, p = _synthetic_probs(ctx, n=n, prevalence=prevalence, k=k)
    precision, recall, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    return precision, recall, ap


def plot_pr_curves():
    fig, ax = plt.subplots(figsize=(8, 6))
    for ctx, lbl, col in zip(CTX_MIN, CTX_LABEL, CTX_COLOR):
        prec, rec, ap = _pr_curve_data(ctx)
        ax.plot(rec, prec, color=col, lw=2, label=f"{lbl} (AP={ap:.3f})")
    ax.axhline(0.35, color="gray", ls="--", lw=1, label="Prevalence baseline")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curves by Context Length\n"
                 "OSA binary — K=50 windows, prevalence=35%")
    ax.legend(title="Context", fontsize=8, title_fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "precision_recall" / "8A_pr_curves.png")
    plt.close(fig)
    print("  8A done")


def plot_aucpr_vs_context():
    fig, ax = plt.subplots(figsize=(8, 5))
    for head, hp in HEADS.items():
        aucprs = []
        for ctx in CTX_MIN:
            _, _, ap = _pr_curve_data(ctx, k=50)
            scale = {"lstm": 1.0, "transformer": 0.99, "mean_pool": 0.93}[head]
            aucprs.append(ap * scale)
        ax.plot(CTX_MIN, aucprs, color=hp["color"], marker=hp["marker"],
                lw=2, ms=8, label=head)
    ax.set_xscale("log")
    ax.set_xticks(CTX_MIN)
    ax.set_xticklabels(CTX_LABEL)
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Average Precision (AUC-PR)")
    ax.set_title("Average Precision vs Context Length\nOSA binary — K=50")
    ax.legend(title="Head")
    fig.tight_layout()
    fig.savefig(OUT / "precision_recall" / "8B_aucpr_vs_context.png")
    plt.close(fig)
    print("  8B done")


def plot_majority_vote_threshold_sweep():
    """
    At fixed context and K=20, sweep majority-vote threshold t from 1 to K.
    t=1: any positive vote → high recall, low precision
    t=K: all votes must agree → low recall, high precision
    """
    K = 20
    ctx_show = [(0.5, "30s"), (40.0, "40m"), (240.0, "240m")]
    fig, ax = plt.subplots(figsize=(8, 6))

    for ctx, lbl, col in zip(
        [c for c, _ in ctx_show], [l for _, l in ctx_show],
        [CTX_COLOR[0], CTX_COLOR[2], CTX_COLOR[5]]
    ):
        tpr_base = 0.5 + 0.45 * (1 - np.exp(-ctx / 60))
        fpr_base = 0.5 - 0.35 * (1 - np.exp(-ctx / 60))
        precs, recs = [], []
        for t in range(1, K + 1):
            # Recall ~ P(votes >= t | positive) using binomial
            recall = float(1 - binom.cdf(t - 1, K, tpr_base))
            fpr    = float(1 - binom.cdf(t - 1, K, fpr_base))
            prevalence = 0.35
            tp = recall * prevalence
            fp = fpr * (1 - prevalence)
            precision = tp / (tp + fp + 1e-9)
            precs.append(precision); recs.append(recall)

        sc = ax.scatter(recs, precs, c=range(1, K + 1), cmap="RdYlGn_r",
                        vmin=1, vmax=K, s=60, zorder=5)
        ax.plot(recs, precs, color=col, lw=2, label=f"L={lbl}", alpha=0.8)
        # Annotate t=1 and t=K
        ax.annotate(f"t=1 (L={lbl})", (recs[0], precs[0]),
                    fontsize=7, color=col, textcoords="offset points", xytext=(5, -8))
        ax.annotate(f"t={K}", (recs[-1], precs[-1]),
                    fontsize=7, color=col, textcoords="offset points", xytext=(5, 3))

    cbar = fig.colorbar(sc, ax=ax, label="Threshold t")
    cbar.set_ticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall via Majority-Vote Threshold Sweep\n"
                 "K=20 windows, OSA binary — t=1→high recall, t=K→high precision")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "precision_recall" / "8C_majority_vote_threshold_sweep.png")
    plt.close(fig)
    print("  8C done")


# ─────────────────────────────────────────────────────────────────────────────
# §9  SUBJECT-LEVEL K-SATURATION
# ─────────────────────────────────────────────────────────────────────────────

def _per_subject_kstar(ctx, n_subj=400, max_k=50):
    """Minimum K at which a subject is first correctly predicted."""
    auc = auroc(ctx, k=max_k, task="sex_binary")
    # P(correct at K=1) varies by subject difficulty
    difficulty = RNG.beta(1.5, 3, n_subj)
    p_correct_k1 = np.clip(auc * (1 - difficulty * 0.6), 0.1, 0.98)

    # K* = geometric-like: each additional window has prob p of fixing it
    # Simplified: K* ~ 1 + Geom(p_fix) where p_fix increases with ctx
    p_fix = 0.12 + 0.35 * (1 - np.exp(-ctx / 40))
    k_stars = np.where(
        p_correct_k1 > 0.7,
        np.ones(n_subj, dtype=int),
        np.clip(RNG.geometric(p_fix, n_subj), 1, max_k + 1)
    )
    return k_stars


def plot_kstar_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    fig.suptitle("Distribution of Per-Subject Minimum K to First Correct Prediction\n"
                 "sex_binary, LSTM head", fontsize=12)
    for ax, (ctx, lbl) in zip(axes, [(0.5, "30s"), (40.0, "40m"), (240.0, "240m")]):
        ks = _per_subject_kstar(ctx)
        ax.hist(ks, bins=range(1, 30), color=CTX_COLOR[CTX_MIN.index(ctx)],
                edgecolor="white", density=True, alpha=0.85)
        ax.axvline(np.median(ks), color="red", ls="--", lw=2,
                   label=f"Median K*={int(np.median(ks))}")
        ax.set_title(f"Context = {lbl}")
        ax.set_xlabel("K* (minimum windows needed)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "kstar" / "9A_kstar_distribution.png")
    plt.close(fig)
    print("  9A done")


def plot_coverage_curves():
    """Fraction of subjects correctly predicted as a function of K."""
    max_k = 60
    ks = np.arange(1, max_k + 1)
    fig, ax = plt.subplots(figsize=(9, 6))

    for ctx, lbl, col in zip(CTX_MIN, CTX_LABEL, CTX_COLOR):
        kstars = _per_subject_kstar(ctx, max_k=max_k)
        coverage = np.array([np.mean(kstars <= k) for k in ks])
        ax.plot(ks, coverage * 100, color=col, lw=2, label=lbl)

    # Reference lines
    for pct in [50, 80, 95]:
        ax.axhline(pct, color="gray", ls=":", lw=1, alpha=0.6)
        ax.text(max_k * 1.01, pct, f"{pct}%", va="center", fontsize=8, color="gray")

    ax.set_xlabel("K (windows per subject at inference)")
    ax.set_ylabel("% of subjects correctly predicted")
    ax.set_title("Cumulative Coverage Curves\n"
                 "Fraction of subjects correctly classified by K windows")
    ax.legend(title="Context", fontsize=8, title_fontsize=9)
    ax.set_xlim(1, max_k); ax.set_ylim(0, 102)
    fig.tight_layout()
    fig.savefig(OUT / "kstar" / "9B_coverage_curves.png")
    plt.close(fig)
    print("  9B done")


# ─────────────────────────────────────────────────────────────────────────────
# §10  HEAD ARCHITECTURE AT EQUAL COMPUTE
# ─────────────────────────────────────────────────────────────────────────────

def plot_flops_normalized_saturation():
    """
    X-axis: total training FLOPs (log), Y-axis: best val AUROC.
    Color = context length, marker/linestyle = head.
    Shows whether temporal modeling (LSTM/Transformer) adds value per FLOP
    relative to MeanPool.
    """
    n_steps_per_epoch = 2031  # 13k subjects × 5 windows / 32 batch
    # Sequence lengths per context (SleepFM 30s embeddings → seq_len)
    sl_map = dict(zip(CTX_MIN, [1, 20, 80, 160, 240, 480]))

    fig, ax = plt.subplots(figsize=(10, 6))

    for head, hp in HEADS.items():
        Fs, aucs_list = [], []
        for ctx, lbl, col in zip(CTX_MIN, CTX_LABEL, CTX_COLOR):
            sl = sl_map[ctx]
            if head == "lstm":
                fps = 75e6 * sl
            elif head == "transformer":
                fps = 3e6 * sl ** 2
            else:
                fps = 5e4 * sl

            _, _, _, es = synthetic_curves(ctx)
            total_F = fps * n_steps_per_epoch * es
            auc = auroc(ctx, k=100, task="sex_binary", head=head, noise=0.003)
            Fs.append(total_F)
            aucs_list.append(auc)
            ax.scatter(total_F, auc, color=col, marker=hp["marker"],
                       s=110, zorder=6, edgecolors=hp["color"], lw=2)

        # Connect points for this head with a line
        order = np.argsort(Fs)
        ax.plot(np.array(Fs)[order], np.array(aucs_list)[order],
                color=hp["color"], lw=1.5, ls="--", alpha=0.5, label=head)

    # Context legend (colors)
    for ctx, lbl, col in zip(CTX_MIN, CTX_LABEL, CTX_COLOR):
        ax.scatter([], [], color=col, s=70, label=lbl)
    # Head legend (markers)
    for head, hp in HEADS.items():
        ax.scatter([], [], color="gray", marker=hp["marker"], s=70, label=head)

    ax.set_xscale("log")
    ax.set_xlabel("Total Training FLOPs (log scale)")
    ax.set_ylabel("Best Val AUROC")
    ax.set_title("Head Architecture at Equal Compute\n"
                 "Color = context length · Marker = head type\n"
                 "Note: Transformer FLOPs scale quadratically with context length")
    ax.legend(fontsize=8, ncol=3, loc="lower right", title="Context / Head")
    fig.tight_layout()
    fig.savefig(OUT / "head_compute" / "10A_flops_normalized_saturation.png")
    plt.close(fig)
    print("  10A done")


# ─────────────────────────────────────────────────────────────────────────────
# BONUS: Summary overview grid (one saturation curve per task, all heads)
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_tasks_saturation_grid():
    n_tasks = len(TASKS)
    ncols = 4
    nrows = (n_tasks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.8), sharey=True)
    fig.suptitle("Saturation Curves — All Tasks × All Heads (K=all)\n"
                 "Mock data: shows expected qualitative pattern", fontsize=13)

    for ax, (task, tp) in zip(axes.flat, TASKS.items()):
        for head, hp in HEADS.items():
            aucs = [auroc(c, k=100, task=task, head=head) for c in CTX_MIN]
            ax.plot(CTX_MIN, aucs, color=hp["color"], marker=hp["marker"],
                    lw=2, ms=7, label=head)
        ax.set_title(tp["label"], fontsize=10)
        ax.set_xscale("log")
        ax.set_xticks(CTX_MIN)
        ax.set_xticklabels(CTX_LABEL, fontsize=7)
        ax.set_xlabel("Context length", fontsize=8)
        ax.set_ylabel("AUROC", fontsize=8)
        ax.set_ylim(0.55, 0.99)

    # Hide unused axes
    for ax in axes.flat[n_tasks:]:
        ax.set_visible(False)

    handles = [mpatches.Patch(color=hp["color"], label=h)
               for h, hp in HEADS.items()]
    fig.legend(handles=handles, loc="lower right", fontsize=10, title="Head")
    fig.tight_layout()
    fig.savefig(OUT / "task_sensitivity" / "BONUS_all_tasks_saturation_grid.png")
    plt.close(fig)
    print("  BONUS saturation grid done")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating all mock plots...\n")

    print("§1 — Overfitting + Scaling Laws")
    plot_overfitting_curves()
    plot_scaling_law()
    plot_optimal_epoch_vs_context()

    print("\n§2 — Model Calibration")
    plot_reliability_diagrams()
    plot_ece_vs_context()
    plot_ece_vs_k()

    print("\n§3 — Bootstrap Confidence Intervals")
    plot_saturation_with_ci()

    print("\n§4 — Window Position")
    plot_window_position_profiles()

    print("\n§5 — Per-Subject Consistency")
    plot_subject_variance_distribution()
    plot_variance_vs_k()
    plot_hard_subject_analysis()

    print("\n§6 — Task × Context Sensitivity")
    plot_task_sensitivity_scatter()
    plot_task_sensitivity_bars()
    plot_lstar_per_task()

    print("\n§7 — Cohort-Stratified Saturation")
    plot_cohort_saturation()

    print("\n§8 — Precision-Recall")
    plot_pr_curves()
    plot_aucpr_vs_context()
    plot_majority_vote_threshold_sweep()

    print("\n§9 — Subject-Level K-Saturation")
    plot_kstar_distribution()
    plot_coverage_curves()

    print("\n§10 — Head Architecture at Equal Compute")
    plot_flops_normalized_saturation()

    print("\nBonus — All-tasks saturation grid")
    plot_all_tasks_saturation_grid()

    # Print summary
    all_pngs = sorted(OUT.rglob("*.png"))
    print(f"\nDone — {len(all_pngs)} plots saved to {OUT}/")
    for p in all_pngs:
        print(f"  {p.relative_to(OUT.parent)}")


if __name__ == "__main__":
    main()
