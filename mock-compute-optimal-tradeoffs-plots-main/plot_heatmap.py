import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# --- Config ---
context_lengths_min = [0.5, 5, 10, 20, 40, 80, 120, 180, 240]
context_labels = ["30s", "5m", "10m", "20m", "40m", "80m", "120m", "180m", "240m"]
budget_min = 16 * 60  # 960 minutes

# For each context length, max k = floor(budget / context_length)
max_ks = [int(budget_min // c) for c in context_lengths_min]
# All possible k values (union of all valid ks, sorted)
all_ks = sorted(set(k for mk in max_ks for k in range(1, mk + 1)))

# --- Mock performance model ---
# perf(context, k) = base(context) + voting_gain(k, context)
# base increases with context (log-like, diminishing returns)
# voting gain: sqrt-like in k, but saturates
# We want values roughly in [30, 90] range (like accuracy %)

def base_perf(ctx_min):
    """Longer context -> better base, diminishing returns."""
    return 35 + 25 * np.log1p(ctx_min) / np.log1p(240)

def voting_boost(k, ctx_min):
    """More votes -> better, but diminishing. Shorter context benefits more from voting."""
    # Shorter context has noisier predictions, so voting helps more
    noise_factor = 1.0 + 0.5 * (1 - np.log1p(ctx_min) / np.log1p(240))
    return 20 * noise_factor * (1 - 1 / np.sqrt(k))

def perf(ctx_min, k):
    return base_perf(ctx_min) + voting_boost(k, ctx_min)

# --- Build the data matrix ---
# Rows: context lengths, Cols: k values
# NaN where k exceeds budget
perf_matrix = np.full((len(context_lengths_min), len(all_ks)), np.nan)
for i, ctx in enumerate(context_lengths_min):
    for j, k in enumerate(all_ks):
        if ctx * k <= budget_min:
            perf_matrix[i, j] = perf(ctx, k)

# --- Subsample k axis for readability ---
# Show a reasonable number of k columns (about 30-40 ticks)
# Use log-ish spacing for k
k_indices_to_show = []
target_ks = sorted(set(
    [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 60, 80, 96,
     120, 160, 192, 240, 320, 384, 480, 640, 768, 960]
))
for tk in target_ks:
    if tk in all_ks:
        k_indices_to_show.append(all_ks.index(tk))

sub_ks = [all_ks[idx] for idx in k_indices_to_show]
sub_matrix = perf_matrix[:, k_indices_to_show]

# --- Plot ---
fig, ax = plt.subplots(figsize=(16, 7))

# Custom colormap
cmap = sns.color_palette("YlOrRd", as_cmap=True)

# Mask NaNs
masked = np.ma.masked_invalid(sub_matrix)

# Use imshow-style but with seaborn heatmap for nicer rendering
sns.heatmap(
    sub_matrix,
    ax=ax,
    cmap=cmap,
    xticklabels=[str(k) for k in sub_ks],
    yticklabels=context_labels,
    cbar_kws={"label": "Accuracy (%)"},
    linewidths=0.5,
    linecolor="white",
    mask=np.isnan(sub_matrix),
    vmin=35,
    vmax=90,
)

ax.set_xlabel("k (number of majority votes)", fontsize=13)
ax.set_ylabel("Context Length", fontsize=13)
ax.set_title("Iso-Compute Heatmap: Context Length vs Majority Voting (k)\nBudget = 16h", fontsize=15)

# --- Iso-compute lines ---
# Total compute = context_length * k
# Draw contour lines for specific compute budgets
iso_computes = [10, 30, 60, 120, 240, 480]  # in minutes
iso_colors = plt.cm.cool(np.linspace(0.2, 0.9, len(iso_computes)))

# We need to map from (context_idx, k_idx) space to axis coordinates
# seaborn heatmap: cell (i,j) is centered at (j+0.5, i+0.5)
for ic, compute_budget in enumerate(iso_computes):
    # For each context length, find which k gives this compute
    xs = []
    ys = []
    for i, ctx in enumerate(context_lengths_min):
        k_needed = compute_budget / ctx
        if k_needed < 1 or k_needed > max_ks[i]:
            continue
        # Find position in sub_ks axis (interpolate)
        # Map k_needed to x-position in the subsampled axis
        for jj in range(len(sub_ks) - 1):
            if sub_ks[jj] <= k_needed <= sub_ks[jj + 1]:
                frac = (k_needed - sub_ks[jj]) / (sub_ks[jj + 1] - sub_ks[jj])
                x_pos = jj + frac + 0.5
                y_pos = i + 0.5
                xs.append(x_pos)
                ys.append(y_pos)
                break
        else:
            # k_needed might equal first sub_k
            if abs(k_needed - sub_ks[0]) < 0.01:
                xs.append(0.5)
                ys.append(i + 0.5)

    if len(xs) >= 2:
        ax.plot(xs, ys, color=iso_colors[ic], linewidth=2.5, linestyle="--", alpha=0.85)
        # Label at the top-left end of the line
        label_text = f"{compute_budget}m" if compute_budget < 60 else f"{compute_budget // 60}h"
        if compute_budget == 10:
            label_text = "10m"
        elif compute_budget == 30:
            label_text = "30m"
        ax.annotate(
            label_text,
            (xs[0], ys[0]),
            fontsize=10,
            fontweight="bold",
            color=iso_colors[ic],
            ha="center",
            va="bottom",
            xytext=(0, -14),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=iso_colors[ic], alpha=0.8),
        )

plt.tight_layout()
plt.savefig("img/iso_compute_heatmap.png", dpi=200, bbox_inches="tight")
plt.savefig("img/iso_compute_heatmap.pdf", bbox_inches="tight")
print("Saved to img/iso_compute_heatmap.png and img/iso_compute_heatmap.pdf")
plt.close()

# --- Plot 2: Accuracy vs k, hue = context length ---
fig2, ax2 = plt.subplots(figsize=(12, 6))
palette = sns.color_palette("viridis", len(context_lengths_min))

for i, (ctx, label) in enumerate(zip(context_lengths_min, context_labels)):
    ks_valid = [k for k in all_ks if ctx * k <= budget_min]
    perfs = [perf(ctx, k) for k in ks_valid]
    ax2.plot(ks_valid, perfs, color=palette[i], linewidth=2, label=label, marker="o",
             markersize=3, markevery=max(1, len(ks_valid) // 15))

# Iso-compute vertical lines: for a given total compute budget, k = budget / ctx
# Since each model has a different ctx, we draw vertical spans at the k values
iso_computes_p2 = [10, 30, 60, 120, 240, 480]
iso_line_colors = plt.cm.Greys(np.linspace(0.3, 0.7, len(iso_computes_p2)))
for ic, compute_budget in enumerate(iso_computes_p2):
    # For each compute budget, plot a vertical line at k = budget/ctx for each model
    # But that's many lines -- instead, shade a region or draw labeled vlines at representative k
    # Draw a single vertical line at k values where the iso-compute crosses the median context
    # Better: for each iso budget, draw the locus of (k, perf) across all models
    ks_iso = []
    perfs_iso = []
    for ctx in context_lengths_min:
        k_at_budget = compute_budget / ctx
        if k_at_budget < 1 or k_at_budget > budget_min / ctx:
            continue
        ks_iso.append(k_at_budget)
        perfs_iso.append(perf(ctx, k_at_budget))
    if len(ks_iso) >= 2:
        # Sort by k for a clean line
        order = np.argsort(ks_iso)
        ks_iso = [ks_iso[o] for o in order]
        perfs_iso = [perfs_iso[o] for o in order]
        label_text = f"{compute_budget}m" if compute_budget < 60 else f"{compute_budget // 60}h"
        ax2.plot(ks_iso, perfs_iso, color=iso_line_colors[ic], linewidth=2, linestyle="--", alpha=0.8)
        ax2.annotate(
            label_text, (ks_iso[-1], perfs_iso[-1]), fontsize=9, fontweight="bold",
            color=iso_line_colors[ic], ha="left", va="bottom",
            xytext=(4, 2), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=iso_line_colors[ic], alpha=0.8),
        )

ax2.set_xscale("log")
ax2.set_xlabel("k (number of majority votes)", fontsize=13)
ax2.set_ylabel("Accuracy (%)", fontsize=13)
ax2.set_title("Accuracy vs k by Context Length", fontsize=15)
ax2.legend(title="Context Length", fontsize=10, title_fontsize=11)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("img/accuracy_vs_k.png", dpi=200, bbox_inches="tight")
plt.savefig("img/accuracy_vs_k.pdf", bbox_inches="tight")
print("Saved to img/accuracy_vs_k.png and img/accuracy_vs_k.pdf")
plt.close()

# --- Plot 3: Accuracy vs total compute, hue = context length ---
fig3, ax3 = plt.subplots(figsize=(12, 6))

for i, (ctx, label) in enumerate(zip(context_lengths_min, context_labels)):
    ks_valid = [k for k in all_ks if ctx * k <= budget_min]
    total_ctx = [ctx * k for k in ks_valid]
    perfs = [perf(ctx, k) for k in ks_valid]
    ax3.plot(total_ctx, perfs, color=palette[i], linewidth=2, label=label, marker="o",
             markersize=3, markevery=max(1, len(ks_valid) // 15))

ax3.set_xscale("log")
ax3.set_xlabel("Total Context (minutes) = context_length x k", fontsize=13)
ax3.set_ylabel("Accuracy (%)", fontsize=13)
ax3.set_title("Accuracy vs Total Context by Model Context Length", fontsize=15)
ax3.legend(title="Context Length", fontsize=10, title_fontsize=11)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("img/accuracy_vs_total_context.png", dpi=200, bbox_inches="tight")
plt.savefig("img/accuracy_vs_total_context.pdf", bbox_inches="tight")
print("Saved to img/accuracy_vs_total_context.png and img/accuracy_vs_total_context.pdf")
plt.close()

# --- Plot 4: Optimal context length vs total budget (Pareto front) ---
# For a dense set of total budgets, find the (ctx, k) pair maximizing perf
budgets_sweep = np.unique(np.concatenate([
    np.arange(1, 20, 1),
    np.arange(20, 100, 5),
    np.arange(100, budget_min + 1, 10),
]))

opt_budgets = []
opt_perfs = []
opt_ctxs = []
opt_ks_chosen = []

for b in budgets_sweep:
    best_p = -1
    best_ctx = None
    best_k = None
    for ctx in context_lengths_min:
        k_max = int(b // ctx)
        if k_max < 1:
            continue
        p = perf(ctx, k_max)
        if p > best_p:
            best_p = p
            best_ctx = ctx
            best_k = k_max
    if best_ctx is not None:
        opt_budgets.append(b)
        opt_perfs.append(best_p)
        opt_ctxs.append(best_ctx)
        opt_ks_chosen.append(best_k)

# Also compute per-context-length curves for background
fig4, ax4 = plt.subplots(figsize=(13, 6))

# Background: faded lines for each context length
for i, (ctx, label) in enumerate(zip(context_lengths_min, context_labels)):
    bs = []
    ps = []
    for b in budgets_sweep:
        k_max = int(b // ctx)
        if k_max < 1:
            continue
        bs.append(b)
        ps.append(perf(ctx, k_max))
    ax4.plot(bs, ps, color=palette[i], linewidth=1, alpha=0.25)

# Optimal frontier: colored segments by which ctx is chosen
# Group consecutive budget points by optimal ctx
segments = []
seg_start = 0
for j in range(1, len(opt_budgets)):
    if opt_ctxs[j] != opt_ctxs[j - 1]:
        segments.append((seg_start, j))
        seg_start = j
segments.append((seg_start, len(opt_budgets)))

ctx_to_color = {ctx: palette[i] for i, ctx in enumerate(context_lengths_min)}
ctx_to_label = {ctx: label for ctx, label in zip(context_lengths_min, context_labels)}
labeled = set()

for s, e in segments:
    ctx = opt_ctxs[s]
    lbl = ctx_to_label[ctx] if ctx not in labeled else None
    ax4.plot(
        opt_budgets[s:e], opt_perfs[s:e],
        color=ctx_to_color[ctx], linewidth=3.5, solid_capstyle="round",
        label=lbl,
    )
    labeled.add(ctx)
    # Mark transition points with context label + maj@k
    mid = (s + e) // 2
    k_val = opt_ks_chosen[mid]
    ann_text = f"{ctx_to_label[ctx]}\nmaj@{k_val}"
    ax4.annotate(
        ann_text,
        (opt_budgets[mid], opt_perfs[mid]),
        fontsize=9, fontweight="bold",
        color=ctx_to_color[ctx],
        ha="center", va="bottom",
        xytext=(0, 8), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ctx_to_color[ctx], alpha=0.85),
    )

# Annotate transition points (where optimal ctx switches) with maj@k
for s, e in segments:
    # Start of segment = transition point (except first)
    if s > 0:
        ctx = opt_ctxs[s]
        k_val = opt_ks_chosen[s]
        ax4.plot(opt_budgets[s], opt_perfs[s], "o", color=ctx_to_color[ctx],
                 markersize=7, zorder=5)

ax4.set_xscale("log")
ax4.set_xlabel("Total Compute Budget (minutes)", fontsize=13)
ax4.set_ylabel("Best Achievable Accuracy (%)", fontsize=13)
ax4.set_title("Optimal Context Length vs Total Budget (Pareto Front)", fontsize=15)
ax4.legend(title="Optimal Context", fontsize=10, title_fontsize=11, loc="lower right")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("img/optimal_tradeoff.png", dpi=200, bbox_inches="tight")
plt.savefig("img/optimal_tradeoff.pdf", bbox_inches="tight")
print("Saved to img/optimal_tradeoff.png and img/optimal_tradeoff.pdf")
plt.close()

# --- Plot 5: Min-cost frontier (cheapest way to reach a target accuracy) ---
fig5, ax5 = plt.subplots(figsize=(14, 7))

target_accs = np.arange(36, 91, 0.5)

# Store per-ctx results for annotation
ctx_results = {}
for i, (ctx, label) in enumerate(zip(context_lengths_min, context_labels)):
    min_costs = []
    min_ks = []
    valid_targets = []
    for target in target_accs:
        for k in range(1, int(budget_min // ctx) + 1):
            if perf(ctx, k) >= target:
                min_costs.append(ctx * k)
                min_ks.append(k)
                valid_targets.append(target)
                break
    ctx_results[i] = (valid_targets, min_costs, min_ks)
    if valid_targets:
        ax5.plot(valid_targets, min_costs, color=palette[i], linewidth=2, label=label)

# For every 10pp in accuracy, find the cheapest option and annotate with k
annot_targets = [40, 60, 70, 80, 90, 95, 99, 99.99]
for target in annot_targets:
    best_cost = float("inf")
    best_k = None
    best_ctx_i = None
    for i, (ctx, label) in enumerate(zip(context_lengths_min, context_labels)):
        for k in range(1, int(budget_min // ctx) + 1):
            if perf(ctx, k) >= target:
                cost = ctx * k
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
                    best_ctx_i = i
                break
    if best_ctx_i is not None and best_cost <= budget_min:
        ax5.plot(target, best_cost, "o", color=palette[best_ctx_i], markersize=8, zorder=5,
                 markeredgecolor="black", markeredgewidth=0.8)
        ax5.annotate(
            f"k={best_k}\n{context_labels[best_ctx_i]}",
            (target, best_cost),
            fontsize=8, fontweight="bold",
            color=palette[best_ctx_i],
            ha="center", va="top",
            xytext=(0, -12), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=palette[best_ctx_i], alpha=0.85),
        )

ax5.set_xscale("log")
ax5.set_yscale("log")
ax5.set_xlabel("Target Accuracy (%)", fontsize=13)
ax5.set_ylabel("Minimum Total Compute (minutes)", fontsize=13)
ax5.set_title("Cheapest Way to Reach a Target Accuracy", fontsize=15)
ax5.legend(title="Context Length", fontsize=10, title_fontsize=11)
ax5.grid(True, alpha=0.3)
ax5.axhline(y=budget_min, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
ax5.annotate("16h budget", (target_accs[0], budget_min), fontsize=10, color="red",
             va="bottom", xytext=(5, 3), textcoords="offset points")
# Iso-compute lines (horizontal with labels)
iso_computes_p5 = [10, 30, 60, 120, 240, 480]
for compute_budget in iso_computes_p5:
    label_text = f"{compute_budget}m" if compute_budget < 60 else f"{compute_budget // 60}h"
    ax5.axhline(y=compute_budget, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax5.annotate(label_text, (target_accs[-1], compute_budget), fontsize=8, color="grey",
                 ha="right", va="bottom", xytext=(-2, 1), textcoords="offset points")
plt.tight_layout()
plt.savefig("img/min_cost_frontier.png", dpi=200, bbox_inches="tight")
plt.savefig("img/min_cost_frontier.pdf", bbox_inches="tight")
print("Saved to img/min_cost_frontier.png and img/min_cost_frontier.pdf")
plt.close()

# --- Plot 6: Marginal gain per additional vote ---
fig6, ax6 = plt.subplots(figsize=(12, 6))

for i, (ctx, label) in enumerate(zip(context_lengths_min, context_labels)):
    k_max = int(budget_min // ctx)
    if k_max < 2:
        continue
    ks = np.arange(1, k_max + 1)
    perfs_arr = np.array([perf(ctx, k) for k in ks])
    # Marginal gain: perf(k) - perf(k-1)
    marginal = np.diff(perfs_arr)
    # Subsample for readability on log scale
    ks_plot = ks[1:]  # k=2,3,...
    ax6.plot(ks_plot, marginal, color=palette[i], linewidth=1.5, label=label, alpha=0.8)

ax6.set_xscale("log")
ax6.set_yscale("log")
ax6.set_xlabel("k (number of majority votes)", fontsize=13)
ax6.set_ylabel("Marginal Accuracy Gain per Vote (%)", fontsize=13)
ax6.set_title("Diminishing Returns: Marginal Gain per Additional Vote", fontsize=15)
ax6.legend(title="Context Length", fontsize=10, title_fontsize=11)
ax6.grid(True, alpha=0.3)
# Reference line for "negligible" threshold
ax6.axhline(y=0.01, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
ax6.annotate("0.01% threshold", (2, 0.01), fontsize=10, color="red",
             va="bottom", xytext=(5, 3), textcoords="offset points")
plt.tight_layout()
plt.savefig("img/marginal_gain.png", dpi=200, bbox_inches="tight")
plt.savefig("img/marginal_gain.pdf", bbox_inches="tight")
print("Saved to img/marginal_gain.png and img/marginal_gain.pdf")
plt.close()

# --- Plot 7: Double context vs double k tradeoff ---
fig7, ax7 = plt.subplots(figsize=(13, 7))

# For each context length (except the largest), compare:
#   - "double k": perf(ctx, 2k) - perf(ctx, k)
#   - "double ctx": perf(next_ctx, k) - perf(ctx, k)  where next_ctx ~ 2*ctx
# We pair each ctx with the next-larger ctx as the "doubled" version
ctx_pairs = []
for i, ctx in enumerate(context_lengths_min[:-1]):
    # Find the closest ctx that is >= 2*ctx
    doubled = [c for c in context_lengths_min if c >= 2 * ctx * 0.8]
    if doubled:
        ctx_pairs.append((i, ctx, context_lengths_min.index(doubled[0]), doubled[0]))

n_pairs = len(ctx_pairs)
fig7, axes7 = plt.subplots(2, (n_pairs + 1) // 2, figsize=(16, 9), sharex=True, sharey=True)
axes7 = axes7.flatten()

for idx, (i_ctx, ctx, i_ctx2, ctx2) in enumerate(ctx_pairs):
    ax = axes7[idx]
    budgets_plot = []
    gain_double_k = []
    gain_double_ctx = []

    for b in budgets_sweep:
        k = int(b // ctx)
        if k < 1:
            continue
        k2 = 2 * k
        # Double k: same ctx, 2x votes (costs 2x compute)
        if ctx * k2 <= budget_min:
            g_k = perf(ctx, k2) - perf(ctx, k)
        else:
            continue
        # Double ctx: next ctx, same k (costs ~2x compute)
        if ctx2 * k <= budget_min:
            g_c = perf(ctx2, k) - perf(ctx, k)
        else:
            continue
        budgets_plot.append(b)
        gain_double_k.append(g_k)
        gain_double_ctx.append(g_c)

    if budgets_plot:
        ax.plot(budgets_plot, gain_double_k, color="steelblue", linewidth=2,
                label=f"Double k")
        ax.plot(budgets_plot, gain_double_ctx, color="coral", linewidth=2,
                label=f"Double ctx ({context_labels[i_ctx2]})")
        ax.axhline(y=0, color="grey", linewidth=0.5)
        # Fill between to highlight which is better
        ax.fill_between(budgets_plot, gain_double_k, gain_double_ctx,
                         where=[gk > gc for gk, gc in zip(gain_double_k, gain_double_ctx)],
                         alpha=0.15, color="steelblue")
        ax.fill_between(budgets_plot, gain_double_k, gain_double_ctx,
                         where=[gc > gk for gk, gc in zip(gain_double_k, gain_double_ctx)],
                         alpha=0.15, color="coral")
        ax.set_xscale("log")
        ax.set_title(f"From {context_labels[i_ctx]}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(len(ctx_pairs), len(axes7)):
    axes7[idx].set_visible(False)

fig7.supxlabel("Current Compute Budget (minutes)", fontsize=13, y=0.02)
fig7.supylabel("Accuracy Gain from Doubling (%)", fontsize=13, x=0.02)
fig7.suptitle("Should You Double Context or Double k?", fontsize=15, y=0.98)
plt.tight_layout(rect=[0.03, 0.04, 1, 0.96])
plt.savefig("img/double_tradeoff.png", dpi=200, bbox_inches="tight")
plt.savefig("img/double_tradeoff.pdf", bbox_inches="tight")
print("Saved to img/double_tradeoff.png and img/double_tradeoff.pdf")
plt.close()
