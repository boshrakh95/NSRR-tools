#!/usr/bin/env python3
"""
plot_scaling_laws.py — §1 Overfitting Curves and Compute Scaling Laws

Reads results/collected/training.csv (output of collect_results_v2.py) and
produces three figures:

  1A: U-shape curves — train_loss and val_loss vs epoch per context length.
      Includes overfit epochs (is_overfit_epoch=True) to show the rising right arm.
      Best epoch marked with a dashed vertical line.

  1B: Compute scaling law — cumulative training FLOPs at best epoch vs test AUROC.
      FLOPs computed analytically from seq_len × steps_per_epoch × FLOPs/token.
      Fit a power-law: AUROC ≈ ceiling − A × FLOPs^(−b).
      Skipped automatically if seq_len or steps_per_epoch are missing from training.csv.

  1C: Optimal epoch vs context length — bar chart showing the epoch at which
      val loss is minimised (= early stopping point) per head and context.

FLOPs formulas (per gradient step, forward + backward ≈ 3× forward):
  LSTM:        3 × seq_len × 4 × hidden_dim × (input_dim + hidden_dim)
  Transformer: 3 × seq_len × (seq_len × hidden_dim + 4 × hidden_dim²)
  MeanPool:    3 × seq_len × input_dim

Usage:
  python scripts/plot_scaling_laws.py \\
      --task sex_binary \\
      --heads lstm transformer mean_pool \\
      --collected-dir results/collected \\
      --results-dir /scratch/boshra95/psg/unified/results/phase0_v2

Output:
  {results_dir}/figures/scaling_laws/{task}_{head}_1A_uShape.{png,pdf}
  {results_dir}/figures/scaling_laws/{task}_1B_compute_scaling.{png,pdf}
  {results_dir}/figures/scaling_laws/{task}_1C_optimal_epoch.{png,pdf}
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Style ─────────────────────────────────────────────────────────────────────

CONTEXT_TO_MIN = {
    "30s": 0.5, "10m": 10.0, "40m": 40.0,
    "80m": 80.0, "120m": 120.0, "240m": 240.0,
}
CTX_ORDER = {c: i for i, c in enumerate(CONTEXT_TO_MIN)}

HEAD_STYLE = {
    "lstm":       {"color": "#4C72B0", "marker": "o", "ls": "-",  "label": "LSTM"},
    "transformer":{"color": "#DD8452", "marker": "s", "ls": "--", "label": "Transformer"},
    "mean_pool":  {"color": "#55A868", "marker": "^", "ls": ":",  "label": "Mean Pool"},
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


# ── FLOPs computation ─────────────────────────────────────────────────────────

def compute_flops_per_step(head: str, seq_len, input_dim, hidden_dim) -> float:
    """Analytical FLOPs per gradient step (forward + backward ≈ 3× forward)."""
    try:
        sl = int(seq_len)
        id_ = int(input_dim)
        hd  = int(hidden_dim)
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


# ── Context colour helper ─────────────────────────────────────────────────────

def ctx_color_map(contexts: list) -> dict:
    n = max(len(contexts), 2)
    cmap = cm.get_cmap("viridis_r", n)
    return {c: cmap(i / (n - 1)) for i, c in enumerate(contexts)}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_training_csv(collected_dir: Path) -> pd.DataFrame:
    p = collected_dir / "training.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"training.csv not found at {p}. "
            "Run collect_results_v2.py first."
        )
    df = pd.read_csv(p)
    # Normalise boolean columns that may have been stored as strings
    for col in ("is_best_epoch", "is_overfit_epoch"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False, "nan": False}
            ).fillna(False).astype(bool)
    if "context_length" in df.columns:
        df["context_min"] = df["context_length"].map(
            lambda s: CONTEXT_TO_MIN.get(str(s).strip(), None)
        )
    return df


# ── Plot 1A: U-shape overfitting curves ───────────────────────────────────────

def plot_uShape(df: pd.DataFrame, task: str, head: str, out_dir: Path) -> None:
    sub = df[(df["task"] == task) & (df["head"] == head)].copy()
    if sub.empty:
        print(f"  [1A] No data for {task}/{head}")
        return

    contexts = sorted(sub["context_length"].dropna().unique(),
                      key=lambda c: CTX_ORDER.get(c, 99))
    if not contexts:
        return

    ncols = min(len(contexts), 3)
    nrows = (len(contexts) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             sharey=False, squeeze=False)
    fig.suptitle(
        f"U-Shape Overfitting Curves — {task}  ({head.upper()})\n"
        "Shaded region = generalisation gap after early stopping",
        fontsize=11, y=1.01,
    )

    for ax_flat, ctx in zip(axes.flat, contexts):
        rows = sub[sub["context_length"] == ctx].sort_values("epoch")
        if rows.empty:
            ax_flat.set_visible(False)
            continue

        epochs     = rows["epoch"].values
        train_loss = rows["train_loss"].values
        val_loss   = rows["val_loss"].values
        is_overfit = rows.get("is_overfit_epoch", pd.Series([False] * len(rows))).values
        best_rows  = rows[rows.get("is_best_epoch", pd.Series([False] * len(rows))).values]
        best_epoch = int(best_rows["epoch"].iloc[0]) if not best_rows.empty else None

        normal_mask  = ~is_overfit
        overfit_mask = is_overfit

        ax_flat.plot(epochs[normal_mask], train_loss[normal_mask],
                     color="#1f77b4", lw=2, label="Train loss")
        ax_flat.plot(epochs[normal_mask], val_loss[normal_mask],
                     color="#d62728", lw=2, label="Val loss")
        if overfit_mask.any():
            ax_flat.plot(epochs[overfit_mask], train_loss[overfit_mask],
                         color="#1f77b4", lw=2, ls=":", alpha=0.7)
            ax_flat.plot(epochs[overfit_mask], val_loss[overfit_mask],
                         color="#d62728", lw=2, ls=":", alpha=0.7)

        # Generalisation gap shading (overfit region)
        if overfit_mask.any():
            tr_o = train_loss[overfit_mask]
            vl_o = val_loss[overfit_mask]
            ep_o = epochs[overfit_mask]
            ax_flat.fill_between(ep_o, tr_o, vl_o,
                                 where=vl_o > tr_o, alpha=0.15, color="red",
                                 label="Generalisation gap")

        if best_epoch is not None:
            ax_flat.axvline(best_epoch, color="gray", ls="--", lw=1.5,
                            label=f"Best epoch ({best_epoch})")

        ctx_min = CONTEXT_TO_MIN.get(ctx)
        ax_flat.set_title(
            f"L = {ctx}" + (f"  ({ctx_min:.0f} min)" if ctx_min else ""),
            fontsize=10,
        )
        ax_flat.set_xlabel("Epoch")
        ax_flat.set_ylabel("Loss")
        ax_flat.set_xlim(left=1)

    axes.flat[0].legend(fontsize=8, loc="upper right")
    # Hide unused subplots
    for ax in axes.flat[len(contexts):]:
        ax.set_visible(False)

    fig.tight_layout()
    stem = f"{task}_{head}_1A_uShape"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}",
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1A] Saved: {stem}.{{png,pdf}}")


# ── Plot 1B: Compute scaling law ──────────────────────────────────────────────

def plot_scaling_law(df: pd.DataFrame, task: str, heads: list, out_dir: Path) -> None:
    best_df = df[(df["task"] == task) & df["is_best_epoch"]].copy()
    if best_df.empty:
        print(f"  [1B] No best-epoch rows for {task}")
        return

    required = {"seq_len", "steps_per_epoch", "input_dim", "hidden_dim"}
    missing = required - set(best_df.columns)
    if missing:
        print(f"  [1B] Skipped — columns missing from training.csv: {missing}")
        return

    # Compute FLOPs and drop rows where computation fails
    best_df["flops_per_step"] = best_df.apply(
        lambda r: compute_flops_per_step(
            str(r["head"]), r["seq_len"], r["input_dim"], r["hidden_dim"]),
        axis=1,
    )
    best_df["total_flops"] = (
        best_df["flops_per_step"] * best_df["steps_per_epoch"] * best_df["epoch"]
    )
    valid = best_df.dropna(subset=["total_flops", "test_auroc"])
    if valid.empty:
        print("  [1B] Skipped — no rows with valid FLOPs and test_auroc")
        return

    contexts = sorted(valid["context_length"].dropna().unique(),
                      key=lambda c: CTX_ORDER.get(c, 99))
    colors = ctx_color_map(contexts)

    fig, ax = plt.subplots(figsize=(10, 6))

    def _power_law(F, a, b, c):
        return c - a * np.asarray(F) ** (-b)

    for head in heads:
        hsub = valid[valid["head"] == head]
        if hsub.empty:
            continue
        hs = HEAD_STYLE.get(head, {"color": "gray", "marker": "o", "label": head})
        Fs, aucs = [], []
        for _, row in hsub.iterrows():
            ctx = str(row["context_length"])
            ax.scatter(row["total_flops"], row["test_auroc"],
                       color=colors.get(ctx, "gray"),
                       marker=hs["marker"], s=90, zorder=5,
                       edgecolors=hs["color"], linewidths=1.5)
            Fs.append(row["total_flops"])
            aucs.append(row["test_auroc"])

        # Power-law fit per head
        if HAS_SCIPY and len(Fs) >= 3:
            Fs_a  = np.array(Fs, dtype=float)
            auc_a = np.array(aucs, dtype=float)
            try:
                norm = Fs_a.min()
                p0   = [0.1, 0.2, auc_a.max()]
                popt, _ = curve_fit(_power_law, Fs_a / norm, auc_a,
                                    p0=p0, maxfev=10000,
                                    bounds=([0, 1e-3, 0], [10, 2, 1]))
                F_grid = np.linspace(Fs_a.min(), Fs_a.max(), 300)
                fit    = _power_law(F_grid / norm, *popt)
                ax.plot(F_grid, fit, color=hs["color"], lw=2, ls="--", alpha=0.6,
                        label=f"{hs['label']} fit")
            except Exception:
                pass

    # Context colour legend
    for ctx in contexts:
        ax.scatter([], [], color=colors[ctx], s=60,
                   label=f"L={ctx}")
    # Head marker legend
    for head in heads:
        hs = HEAD_STYLE.get(head, {"color": "gray", "marker": "o", "label": head})
        ax.scatter([], [], color="gray", marker=hs["marker"], s=70, label=hs["label"])

    ax.set_xscale("log")
    ax.set_xlabel("Total Training FLOPs (log scale)", fontsize=11)
    ax.set_ylabel("Test AUROC at best epoch", fontsize=11)
    ax.set_title(
        f"Compute Scaling Law — {task}\n"
        "Color = context length · Marker = head · Dashed = power-law fit",
        fontsize=11,
    )
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    fig.tight_layout()

    stem = f"{task}_1B_compute_scaling"
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}",
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1B] Saved: {stem}.{{png,pdf}}")


# ── Plot 1C: Optimal epoch vs context length ──────────────────────────────────

def plot_optimal_epoch(df: pd.DataFrame, task: str, heads: list, out_dir: Path) -> None:
    best_df = df[(df["task"] == task) & df["is_best_epoch"]].copy()
    if best_df.empty:
        print(f"  [1C] No best-epoch rows for {task}")
        return

    contexts = sorted(best_df["context_length"].dropna().unique(),
                      key=lambda c: CTX_ORDER.get(c, 99))
    if not contexts:
        return

    n_heads = len([h for h in heads if not best_df[best_df["head"] == h].empty])
    x = np.arange(len(contexts))
    width = 0.8 / max(n_heads, 1)

    fig, ax = plt.subplots(figsize=(max(8, len(contexts) * 1.4), 5))
    plotted = 0
    for i, head in enumerate(heads):
        hsub = best_df[best_df["head"] == head]
        if hsub.empty:
            continue
        hs = HEAD_STYLE.get(head, {"color": "gray", "label": head})
        epochs = [
            int(hsub[hsub["context_length"] == ctx]["epoch"].iloc[0])
            if not hsub[hsub["context_length"] == ctx].empty else 0
            for ctx in contexts
        ]
        offset = (i - n_heads / 2 + 0.5) * width
        bars = ax.bar(x + offset, epochs, width * 0.9,
                      color=hs["color"], label=hs["label"], alpha=0.85,
                      edgecolor="white")
        for bar, ep in zip(bars, epochs):
            if ep > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3, str(ep),
                        ha="center", va="bottom", fontsize=8, color=hs["color"])
        plotted += 1

    ax.set_xticks(x)
    ax.set_xticklabels(contexts, fontsize=10)
    ax.set_xlabel("Context Length", fontsize=11)
    ax.set_ylabel("Optimal Epoch (val-loss minimum)", fontsize=11)
    ax.set_title(
        f"Optimal Epoch vs Context Length — {task}\n"
        "Longer context may require more epochs to converge",
        fontsize=11,
    )
    ax.legend()
    fig.tight_layout()

    stem = f"{task}_1C_optimal_epoch"
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}",
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1C] Saved: {stem}.{{png,pdf}}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot U-shape overfitting curves, compute scaling law, "
                    "and optimal epoch chart."
    )
    parser.add_argument("--task", required=True,
                        help="Task name, e.g. sex_binary")
    parser.add_argument("--heads", nargs="+",
                        default=["lstm", "transformer", "mean_pool"],
                        help="Heads to include")
    parser.add_argument("--collected-dir", type=Path,
                        default=Path("results/collected"),
                        dest="collected_dir",
                        help="Directory containing training.csv (default: results/collected)")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0_v2"),
                        dest="results_dir")
    parser.add_argument("--plots", nargs="+",
                        default=["1A", "1B", "1C"],
                        help="Which plots to generate (1A 1B 1C, default: all)")
    args = parser.parse_args()

    out_dir = args.results_dir / "figures" / "scaling_laws"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training.csv from {args.collected_dir} ...")
    try:
        df = load_training_csv(args.collected_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    task_rows = df[df["task"] == args.task]
    if task_rows.empty:
        print(f"ERROR: No rows for task '{args.task}' in training.csv")
        return

    print(f"Task: {args.task}  Heads: {args.heads}")
    print(f"Total rows: {len(task_rows):,}  "
          f"Contexts: {sorted(task_rows['context_length'].unique())}")

    if "1A" in args.plots:
        for head in args.heads:
            plot_uShape(df, args.task, head, out_dir)

    if "1B" in args.plots:
        plot_scaling_law(df, args.task, args.heads, out_dir)

    if "1C" in args.plots:
        plot_optimal_epoch(df, args.task, args.heads, out_dir)

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
