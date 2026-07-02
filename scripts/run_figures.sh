#!/usr/bin/env bash
# run_figures.sh — Paper figure and table regeneration pipeline (figures-only)
#
# Assumes analyze --k-dense (and optionally --bootstrap) has already been run
# for all experiments, and collect + build-heatmap are up-to-date.
# Starts directly at the figure-generation steps.
#
# Usage:
#   bash scripts/run_figures.sh [options]
#
# Options:
#   --split test|val      Evaluation split (default: test)
#   --dry-run             Print commands without running them
#   --skip-tables         Skip Table 3/5/6 regeneration
#   --skip-cross-round    Skip cross-round figures (Fig 4, S-Fig 2, S-Fig 12)
#   --skip-iso            Skip iso-compute plots (slowest step; ~20 min total)
#
# Steps executed (in order):
#   1.   iso-plots                 per experiment (21 total)        → Fig 2 / S-Fig 3 inputs
#   2.   saturation                per task (7 tasks)               → Fig 1
#   3.   scaling-laws              per task (7 tasks, 1B only)      → S-Fig 8
#   4.   calibration               per experiment                   → S-Fig 4a / 4b
#   5.   window-position           per experiment (lstm only)       → S-Fig 7
#   6.   subject-consistency       per experiment (transformer)     → S-Fig 6a / 6b
#   7.   cohort-saturation         per experiment (lstm only)       → S-Fig 5
#   8.   precision-recall          per experiment                   → S-Fig 10
#   9.   subject-kstar             per experiment (transformer)     → S-Fig 9
#  10.   task-comparison           all 7 tasks                      → Fig 3 (6A + 6C)
#  11.   plot_modality_bar.py      cross-round (abl+v3+v3_full)     → Fig 4
#  12.   plot_channel_comparison.py cross-round (v3+v3_full)        → S-Fig 2
#  13.   plot_aggregate_scaling.py  v3 only                         → S-Fig 12 / Fig 5 (TBD)
#  14.   make_table1_peak_auroc.py  fast-ch + full-ch               → paper Table II
#  14b.  make_table2_lstar.py       fast-ch                         → paper Table III
#  15.   make_table4_sensitivity.py fast-ch                         → supp Table 4
#  16.   make_table9_cohort.py × 3  sex/bmi/apnea LSTM              → supp Table 9
#  17.   make_table10_ci.py         fast-ch                         → supp Table 10
#  18.   make_table3_kgrid.py       sex_binary_lstm                 → supp Table 3
#  19.   make_table5_heads.py       fast-ch                         → paper Table IV
#  20.   make_table6_modality.py    v3 + v3_abl                     → paper Table V
#
# Outputs:
#   v3 per-experiment figures  → /scratch/boshra95/psg/unified/results/phase0_v3/figures/
#   Fig 4                      → /scratch/.../phase0_v3_abl/figures/phase0_v3_abl/
#   S-Fig 2                    → /scratch/.../phase0_v3_full/figures/phase0_v3_full/
#   S-Fig 12                   → /scratch/.../phase0_v3/figures/aggregate/
#   Tables 3/5/6               → results/tables/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

GEN="python scripts/gen_commands.py"
SPLIT=test
DRY_RUN=false
SKIP_TABLES=false
SKIP_CROSS_ROUND=false
SKIP_ISO=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)          SPLIT="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --skip-tables)    SKIP_TABLES=true; shift ;;
        --skip-cross-round) SKIP_CROSS_ROUND=true; shift ;;
        --skip-iso)       SKIP_ISO=true; shift ;;
        --help|-h)        sed -n '2,42p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Experiment sets ────────────────────────────────────────────────────────────
# All 7 paper tasks × 3 heads (Tier 1: 5 tasks; Tier 2 paper tasks: depression + OSA)
ALL_EXPS=(
  sex_binary_lstm              sex_binary_transformer              sex_binary_mean_pool
  apnea_binary_lstm            apnea_binary_transformer            apnea_binary_mean_pool
  sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer sleep_efficiency_binary_mean_pool
  bmi_binary_lstm              bmi_binary_transformer              bmi_binary_mean_pool
  age_class_lstm               age_class_transformer               age_class_mean_pool
  depression_extreme_binary_lstm depression_extreme_binary_transformer depression_extreme_binary_mean_pool
  osa_binary_apples_postqc_lstm  osa_binary_apples_postqc_transformer  osa_binary_apples_postqc_mean_pool
)

# Subset: LSTM experiments only (calibration, window-position, cohort-saturation)
LSTM_EXPS=(
  sex_binary_lstm
  apnea_binary_lstm
  sleep_efficiency_binary_lstm
  bmi_binary_lstm
  age_class_lstm
  depression_extreme_binary_lstm
  osa_binary_apples_postqc_lstm
)

# Subset: Transformer experiments only (subject-consistency, subject-kstar)
TRANSFORMER_EXPS=(
  sex_binary_transformer
  apnea_binary_transformer
  sleep_efficiency_binary_transformer
  bmi_binary_transformer
  age_class_transformer
  depression_extreme_binary_transformer
  osa_binary_apples_postqc_transformer
)

# Unique tasks (in paper order)
TASKS=(
  sex_binary apnea_binary sleep_efficiency_binary bmi_binary age_class
  depression_extreme_binary osa_binary_apples_postqc
)

V3_RESULTS="/scratch/boshra95/psg/unified/results/phase0_v3"
V3_COLLECTED="results/collected/phase0_v3"

# ── Helpers ────────────────────────────────────────────────────────────────────
gen_cmd() { $GEN "$@" | sed '/^#/d; /^[[:space:]]*$/d'; }

run_step() {
    local cmd="$1"
    [[ -z "$cmd" ]] && { echo "  [skip — empty command]"; return; }
    if $DRY_RUN; then
        echo "  [DRY-RUN] $cmd"
    else
        echo "  >>> $cmd"
        eval "$cmd"
    fi
}

run_py() {
    if $DRY_RUN; then
        echo "  [DRY-RUN] python $*"
    else
        echo "  >>> python $*"
        python "$@"
    fi
}

echo "============================================================"
echo "  Paper figure generation pipeline"
echo "  SPLIT   : $SPLIT"
$DRY_RUN && echo "  MODE    : DRY-RUN"
echo "============================================================"

# ── Step 1: Iso-compute plots ──────────────────────────────────────────────────
# Outputs: figures/{task}_{head}/auroc_test/{heatmap,metric_vs_k,pareto,...}.png
# Used by: Fig 2 (sex_binary_transformer panels), S-Fig 3 (sex + apnea transformer)
echo ""
echo "── Step 1: iso-plots ────────────────────────────────────────────────────────"
if $SKIP_ISO; then
    echo "  [skipped via --skip-iso]"
else
    for exp_id in "${ALL_EXPS[@]}"; do
        echo "   $exp_id"
        run_step "$(gen_cmd iso-plots "$exp_id" --split "$SPLIT")"
    done
fi

# ── Step 2: Saturation curves (Fig 1) ─────────────────────────────────────────
# Outputs: figures/saturation/saturation_{task}_auroc_test.{png,pdf}
# Used by: Fig 1 (main paper — 7 panels, 3 lines each)
echo ""
echo "── Step 2: saturation (Fig 1) ───────────────────────────────────────────────"
for task in "${TASKS[@]}"; do
    echo "   $task"
    run_step "$(gen_cmd saturation "$task" --heads lstm transformer mean_pool --split "$SPLIT")"
done

# ── Step 3: Scaling laws (S-Fig 8) ────────────────────────────────────────────
# Outputs: figures/scaling_laws/{task}_1B_compute_scaling.{png,pdf}
# Note: 1A uShape is PENDING (rerun with BA metric). 1C is blacklisted.
echo ""
echo "── Step 3: scaling-laws 1B (S-Fig 8) ───────────────────────────────────────"
for task in "${TASKS[@]}"; do
    echo "   $task"
    run_step "$(gen_cmd scaling-laws "$task" --heads lstm transformer mean_pool --plots 1B)"
done

# ── Step 4: Calibration (S-Fig 4a, 4b) ────────────────────────────────────────
# Outputs: figures/{task}_{head}/{task}_{head}_calibration_2A_reliability.png
#          figures/{task}_{head}/{task}_calibration_2B_ece_vs_context.png
# 2C (ECE vs K) is blacklisted and excluded from default --plots.
echo ""
echo "── Step 4: calibration (S-Fig 4a / 4b) ────────────────────────────────────"
for exp_id in "${ALL_EXPS[@]}"; do
    echo "   $exp_id"
    run_step "$(gen_cmd calibration "$exp_id" --split "$SPLIT")"
done

# ── Step 5: Window position profiles (S-Fig 7) ────────────────────────────────
# Outputs: figures/{task}_{head}/{task}_{head}_window_position_4A_profiles.png
#          figures/{task}_{head}/{task}_{head}_window_position_4B_variance.png
# Paper uses: sleep_efficiency_binary_lstm + sex_binary_lstm (null control)
echo ""
echo "── Step 5: window-position (S-Fig 7) ───────────────────────────────────────"
for exp_id in "${LSTM_EXPS[@]}"; do
    echo "   $exp_id"
    run_step "$(gen_cmd window-position "$exp_id" --split "$SPLIT")"
done

# ── Step 6: Subject consistency (S-Fig 6a / 6b) ───────────────────────────────
# Outputs: {task}_transformer_subject_consistency_5A_variance.png  (5A: violins)
#          {task}_transformer_subject_consistency_5C_hard_subjects.png (5C: bars)
# 5B (variance vs K) is blacklisted.
echo ""
echo "── Step 6: subject-consistency (S-Fig 6a / 6b) ─────────────────────────────"
for exp_id in "${TRANSFORMER_EXPS[@]}"; do
    echo "   $exp_id"
    run_step "$(gen_cmd subject-consistency "$exp_id" --split "$SPLIT")"
done

# ── Step 7: Cohort saturation (S-Fig 5) ───────────────────────────────────────
# Outputs: {task}_lstm_cohort_saturation_7A.{png,pdf}
# Paper uses: sex_binary, apnea_binary, sleep_efficiency_binary (LSTM)
# 7B (N per cohort) is blacklisted.
echo ""
echo "── Step 7: cohort-saturation (S-Fig 5) ─────────────────────────────────────"
for exp_id in "${LSTM_EXPS[@]}"; do
    echo "   $exp_id"
    run_step "$(gen_cmd cohort-saturation "$exp_id" --split "$SPLIT")"
done

# ── Step 8: Precision-recall (S-Fig 10) ───────────────────────────────────────
# Outputs: {task}_{head}_pr_8A_curves.{png,pdf}
#          {task}_pr_8B_aucpr_vs_context.{png,pdf}
# 8C (vote sweep) is blacklisted.
echo ""
echo "── Step 8: precision-recall (S-Fig 10) ──────────────────────────────────────"
for exp_id in "${ALL_EXPS[@]}"; do
    echo "   $exp_id"
    run_step "$(gen_cmd precision-recall "$exp_id" --split "$SPLIT")"
done

# ── Step 9: Subject K* histograms (S-Fig 9) ───────────────────────────────────
# Outputs: {task}_transformer_kstar_9A_histogram.{png,pdf}
# 9B (coverage) is blacklisted.
echo ""
echo "── Step 9: subject-kstar (S-Fig 9) ─────────────────────────────────────────"
for exp_id in "${TRANSFORMER_EXPS[@]}"; do
    echo "   $exp_id"
    run_step "$(gen_cmd subject-kstar "$exp_id" --split "$SPLIT")"
done

# ── Step 10: Task comparison (Fig 3) ──────────────────────────────────────────
# Outputs: task_comparison/task_comparison_6A_scatter.{png,pdf}
#          task_comparison/task_comparison_6C_lstar.{png,pdf}
# 6B (bars) is blacklisted.
echo ""
echo "── Step 10: task-comparison (Fig 3: 6A + 6C) ───────────────────────────────"
run_step "$(gen_cmd task-comparison --tasks "${TASKS[@]}" --head lstm --split "$SPLIT")"

# ── Cross-round figures ────────────────────────────────────────────────────────
echo ""
if $SKIP_CROSS_ROUND; then
    echo "── Steps 11–13: cross-round figures [SKIPPED via --skip-cross-round] ───────"
else

# ── Step 11: Modality bar chart (Fig 4) ───────────────────────────────────────
# Reads: results/collected/phase0_v3_abl/analysis.csv  (ablation conditions)
#        results/collected/phase0_v3/analysis.csv       (fast-ch baseline)
#        results/collected/phase0_v3_full/analysis.csv  (full-ch reference)
# Output: /scratch/.../phase0_v3_abl/figures/phase0_v3_abl/modality_ablation_bar.{png,pdf}
echo "── Step 11: Fig 4 — modality bar chart (plot_modality_bar.py) ──────────────"
run_py scripts/plot_modality_bar.py --split "$SPLIT"

# ── Step 12: Fast vs full channel comparison (S-Fig 2) ────────────────────────
# Reads: results/collected/phase0_v3/analysis.csv       (fast-ch)
#        results/collected/phase0_v3_full/analysis.csv  (full-ch)
# Output: /scratch/.../phase0_v3_full/figures/phase0_v3_full/channel_comparison.{png,pdf}
echo ""
echo "── Step 12: S-Fig 2 — channel comparison (plot_channel_comparison.py) ──────"
run_py scripts/plot_channel_comparison.py --split "$SPLIT"

# ── Step 13: Aggregate scaling law (S-Fig 12 / Fig 5 TBD) ────────────────────
# Reads: results/collected/phase0_v3/analysis.csv
# Output: /scratch/.../phase0_v3/figures/aggregate/aggregate_scaling.{png,pdf}
# Note: use --exclude-tasks depression_extreme_binary to check robustness without
#       the non-monotonic outlier when deciding main vs. supplementary placement.
echo ""
echo "── Step 13: S-Fig 12 / Fig 5 — aggregate scaling (plot_aggregate_scaling.py)"
run_py scripts/plot_aggregate_scaling.py \
    --collected-dir "$V3_COLLECTED" \
    --results-dir   "$V3_RESULTS"   \
    --split "$SPLIT"

fi  # end skip-cross-round

# ── Table regeneration ─────────────────────────────────────────────────────────
# All scripts default to results/collected/phase0_v3/analysis.csv (v3 fast-channel).
# Paper table mapping:
#   Table 1 script → paper Table II  (peak AUROC per task × head)
#   Table 2 script → paper Table III (L* saturation context per task)
#   Table 3 script → supp K-grid     (K × context AUROC pivot, sex_binary_lstm example)
#   Table 4 script → supp sensitivity (context-sensitivity ranking)
#   Table 5 script → paper Table IV  (head comparison at LSTM's L*)
#   Table 6 script → paper Table V   (modality ablation Δ, cross-round: v3 + v3_abl)
#   Table 9 script → supp cohort     (per-dataset AUROC at L*, reads parquets)
#   Table 10 script→ supp CI         (bootstrap 95% CIs; requires --bootstrap in analyze)
echo ""
if $SKIP_TABLES; then
    echo "── Steps 14–21: table regeneration [SKIPPED via --skip-tables] ─────────────"
else

# ── Step 14: Tables 1 + 2 — Primary paper tables (peak AUROC and L*) ─────────
# Table 1 needs BOTH fast-ch and full-ch: paper Table II shows both as columns
#   (paper_figures.md: "Peak AUROC: fast-ch + full-ch columns, K=5 & K=all at best L")
# Table 2 (L*) is fast-ch only in the main paper.
# Output: results/tables/table1_peak_auroc_fast.{csv,md,tex}
#         results/tables/table1_peak_auroc_full.{csv,md,tex}
#         results/tables/table2_lstar_fast.{csv,md,tex}
echo "── Step 14: Table 1 — peak AUROC fast-ch (make_table1_peak_auroc.py) ───────"
run_py scripts/make_table1_peak_auroc.py --latex

echo ""
echo "── Step 14b: Table 1 — peak AUROC full-ch ──────────────────────────────────"
run_py scripts/make_table1_peak_auroc.py \
    --collected-dir results/collected/phase0_v3_full \
    --channel full \
    --latex

echo ""
echo "── Step 14c: Table 2 — L* saturation (make_table2_lstar.py) ────────────────"
run_py scripts/make_table2_lstar.py --latex

# ── Step 15: Table 4 — Context sensitivity ranking (supplementary) ────────────
# Output: results/tables/table4_sensitivity_fast_lstm.{csv,md,tex}
echo ""
echo "── Step 15: Table 4 — sensitivity (make_table4_sensitivity.py) ─────────────"
run_py scripts/make_table4_sensitivity.py --latex

# ── Step 16: Table 9 — Per-cohort AUROC breakdown (supplementary) ─────────────
# Reads inference parquets (not collected CSVs) at the L* context for each task.
# PAPER_TABLES.md: "representative tasks: sex_binary, bmi_binary, apnea_binary"
TABLE9_EXPS=(sex_binary_lstm bmi_binary_lstm apnea_binary_lstm)
echo ""
echo "── Step 16: Table 9 — per-cohort AUROC (make_table9_cohort.py) ─────────────"
for exp_id in "${TABLE9_EXPS[@]}"; do
    echo "   $exp_id"
    run_py scripts/make_table9_cohort.py "$exp_id" --latex
done

# ── Step 17: Table 10 — Bootstrap CI summary (supplementary) ─────────────────
# Requires analysis.csv to have CI columns (from analyze_windows.py --bootstrap N).
# Output: results/tables/table10_ci_fast.{csv,md,tex}
echo ""
echo "── Step 17: Table 10 — bootstrap CIs (make_table10_ci.py) ──────────────────"
run_py scripts/make_table10_ci.py --latex

# ── Step 18: Table 3 — K-grid supplementary table ─────────────────────────────
# sex_binary_lstm is the standard supplementary example.
# Output: results/tables/table3_kgrid_sex_binary_lstm_fast.{csv,md,tex}
echo ""
echo "── Step 18: Table 3 — K-grid (make_table3_kgrid.py sex_binary_lstm) ────────"
run_py scripts/make_table3_kgrid.py sex_binary_lstm --latex

# ── Step 19: Table 5 — Head architecture comparison at L* ─────────────────────
# Reads: results/collected/phase0_v3/analysis.csv
# Output: results/tables/table5_heads_fast.{csv,md,tex}
echo ""
echo "── Step 19: Table 5 — head comparison (make_table5_heads.py) ───────────────"
run_py scripts/make_table5_heads.py --latex

# ── Step 20: Table 6 — Modality ablation ΔAUROC ───────────────────────────────
# Reads: results/collected/phase0_v3/analysis.csv   (Full baseline)
#        results/collected/phase0_v3_abl/analysis.csv (ablation conditions)
# Output: results/tables/table6_modality.{csv,md,tex}
echo ""
echo "── Step 20: Table 6 — modality ablation (make_table6_modality.py) ──────────"
run_py scripts/make_table6_modality.py --latex

fi  # end skip-tables

echo ""
echo "============================================================"
echo "  Figure generation pipeline complete."
echo "============================================================"
