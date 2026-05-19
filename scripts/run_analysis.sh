#!/usr/bin/env bash
# run_analysis.sh — Full post-inference analysis pipeline
#
# Usage:
#   bash scripts/run_analysis.sh EXP_ID [EXP_ID ...] [options]
#
# Positional args:
#   EXP_ID ...        One or more experiment IDs from v2_registry.yaml
#                     (e.g. sex_binary_lstm sleep_efficiency_binary_transformer)
#
# Options:
#   --bootstrap N     Bootstrap resamples for CIs in analyze step (default: 0 = off)
#   --split test|val  Evaluation split (default: test)
#   --heads h1 h2...  Heads for saturation/scaling-laws (default: lstm transformer mean_pool)
#   --dry-run         Print commands without running them
#
# Pipeline executed (in order):
#   1. analyze --k-dense [--bootstrap N]       per exp_id
#   2. collect                                 all exp_ids together
#   3. build-heatmap                           per exp_id
#   4. iso-plots                               per exp_id
#   5. saturation (head comparison)            per unique task
#   6. scaling-laws                            per unique task
#   7. calibration                             per exp_id
#   8. window-position                         per exp_id
#   9. subject-consistency                     per exp_id
#  10. cohort-saturation                       per exp_id
#  11. precision-recall                        per exp_id
#  12. subject-kstar                           per exp_id
#  13. task-comparison (multi-task only)       all tasks together
#
# Examples:
#   bash scripts/run_analysis.sh sex_binary_lstm --bootstrap 1000
#   bash scripts/run_analysis.sh sex_binary_lstm sex_binary_transformer sex_binary_mean_pool \
#       --bootstrap 1000 --split test
#   bash scripts/run_analysis.sh sleep_efficiency_binary_lstm --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

GEN="python scripts/gen_commands.py"
BOOTSTRAP=0
SPLIT=test
HEADS=()
DRY_RUN=false
EXP_IDS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bootstrap)
            BOOTSTRAP="$2"; shift 2 ;;
        --split)
            SPLIT="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --heads)
            shift
            while [[ $# -gt 0 && "${1:0:2}" != "--" ]]; do
                HEADS+=("$1"); shift
            done ;;
        --help|-h)
            sed -n '2,35p' "$0"; exit 0 ;;
        --*)
            echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            EXP_IDS+=("$1"); shift ;;
    esac
done

if [[ ${#EXP_IDS[@]} -eq 0 ]]; then
    echo "Error: no experiment IDs provided." >&2
    echo "Usage: bash scripts/run_analysis.sh EXP_ID [EXP_ID ...] [--bootstrap N] [--split test|val] [--heads lstm ...] [--dry-run]" >&2
    exit 1
fi

[[ ${#HEADS[@]} -eq 0 ]] && HEADS=(lstm transformer mean_pool)

# Derive unique tasks by stripping the head suffix (handles optional run_tag after head)
declare -A _TASK_SEEN
TASKS=()
for exp_id in "${EXP_IDS[@]}"; do
    task="$(echo "$exp_id" | sed 's/_\(lstm\|transformer\|mean_pool\).*//')"
    if [[ -z "${_TASK_SEEN[$task]+x}" ]]; then
        _TASK_SEEN[$task]=1
        TASKS+=("$task")
    fi
done

# Extract the actual command from gen_commands.py output (strip comment/blank lines)
gen_cmd() {
    $GEN "$@" | sed '/^#/d; /^[[:space:]]*$/d'
}

run_step() {
    local cmd="$1"
    if [[ -z "$cmd" ]]; then
        echo "  [skip — empty command]"
        return
    fi
    if $DRY_RUN; then
        echo "  [DRY-RUN] $cmd"
    else
        echo "  >>> $cmd"
        eval "$cmd"
    fi
}

echo "============================================================"
echo "  Post-inference analysis pipeline"
echo "  EXP_IDS : ${EXP_IDS[*]}"
echo "  TASKS   : ${TASKS[*]}"
echo "  SPLIT   : $SPLIT   BOOTSTRAP: $BOOTSTRAP   HEADS: ${HEADS[*]}"
$DRY_RUN && echo "  MODE    : DRY-RUN"
echo "============================================================"

# ── 1. Window analysis (k-dense, optional bootstrap) ──────────────────────
echo ""
echo "── Step 1: analyze --k-dense ────────────────────────────────────────────"
for exp_id in "${EXP_IDS[@]}"; do
    echo "   $exp_id"
    ANALYZE_ARGS=(analyze "$exp_id" --k-dense)
    [[ "$BOOTSTRAP" -gt 0 ]] && ANALYZE_ARGS+=(--bootstrap "$BOOTSTRAP")
    run_step "$(gen_cmd "${ANALYZE_ARGS[@]}")"
done

# ── 2. Collect ────────────────────────────────────────────────────────────
echo ""
echo "── Step 2: collect ──────────────────────────────────────────────────────"
run_step "$(gen_cmd collect "${EXP_IDS[@]}")"

# ── 3. Build heatmap DataFrames ───────────────────────────────────────────
echo ""
echo "── Step 3: build-heatmap ────────────────────────────────────────────────"
for exp_id in "${EXP_IDS[@]}"; do
    echo "   $exp_id"
    run_step "$(gen_cmd build-heatmap "$exp_id" --split "$SPLIT")"
done

# ── 4. Iso-compute plots ──────────────────────────────────────────────────
echo ""
echo "── Step 4: iso-plots ────────────────────────────────────────────────────"
for exp_id in "${EXP_IDS[@]}"; do
    echo "   $exp_id"
    run_step "$(gen_cmd iso-plots "$exp_id" --split "$SPLIT")"
done

# ── 5. Saturation curves (head comparison, per task) ─────────────────────
echo ""
echo "── Step 5: saturation ───────────────────────────────────────────────────"
for task in "${TASKS[@]}"; do
    echo "   $task  heads: ${HEADS[*]}"
    run_step "$(gen_cmd saturation "$task" --heads "${HEADS[@]}" --split "$SPLIT")"
done

# ── 6. Scaling law plots (per task) ──────────────────────────────────────
echo ""
echo "── Step 6: scaling-laws ─────────────────────────────────────────────────"
for task in "${TASKS[@]}"; do
    echo "   $task  heads: ${HEADS[*]}"
    run_step "$(gen_cmd scaling-laws "$task" --heads "${HEADS[@]}")"
done

# ── 7. Calibration plots ─────────────────────────────────────────────────
echo ""
echo "── Step 7: calibration ──────────────────────────────────────────────────"
for exp_id in "${EXP_IDS[@]}"; do
    echo "   $exp_id"
    run_step "$(gen_cmd calibration "$exp_id" --split "$SPLIT")"
done

# ── 8. Window position plots ──────────────────────────────────────────────
echo ""
echo "── Step 8: window-position ──────────────────────────────────────────────"
for exp_id in "${EXP_IDS[@]}"; do
    echo "   $exp_id"
    run_step "$(gen_cmd window-position "$exp_id" --split "$SPLIT")"
done

# ── 9. Subject consistency plots ─────────────────────────────────────────
echo ""
echo "── Step 9: subject-consistency ──────────────────────────────────────────"
for exp_id in "${EXP_IDS[@]}"; do
    echo "   $exp_id"
    run_step "$(gen_cmd subject-consistency "$exp_id" --split "$SPLIT")"
done

# ── 10. Cohort saturation plots ───────────────────────────────────────────
echo ""
echo "── Step 10: cohort-saturation ───────────────────────────────────────────"
for exp_id in "${EXP_IDS[@]}"; do
    echo "   $exp_id"
    run_step "$(gen_cmd cohort-saturation "$exp_id" --split "$SPLIT")"
done

# ── 11. Precision-recall plots ────────────────────────────────────────────
echo ""
echo "── Step 11: precision-recall ────────────────────────────────────────────"
for exp_id in "${EXP_IDS[@]}"; do
    echo "   $exp_id"
    run_step "$(gen_cmd precision-recall "$exp_id" --split "$SPLIT")"
done

# ── 12. Subject K* plots ──────────────────────────────────────────────────
echo ""
echo "── Step 12: subject-kstar ───────────────────────────────────────────────"
for exp_id in "${EXP_IDS[@]}"; do
    echo "   $exp_id"
    run_step "$(gen_cmd subject-kstar "$exp_id" --split "$SPLIT")"
done

# ── 13. Task comparison (only when multiple tasks present) ────────────────
if [[ ${#TASKS[@]} -gt 1 ]]; then
    echo ""
    echo "── Step 13: task-comparison ─────────────────────────────────────────────"
    echo "   tasks: ${TASKS[*]}  head: ${HEADS[0]}"
    run_step "$(gen_cmd task-comparison --tasks "${TASKS[@]}" --head "${HEADS[0]}" --split "$SPLIT")"
fi

echo ""
echo "============================================================"
echo "  Analysis pipeline complete."
echo "============================================================"
