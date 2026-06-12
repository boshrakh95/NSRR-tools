#!/usr/bin/env bash
# gen_tables.sh — Generate paper tables from collected analysis results.
#
# Usage:
#   bash scripts/gen_tables.sh [EXP_ID ...] [options]
#
# Positional args:
#   EXP_ID ...        One or more experiment IDs (e.g. sex_binary_lstm bmi_binary_lstm).
#                     Used for per-experiment tables (Table 3: K grid, Table 9: cohort).
#                     Multi-task tables (1, 2, 4, 5, 10) use all tasks found in analysis.csv
#                     by default, or the subset specified by --tasks.
#
# Options:
#   --tasks t1 t2...  Override which tasks appear in multi-task tables
#                     (default: all tasks in analysis.csv)
#   --registry PATH   Registry YAML (default: experiments/v2_registry.yaml)
#                     Use experiments/v2_full_registry.yaml for full-channel tables
#   --tables N [N...] Which tables to generate (default: 1 2 3 4 5 9 10)
#                     Available: 1 2 3 4 5 9 10
#   --heads h1 h2...  Heads for multi-task tables (default: lstm transformer mean_pool)
#   --k-deploy N      K value for deployment column in Tables 1, 5, 10 (default: 5)
#   --split test|val  Evaluation split (default: test)
#   --out DIR         Output directory for table files (default: results/tables/)
#   --dry-run         Print commands without running them
#
# How it works:
#   - Tables 1, 2, 4, 5, 10 are multi-task: one row per task (use --tasks to restrict rows)
#   - Table 3 (K grid) is per-experiment: run once per EXP_ID provided
#   - Table 9 (cohort) is per-experiment: run once per EXP_ID provided
#   - All tables read from results/collected/<channel>/analysis.csv (fast or full channel
#     determined automatically from the registry's results_dir)
#   - Table 9 also reads inference parquets from scratch
#
# Single-task examples (just one experiment):
#   bash scripts/gen_tables.sh sex_binary_lstm
#   bash scripts/gen_tables.sh sex_binary_lstm --tables 3 9
#
# Multi-task examples:
#   bash scripts/gen_tables.sh sex_binary_lstm bmi_binary_lstm apnea_binary_lstm \
#       --tasks sex_binary bmi_binary apnea_binary
#
# Full-channel tables:
#   bash scripts/gen_tables.sh sex_binary_lstm bmi_binary_lstm \
#       --registry experiments/v2_full_registry.yaml
#
# Loop example — run tables for a list of tasks one at a time:
#   for TASK in sex_binary bmi_binary apnea_binary sleep_efficiency_binary; do
#       bash scripts/gen_tables.sh ${TASK}_lstm ${TASK}_transformer \
#           --tasks "$TASK" --tables 3 9
#   done
#
# All tasks, fast channel, all tables:
#   EXP_IDS=(sex_binary_lstm sex_binary_transformer bmi_binary_lstm bmi_binary_transformer
#            apnea_binary_lstm apnea_binary_transformer age_class_lstm age_class_transformer
#            sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer
#            cvd_binary_lstm sleepiness_binary_lstm depression_extreme_binary_lstm
#            osa_binary_apples_postqc_lstm psqi_binary_lstm)
#   bash scripts/gen_tables.sh "${EXP_IDS[@]}"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

GEN="python scripts/gen_commands.py"
REGISTRY=""
TASKS=()
EXP_IDS=()
HEADS=()
TABLES=(1 2 3 4 5 9 10)
K_DEPLOY=5
SPLIT=test
OUT=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry)   REGISTRY="$2";   shift 2 ;;
        --split)      SPLIT="$2";      shift 2 ;;
        --k-deploy)   K_DEPLOY="$2";   shift 2 ;;
        --out)        OUT="$2";         shift 2 ;;
        --dry-run)    DRY_RUN=true;     shift ;;
        --tasks)
            shift
            while [[ $# -gt 0 && "${1:0:2}" != "--" ]]; do
                TASKS+=("$1"); shift
            done ;;
        --heads)
            shift
            while [[ $# -gt 0 && "${1:0:2}" != "--" ]]; do
                HEADS+=("$1"); shift
            done ;;
        --tables)
            TABLES=()
            shift
            while [[ $# -gt 0 && "${1:0:2}" != "--" ]]; do
                TABLES+=("$1"); shift
            done ;;
        --help|-h)
            sed -n '2,60p' "$0"; exit 0 ;;
        --*)
            echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            EXP_IDS+=("$1"); shift ;;
    esac
done

[[ -n "$REGISTRY" ]] && GEN="$GEN --registry $REGISTRY"
[[ ${#HEADS[@]} -eq 0 ]] && HEADS=(lstm transformer mean_pool)

# Helper: extract command from gen output (strip comment/blank lines)
gen_cmd() { $GEN "$@" | sed '/^#/d; /^[[:space:]]*$/d'; }

run_step() {
    local label="$1"
    local cmd="$2"
    echo "  ── $label"
    if [[ -z "$cmd" ]]; then
        echo "     [skip — empty command]"; return
    fi
    if $DRY_RUN; then
        echo "     [DRY-RUN] $cmd"
    else
        echo "     >>> $cmd"
        eval "$cmd"
    fi
}

# Build optional flag strings
TASKS_FLAG=""
[[ ${#TASKS[@]} -gt 0 ]] && TASKS_FLAG="--tasks ${TASKS[*]}"
HEADS_FLAG="--heads ${HEADS[*]}"
OUT_FLAG=""
[[ -n "$OUT" ]] && OUT_FLAG="--out $OUT"

echo "============================================================"
echo "  Paper table generation"
echo "  EXP_IDS : ${EXP_IDS[*]:-<none>}"
echo "  TASKS   : ${TASKS[*]:-all in analysis.csv}"
echo "  TABLES  : ${TABLES[*]}"
echo "  SPLIT   : $SPLIT   K_DEPLOY: $K_DEPLOY   HEADS: ${HEADS[*]}"
echo "  REGISTRY: ${REGISTRY:-experiments/v2_registry.yaml (default)}"
$DRY_RUN && echo "  MODE    : DRY-RUN"
echo "============================================================"

# ── Table 1: Peak AUROC ────────────────────────────────────────────────────────
if printf '%s\n' "${TABLES[@]}" | grep -qx "1"; then
    echo ""
    echo "── Table 1: Peak AUROC across tasks ─────────────────────────────────────────"
    ARGS=(table-1 --split "$SPLIT" --k-deploy "$K_DEPLOY")
    [[ ${#TASKS[@]}   -gt 0 ]] && ARGS+=(--tasks "${TASKS[@]}")
    [[ ${#HEADS[@]}   -gt 0 ]] && ARGS+=(--heads "${HEADS[@]}")
    [[ -n "$OUT_FLAG"       ]] && { CMD="$(gen_cmd "${ARGS[@]}") $OUT_FLAG"; } || CMD="$(gen_cmd "${ARGS[@]}")"
    run_step "table-1" "$CMD"
fi

# ── Table 2: Saturation L* ────────────────────────────────────────────────────
if printf '%s\n' "${TABLES[@]}" | grep -qx "2"; then
    echo ""
    echo "── Table 2: Saturation L* per task ──────────────────────────────────────────"
    ARGS=(table-2 --split "$SPLIT")
    [[ ${#TASKS[@]} -gt 0 ]] && ARGS+=(--tasks "${TASKS[@]}")
    [[ -n "$OUT_FLAG"      ]] && { CMD="$(gen_cmd "${ARGS[@]}") $OUT_FLAG"; } || CMD="$(gen_cmd "${ARGS[@]}")"
    run_step "table-2" "$CMD"
fi

# ── Table 3: K grid (per experiment) ─────────────────────────────────────────
if printf '%s\n' "${TABLES[@]}" | grep -qx "3"; then
    echo ""
    echo "── Table 3: AUROC×K grid (per experiment) ───────────────────────────────────"
    if [[ ${#EXP_IDS[@]} -eq 0 ]]; then
        echo "  [skip — no EXP_IDs provided; pass at least one experiment ID for Table 3]"
    else
        for exp_id in "${EXP_IDS[@]}"; do
            echo "   $exp_id"
            ARGS=(table-3 "$exp_id" --split "$SPLIT")
            [[ -n "$OUT_FLAG" ]] && { CMD="$(gen_cmd "${ARGS[@]}") $OUT_FLAG"; } || CMD="$(gen_cmd "${ARGS[@]}")"
            run_step "table-3 $exp_id" "$CMD"
        done
    fi
fi

# ── Table 4: Cross-task sensitivity ──────────────────────────────────────────
if printf '%s\n' "${TABLES[@]}" | grep -qx "4"; then
    echo ""
    echo "── Table 4: Cross-task context sensitivity ──────────────────────────────────"
    ARGS=(table-4 --head lstm --split "$SPLIT")
    [[ ${#TASKS[@]} -gt 0 ]] && ARGS+=(--tasks "${TASKS[@]}")
    [[ -n "$OUT_FLAG"      ]] && { CMD="$(gen_cmd "${ARGS[@]}") $OUT_FLAG"; } || CMD="$(gen_cmd "${ARGS[@]}")"
    run_step "table-4" "$CMD"
fi

# ── Table 5: Head comparison ─────────────────────────────────────────────────
if printf '%s\n' "${TABLES[@]}" | grep -qx "5"; then
    echo ""
    echo "── Table 5: Head comparison at L* ───────────────────────────────────────────"
    ARGS=(table-5 --split "$SPLIT" --k-deploy "$K_DEPLOY" --heads "${HEADS[@]}")
    [[ ${#TASKS[@]} -gt 0 ]] && ARGS+=(--tasks "${TASKS[@]}")
    [[ -n "$OUT_FLAG"      ]] && { CMD="$(gen_cmd "${ARGS[@]}") $OUT_FLAG"; } || CMD="$(gen_cmd "${ARGS[@]}")"
    run_step "table-5" "$CMD"
fi

# ── Table 9: Cohort breakdown (per experiment) ────────────────────────────────
if printf '%s\n' "${TABLES[@]}" | grep -qx "9"; then
    echo ""
    echo "── Table 9: Per-cohort AUROC breakdown (per experiment) ─────────────────────"
    if [[ ${#EXP_IDS[@]} -eq 0 ]]; then
        echo "  [skip — no EXP_IDs provided; pass at least one experiment ID for Table 9]"
    else
        for exp_id in "${EXP_IDS[@]}"; do
            echo "   $exp_id"
            ARGS=(table-9 "$exp_id" --split "$SPLIT")
            [[ -n "$OUT_FLAG" ]] && { CMD="$(gen_cmd "${ARGS[@]}") $OUT_FLAG"; } || CMD="$(gen_cmd "${ARGS[@]}")"
            run_step "table-9 $exp_id" "$CMD"
        done
    fi
fi

# ── Table 10: Bootstrap CI summary ────────────────────────────────────────────
if printf '%s\n' "${TABLES[@]}" | grep -qx "10"; then
    echo ""
    echo "── Table 10: Bootstrap CI summary ───────────────────────────────────────────"
    ARGS=(table-10 --split "$SPLIT" --k-deploy "$K_DEPLOY")
    [[ ${#TASKS[@]}  -gt 0 ]] && ARGS+=(--tasks "${TASKS[@]}")
    [[ ${#HEADS[@]}  -gt 0 ]] && ARGS+=(--heads "${HEADS[@]}")
    [[ -n "$OUT_FLAG"       ]] && { CMD="$(gen_cmd "${ARGS[@]}") $OUT_FLAG"; } || CMD="$(gen_cmd "${ARGS[@]}")"
    run_step "table-10" "$CMD"
fi

echo ""
echo "============================================================"
echo "  Table generation complete."
echo "  Output files are in: ${OUT:-results/tables/}"
echo "============================================================"
