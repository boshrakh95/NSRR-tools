#!/usr/bin/env python3
"""Generate LaTeX Table-1-style (full sweep) and Table-3-style (head comparison)
tables for the OSF / PhysioOmni / Mantis TSFM baselines, pulling numbers
directly from the collected analysis.csv files. No values are hand-typed.

Written 2026-09-17 for the TSFM baseline paper-draft work
(docs/npj_paper_md_files/TSFM_BASELINE_RESULTS_DRAFT.md, Section 6). The
tables it prints there were captured verbatim from this script's stdout.

Usage: python3 scripts/gen_tsfm_baseline_tables.py
(reads results/collected/{phase0_osf,phase0_physioomni}/analysis.csv from
this repo, and results/collected/phase0_mantis/analysis.csv from the
NSRR-tools-mantis worktree via a hardcoded absolute path below — update that
path if the worktree moves, or once Mantis results are merged into main).

Re-run this whenever the underlying analysis.csv files change (e.g. once
Mantis's mean_pool head or a later LoRA pass lands) rather than hand-editing
the tables in the draft doc.
"""
import pandas as pd
from pathlib import Path

CONTEXTS = ["30s", "10m", "40m", "80m", "120m", "240m"]
CTX_HEAD = {"30s": "30~s", "10m": "10~min", "40m": "40~min",
            "80m": "80~min", "120m": "120~min", "240m": "240~min"}

TASK_LABEL = {
    "sleep_efficiency_binary": "Sleep efficiency",
    "apnea_binary": "Apnea detection",
    "sex_binary": "Sex classification",
    "age_class": "Age-group prediction",
    "bmi_binary": "BMI (obese)",
    "osa_binary_apples_postqc": "OSA severity (APPLES)",
    "depression_extreme_binary": "Depression screening",
}
SECONDARY = {"osa_binary_apples_postqc", "depression_extreme_binary"}

SOURCES = {
    "OSF": Path("/Users/boshra/NSRR-workspace/NSRR-tools/results/collected/phase0_osf/analysis.csv"),
    "PhysioOmni": Path("/Users/boshra/NSRR-workspace/NSRR-tools/results/collected/phase0_physioomni/analysis.csv"),
    "Mantis": Path("/Users/boshra/NSRR-workspace/NSRR-tools-mantis/results/collected/phase0_mantis/analysis.csv"),
}

TASK_ORDER = {
    "OSF": ["sleep_efficiency_binary", "apnea_binary", "sex_binary", "age_class", "bmi_binary",
            "osa_binary_apples_postqc", "depression_extreme_binary"],
    "PhysioOmni": ["sleep_efficiency_binary", "sex_binary", "age_class", "bmi_binary",
                   "depression_extreme_binary"],
    "Mantis": ["sleep_efficiency_binary", "apnea_binary", "sex_binary", "age_class", "bmi_binary",
               "osa_binary_apples_postqc", "depression_extreme_binary"],
}


def load(model):
    df = pd.read_csv(SOURCES[model])
    df = df[df["split"] == "test"].copy()
    return df


def get(df, task, head, ctx, k):
    sub = df[(df["task"] == task) & (df["head"] == head) &
             (df["context_length"] == ctx) & (df["k"].astype(str) == str(k))]
    if sub.empty:
        return None, None
    row = sub.iloc[0]
    return row["mean_prob_auroc"], int(row["n_subjects"])


def kmax_for(df, task, head, ctx):
    """Return the largest available numeric k for this cell (its own K_max),
    same definition as the paper's own K_max = floor(T/N) (subject-varying;
    'all' row aggregates each subject's own true max)."""
    sub = df[(df["task"] == task) & (df["head"] == head) & (df["context_length"] == ctx)]
    if sub.empty:
        return None
    allrow = sub[sub["k"].astype(str) == "all"]
    if allrow.empty:
        return None
    n_subj = allrow.iloc[0]["n_subjects"]
    n_seg = allrow.iloc[0]["n_segments"]
    return n_seg / n_subj  # mean windows/subject actually used, for the header


def has_k5(df, task, head, ctx):
    sub = df[(df["task"] == task) & (df["head"] == head) &
             (df["context_length"] == ctx) & (df["k"].astype(str) == "5")]
    return not sub.empty


def lstar(df, task, head):
    """Same definition as npj_main.tex eq. (lstar): smallest L within 0.005
    AUROC of the max, evaluated at K=Kmax ('all') across the 6 contexts."""
    vals = {}
    for ctx in CONTEXTS:
        v, _ = get(df, task, head, ctx, "all")
        if v is not None:
            vals[ctx] = v
    if not vals:
        return None, vals
    peak = max(vals.values())
    for ctx in CONTEXTS:
        if ctx in vals and vals[ctx] >= peak - 0.005:
            return ctx, vals
    return None, vals


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1 style — full sweep, K=1/5/Kmax, Transformer head
# ─────────────────────────────────────────────────────────────────────────────

def make_table1(model):
    df = load(model)
    head = "transformer"
    tasks = TASK_ORDER[model]

    # header K_max-per-context (approx, using sleep_efficiency_binary as a
    # representative large-N task, same convention as npj_main.tex's own
    # Table 1 header, which uses one task's subject-mean K_max per column)
    ref_task = tasks[0]
    kmax_hdr = []
    for ctx in CONTEXTS:
        km = kmax_for(df, ref_task, head, ctx)
        kmax_hdr.append("---" if km is None else f"{km:.0f}" if km >= 1.5 else f"{km:.2g}")

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(f" \\caption{{Full context-length sweep, {model} frozen encoder "
                 "(Transformer, test split). Same format and $L^*$ definition as "
                 "Table~\\ref{tab:sweep}. $K_{\\max}$ at each context shown in "
                 "parentheses (approximate; " + ref_task.replace('_', r'\_') + " used as "
                 "reference). --- denotes no subject reaches $K{=}5$ windows at that "
                 "context. $\\dagger$: small test sets ($N{<}250$).}")
    lines.append(f" \\label{{tab:sweep_{model.lower()}}}")
    lines.append(r" \centering")
    lines.append(r" \footnotesize")
    lines.append(r" \setlength{\tabcolsep}{3pt}")
    lines.append(r" \begin{tabular}{ll@{\hskip8pt}*{6}{>{\centering\arraybackslash}p{32pt}}}")
    lines.append(r"   \toprule")
    lines.append(r"   & & \multicolumn{6}{c}{AUROC at training context length~$L$} \\")
    lines.append(r"   \cmidrule(lr){3-8}")
    lines.append(r"   Task & $K$ & 30~s & 10~min & 40~min & 80~min & 120~min & 240~min \\")
    lines.append(f"   & ($K_{{\\max}}{{\\approx}}$) & " + " & ".join(f"({v})" for v in kmax_hdr) + r" \\")
    lines.append(r"   \midrule")

    n_test = {}
    primary_done = False
    for ti, task in enumerate(tasks):
        if task in SECONDARY and not primary_done:
            lines.append(r"   \midrule")
            primary_done = True
        lstar_ctx, allvals = lstar(df, task, head)
        label = TASK_LABEL[task] + (r"$^\dagger$" if task in SECONDARY else "")
        rows_k = {}
        for k in ("1", "5", "all"):
            vals = []
            for ctx in CONTEXTS:
                v, n = get(df, task, head, ctx, k)
                n_test[task] = n if n is not None else n_test.get(task)
                if k == "5" and not has_k5(df, task, head, ctx):
                    vals.append("---")
                elif v is None:
                    vals.append("---")
                else:
                    bold = (k == "all" and ctx == lstar_ctx)
                    s = f"{v:.3f}"
                    if bold:
                        s = r"\textbf{" + s + "}"
                    vals.append(s)
            rows_k[k] = vals
        lines.append(f"   {label}")
        lines.append(f"     & 1           & " + " & ".join(rows_k["1"]) + r" \\")
        lines.append(f"     & 5           & " + " & ".join(rows_k["5"]) + r" \\")
        lines.append(f"     & $K_{{\\max}}$  & " + " & ".join(rows_k["all"]) + r" \\")
        if ti != len(tasks) - 1:
            lines.append(r"   \addlinespace")
    lines.append(r"   \bottomrule")
    lines.append(r" \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines), n_test


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3 style — head comparison (LSTM vs Transformer; no MeanPool run), K=Kmax
# ─────────────────────────────────────────────────────────────────────────────

def make_table3(model):
    df = load(model)
    tasks = TASK_ORDER[model]

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(f" \\caption{{Head comparison across all context lengths~$L$, {model} "
                 "frozen encoder ($K{=}K_{\\max}$, test split). Same format as "
                 "Table~\\ref{tab:heads}, restricted to LSTM and Transformer: the "
                 "MeanPool head was not run for any TSFM baseline (Methods, "
                 "\\nameref{sec:tsfm_baselines}), so no MeanPool column or "
                 "Transformer-vs-MeanPool advantage can be shown here. "
                 "$L^*$ markers: $^\\ast$ LSTM, $^\\dagger$ Transformer. "
                 "$\\Delta_{TL}$: Transformer$\\,-\\,$LSTM at $K{=}K_{\\max}$.}")
    lines.append(f" \\label{{tab:heads_{model.lower()}}}")
    lines.append(r" \centering")
    lines.append(r" \small")
    lines.append(r" \setlength{\tabcolsep}{2pt}")
    lines.append(r" \begin{tabular}{llccr}")
    lines.append(r"   \toprule")
    lines.append(r"   Task & $L$ & LSTM & Transformer & $\Delta_{TL}$ \\")
    lines.append(r"   \midrule")

    for ti, task in enumerate(tasks):
        lstar_lstm, _ = lstar(df, task, "lstm")
        lstar_tf, _ = lstar(df, task, "transformer")
        label = TASK_LABEL[task] + (r"$^\dagger$" if task in SECONDARY else "")
        lines.append(f"      {label}")
        for ci, ctx in enumerate(CONTEXTS):
            vl, _ = get(df, task, "lstm", ctx, "all")
            vt, _ = get(df, task, "transformer", ctx, "all")
            markers = ""
            if ctx == lstar_lstm:
                markers += r"$^\ast$"
            if ctx == lstar_tf:
                markers += r"$^\dagger$"
            ctx_lbl = CTX_HEAD[ctx] + markers
            vl_s = f"{vl:.3f}" if vl is not None else "---"
            vt_s = f"{vt:.3f}" if vt is not None else "---"
            if vl is not None and vt is not None:
                d = vt - vl
                d_s = f"${'+' if d >= 0 else ''}{d:.3f}$"
            else:
                d_s = "---"
            lines.append(f"      & {ctx_lbl}  & {vl_s} & {vt_s} & {d_s} \\\\")
        if ti != len(tasks) - 1:
            lines.append(r"   \addlinespace")
    lines.append(r"   \bottomrule")
    lines.append(r" \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    for model in ["OSF", "PhysioOmni", "Mantis"]:
        print(f"\n\n{'='*100}\n{model} — TABLE 1 STYLE (full sweep)\n{'='*100}\n")
        tex, n_test = make_table1(model)
        print(tex)
        print("\n% n_test per task (sanity check):", n_test)

    for model in ["OSF", "PhysioOmni", "Mantis"]:
        print(f"\n\n{'='*100}\n{model} — TABLE 3 STYLE (head comparison)\n{'='*100}\n")
        print(make_table3(model))


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-MODEL TABLE — all 5 model variants x all 6 contexts, one per head
# (added 2026-09-17, second request: "all contexts and all models" in one
# table so they're easy to compare row-by-row at a glance)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_SOURCES = [
    ("SleepFM (reduced-ch.)",
     Path("/Users/boshra/NSRR-workspace/NSRR-tools/results/collected/phase0_v3/analysis.csv")),
    ("SleepFM (full-ch.)",
     Path("/Users/boshra/NSRR-workspace/NSRR-tools/results/collected/phase0_v3_full/analysis.csv")),
    ("OSF (full-ch.)", SOURCES["OSF"]),
    ("PhysioOmni (reduced-ch.)", SOURCES["PhysioOmni"]),
    ("Mantis (reduced-ch.)", SOURCES["Mantis"]),
]

ALL_TASKS_ORDER = ["sleep_efficiency_binary", "apnea_binary", "sex_binary", "age_class",
                   "bmi_binary", "osa_binary_apples_postqc", "depression_extreme_binary"]


def load_generic(path):
    df = pd.read_csv(path)
    return df[df["split"] == "test"].copy()


def make_cross_model_table(head):
    dfs = {name: load_generic(path) for name, path in MODEL_SOURCES}

    lines = []
    lines.append(r"\begin{table*}[!t]")
    head_label = "LSTM" if head == "lstm" else "Transformer"
    lines.append(f" \\caption{{Cross-model comparison, {head_label} head, $K{{=}}K_{{\\max}}$, "
                 "test split. All four encoders (SleepFM shown at both channel "
                 "configurations; OSF ran only on full-channel, PhysioOmni and Mantis "
                 "only on reduced-channel, Methods, \\nameref{sec:tsfm_baselines}) at "
                 "every context length, for direct row-by-row comparison. \\textbf{Bold}: "
                 "each row's own saturation context $L^*$ (same 0.005-AUROC-of-peak "
                 "definition as Table~\\ref{tab:sweep}, computed independently per "
                 "model/task/head). $\\Delta$: AUROC at $L^*$ minus AUROC at 30~s (same "
                 "convention as Table~\\ref{tab:saturation}; not simply the global max "
                 "minus 30~s, which can differ slightly when the true peak lies just "
                 "beyond $L^*$ but within its 0.005 tolerance). ---: task not run for "
                 "that encoder (PhysioOmni has no "
                 "respiratory pathway, so no apnea row; osa\\_binary\\_apples\\_postqc is "
                 "OSF/Mantis only). $\\dagger$: small test sets ($N{<}250$). No "
                 "bootstrap confidence intervals exist for OSF/PhysioOmni/Mantis cells "
                 "yet (Methods)." +
                 (" See Supplementary~Table~S-XX for the LSTM counterpart."
                  if head == "transformer" else "") +
                 "}")
    lines.append(f" \\label{{tab:crossmodel_{head}}}")
    lines.append(r" \centering")
    lines.append(r" \resizebox{\linewidth}{!}{%")
    lines.append(r" \footnotesize")
    lines.append(r" \setlength{\tabcolsep}{4pt}")
    lines.append(r" \begin{tabular}{ll*{6}{c}r}")
    lines.append(r"   \toprule")
    lines.append(r"   Task & Model & 30~s & 10~min & 40~min & 80~min & 120~min & 240~min & $\Delta$ \\")
    lines.append(r"   \midrule")

    primary_done = False
    for ti, task in enumerate(ALL_TASKS_ORDER):
        if task in SECONDARY and not primary_done:
            lines.append(r"   \midrule")
            primary_done = True
        label = TASK_LABEL[task] + (r"$^\dagger$" if task in SECONDARY else "")
        lines.append(f"   {label}")
        for mi, (mname, _) in enumerate(MODEL_SOURCES):
            df = dfs[mname]
            lstar_ctx, vals = lstar(df, task, head)
            cells = []
            for ctx in CONTEXTS:
                v, n = get(df, task, head, ctx, "all")
                if v is None:
                    cells.append("---")
                else:
                    s = f"{v:.3f}"
                    if ctx == lstar_ctx:
                        s = r"\textbf{" + s + "}"
                    cells.append(s)
            if vals and "30s" in vals and lstar_ctx is not None:
                # Same convention as npj_main.tex Table~2 (tab:saturation):
                # Delta uses the value AT L*, not the global max across all
                # contexts -- these differ whenever the true peak lies beyond
                # L* but within the 0.005 saturation tolerance (e.g. SleepFM
                # age_class: true max is 240m/0.905, but L*=120m/0.902 is
                # already "within 0.005 of peak" and is the reported point,
                # so Delta = 0.902-0.854 = +0.048, not 0.905-0.854 = +0.051).
                d = vals[lstar_ctx] - vals["30s"]
                delta_s = f"${'+' if d >= 0 else ''}{d:.3f}$"
            else:
                delta_s = "---"
            lines.append(f"     & {mname}  & " + " & ".join(cells) + f" & {delta_s} \\\\")
        if ti != len(ALL_TASKS_ORDER) - 1:
            lines.append(r"   \addlinespace")
    lines.append(r"   \bottomrule")
    lines.append(r" \end{tabular}%")
    lines.append(r" }% end resizebox")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


if __name__ == "__main__" and "--cross-model" in __import__("sys").argv:
    for head in ["transformer", "lstm"]:
        print(f"\n\n{'='*100}\nCROSS-MODEL TABLE — {head.upper()} HEAD\n{'='*100}\n")
        print(make_cross_model_table(head))
