"""TBME figure style constants and helpers."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── TBME layout constants ─────────────────────────────────────────────────────
FULL_W  = 7.0    # double-column width (inches)
HALF_W  = 3.5    # single-column width (inches)
DPI_OUT = 300    # output DPI (PDF is vector; PNG preview uses this)

# ── Typography ─────────────────────────────────────────────────────────────────
# FONT_BASE  = 8    # axes tick/label baseline
# FONT_TITLE = 8    # panel title
# FONT_ANNOT = 6    # small annotations
# FONT_LABEL = 7    # x/y axis labels

FONT_BASE  = 11    # axes tick/label baseline
FONT_TITLE = 11    # panel title
FONT_ANNOT = 11    # small annotations
FONT_LABEL = 11    # x/y axis labels

# ── Head colour/marker scheme ─────────────────────────────────────────────────
HEAD_STYLE = {
    "lstm":        {"color": "#3A7EBF", "marker": "o", "ls": "-",  "label": "LSTM"},
    "transformer": {"color": "#E86A33", "marker": "s", "ls": "--", "label": "Transformer"},
    "mean_pool":   {"color": "#44A15E", "marker": "^", "ls": ":",  "label": "MeanPool"},
}

# ── Task display names ────────────────────────────────────────────────────────
TASK_LABEL = {
    "sex_binary":                  "Sex",
    "age_class":                   "Age",
    "apnea_binary":                "Apnea (AHI≥15)",
    "bmi_binary":                  "BMI",
    "sleep_efficiency_binary":     "Sleep Efficiency",
    "depression_extreme_binary":   "Depression",
    "osa_binary_apples_postqc":    "OSA (APPLES)",
    # "cvd_binary":                  "CVD",
}

TASK_LABEL_LONG = {
    "sex_binary":                  "Sex (binary)",
    "age_class":                   "Age (3-class)",
    "apnea_binary":                "Sleep apnea (AHI≥15)",
    "bmi_binary":                  "BMI (obese)",
    "sleep_efficiency_binary":     "Sleep efficiency",
    "depression_extreme_binary":   "Depression (extreme)",
    "osa_binary_apples_postqc":    "OSA (APPLES only)",
    # "cvd_binary":                  "CVD",
}

# ── Task lists ────────────────────────────────────────────────────────────────
MAIN_TASKS = ["sex_binary", "bmi_binary", "age_class",
              "sleep_efficiency_binary", "apnea_binary"]

SUPP_TASKS = ["depression_extreme_binary", "osa_binary_apples_postqc"] #, "cvd_binary"]

ALL_TASKS  = MAIN_TASKS + SUPP_TASKS

BINARY_MAIN = [t for t in MAIN_TASKS if t != "age_class"]

# ── Context mapping ───────────────────────────────────────────────────────────
CONTEXT_TO_MIN = {
    "30s": 0.5, "10m": 10.0, "40m": 40.0,
    "80m": 80.0, "120m": 120.0, "240m": 240.0,
}
CTX_ORDER = list(CONTEXT_TO_MIN.keys())
MIN_TO_CTX = {v: k for k, v in CONTEXT_TO_MIN.items()}

# ── Ablation conditions ───────────────────────────────────────────────────────
ABL_CONDITIONS = [
    ("abl_no_bas",   "No BAS",     "#E86A33"),
    ("abl_no_resp",  "No RESP",    "#3A7EBF"),
    ("abl_no_ekg",   "No EKG",     "#44A15E"),
    ("abl_cardio",   "Cardio only","#C94040"),
    ("abl_bas_only", "BAS only",   "#7B5EA7"),
]

# ── Ablation context per task ─────────────────────────────────────────────────
ABL_CONTEXT = {
    "sex_binary":              "120m",
    "apnea_binary":            "120m",
    "sleep_efficiency_binary": "120m",
    "age_class":               "120m",
    "bmi_binary":              "40m",
}

# ── Cohort colour scheme (per-dataset breakdowns) ─────────────────────────────
COHORT_COLOR = {
    "apples": "#3A7EBF", "stages": "#E86A33", "mros": "#44A15E", "shhs": "#7B5EA7",
}

# ── Number of classes per task (for multi-class AUROC aggregation) ───────────
NUM_CLASSES = {
    "age_class":                   3,
    "apnea_binary":                2,
    "sleep_efficiency_binary":     2,
    "bmi_binary":                  2,
    "sex_binary":                  2,
    "depression_extreme_binary":   2,
    "osa_binary_apples_postqc":    2,
}


def apply_tbme_style():
    """Apply global rcParams for TBME-style figures."""
    plt.rcParams.update({
        "figure.dpi":          DPI_OUT,
        "savefig.bbox":        "tight",
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "font.family":         "serif",
        "font.size":           FONT_BASE,
        "axes.labelsize":      FONT_LABEL,
        "xtick.labelsize":     FONT_BASE,
        "ytick.labelsize":     FONT_BASE,
        "legend.fontsize":     FONT_BASE,
        "axes.titlesize":      FONT_TITLE,
    })


def log_ctx_axis(ax, xs=None):
    """Set log-scale x-axis with context-length tick labels."""
    ax.set_xscale("log")
    ticks = xs if xs is not None else list(CONTEXT_TO_MIN.values())
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [MIN_TO_CTX.get(float(x), f"{x:.0f}m") for x in ticks],
        fontsize=FONT_BASE,
    )
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def save_figure(fig, out_path, stem):
    """Save fig as PDF (and PNG) to out_path/stem.{pdf,png}."""
    from pathlib import Path
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(str(out_path / f"{stem}.{ext}"),
                    dpi=DPI_OUT, bbox_inches="tight")
    print(f"  saved → {out_path}/{stem}.pdf")
