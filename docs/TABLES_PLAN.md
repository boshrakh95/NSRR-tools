# TABLES_PLAN.md — Comprehensive Table Redesign
*Agreed: 2026-07-04. Revised: 2026-07-06 (full restructure implemented).*
*See results.md → [RENAME NOTE] and methods.md → [RENAME NOTE] for the training-w rename.*

---

## Design principles

1. **Each hypothesis has at least one dedicated table or dedicated columns** — H1, H2, H3, H4 all visible in the main paper tables.
2. **AUROC@30s is always shown** so the reader never has to back-calculate the baseline.
3. **K unambiguously means inference-time aggregation count** throughout all tables. Training windows are described as w=5 in Methods, never K.
4. **No table repeats another**; supplementary tables add at least one new dimension (more tasks, more K values, bootstrap CIs, per-cohort breakdown).
5. **K_max at long contexts is small** (≈2–6 windows at 80–240m). Tables that show K=5 and K=K_max will appear nearly identical at these contexts, and a footnote or column header explains this rather than hiding it.
6. **K_max shown explicitly** in each table where relevant (column header or explicit K_max column).

---

## Parameter notation (used consistently in all tables)

| Symbol | Meaning |
|--------|---------|
| $L$ | Training context length (minutes) |
| $L^*$ | Saturation threshold: smallest $L$ within 0.005 AUROC of peak (defined at K=K_max) |
| $K$ | Inference-time aggregation count (windows averaged per subject) |
| $K_{\max}$ | Maximum $K$ available at context $L$ = $\lfloor T/N \rfloor$ |
| $w$ | Training windows per subject per epoch (fixed at 5; *not* $K$) |
| $\Delta$ | AUROC gain (definition varies by table — see each caption) |

---

## Main paper tables (7 total)

### Table I — Task Definitions (UNCHANGED)
*Label: tab:tasks. No structural change needed.*

---

### Table II — Full Context-Length Sweep (NEW)
*Answers H1+H3 together by showing the full 6L × 3K matrix.*
*Label: tab:sweep*

**What it shows:** LSTM, all 7 tasks, all 6 context lengths, K=1 / K=5 / K=K_max.
**Bold entries** mark L* per task (L* defined at K=K_max; same bold across all 3 K-rows).
**K_max ≈ (1062, 53, 13, 6, 4, 2)** shown in parentheses in the context header row.
"---" entries: K=5 unavailable at 240m for most tasks (K_max≈2).

**Columns:** Task | K | 30s | 10m | 40m | 80m | 120m | 240m

**Key numbers (from analysis.csv):**
- Sex (L*=120m): K=1: 0.687, 0.724, 0.750, 0.785, **0.807**, 0.844
- Age (L*=80m): K=all: 0.865, 0.870, 0.887, **0.890**, 0.893, 0.885
- Apnea (L*=120m): K=1: 0.614, 0.635, 0.645, 0.688, **0.727**, 0.787
- BMI (L*=10m): K=all: 0.760, **0.762**, 0.756, 0.767, 0.756, 0.748
- Sleep eff. (L*=240m): K=all: 0.697, 0.717, 0.731, 0.759, 0.778, **0.788**

---

### Table III — Saturation Context Comparison (REPLACES OLD TABLE II)
*Answers H1 (context gain) and H3 (aggregation gain) in one compact table.*
*Label: tab:saturation*

**What it shows:** For each task, AUROC at 30s (K=1, K=5, K=all) AND at L* (K=1, K=5, K=all).
L* and K_max@L* are explicit columns.
Δ = AUROC@L*,K=K_max − AUROC@30s,K=1 (total gain from worst to ceiling at L*).

**Column spec:** Task | N_test | @30s K=1,5,all | L* | K_max@L* | @L* K=1,5,all | Δ

**Why better than old Table II:** 
- Old table only showed @30s K=1 and @L* K=1,5,all — incomplete 30s picture
- L* was right next to @30s column, confusing readers about which baseline it belongs to
- New table clearly separates the two blocks with a vertical line and explicit block headers
- K_max@L* shown explicitly so reader understands why K=5≈K_max at long L*

**Δ definition note:** Different from Table IV Δ (which uses K=K_max at both ends for a pure context comparison). Table III Δ shows total gain including aggregation benefit.

---

### Table IV — L* and Context Gain per Head (UNCHANGED from old Table III)
*Label: tab:lstar*

**Δ here** = best AUROC(K=K_max) − AUROC@30s(K=K_max) — pure context gain with K fixed.
Tasks sorted by LSTM Δ descending. Transformer Δ > LSTM Δ for tasks with L*≥80m.

---

### Table V — Head Comparison Across All Contexts (REPLACES OLD TABLE IV)
*Answers H4 fully, showing how temporal advantage grows with L.*
*Label: tab:heads*

**What it shows:** LSTM / Transformer / MeanPool at K=K_max, all 6 context lengths per task.
Trans. adv. = Transformer − MeanPool (positive = temporal integration helps).
**Bold rows** mark L* per task.

**Column spec:** Task | L | LSTM | Transformer | MeanPool | Trans. adv.

**Key findings visible in table:**
- Trans. adv. grows with L for apnea (0.028@30s → 0.093@120m) and sex (0.052 → 0.090)
- BMI: flat Trans. adv. (0.018–0.031) across all L → no temporal structure
- Age at 30s: LSTM (0.865) > Transformer (0.854) — only task/context where LSTM leads
- Sleep eff.: Trans. adv. rises from 0.013@30s to 0.071@240m (L*)

**Change from old Table IV:**
- Was single-row per task at L* only; now 6 rows per task (all contexts)
- Temp. adv. was LSTM−MeanPool; now Transformer−MeanPool (more informative for H4)
- Made table* (full width) for readability of 30-row table

---

### Table VI — Iso-Compute Comparison, 5 Tasks (REPLACES OLD TABLE V)
*Answers H2 with full task coverage.*
*Label: tab:isocompute*

**What it shows:** Best AUROC and optimal (L, K) at each budget for 5 tasks.
Uses compact "Lm/K=N" notation. Table* (full width). Uses \footnotesize for fit.

**Tasks:** Sex, Apnea, Sleep efficiency, Age group, BMI
**Budgets:** 40, 80, 120, 240, 480 min

**Key correction from old Table V:**
- Apnea B=240: was (120m, K=2)→0.816; corrected to (80m, K=3)→0.822 (actual best)

**Key new findings:**
- Sleep efficiency at 40-80m: best is (30s, K=80/160)→0.707 = K_max ceiling → short context useless for this task
- BMI: near-flat across all budgets (0.747–0.775) → context-insensitive
- Age at B=120m: single 120m window (120m/K=1→0.874) matches 10m×12 windows → context efficient

---

### Table VII — Modality Group Ablation (UNCHANGED from old Table VI)
*Label: tab:modality. Numbers verified correct.*
*Note: Sleep efficiency evaluated at L=120m (not L*=240m) to preserve inference windows.*

---

## Supplementary tables (all retained)

All supplementary table labels and content unchanged. Updated references:
- "main paper Table~V" → "main paper Table~VII" (3 locations in supplementary.tex)
- Supp Table S-IX (aggregation saturation, all tasks): still pending

---

## Cross-reference between hypotheses and table locations

| Hypothesis | Primary evidence | Table | Section |
|------------|-----------------|-------|---------|
| H1 Context saturation | Full sweep + L* per task | Table II (sweep) + III (saturation) + IV (lstar Δ) | IV-A |
| H2 Iso-compute (partial reject) | Best (L,K) per budget, 5 tasks | Table VI (isocompute) + Fig.3 heatmap | IV-D |
| H3 Aggregation saturates | K=1→5→all at L* | Table III (saturation) + Supp S-IX | IV-C |
| H4 Temporal heads | Trans adv. grows with L | Table V (heads, all contexts) | IV-B |
| Ablation | Modality contributions | Table VII (modality) | IV-I |

---

## Implementation checklist

- [x] generic-color.tex: Table I caption — no change needed
- [x] generic-color.tex: Table II (tab:sweep) — NEW full sweep table added
- [x] generic-color.tex: Table III (tab:saturation) — new 30s+L* comparison
- [x] generic-color.tex: Table IV (tab:lstar) — unchanged
- [x] generic-color.tex: Table V (tab:heads) — all contexts, Trans-MP temp. adv.
- [x] generic-color.tex: Table VI (tab:isocompute) — 5 tasks, table*, apnea B=240 corrected
- [x] generic-color.tex: Table VII (tab:modality) — unchanged content, auto-renumbered
- [x] generic-color.tex Section IV-A text — references new Tables II+III+IV
- [x] generic-color.tex Section IV-B text — updated for expanded head comparison
- [x] generic-color.tex Section IV-D text — updated for 5-task iso-compute
- [x] supplementary.tex — Table~V → Table~VII (3 locations)
- [ ] supplementary.tex: add Supp Table S-IX (aggregation saturation, all tasks) — PENDING
- [ ] Final review: compile both documents, check all numbers vs CSV

---

## Numbers source reference

All numbers from:
- `results/collected/phase0_v3/analysis.csv` (fast-channel LSTM/Transformer/MeanPool)
- `results/tables/table5_heads_fast.csv` (secondary check for Table V values)
- `results/tables/table2_lstar_fast.csv` (secondary check for Table IV Δ values)
- `results/tables/table6_modality.csv` (Table VII values)

Do NOT use numbers from supplementary figures directly — always verify against CSV.
