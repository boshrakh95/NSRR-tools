# TABLES_PLAN.md — Comprehensive Table Redesign
*Agreed: 2026-07-04. Implements the restructured table set for TBME submission.*
*See results.md → [RENAME NOTE] and methods.md → [RENAME NOTE] for the training-w rename.*

---

## Design principles

1. **Each hypothesis has at least one dedicated table or dedicated columns** — H1, H2, H3, H4 all visible in the main paper tables.
2. **AUROC@30s is always shown** so the reader never has to back-calculate the baseline.
3. **K unambiguously means inference-time aggregation count** throughout all tables. Training windows are described as w=5 in Methods, never K.
4. **No table repeats another**; supplementary tables add at least one new dimension (more tasks, more K values, bootstrap CIs, per-cohort breakdown).
5. **K_max at long contexts is small** (≈2–6 windows at 80–240m). Tables that show K=5 and K=K_max will appear nearly identical at these contexts, and a footnote or column header explains this rather than hiding it.

---

## Parameter notation (used consistently in all tables)

| Symbol | Meaning |
|--------|---------|
| $L$ | Training context length (minutes) |
| $L^*$ | Saturation threshold: smallest $L$ within 0.005 AUROC of peak |
| $K$ | Inference-time aggregation count (windows averaged per subject) |
| $K_{\max}$ | Maximum $K$ available at context $L$ = $\lfloor T/N \rfloor$ |
| $w$ | Training windows per subject per epoch (fixed at 5; *not* $K$) |
| $\Delta$ | AUROC gain from 30s,K=1 baseline to peak (best L, K=K_max) |

---

## Main paper tables

### Table I — Task Definitions (KEEP, minor caption update)
*Current: tab:tasks. No structural change needed.*

**Caption update:** Add footnote clarifying that $K$ throughout the paper refers to inference-time aggregation count, distinct from the training window count $w=5$ (Section~III-F).

---

### Table II — Saturation and Aggregation (NEW unified table; replaces current Tables II+III)
*Answers H1 (context saturation) and H3 (aggregation saturation) together.*
*Label: tab:saturation*

**What it shows:** For LSTM head, each row is one task. Columns show:
- AUROC@30s, K=1 — the absolute worst-case baseline (single 30s window, no aggregation)
- L\* — saturation threshold
- AUROC@L\*, K=1 — single-window deployment at the recommended context
- AUROC@L\*, K=5 — 5-window deployment (practical scenario)
- AUROC@L\*, K=K_max — full-night ceiling at L\*
- Δ — peak AUROC minus AUROC@30s,K=1 (total gain)

**Why this works:** A reader can read across one row to see:
- Column 1→3: effect of training context alone (K=1 at both ends)
- Column 3→4→5: effect of inference aggregation at fixed L\*
- The K=5≈K_max near-equality at long contexts is *visible and expected*: K_max≈2–6 at 120–240m

**LaTeX structure:** `table*` (full width, two columns).

**Numbers (LSTM, fast-channel, test split):**

| Task | N_test | AUROC@30s,K=1 | L\* | AUROC@L\*,K=1 | AUROC@L\*,K=5 | AUROC@L\*,K_max | K_max@L\* | Δ |
|------|--------|---------------|-----|----------------|----------------|-----------------|----------|---|
| Sex | 1430 | 0.687 | 120m | 0.807 | 0.872 | 0.872 | 4 | +0.047 |
| Age group | 1859 | 0.798 | 80m | 0.843 | 0.890 | 0.890 | 6 | +0.028 |
| Apnea | 2054 | 0.614 | 120m | 0.727 | 0.831 | 0.832 | 4 | +0.074 |
| BMI | 1856 | 0.676 | 10m | 0.678 | 0.745 | 0.762 | 56 | +0.006 |
| Sleep eff. | 2023 | 0.649 | 240m | 0.737 | 0.788 | 0.788 | 2 | +0.091 |
| Depression† | 229 | 0.777 | 10m | 0.788 | 0.776 | 0.770 | 50 | +0.013 |
| OSA-APPLES† | 161 | 0.555 | 40m | 0.641 | 0.792 | 0.834 | 12 | +0.064 |

*†Supplementary tasks: small test set, interpret cautiously.*

**Caption template:**
> Saturation and aggregation results (LSTM, fast-channel, test split). Each row shows the
> progression from the single-window 30s baseline (first context, K=1) to the saturation point
> L\* (K=1, 5, K_max). K denotes the \emph{inference-time} aggregation count — number of
> non-overlapping windows averaged per subject; training used w=5 overlapping windows per
> subject per epoch (Section~III-F). K_max = floor(T/N) is small at long contexts (≈2–4
> at L=120–240m), explaining why K=5 ≈ K_max there. Δ = peak AUROC − AUROC@30s,K=1.
> Tasks marked † have small test sets (N<250); treat with caution.

**Story the reader extracts:**
- Going from 30s,K=1 to L\*,K=1: pure effect of training context (H1 — large for apnea, sleep eff.)
- Going from L\*,K=1 to L\*,K=5: aggregation gain at the right context (H3 — large for BMI/OSA at short L\*, negligible at long L\*)
- Going from L\*,K=5 to L\*,K_max: additional gain beyond 5 windows (negligible at long L\*)

---

### Table III — L\* and Context Gain per Head (replaces current Table III/tab:lstar)
*Answers H1 for LSTM and Transformer side-by-side with explicit AUROC@30s.*
*Label: tab:lstar*

**What it shows:** For each task and both heads, L\* and the absolute AUROC gain from baseline to best context.

**Columns:** Task | LSTM L\* | LSTM AUROC@30s | LSTM best AUROC | LSTM Δ | Transformer L\* | Transformer AUROC@30s | Transformer best AUROC | Transformer Δ

This is wide — consider two sub-tables or compressing. A compact version:

| Task | LSTM L\* | LSTM Δ | Transformer L\* | Transformer Δ |
|------|----------|--------|-----------------|----------------|
| Sleep efficiency | 240m | +0.091 | 240m | +0.124 |
| Apnea | 120m | +0.074 | 120m | +0.103 |
| OSA-APPLES† | 40m | +0.064 | 80m | +0.098 |
| Sex | 120m | +0.047 | 240m | +0.079 |
| Age group | 80m | +0.028 | 120m | +0.051 |
| Depression† | 10m | +0.013 | 30s | +0.000 |
| BMI | 10m | +0.006 | 240m | +0.030 |

**Caption template:**
> Saturation context L\* and AUROC gain from 30-second baseline to best context (K=K_max).
> Δ = best AUROC − AUROC@30s. Tasks sorted by LSTM Δ descending. Transformer Δ generally
> exceeds LSTM Δ; both heads agree on task ordering. Tasks marked † have small test sets.

**Note:** The Δ here is AUROC_peak − AUROC@30s (total gain, "peak gain" definition —
consistent with supplementary sensitivity table and the IV-F/IV-G text). This is distinct
from gain-to-L\* used in Table II.

---

### Table IV — Head Comparison at LSTM L\* (simplify current Table IV/tab:heads)
*Answers H4: temporal heads vs MeanPool.*
*Label: tab:heads*

**Columns:** Task | L\* | LSTM | Transformer | MeanPool | Temporal adv.

All values at K=K_max (K_max at LSTM's L\*). Caption must note K_max is small at long L\*.

| Task | L\* | LSTM | Transformer | MeanPool | Temp. adv. |
|------|-----|------|-------------|----------|------------|
| Sex | 120m | 0.872 | 0.905 | 0.815 | +0.057 |
| Age group | 80m | 0.890 | 0.900 | 0.843 | +0.047 |
| Apnea | 120m | 0.832 | 0.857 | 0.764 | +0.068 |
| BMI | 10m | 0.762 | 0.755 | 0.734 | +0.028 |
| Sleep eff. | 240m | 0.788 | 0.831 | 0.760 | +0.028 |

All values from table5_heads_fast.csv (K=all at LSTM L\*). Temporal adv. = LSTM − MeanPool.

---

### Table V — Modality Ablation (KEEP current Table V/tab:modality)
*No structural changes needed. Numbers verified correct.*
*Only change: add the comment about sleep efficiency using L=120m ≠ L\*=240m (already done as LaTeX comment).*

---

## Supplementary tables

### Supp Table S-I — Excluded Subjects (KEEP current tab:supp-excluded)

### Supp Table S-II — Per-Task Subject Counts (KEEP current tab:supp-task-n)

### Supp Table S-III — Bandpass Filter Parameters (KEEP current tab:supp-filters)

### Supp Table S-IV — K-Grid for Sex Classification (KEEP current tab:supp-kgrid)
*This is the detailed K×L aggregation table. Already correct.*

**Optionally extend:** Add a second row block for one more task (e.g., apnea or sleep eff.) to show the contrast between a task with high vs low context sensitivity.

### Supp Table S-V — Cross-Task Context Sensitivity (UPDATE to add AUROC@30s explicitly)
*Current shows: AUROC@30s | Best AUROC | Δ | L\**
*Keep as-is — the "Best AUROC" column is peak AUROC (same Δ convention as Table III in main).*

### Supp Table S-VI — Extended Performance with Bootstrap CIs (KEEP current tab:supp-ci)

### Supp Table S-VII — Per-Cohort AUROC Breakdown (KEEP current tab:supp-cohort)

### Supp Table S-VIII — Post-Hoc Threshold Tuning (KEEP current tab:supp-threshold)

### Supp Table S-IX (NEW) — Aggregation Saturation Summary, All Tasks
*Currently only sex classification is shown (S-IV). Extend to cover all 5 main tasks.*

**What it shows:** For each of the 5 main tasks, AUROC at K=1, K=5, K=10, K_max at L\*.
Illustrates H3 (aggregation saturation) across tasks consistently.

| Task | L\* | K_max | K=1 | K=5 | K=10 | K=K_max |
|------|-----|-------|-----|-----|------|---------|
| Sex (LSTM) | 120m | 4 | 0.807 | 0.872 | — | 0.872 |
| Age (LSTM) | 80m | 6 | 0.843 | 0.890 | 0.890 | 0.890 |
| Apnea (LSTM) | 120m | 4 | 0.727 | 0.831 | — | 0.832 |
| BMI (LSTM) | 10m | 56 | 0.678 | 0.745 | 0.762 | 0.762 |
| Sleep eff. (LSTM) | 240m | 2 | 0.737 | 0.788† | — | 0.788 |

*†At L=240m K_max=2, so K=5 is not available; the value shown is K=K_max=2.*

Footnote: K=10 shown with "—" where K_max < 10 for that context.

This table directly supports the claim "K=5 recovers >99% of K_max AUROC at L≥40m" with numbers.

---

## Cross-reference between hypotheses and table locations

| Hypothesis | Primary evidence | Table | Section |
|------------|-----------------|-------|---------|
| H1 Context saturation | AUROC rises with L, reaches L\* | Table II (K=1 col), Table III (Δ) | IV-A |
| H2 Iso-compute (rejected) | Longer L wins at equal L×K | main_fig3 (heatmap) | IV-D |
| H3 Aggregation saturates | K=5 ≈ K_max at L≥40m | Table II (K=5 vs K_max cols), Supp S-IV, Supp S-IX | IV-C |
| H4 Temporal heads | LSTM/Transformer > MeanPool | Table IV (tab:heads) | IV-B |
| Ablation | Modality contributions | Table V (tab:modality) | IV-I |

---

## Implementation checklist

- [x] methods.md: training-K rename note added
- [x] results.md: rename note + table redesign reference added
- [x] generic-color.tex Methods: w=5 introduced, defined clearly
- [x] generic-color.tex Table I caption: updated to use w=5
- [ ] generic-color.tex: implement new Table II (unified saturation+aggregation)
- [ ] generic-color.tex: update Table III (tab:lstar) caption to reflect "peak gain" definition
- [ ] generic-color.tex: Table IV (tab:heads) — verify age transformer 0.900 already corrected ✓
- [ ] generic-color.tex: add cross-reference sentences to results sections pointing to new Table II
- [ ] supplementary.tex: add Supp Table S-IX (aggregation saturation, all tasks)
- [ ] supplementary.tex: update S-IV caption to note K=5 unavailable at long L
- [ ] Final review: all numbers cross-checked against collected CSV

---

## Numbers source reference

All numbers from:
- `results/collected/phase0_v3/analysis.csv` (fast-channel LSTM/Transformer/MeanPool)
- `results/tables/table5_heads_fast.csv` (Table IV values)
- `results/tables/table6_modality.csv` (Table V values)
- `results/tables/table2_lstar_fast.csv` (Table III Δ values)

Do NOT use numbers from supplementary figures directly — always verify against CSV.
