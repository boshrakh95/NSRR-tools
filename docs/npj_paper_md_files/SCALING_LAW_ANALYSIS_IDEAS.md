# Scaling-Law Analysis Ideas — Candidates for the Paper

Written 2026-07-28. Two questions were asked: (1) is anything in the old
`plot_scaling_laws.py` figure set (`results/figures/phase0_v3/scaling_laws/`)
worth pulling into the paper, and (2) what *new*, cheap-to-run analyses on the
existing `analysis.csv` / `training.csv` / prediction parquets would strengthen
the "context efficiency" framing, particularly anything in the spirit of the
LLM-scaling-law practice of fitting small/cheap runs and using the fit to
predict expensive ones.

---

## Part 1 — Existing `scaling_laws/` figures: verdict

Looked at 1A (train/val balanced-accuracy curves per task×head×context,
21 files), 1B (FLOPs vs AUROC scatter per task, 8 files — superseded, see
below), and 1C (best-epoch bar chart, 1 file only, explicitly blacklisted in
`EXPERIMENTS_GUIDE.md`).

**1A (uShape curves).** Checked `sex_binary` (large train/val gap, ~0.15–0.19
BA, at every context length) against `sleep_efficiency_binary` (near-zero gap
at short context, growing only slightly at long context). The overfitting gap
is real and task-dependent, but it doesn't form a clean monotonic trend against
context length in either task (best-epoch index is noisy: 8, 11, 9, 16, 20, 14
for sex; 27, 18, 17, 8, 11, 6 for sleep efficiency — opposite-looking trends
between two tasks, most likely early-stopping-patience noise rather than a
real effect). **Verdict: not worth adding.** It's a training-diagnostic
confirming the early-stopping methodology is doing its job, not a new
scientific finding, and a clean cross-task pattern isn't there.

**1B (FLOPs vs AUROC per task).** This is exactly what Extended Data Fig. 2 /
S-Fig 19 already do, pooled across tasks, with the corrected FLOPs formula.
The old per-task 1B files are strictly superseded (and were generated before
the FLOPs bug fix, so numerically stale too). **Verdict: superseded, ignore.**

**1C (best-epoch bar).** Only one file exists; the guide explicitly
blacklisted it. **Verdict: correctly dropped, nothing to recover.**

**Bottom line for Part 1:** nothing in the existing scaling-law figure set is
worth pulling into the paper as-is. The ideas below are new analyses, not
salvage of old figures.

---

## Part 2 — Literature grounding

The paper's own iso-compute analysis (main Fig. 4/5, Table 4) is already a
domain-specific analog of Chinchilla-style compute-optimal analysis, without
being framed that way. A few recent papers make the connection worth drawing
out explicitly and suggest concrete new analyses:

- **Hoffmann et al., "Training Compute-Optimal Large Language Models"
  (Chinchilla)** — fits parabolas to IsoFLOP slices to find the loss-optimal
  model size at each compute budget, then fits a *power law* between compute
  and the optimal allocation (`N_opt ∝ C^0.49`, `D_opt ∝ C^0.51`), and
  validates the extrapolation by training a new model at a much larger budget
  than any point used for the fit. This is the canonical "small-scale fit
  predicts large-scale behavior" result the user is thinking of.
  ([NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf))

- **Montgomery et al., "Predicting Task Performance with Context-aware
  Scaling Laws" (2025)** — fits a single joint function of *training compute
  and context length* to downstream task accuracy for Llama-2 variants, and
  shows it extrapolates across compute *and* generalizes to unseen amounts of
  context. This is the closest existing precedent to what Idea E below
  proposes, just applied to a clinical PSG encoder instead of an LLM.
  ([arXiv:2510.14919](https://arxiv.org/abs/2510.14919),
  [code](https://github.com/wang-research-lab/context-scaling))

- **Yang et al./others, "Scaling Laws for Downstream Task Performance of
  LLMs" (2402.04177)** — shows downstream-metric scaling can be smooth
  (log-law) or non-monotonic depending on distribution alignment between
  pretraining and downstream data; relevant caveat for framing any
  extrapolation claims we make. ([arXiv:2402.04177](https://arxiv.org/abs/2402.04177))

- **Time-series / EHR / EEG foundation-model scaling papers** — direct
  precedent that "scaling laws" is an active, citable framing in exactly our
  kind of domain, not just LLMs:
  - "Towards Neural Scaling Laws for Time Series Foundation Models"
    ([arXiv:2410.12360](https://arxiv.org/abs/2410.12360))
  - "Scaling-laws for Large Time-series Models"
    ([arXiv:2405.13867](https://arxiv.org/pdf/2405.13867))
  - "Exploring Scaling Laws for EHR Foundation Models" — power-law relations
    between compute/data/params and clinical utility, same functional form
    as LLM scaling laws ([arXiv:2505.22964](https://arxiv.org/pdf/2505.22964))
  - EEG foundation model scaling work confirms power-law data scaling with
    diminishing but still-positive returns even at large data volumes
    (relevant to the "no ceiling observed" framing already in our Discussion)

- **On fitting small/noisy data properly**: "rectified" scaling laws add an
  irreducible-error asymptote term (`AUROC ≈ ceiling − A·x^(−b) − floor`, or
  similar) rather than a bare power law, because omitting it "can easily
  underestimate scaling exponents by over 2×" on noisy/small data — directly
  relevant to our small-*N* tasks (OSA, depression) whose curves are
  currently just described qualitatively as "noisy." See Lilian Weng's
  survey (["Scaling Laws, Carefully"](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/))
  and the ICLR 2025 survey ["(Mis)fitting: A Survey of Scaling Laws"](https://proceedings.iclr.cc/paper_files/paper/2025/file/7f77492bb8070a5c825a87c0c5181da2-Paper-Conference.pdf).

---

## Part 3 — Candidate new analyses (existing data only, no retraining)

Checked feasibility directly against the actual files. Prediction parquets
already carry a `dataset` column (cohort identity per subject), confirmed via:

```
apnea_binary (240m, test):  apples=168  mros=408  shhs=1278  stages=200
sleep_efficiency (240m, test): apples=168  mros=571  shhs=1281
bmi_binary (240m, test):    apples=167  mros=435  shhs=1254
age_class (240m, test):     apples=165  mros=439  shhs=1255
sex_binary (240m, test):    apples + shhs only (mros/stages not used for this task)
osa (240m, test):           apples=161 only (single-cohort — not usable for Idea A)
depression (240m, test):    apples=130  stages=99 (comparable sizes, not small-vs-large)
```

### Idea A — Cross-cohort replication check (HIGH feasibility, most direct match to the user's LLM analogy)

For apnea/sleep-efficiency/BMI/age (all four cohorts present, sizes ranging
~165–1281, a ~7–8× spread), recompute the existing AUROC-vs-context
saturation curve **stratified by `dataset`** using only the already-collected
predictions — no new training. Check whether the saturation context $L^*$ and
qualitative pattern (e.g. sleep efficiency's context-irreplaceable shape)
seen in the *small* cohorts (APPLES ~168, STAGES ~200) match the pattern in
the *large* cohort (SHHS ~1278). This is not "predict large from small" in
the LLM sense of extrapolating a fitted curve, but it is a direct empirical
answer to "does the small-scale pattern generalize to the large scale,"
using cohorts that were collected independently (different sites, eras,
populations) — arguably a *stronger* generalization claim than an LLM
scaling-law fit, since it's not extrapolation but replication across
genuinely independent samples.
**Effort:** low — groupby on existing parquets + existing plotting code, one
new notebook. **Risk:** small-cohort curves may be too noisy per-context to
draw a clean $L^*$ estimate; may need coarser context bins.

### Idea B — Held-out context-length extrapolation test (HIGH feasibility, closest to "train cheap, predict expensive")

Fit the per-task saturation curve (AUROC vs $L$) using only the four
*cheaper* context lengths (30s, 10m, 40m, 80m — all fast to train) and use
the fit to predict AUROC at the two *expensive* ones (120m, 240m). Compare
predicted vs. actual. This is the literal analog of the LLM practice: use
small/cheap runs to forecast expensive ones, and report the prediction error.
A clean result ("held-out 240m prediction within X AUROC points using only
≤80 min runs") would be a genuinely novel, quotable finding tying directly
into the "context efficiency" title — it reframes context efficiency as
applying to the *research/deployment process itself*, not just inference.
**Effort:** low — pure curve-fit exercise on existing `analysis.csv`, per
task/head. **Risk:** only 4 points to fit on for some tasks; should report
per-task fit quality, not force one functional form on all seven tasks.

### Idea C — Compute-optimal frontier extrapolation beyond tested budgets (MEDIUM feasibility)

Chinchilla-style: fit power laws relating the optimal $(L,K)$ allocation to
total compute budget, using the 5 budget levels already in Table 4
(40/80/120/240/480 min), then extrapolate to a budget beyond anything
tested (e.g., 960 min / 16h, simulating extended or multi-night monitoring).
Report the *predicted* optimal $L$ at that budget as a forward-looking
deployment recommendation, explicitly flagged as extrapolation (per the
"don't extrapolate far below your smallest point" caution above — here we'd
be extrapolating *above* our largest tested budget, which carries the same
risk in reverse).
**Effort:** medium — needs the iso-compute heatmap data reshaped and two
power-law fits (one for $L_{opt}(C)$, one for $K_{opt}(C)$) per task.
**Risk:** the existing iso-compute grid (Fig. 3) already shows the optimal
$L$ saturates at the longest tested context for several tasks (sex, age) —
extrapolation may just say "keep using the max," a less interesting result
than for tasks with a genuine crossover (apnea, sleep efficiency).

### Idea D — Rectified (irreducible-error) power-law refit for small-*N* tasks (MEDIUM feasibility)

Refit OSA and depression's AUROC-vs-context curves with a power-law-plus-floor
form instead of treating them as merely "noisy" or "near-chance," using
bootstrap CIs already available via `mean_prob_auroc_ci_lo/hi` in
`analysis.csv`. This would let us say something quantitatively sharper than
"OSA is noisy due to small N" — e.g., whether the fitted floor is
statistically distinguishable from chance, and how wide the CI on the
scaling exponent is. Directly citable to the rectified-scaling-law
literature above.
**Effort:** medium — nonlinear curve fit with an extra free parameter on
noisy small-N data; needs care (per the literature, this can be numerically
unstable without a good initialization/prior).
**Risk:** may not produce a cleaner story than what's already said; worth
trying but not guaranteed to pay off.

### Idea E — Joint context×compute scaling surface (MEDIUM–HIGH feasibility, most novel)

> **Implemented — see Part 7 below for results.** Short version: it works
> reasonably as an interior-interpolation tool (~1-3 AUROC points off for
> most held-out context lengths) but the hoped-for "one clean equation"
> didn't pan out — the fitted ceiling parameter saturates at an
> unphysical value (≥1.0) for all three tasks tried under the simplest
> functional form. Mixed result, not a clean win.

Directly analogous to Montgomery et al. (2510.14919): instead of the current
per-context-length iso-compute heatmap (a lookup table), fit one smooth joint
functional form `AUROC ≈ f(L, C)` (context length and total compute as two
covariates of a single equation, e.g. a bivariate power law or the
context-aware form from that paper) across all six context lengths and the
full $K$ sweep simultaneously. Report the fitted form and its
goodness-of-fit. If it fits well, this becomes a single interpretable
equation replacing the heatmap — a strong, quotable "context efficiency law"
result, and lets the Discussion cite 2510.14919 as "recent work extended
scaling laws to jointly model compute and context for LLMs; we find an
analogous relationship holds for a clinical PSG foundation model" — good
related-work positioning, and to our knowledge not yet done for PSG/clinical
time-series data specifically.
**Effort:** medium-high — reuses existing `heatmap_df`/`analysis.csv`, but
the joint fit and validation (train/test split across the 2D grid, e.g.
leave-one-context-out) is more involved than Ideas A–D.
**Risk:** highest payoff but also highest chance the fit just doesn't work
well given only 6 context lengths × a handful of $K$ values per task —
recommend prototyping on the 2 highest-signal tasks (sex, sleep efficiency)
before committing to all seven.

---

## Suggested priority if pursuing any of this

1. **Idea A** (cross-cohort replication) — cheapest, most directly answers
   the "does small-scale pattern generalize" question the user asked, and
   produces a Discussion-strengthening robustness argument even if the paper
   doesn't otherwise change.
2. **Idea B** (held-out context extrapolation) — cheapest genuinely novel
   result, ties directly to the paper's title.
3. **Idea E** (joint scaling surface) — highest ceiling, but prototype on 1–2
   tasks before committing, and only pursue if A/B go well and there's
   appetite for another figure/table.
4. **Idea C, D** — lower priority; C mostly reconfirms Fig. 3/Table 4, D is a
   statistical-rigor improvement rather than a new finding.

None of these require new training runs or GPU time — everything reuses
`final_results/phase0_v3/collected/{analysis.csv,training.csv,predictions/*.parquet}`.

---

## Part 4 — Idea A implemented: results, and an honest reassessment of what it shows

Implemented on branch `scaling-law-ideas-ab` (NSRR-tools) as
`scaling_law_ideas/idea_a_cross_cohort.py`. Nothing existing was modified;
outputs went to a new `scaling_law_ideas/output/` directory only.

**What this is NOT, stated up front because an earlier draft of this
section undersold the caveat.** This is **not** external validation and
**not** a generalization or scaling-law check in any meaningful sense. The
Transformer model being evaluated here was trained on a *pool* that
already includes subjects from all four cohorts (APPLES, MrOS, SHHS,
STAGES) mixed together. What follows just takes that single already-fully-
trained model and looks at its existing test-set predictions broken down
by which cohort each test subject happens to come from. The model's
weights were already shaped by seeing each cohort's demographics, signal
characteristics, and label distribution during training — nothing here is
held out from training. Do not describe this in the paper as showing the
finding "generalizes across cohorts" or "replicates independently" —
those are training-blind-holdout claims this analysis cannot support.

**What this actually is: a subgroup-consistency check.** The real,
answerable question is narrower: *within the pooled test set, is the
model's context-sensitivity behaviour uniform across the different
recruitment populations it was trained and tested on, or is the pooled
result actually driven by whichever cohort dominates the pooled subject
count (usually SHHS)?* This is analogous to the subgroup-by-site or
subgroup-by-demographic breakdowns clinical ML papers often report as a
robustness check, not to a scaling-law or held-out generalization result.
It can still surface something worth knowing (see the MrOS finding below),
but the bar for what it can claim is much lower than "predicts unseen
scale."

**Method.** Exactly reproduces the pipeline's own mean-prob AUROC
convention (`scripts/analyze_windows.py: evaluate_at_k`, K="all": per
subject, average the probability vector over *all* available windows,
then AUROC over subjects — binary via `prob_class1`, multi-class via
`roc_auc_score(..., multi_class="ovr", average="macro")`), with one added
group-by on the parquets' `dataset` column before computing AUROC. Sanity
check: the pooled (all-cohorts) recomputed AUROC was verified to match
`analysis.csv`'s `mean_prob_auroc` **exactly** (max abs diff `0.000000`
across all 42 task×context cells checked), so the per-cohort breakdown
uses the identical, already-validated methodology — the numbers below are
trustworthy even though the *interpretation* needs to stay modest.

**Cohort sizes available at test time** (four tasks have 3–4 cohorts; sex
has 2; OSA is APPLES-only and depression is APPLES+STAGES only, so both
were excluded — no meaningful size spread):

| Task | apples | mros | shhs | stages |
|---|---|---|---|---|
| Apnea | 168 | 408 | 1278 | 200 |
| Sleep efficiency | 169 | 571 | 1281 | — |
| BMI | 167 | 435 | 1254 | — |
| Age | 165 | *(excluded, single-class)* | 1255 | — |
| Sex | 166 | — | 1264 | — |

**Results, AUROC at 30s → 240m (peak in parentheses if not at 240m):**

| Task | Cohort | AUROC@30s | AUROC@240m | Peak |
|---|---|---|---|---|
| Apnea | apples (n=168) | 0.793 | 0.912 | 0.919 @ 80m |
| | mros (n=408) | 0.706 | 0.855 | 0.862 @ 80m |
| | shhs (n=1278) | 0.742 | 0.830 | 0.835 @ 120m |
| | stages (n=200) | 0.711 | 0.845 | 0.863 @ 120m |
| Sleep efficiency | apples (n=169) | 0.667 | **0.870** | (at 240m) |
| | mros (n=571) | 0.671 | 0.777 | 0.777 @ 120m (plateaus) |
| | shhs (n=1281) | 0.681 | 0.824 | (at 240m) |
| BMI | apples (n=167) | 0.723 | 0.776 | (at 240m, flat throughout) |
| | mros (n=435) | 0.706 | 0.742 | 0.772 @ 80m (noisy/flat) |
| | shhs (n=1254) | 0.726 | 0.758 | (at 240m, flat throughout) |
| Age | apples (n=165) | 0.720 | 0.776 | 0.795 @ 80m (noisy) |
| | shhs (n=1255) | 0.788 | 0.859 | (at 240m) |
| Sex | apples (n=166) | 0.746 | 0.872 | (at 240m) |
| | shhs (n=1264) | 0.838 | 0.916 | (at 240m) |

**Observation: within the pooled test set, the *shape* of the
context-length curve looks similar across the cohort subgroups (spanning
a ~7.5× size range, 165 to 1278 subjects); the *absolute AUROC level*
does not.** ("Similar across subgroups of the same trained model's test
set" — not "replicates" or "generalizes," to avoid implying a held-out
claim this data can't support.)

- **BMI**: every cohort (apples, mros, shhs) shows the same flat,
  context-insensitive shape already reported for the pooled result — the
  clearest, cleanest replication of the three multi-cohort tasks.
- **Apnea**: every cohort (including the smallest, apples n=168 and stages
  n=200) shows the same rise-then-plateau shape peaking around 80–120 min.
  But APPLES sits 6–9 points *higher* than SHHS throughout (e.g. 0.919 vs
  0.835 at peak) — plausibly because APPLES is a CPAP-efficacy trial cohort
  enriched for more clear-cut/severe OSA, unlike SHHS's general-population
  case-mix. Shape replicates; absolute level is cohort/case-mix dependent.
- **Sleep efficiency**: APPLES and SHHS both show the paper's
  "context-irreplaceable" monotonic-rising shape with no plateau through
  240 min (APPLES rises *most* steeply of all, ending highest at 0.870).
  MrOS is the one partial exception: it plateaus by 120 min (0.777) rather
  than continuing to rise — a genuine, worth-flagging divergence from the
  pooled story, plausibly related to MrOS's older, all-male, comorbidity
  profile differing from the other cohorts.
- **Age / Sex**: SHHS closely tracks the pooled curve in both (expected,
  since SHHS dominates the pooled subject count). APPLES shows the same
  qualitative upward trend but at a visibly lower, noisier ceiling in both
  — consistent with its much smaller test-set size (165–166 subjects, vs.
  ~1260+ for SHHS) rather than a genuine shape difference.

**Assessment: modest supporting value at most — not a headline result, and
only worth including (if at all) as a clearly-scoped subgroup-consistency
footnote, not as evidence of generalization.** On reflection (and per
feedback that an earlier version of this section oversold it), the
honest claim this analysis supports is limited to: *"the pooled-trained
model's test-set behaviour, when broken down by recruitment cohort, does
not appear to be driven entirely by the largest cohort (SHHS) — the
smaller cohorts show qualitatively similar context-length curves in most
cases, with one documented exception (MrOS/sleep efficiency, below)."*
That is a much narrower and more defensible claim than "the finding
generalizes" or "replicates independently," and it is genuinely uncertain
whether it is interesting enough to spend supplementary-figure space on,
given a similar figure was already considered and left out of an earlier
paper draft. If included at all, it should be one or two sentences with a
small table (not a full figure) and language matching the framing above,
not the stronger claims in the git commit messages/earlier draft of this
section. The one thing worth stating plainly regardless of whether this
becomes a figure: **MrOS's sleep-efficiency curve plateaus by 120 min
(0.777) instead of continuing to rise like the other cohorts** — this is
a real divergence from the pooled "context-irreplaceable" story for one
specific subgroup, and worth a one-sentence caveat wherever that claim is
made, independent of whether the rest of this analysis is used.

**What would actually answer the "does a small-scale pattern predict
large-scale behaviour" question** is a genuinely different, stronger
experiment: retrain the model excluding one cohort entirely from
training, then evaluate only on that fully-unseen cohort (leave-one-
cohort-out). That is real external validation and would support a
generalization claim; the analysis above cannot. See Part 7 (Idea F) for
a scoped-down, feasible version of this that does require new training
runs, proposed per your go-ahead to consider retraining-based ideas.

**Caveats:** (1) this is in-distribution stratified test evaluation, not
a holdout test, per the reassessment above; (2) APPLES/age and APPLES/sex
cells are noisy given small n (165–166 test subjects); (3) MrOS was
excluded from the age-group breakdown because MrOS enrolls only older
men, so age-group label variance within that cohort is degenerate (all
subjects fall in one class) — a cohort-demographic fact, not a bug.

Files: `NSRR-tools/scaling_law_ideas/idea_a_cross_cohort.py`,
`NSRR-tools/scaling_law_ideas/output/idea_a_cross_cohort.png`,
`NSRR-tools/scaling_law_ideas/output/idea_a_per_cohort_auroc.csv`.

---

## Part 5 — Idea B implemented: results and verdict

Implemented on branch `scaling-law-ideas-ab` (NSRR-tools) as
`scaling_law_ideas/idea_b_context_extrapolation.py`. Add-only, nothing
existing modified.

**Method.** For each of the seven tasks (Transformer, test split, K=all,
from `analysis.csv` — the exact numbers already used for the paper's main
saturation-curve results), fit the same power-law form already used
elsewhere in the codebase for the FLOPs scaling figure,
`AUROC(L) = c − a·L^(−b)` (`scipy.optimize.curve_fit`, same bounds/init as
`utils/panels.py`'s `_power_law`), **using only the four cheap context
lengths (30s, 10m, 40m, 80m)**. Evaluate the fitted curve at 120m and 240m
— genuinely held out, never seen by the fit — and compare to the real
values. A second fit on all six points is plotted alongside as a
reference (how good the functional form could ever do in-sample), to
separate "extrapolation is hard" from "this functional form doesn't suit
this task's curve shape."

**Results — prediction error (predicted − actual AUROC), Transformer, test split:**

| Task | Error @ 120m | Error @ 240m |
|---|---|---|
| Sex | −0.016 | −0.013 |
| Age | −0.009 | −0.007 |
| Apnea | −0.012 | +0.003 |
| BMI | −0.001 | −0.010 |
| Sleep efficiency | −0.035 | −0.042 |
| Depression | −0.006 | +0.003 |
| OSA (APPLES) | +0.018 | +0.024 |

**Mean absolute error across all 7 tasks: 0.014 AUROC points at 120m,
0.014 at 240m (max 0.042, sleep efficiency@240m). Restricted to the 5
"well-behaved" tasks (excluding sleep efficiency and OSA, both of which
have known, already-documented reasons to misbehave — see below): mean
absolute error drops to 0.008 AUROC points, max 0.016.** That is: for
5 of 7 tasks, training only the four cheapest context lengths and
extrapolating with a 3-parameter power law predicts the two most
expensive, longest-context runs to within ~1–1.5 AUROC points.

**The two exceptions are individually explainable, not arbitrary
failures, and both explanations independently reinforce findings already
in the paper:**
- **Sleep efficiency** undershoots by the largest margin (4.2 points at
  240m) because its curve is genuinely still rising with no sign of a
  plateau through the longest tested context — exactly the paper's own
  "context-irreplaceable" characterization of this task. A curve that
  hasn't shown its bend yet in the cheap-context data *should* be hard to
  extrapolate; the size of the miss is itself corroborating evidence for
  the qualitative claim, not a contradiction of it.
- **OSA (APPLES-only, n=161)** overshoots by 1.8–2.4 points because its
  AUROC is non-monotonic (dips at 120m before rising again at 240m), a
  known small-*N* noise artifact already flagged elsewhere in the paper.
  A smooth power law cannot reproduce a non-monotonic dip by construction;
  this is an expected, not surprising, limitation.

**Verdict: worth adding**, and arguably the stronger of the two candidates
implemented so far, because it makes a directly quantitative, falsifiable,
title-relevant claim: *cheap short-context experiments can forecast
expensive long-context outcomes to within ~1 AUROC point for most tasks,
and the cases where they can't are exactly the cases the paper already
calls out as exceptional (sleep efficiency, small-*N* OSA).* This reframes
"context efficiency" to apply to the experimental process itself, not just
model inference: you would not have needed to run the 120m/240m arms at
all to know, within ~1 point, what they'd show — except for the one task
where that's precisely the interesting finding. Recommend either a new
supplementary figure (the 7-panel plot already produced) or a compact
table in the main text/supplementary, with 3–4 sentences in Results or
Discussion.

**Caveats to carry into the paper if added:** (1) only 4 points feed a
3-parameter fit (1 degree of freedom) — the fit quality figure shows this
is enough for 5/7 tasks but flag it explicitly; (2) this uses the already-
completed 120m/240m runs as ground truth to validate against — it is a
retrospective demonstration that the approach *would have* worked, not a
prospective one (we are not proposing to actually skip the expensive runs
now that we have them); (3) only tested with the Transformer head — LSTM/
MeanPool were not checked and might extrapolate differently (worth doing
if this analysis is adopted, low additional effort).

Files: `NSRR-tools/scaling_law_ideas/idea_b_context_extrapolation.py`,
`NSRR-tools/scaling_law_ideas/output/idea_b_context_extrapolation.png`,
`NSRR-tools/scaling_law_ideas/output/idea_b_extrapolation_errors.csv`.

### Idea B follow-up: sensitivity to how few cheap points are used

Asked and answered before implementing: fitting the same 3-parameter power
law on only 3 points (30s, 10m, 40m), or 2 points (30s, 10m), leaves 0 or
negative residual degrees of freedom respectively — an exact interpolation
(3 points) or an under-determined fit relying entirely on the parameter
bounds to pick a solution (2 points), neither of which is validated by the
training data itself the way the original 4-point fit was. Implemented as
`idea_b_sensitivity_num_cheap_points.py` (imports the original script's
fitting function rather than duplicating or modifying it) to quantify
exactly how much this matters, predicting *all* remaining longer contexts
in each case.

**Result — mean absolute error (AUROC points) across all held-out
predictions, by number of cheap training points used:**

| Points used | Contexts | Mean abs. error | Max abs. error |
|---|---|---|---|
| 4 (original) | 30s,10m,40m,80m | 0.014 | 0.042 |
| 3 | 30s,10m,40m | 0.021 | 0.069 |
| 2 | 30s,10m | 0.032 | 0.117 |

Degrades monotonically and roughly linearly as predicted — error
approximately doubles from 4→2 points, worst case (max error) nearly
triples. `curve_fit` converged in every case (no outright failures), but
threw `OptimizeWarning: Covariance of the parameters could not be
estimated` for the poorly-constrained 2-3 point fits, exactly the
numerical-instability symptom flagged beforehand.

Per-task breakdown confirms this isn't uniform: **sleep efficiency is the
worst case at every point count and degrades fastest** (0.039 → 0.057 →
0.087 as points drop from 4 to 2 — more than doubling), consistent with
its still-rising, not-yet-plateaued curve shape. **OSA (small-*N*,
APPLES-only) jumps sharply between 3 and 2 points** (0.015 → 0.051),
consistent with its already-known small-sample instability compounding
with an under-determined fit. **BMI, age, and depression stay low and
roughly flat across all three point counts** (all have simple,
near-saturating or near-flat curves that a power law captures easily even
from very little data) — for these tasks, going cheaper genuinely costs
little.

**Verdict:** confirms the original prediction rather than changing it —
this is worth keeping as a **supporting sensitivity panel for Idea B**
(one extra plot: error vs. number of training points), not a replacement
for the 4-point result. It strengthens the "context efficiency of the
research process" framing by showing precisely where the cliff is: for
tasks with a simple/flat curve, 2 cheap runs are almost as good as 4; for
tasks still actively benefiting from context (sleep efficiency) or with
small, noisy samples (OSA), the 4-point version is close to the minimum
viable amount of cheap data, and going cheaper trades away real accuracy.

Files: `NSRR-tools/scaling_law_ideas/idea_b_sensitivity_num_cheap_points.py`,
`NSRR-tools/scaling_law_ideas/output/idea_b_sensitivity_num_cheap_points.png`,
`NSRR-tools/scaling_law_ideas/output/idea_b_sensitivity_errors.csv`.

---

## Part 6 — Overall recommendation after implementing A and B (revised)

**Idea B is the solid result of the two** — a genuinely useful,
appropriately-scoped finding (predicting expensive long-context results
from cheap short-context fits to within ~1 AUROC point for most tasks),
worth a supplementary figure and a paragraph in Discussion or Results.

**Idea A should not be sold as a finding on its own.** As reassessed in
Part 4: it's an in-distribution subgroup breakdown of one already
pooled-trained model, not a generalization or held-out check, and a
similar analysis was already considered and left out of an earlier paper
draft for what was likely the same reason. The one piece of it worth
keeping is the single-sentence MrOS/sleep-efficiency caveat, independent
of whether the rest becomes a figure.

Both analyses do happen to single out the same two tasks (sleep
efficiency, small-*N* cohorts/OSA) as the exceptions to otherwise clean
patterns — that convergence is a mildly interesting observation, but
should not be oversold as two independent lines of evidence when one of
the two lines is methodologically much weaker than the other.

Ideas C and D from Part 3 remain unimplemented. Idea E is implemented
below (Part 7). Part 8 adds retraining-based ideas (a category explicitly
ruled in-scope now), including a properly-scoped version of what Idea A
was trying to get at.

---

## Part 7 — Idea E implemented: joint context×compute scaling surface

Implemented on branch `scaling-law-ideas-ab` (NSRR-tools) as
`scaling_law_ideas/idea_e_joint_scaling_surface.py`. Add-only, nothing
existing modified. Prototyped on the 2 tasks the original proposal
recommended (sex, sleep efficiency) plus BMI as a third, deliberately
chosen as a context-*insensitive* reference case for contrast.

**Method.** Fit one joint function of context length ($L$, minutes) and
inference-time aggregation ($K$, windows) to the entire (L, K) → AUROC
grid from `analysis.csv` (Transformer, test split, all numeric $K$ values
— 70-79 points per task, excluding the `k="all"` rows since "all" means a
different numeric $K$ at each $L$). Primary functional form, chosen to
mirror the paper's own established power-law convention (same shape as
the FLOPs fit) rather than inventing something new — an additive,
separable two-term deficit:

$$\mathrm{AUROC}(L,K) = c - a\cdot L^{-p} - b\cdot K^{-q}$$

Validated via **leave-one-context-length-out**: refit on 5 of the 6
context lengths (all their $K$ values), predict the 6th, entirely
held-out context length's whole $K$-curve, compare to actual. This is a
genuine held-out test — unlike Idea A, the held-out context length's data
never enters that fold's fit at all.

**Result 1 — the additive form has a real problem: it wants an
impossible ceiling.** For all three tasks tried, the fitted ceiling
parameter $c$ saturates exactly at the physical upper bound I imposed
(1.0 — AUROC cannot exceed 1.0), regardless of the initial parameter
guess (checked 3 different starting points per task; all converged to
the identical solution, so this is a genuine global optimum under this
functional form, not a local-optimum artifact). A quick side-check with
an alternative multiplicative form, $c - a\cdot L^{-p} K^{-q}$ (single
combined deficit term instead of two additive ones), gives a sensible,
non-saturated ceiling for BMI ($c=0.765$, close to BMI's real ceiling of
~0.777 from Table 3) but **still** saturates at 1.0 for sex and sleep
efficiency under either form. **Interpretation: a simple two-parameter-
per-axis joint power law is not quite the right functional form for this
data** — it is not just "harder to fit for context-sensitive tasks";
even the simplest, flattest task (BMI) only got a sensible ceiling under
one of the two forms tried. This means Part 3's original hope — "one
clean interpretable equation replacing the heatmap" — did not pan out
as cleanly as proposed. It is not a failed idea, but the honest
conclusion is narrower than originally hoped.

**Result 2 — despite the ceiling problem, held-out predictive accuracy is
good for "interior" context lengths and poor at the two extremes (30s and
240m), a sensible U-shaped pattern, not a random failure.**

| Held out | Sex | Sleep Efficiency | BMI |
|---|---|---|---|
| 30s | 0.157 | 0.213 | 0.046 |
| 10m | 0.030 | 0.051 | 0.013 |
| 40m | 0.016 | 0.017 | 0.011 |
| 80m | 0.012 | 0.020 | 0.007 |
| 120m | 0.032 | 0.036 | 0.007 |
| 240m | 0.061 | 0.064 | 0.029 |
| **Mean (all 6)** | **0.073** | **0.092** | **0.023** |
| **Mean (excl. 30s)** | **0.026** | **0.037** | **0.012** |

(AUROC points, i.e. 0.073 = 7.3 points.) The worst case by far in every
task is holding out **30s** — the shortest, most extreme context length,
an order of magnitude below the next-shortest tested point (10m) — which
makes the leave-one-out fold for it a genuine extrapolation to a data
boundary from a fit that never saw anything nearby, not an interpolation.
**240m (the other extreme) is consistently the second-worst.** Dropping
just the 30s fold roughly **triples the accuracy** for sex and sleep
efficiency (7.3→2.6 points, 9.2→3.7 points) and nearly halves it for BMI.
This U-shape (worse at both boundaries, better for the four interior
context lengths) is exactly what you'd expect from curve-fit
extrapolation in general, and matches the same "extrapolation is harder
the further from the training data" pattern already documented in Idea B.

**Verdict: mixed, and more useful as a diagnostic than as a headline
"context efficiency law."** What Idea E delivers, honestly:
- **Not** a single clean equation that can replace the iso-compute
  heatmap — the ceiling-saturation problem means the fitted parameters
  themselves aren't fully trustworthy as physical quantities, even where
  predictive accuracy is good.
- **Is** a working interior-interpolation tool: given 5 of 6 context
  lengths' full $(L,K)$ behaviour, the 6th (if it's not the shortest or
  longest) can be predicted to within ~1-3 AUROC points.
- **Is** a clean quantitative illustration that context length and
  aggregation are not simply, separably substitutable in a textbook
  Chinchilla sense for any of these three tasks — which is a more
  rigorous, model-based way of stating something the paper already argues
  qualitatively (H2 and its "context-irreplaceable" exception), even
  though it doesn't produce the hoped-for standalone equation.
- Recommend, if used at all: a supplementary figure showing the
  data-vs-fit curves and the leave-one-out scatter (`idea_e_joint_scaling_surface.png`
  already shows both), framed honestly as "a joint scaling-law attempt
  that predicts interior context lengths well but does not admit a single
  physically well-behaved equation across the full range" — not as "we
  found a context efficiency law." Whether that's worth a figure given
  the more mixed framing is a judgment call for you; happy to extend to
  more tasks first if that would help decide.

**Caveats:** (1) only 3 tasks tried (sex, sleep efficiency, BMI), not the
full seven; (2) only two functional forms tried (additive, multiplicative)
— there may be a better-behaved form not yet tried (e.g. explicitly
modelling $L$ and $K$'s known interaction via total-compute $L\times K$ as
a third term); (3) leave-one-context-out is a single-fold-per-length test,
not a full cross-validation with repeated resampling.

Files: `NSRR-tools/scaling_law_ideas/idea_e_joint_scaling_surface.py`,
`NSRR-tools/scaling_law_ideas/output/idea_e_joint_scaling_surface.png`,
`NSRR-tools/scaling_law_ideas/output/idea_e_loo_errors.csv`.

---

## Part 8 — New ideas that require retraining (not yet implemented — proposals for you to consider)

You said it's fine for scaling-law ideas to need retraining on a *subset*
of context lengths and/or tasks (not the full grid). Three proposals,
roughly in order of how directly they answer "does a small-scale pattern
predict large-scale behaviour":

### Idea F — Leave-one-cohort-out: the real version of Idea A

Retrain (not just re-evaluate) excluding one cohort entirely from the
training set, then evaluate **only** on that fully-unseen cohort. This is
actual external validation and can properly support a "generalizes to an
unseen population" claim, unlike Idea A.

**Scoped-down proposal** (not the full 4-cohort × 7-task × 6-context
grid, which would be prohibitively expensive): pick the **2 tasks with
the cleanest existing cross-cohort signal from Idea A** — apnea (clean
rise-then-plateau shape in every cohort, but a big absolute-level gap
between APPLES and SHHS worth explaining) and sleep efficiency (where
MrOS already showed a genuine divergence, worth confirming isn't just
noise) — and **one held-out cohort per task**: exclude APPLES for apnea
(it's the cohort with the most different absolute level — the
interesting case), exclude MrOS for sleep efficiency (the one that
diverged). Train on a **reduced context grid** (e.g. 3 of the 6 lengths:
30s, 80m, 240m — enough to see whether the saturation shape still holds)
with just the Transformer head. That's ~6 new training runs total
(2 tasks × 3 contexts × 1 exclusion × 1 head), not 126.

**What it would show:** does the context-length curve shape, and the
saturation point specifically, learned from the *other three* cohorts
correctly predict what happens on the held-out cohort's own subjects
(never seen in training at all)? If yes, that's a real, quotable
generalization claim — actually addressing what Idea A was trying and
failing to show. If the held-out cohort's curve looks different in
*shape* (not just absolute level), that's an important, honest limitation
to report.

**Cost:** ~6 training runs, modest compute (reduced context grid).
**Risk:** with only 1 cohort excluded per task, this is still a
single-fold check, not a full cross-validation — frame results as
"a representative case," not as proof it holds for every cohort.

### Idea G — Data-size scaling law (new axis: training-set size, not context length)

The paper currently has two scaling axes: context length (main result)
and training compute/FLOPs (Ext. Fig. 2 / S-Fig 19, the one just fixed).
It does not have the third, most classically "LLM scaling law" axis:
**how does performance scale with training set size** at fixed context
length? This is the direct analog of Kaplan/Chinchilla's `D_opt` (data)
axis, distinct from everything else in the paper.

**Proposal:** pick 1–2 tasks and **one representative context length**
per task (e.g. each task's own $L^*$, already known from Table 3), and
retrain at several random subsamples of the pooled training set (e.g.
25%, 50%, 75%, 100%), Transformer only. Fit `AUROC ≈ ceiling − A·N^(−b)`
against training-set size $N$ (same functional form already used
elsewhere), and report the fitted exponent and whether performance has
plateaued at current $N$ or would still improve with more subjects.

**Why this is a good complement, not a duplicate, of what's already
there:** it answers a different, forward-looking question the current
paper doesn't touch — "is ~2,000 training subjects (per main task) enough,
or is the model still data-starved?" — directly relevant to anyone
planning a follow-up study or larger NSRR pull, and a natural item for a
"scaling laws" section since it would give the paper all three canonical
axes (context, compute, data) in one place.

**Cost:** needs a small addition to `train_context_sweep.py` (a
`--train_frac` subsampling option doesn't currently exist — quick to add,
subsample `train_ds` before the DataLoader is built, fixed seed for
reproducibility). Then 4 subsample fractions × 1–2 tasks × 1 context ×
1 head = 4–8 new training runs.
**Risk:** low-to-moderate — this is a well-trodden, low-risk methodology
(same as everything else already in the paper), main uncertainty is
whether 4 data points are enough to fit a stable exponent (same
degrees-of-freedom caveat as Idea B).

### Idea I — Prospective context-length extrapolation (stretch goal)

Idea B validates the "cheap predicts expensive" claim *retrospectively*
(120m/240m were already run; we just checked the cheap-only fit would
have predicted them). The prospective version — genuinely more
convincing, closer to Chinchilla's own validation (they trained a new,
larger model specifically to test their extrapolated prediction) — would
be to fit the curve on the existing 6 points and then **actually train at
a new, longer context length never run before** (e.g. 480 min / 8h, if
full-night recordings are long enough) and check whether the earlier
fit's prediction holds.

**Feasibility caveat, flagged honestly:** `TransformerHead`'s docstring
notes it's memory-capped around 80 min *unless* Flash Attention kicks in
(which the existing 240m runs already rely on, per
`sequence_head.py`'s comments) — so 480m may or may not be feasible
without further engineering, and would also need the cohort filter
(`min_recording_patches`) loosened, which could shrink the eligible
subject pool. This is the most expensive and technically riskiest of the
three proposals; I'd only pursue it if F and G don't fill out the section
enough on their own.

**None of these are implemented yet** — flagging for you to pick from
before I start. F is the most direct fix for what Idea A couldn't
deliver; G is the most novel/complementary new result; I is the most
ambitious and technically uncertain.

---

## Part 9 — Ideas A and B written into the paper

Added to both `npj_main.tex` and `npj_supplementary.tex` (Idea E was not
added — its mixed result, per Part 7, didn't seem worth a place in the
paper without further work; happy to revisit if you disagree).

**Placement decision:** did not create a dedicated "Scaling Laws"
section. Idea A and Idea B are conceptually different enough (subgroup
consistency vs. a genuine curve-fit/extrapolation exercise) that forcing
them into one new section alongside the already-placed FLOPs analysis
(Extended Data Fig. 2, already living in the H3/head-comparison context)
would have meant relocating already-correctly-placed content for no clear
gain. Instead, each was added to the existing section it fits best:

- **Idea A** → expanded the existing "Per-Cohort Breakdown" section
  (`sec:supp-cohort`, Supplementary Section S-X) from its original
  single-task, single-context table into the full multi-task analysis,
  with the honest subgroup-consistency framing from Part 4 (not
  generalization language). New Figure S-1.
- **Idea B** → added as a new subsection ("Predicting the Expensive
  Contexts from the Cheap Ones", S-XVIII.A) directly under the existing
  "Saturation Curves — All Heads Overlaid" section, since it's a direct
  extension of that section's own content. New Figure S-3. The sensitivity
  check (fewer cheap points) is described in the same subsection as text,
  not given its own figure.
- **Main text**: two short paragraphs added to the H1 section
  ("Context-length saturation is task-specific"), right before the
  existing "H1 is confirmed" summary sentence, each pointing to its
  supplementary section.

**Figure renumbering required.** Inserting 2 new supplementary figures
before the existing sequence shifts every one of the current 23 figures.
Mapping (also applied to every in-text cross-reference in both `.tex`
files already, via a scripted find-and-replace, not manual): old S-Fig 1
→ new S-Fig 2; old S-Fig $N$ ($N \geq 2$) → new S-Fig $N+2$.

| Old filename | New filename |
|---|---|
| `sfig1_saturation.pdf` | `sfig2_saturation.pdf` |
| `sfig2_threshold_unlock.pdf` | `sfig4_threshold_unlock.pdf` |
| `sfig3_kvsk.pdf` | `sfig5_kvsk.pdf` |
| `sfig4_heatmap.pdf` | `sfig6_heatmap.pdf` |
| `sfig5_iso.pdf` | `sfig7_iso.pdf` |
| `sfig6_mincost.pdf` | `sfig8_mincost.pdf` |
| `sfig7_deployment_grid.pdf` | `sfig9_deployment_grid.pdf` |
| `sfig8_waterfall.pdf` | `sfig10_waterfall.pdf` |
| `sfig9_pr_curves.pdf` | `sfig11_pr_curves.pdf` |
| `sfig10_task_landscape.pdf` | `sfig12_task_landscape.pdf` |
| `sfig11_task_clustermap.pdf` | `sfig13_task_clustermap.pdf` |
| `sfig12_aggregate_scaling.pdf` | `sfig14_aggregate_scaling.pdf` |
| `sfig13_channel_comparison.pdf` | `sfig15_channel_comparison.pdf` |
| `sfig14_modality_ablation.pdf` | `sfig16_modality_ablation.pdf` |
| `sfig15_modality_radar.pdf` | `sfig17_modality_radar.pdf` |
| `sfig16_subject_stability.pdf` | `sfig18_subject_stability.pdf` |
| `sfig17_k_aggregation.pdf` | `sfig19_k_aggregation.pdf` |
| `sfig18_pr_extended.pdf` | `sfig20_pr_extended.pdf` |
| `sfig19_compute_scaling.pdf` | `sfig21_compute_scaling.pdf` |
| `sfig20_variance_violins.pdf` | `sfig22_variance_violins.pdf` |
| `sfig21_reliability.pdf` | `sfig23_reliability.pdf` |
| `sfig22_hard_subjects.pdf` | `sfig24_hard_subjects.pdf` |
| `sfig23_position_variance.pdf` | `sfig25_position_variance.pdf` |

**Two new files needed** (already generated, with both PDF and PNG, at
`NSRR-tools/scaling_law_ideas/output/`):
- `idea_a_cross_cohort.pdf` → copy into the npj repo root as
  **`sfig1_cross_cohort.pdf`**
- `idea_b_context_extrapolation.pdf` → copy into the npj repo root as
  **`sfig3_context_extrapolation.pdf`**

Note this leaves gaps at S-Fig 1 and S-Fig 3 being new/renamed files while
everything else is a rename of an existing file — the numbering is
contiguous (1 through 25) once all 25 files are in place, this table just
separates "rename" from "new."

**Bonus fixes made while in this part of the document** (found, not
introduced, while editing the exact paragraphs involved):
- A second stale TBME-era cross-reference, "Section IV-C of the main
  paper" in the Saturation Curves section of the supplement, fixed the
  same way as the earlier "Section III-G" fix (named pointer instead of a
  numbered section that no longer exists).
- A pre-existing Contents-page bug: the "Balanced Accuracy at
  Validation-Optimised Threshold" section (S-XV) existed in the document
  body but had no Contents entry at all, making the Contents page skip
  straight from S-XIV to S-XVI. Added the missing entry. Confirmed via
  `git diff` that this predates today's changes, not something introduced
  by the figure renumbering.

Both documents compile cleanly (checked after every edit: no `!` errors
other than the expected "file not found" for the not-yet-renamed images,
no undefined references, no multiply-defined labels).
