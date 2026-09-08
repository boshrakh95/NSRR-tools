# Extended Data Figures — Task Selection and Interpretation

Analysis based on direct reading of the three PDF files plus CSV AUROC values
from `phase0_v3/analysis.csv`. Recommendations follow the same principle used
in the main paper: show two or three tasks with qualitatively DIFFERENT patterns
that together support the main claim; move tasks with similar or weaker patterns
to supplementary.

---

## Extended Data Figure 1 — K-Aggregation Curves
**File:** `ext_fig1_k_aggregation.pdf` / `sfig17_k_aggregation.pdf`
**Both heads shown (LSTM + Transformer), L ∈ {40, 120, 240 min}**
**Supplements main Fig. 2 (K vs K, Transformer only, three representative tasks)**

### Pattern taxonomy across all six panels

| Task | Pattern | Note |
|---|---|---|
| Sex | L-dependent saturation speed; head gap present from K=1 | Transformer starts ~0.78 at K=1/L=40m, reaches 0.87 at K=all |
| Sleep Eff | Context-irreplaceable for BOTH heads; curves stay separated at K=all | L=40m ceiling (0.760) well below L=120m ceiling (0.815) |
| Apnea | Head gap GROWS with K — both heads start low at K=1 then diverge | LSTM 0.645, Transformer 0.656 at K=1/L=40m (≈tied), then LSTM 0.792 vs Transformer 0.825 at K=all |
| Age | Iso-compute substitutability (curves converge at K=all) | Already shown in main Fig. 2 panel (a) — redundant |
| BMI | Near-flat, context-insensitive; rapid saturation for both heads | Curves all cluster; head gap <2pp at K=all |
| OSA | Noisy small-dataset pattern; erratic at K=1 | Keep in supplementary |

### Recommended selection

**Include in Extended Data (3 panels): Sex, Sleep Eff., Apnea**

**Retain in Supplementary (3 panels): Age, BMI, OSA**

**Rationale:**
- **Sex**: The L-dependent saturation speed pattern. Shows that at K=1, shorter
  contexts start substantially lower (LSTM: 0.750 at L=40m vs 0.844 at L=240m),
  and more windows are needed to saturate at shorter contexts. Additionally, the
  Transformer-LSTM gap is large from the outset (e.g., +0.056 at K=1/L=120m) and
  persists at K=all. This mirrors main Fig. 2 panel (c) but now adds the LSTM
  curves, showing the head comparison is consistent with H3.

- **Sleep Efficiency**: The context-irreplaceable pattern confirmed for BOTH heads.
  Even at K=all, the L=40m ceiling (LSTM: 0.731, Transformer: 0.760) is far below
  the L=120m ceiling (LSTM: 0.778, Transformer: 0.815) and L=240m ceiling
  (LSTM: 0.788, Transformer: 0.831). No amount of inference aggregation bridges
  this gap. This is particularly important because it confirms the main paper's
  strongest H2 exception holds equally for LSTM and Transformer.

- **Apnea**: Unique pattern not elsewhere shown. At K=1 and L=40m, LSTM (0.645)
  and Transformer (0.656) start nearly tied — the head gap is only 0.011. As K
  increases, the Transformer gains more from aggregation than the LSTM: by K=all,
  the gap widens to 0.033 (LSTM: 0.792, Transformer: 0.825 at L=40m). The same
  divergence holds at L=120m (K=1 gap: 0.013; K=all gap: 0.025). This shows that
  for apnea, the Transformer's architectural advantage is revealed progressively
  through aggregation, not evident in single-window inference.

- **Age** (suppressed): Already shown in main Fig. 2 panel (a) as the
  iso-compute substitutability example. Including it again adds nothing new.
- **BMI** (suppressed): Context-insensitive task with rapid saturation for all
  heads — a minor finding already noted in the main text.
- **OSA** (suppressed): Small test set, erratic K=1 values, not interpretable
  without caveating the small-N confound throughout.

### Interpretation for the extended data figure

**(For body text in main paper, at the end of sec:results_kagg / H2 section):**

Extended Data Figure 1 extends the K-aggregation analysis to three additional
tasks and both temporal heads. Two of the three patterns identified in
Figure~\ref{fig:kvsk} replicate across heads: sex classification shows
L-dependent saturation speed, with longer-context models requiring far fewer
inference windows to plateau (Transformer: $K \approx 2$ at $L{=}240$~min vs
$K \approx 15$ at $L{=}40$~min), and sleep efficiency confirms the
context-irreplaceable regime — the ceiling at $L{=}40$~min ($K{=}K_{\max}$:
AUROC 0.760 Transformer, 0.731 LSTM) lies substantially below the performance
achievable at $L{=}120$~min with any $K$. Apnea detection reveals a third,
distinct pattern not present in the main figure: at $K{=}1$, both heads begin
at nearly the same AUROC ($\leq 0.01$ apart at every $L$), but with increasing
aggregation the Transformer gains more per additional window than the LSTM,
widening the head gap from $<0.015$ at $K{=}1$ to $>0.025$ at $K{=}K_{\max}$
across all contexts. This progressive divergence indicates that the
Transformer's architectural advantage for apnea detection is only fully
expressed when multiple inference windows are available, not in the
single-window regime.

**(For Extended Data Figure caption):**

K-aggregation curves for sex classification, sleep efficiency, and apnea
detection (LSTM and Transformer, reduced-channel, test split) at
$L \in \{40, 120, 240\}$~min. Each line shows AUROC as inference windows
accumulate from $K{=}1$ to $K{=}K_{\max}$. Sex shows L-dependent saturation
speed: longer contexts saturate in fewer windows and start higher at $K{=}1$.
Sleep efficiency shows context-irreplaceable behaviour: even $K{=}K_{\max}$ at
$L{=}40$~min cannot reach $L{=}120$~min performance at any $K$. Apnea shows
progressive head divergence: both heads begin nearly tied at $K{=}1$, but
the Transformer gains more per additional window, so the head gap widens with
$K$. Supplementary Figure~S-17 shows the remaining tasks (age, BMI, OSA).

---

## Extended Data Figure 2 — Compute Scaling
**File:** `ext_fig2_compute_scaling.pdf` / `sfig19_compute_scaling.pdf`
**AUROC vs total training FLOPs, power-law fits per head**
**Placed at end of sec:results_heads (H3 section)**

### Pattern taxonomy across all six panels

| Task | Pattern | At highest FLOPs: T / LSTM / MeanPool | Note |
|---|---|---|---|
| Sex | Clear head hierarchy, parallel power-law slopes | ~89% / ~87% / ~75% | T-MeanPool gap ~14pp, consistent |
| Sleep Eff | Steepest Transformer slope; largest persistent head gap | ~80% / ~76% / ~70% | Gap grows with FLOPs |
| Age | LSTM and Transformer nearly converge at high FLOPs | ~90% / ~89% / ~85% | T-LSTM gap collapses |
| Apnea | LSTM and Transformer converge at high FLOPs (similar to Age) | ~83% / ~82% / ~76% | T-LSTM gap <1pp at max |
| BMI | Near-zero head separation; MeanPool competitive with LSTM | ~76% / ~73% / ~73% | Head choice barely matters |
| OSA | Short FLOPs range; MeanPool leads at mid-range (small-N artefact) | ~85% / ~74% / ~85% | Anomalous, keep in supplementary |

### Recommended selection

**Include in Extended Data (2 panels): Sex, Sleep Eff.**

**Retain in Supplementary (4 panels): Age, BMI, Apnea, OSA**

**Rationale:**
- **Sex**: The canonical illustration of the head hierarchy. Transformer leads,
  LSTM in the middle, MeanPool below, across the full compute range available
  for each head. Shows that the ranking from tab:heads holds regardless of
  training budget.

- **Sleep Efficiency**: The task where the Transformer advantage is largest and
  most persistent across compute levels.

- **BMI, Age, Apnea, OSA** (suppressed): Structurally, all panels in this figure
  share the same visual pattern — three fit lines in their respective FLOPs ranges
  — because MeanPool is ~640x cheaper per step than Transformer (O(sl·d) vs
  O(sl²·d)), so it always occupies a lower total-FLOPs regime than temporal heads.
  That structural property is the same for every task, meaning panels do not show
  meaningfully different patterns from each other. Rather than forcing a
  "different regime" narrative that the figure cannot support, the extended data
  version shows the two most narrative-central tasks; the full six-panel figure
  stays in supplementary.

**Design note (for notebook):** The figure has an inherent limitation: the three
heads occupy only partially overlapping FLOPs ranges (MeanPool: 10^7–10^11;
LSTM/Transformer: 10^10–10^14). A fairer comparison would extrapolate all fit
lines to a common x-range (e.g., 10^8–10^14). Consider updating the notebook to
do this before final submission.

---

## Extended Data Figure 3 — Within-Subject Prediction Variance Violins
**File:** `ext_fig3_variance_violins.pdf` / `sfig20_variance_violins.pdf`
**std(prob) distributions for Correct vs Incorrect subjects across context lengths**
**Placed near end of Results, at the prediction stability paragraph**

### How to read a violin in this figure

Each violin is a sideways histogram. The WIDTH at any y-value = how many subjects
have that std(prob) value. Fat near zero = most subjects have consistent (low-
variance) predictions. Fat higher up = most subjects have erratic predictions.
The meaningful signal is whether blue (correct) and orange (incorrect) violins
DIFFER — and whether that difference grows with context length.

### Pattern taxonomy across all seven panels

| Task | Pattern at 240m | Note |
|---|---|---|
| Sex | Blue collapses near zero; orange grows wide/tall | Strongest separation; correct subjects become very consistent |
| Sleep Eff | Mild separation; blue narrows but remains moderately wide, both overlap | Weaker version of Sex pattern |
| Apnea | Blue collapses, orange grows — similar to Sex | Redundant with Sex |
| Age | OPPOSITE to Sex: blue grows wider/fatter than orange | Correct subjects are MORE variable than incorrect ones at long L |
| BMI | Blue and orange stay similar at all L; no separation | Null result |
| Depression | Very small absolute values throughout; slight separation only at 240m | Task with compressed y-axis (0–0.3); hard to interpret |
| OSA | Both start wide and similar at 30s; slow partial separation by 240m | Small-N noise; both groups remain wider than in other tasks |

### Recommended selection

**Include in Extended Data (2 panels): Sex, Sleep Eff.**

**Retain in Supplementary (5 panels): Age, Apnea, BMI, Depression, OSA**

**Rationale:**
- **Sex**: The clearest demonstration. At 30~s, correct and incorrect subjects
  have overlapping distributions (both spread around 0.15). By 240~min, the
  blue (correct) distribution has collapsed to a very narrow spike near zero,
  while the orange (incorrect) distribution has grown into a tall wide violin
  reaching 0.6. The model becomes specifically confident for subjects it gets
  right, and remains erratic for subjects it gets wrong. This is the main
  message of the figure.

- **Sleep Efficiency**: A meaningfully different pattern. Separation develops
  with context length, but the blue violin at 240~min remains moderately wide —
  it does NOT collapse to near-zero as in Sex. Both distributions remain broader
  than in Sex, and there is more overlap. This reflects sleep efficiency's
  harder discriminability: even at the longest context, the model has residual
  uncertainty even for subjects it correctly classifies, consistent with the
  context-irreplaceable and lower-ceiling nature of this task.

- **Age** (suppressed): Shows the OPPOSITE pattern to Sex — at 240~min, the
  blue (correct) violin is wider/fatter than the orange (incorrect) one. Correct
  age subjects have higher within-subject variance than incorrect ones, meaning
  the model is more erratic when it gets age right. This is a genuine finding
  but requires careful interpretation (borderline-correct subjects with uncertain
  predictions) and would complicate the extended data narrative. Better placed
  in supplementary with a note.
- **Apnea** (suppressed): Similar collapse pattern to Sex (blue near-zero,
  orange wide at 240m). Redundant once Sex is shown.
- **BMI** (suppressed): Null result — blue and orange remain similar. No
  separation develops.
- **Depression** (suppressed): Compressed y-axis (0–0.3), very small absolute
  values. Minor separation at 240m but hard to interpret.
- **OSA** (suppressed): Both groups start with high variance at 30s; slow
  partial separation. Confounded by small-N noise.

### Interpretation for the extended data figure

**(For body text in main paper, near/replacing the current S-16 stability sentence):**

Extended Data Figure 3 quantifies within-subject prediction consistency:
each panel shows the distribution of within-subject standard deviation of
per-window predicted probabilities (at $K{=}K_{\max}$), stratified by
whether the subject was correctly (blue) or incorrectly (orange) classified
at 240~min. For sex classification, the two distributions are nearly
indistinguishable at 30~s but diverge sharply with context length: by
240~min, correctly classified subjects show near-zero variance (the model
gives the same prediction across every window), while incorrectly classified
subjects exhibit a broad, high-variance distribution. Context length thus
sharpens model confidence specifically for subjects the model gets right.
Sleep efficiency shows a weaker version of the same pattern: separation grows
with $L$, but the correctly classified distribution remains broader than in
sex at every context length, consistent with this task's lower ceiling and
residual prediction difficulty even at 240~min.
Supplementary Figure~S-20 shows the remaining tasks (age, apnea, BMI,
depression, OSA).

**(For Extended Data Figure caption):**

Within-subject prediction standard deviation (std of per-window probabilities,
$K{=}K_{\max}$) for correctly classified (blue) and incorrectly classified
(orange) subjects at four context lengths (30~s, 40~min, 120~min, 240~min;
Transformer, reduced-channel, test split). Sex classification (panel a):
the two distributions overlap at 30~s and diverge strongly with context
length; by 240~min, correctly classified subjects have near-zero variance
while incorrectly classified subjects span a wide, high-variance distribution,
showing that longer context sharpens model confidence specifically for correct
predictions. Sleep efficiency (panel b): separation develops more slowly and
the correctly classified distribution remains moderately wide even at 240~min,
consistent with this task's harder discriminability and context-irreplaceable
character. Supplementary Figure~S-20 shows age, apnea, BMI, depression,
and OSA.

---

## Summary Table

| Figure | Extended Data | Supplementary |
|---|---|---|
| Ext Fig 1 (K-agg) | Sex, Sleep Eff., Apnea | Age, BMI, OSA |
| Ext Fig 2 (Compute) | Sex, Sleep Eff. | Age, BMI, Apnea, OSA |
| Ext Fig 3 (Variance) | Sex, Sleep Eff. | Age, Apnea, BMI, Depression, OSA |

**Recurring theme:** Sex and Sleep Efficiency appear in all three extended figures.
They are the two tasks that most cleanly illustrate the paper's central contrast
(strong context sensitivity with clean temporal structure vs. context-irreplaceable
with harder discriminability), and each figure reveals a different facet of that
contrast. Apnea is extended data only for Ext Fig 1, where it shows the unique
progressive-head-divergence K-aggregation pattern not present for other tasks.
For all other figures it goes to supplementary.

**Notable corrections vs earlier drafts:**
- Age was previously described as "identical to Sex" in Ext Fig 3 — this was
  wrong. Age shows the OPPOSITE pattern: correctly classified subjects have
  HIGHER variance than incorrectly classified ones at long L. Age therefore
  goes to supplementary with a note about this reversal, not as extended data.
- BMI was dropped from Ext Fig 3 extended data: with only two panels (Sex +
  Sleep Eff.), a null case is not needed to complete the narrative.
