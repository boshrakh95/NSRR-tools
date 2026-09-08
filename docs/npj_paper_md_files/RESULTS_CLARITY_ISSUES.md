# Results Clarity Issues — Notation and Comprehension

Analysis of Introduction + Results for a reader proceeding linearly through the paper
(Introduction → Results) without reading Methods.
Issues ranked by severity; each entry gives location, problem, suggested fix, and
implementation status.

---

## CRITICAL — Would confuse or block the reader

---

### C1. `$\Delta$` undefined at first use ✅ FIXED

**Location:** `tab:sweep` caption (line 283); prose body (line 344)

**Problem:** The tab:sweep caption ended with "Tasks sorted by $\Delta$ descending" but
never defined $\Delta$. The body text immediately after the table used "$\Delta = {+}0.124$"
as the first prose occurrence. The definition only appeared later in the tab:saturation
caption: "$\Delta$: AUROC at $L^*$,$K_{\max}$ minus AUROC at 30s,$K_{\max}$."

**Fix applied:** Added full definition to the tab:sweep caption:
> "$\Delta$: AUROC gain from 30~s to $L^*$ at $K{=}K_{\max}$; tasks sorted by $\Delta$ descending."
Also extended "---" definition in the same caption to "$K{=}5 > K_{\max}$ (unavailable)".

---

### C2. Sleep staging absent from all Results tables — DEFERRED BY USER

**Location:** Results overview, H1 opening, `tab:sweep`

**Status:** Kept as-is per user instruction. Sleep staging experiments are still running;
the section will be updated (or the task removed) once results are available.

---

### C3. "Five primary tasks" unexplained at H1 opening ✅ FIXED

**Location:** Study overview, line 239–241; H1 section opening

**Problem:** Paper stated seven tasks, but H1 opened with "five primary tasks" with no
explanation of which five or why two were excluded.

**Fix applied:** Added two sentences to the overview paragraph after the task-count sentence:
> "Five tasks (sleep efficiency, apnea, sex, age group, BMI) constitute the primary
> analysis; OSA severity and depression screening are secondary analyses with smaller
> available cohorts ($N_{\text{test}} < 250$), and their results should be interpreted
> with corresponding caution."

---

### C4. `tab:modality` placed before BAS / RESP / EKG / EMG are defined ✅ FIXED

**Location:** `tab:modality` footnote (line 794)

**Problem:** The table column headers used BAS, RESP, EKG, EMG before these acronyms
were defined in the modality ablation running text (several pages later).

**Fix applied:** Expanded the table footnote to define all four acronyms at the point of
first use in the table:
> "BAS: brain activity signal group (EEG + EOG); RESP: respiratory channels;
> EKG: cardiac channels; EMG: electromyography."

---

## MAJOR — Likely to cause confusion or misreading

---

### M1. `$t^*$` undefined at first use ✅ FIXED

**Location:** Results H1 section, first use of $t^*$

**Problem:** "validation-optimised threshold~$t^*$" appeared without definition; formal
definition is in Methods.

**Fix applied:** Added parenthetical at first use:
> "Balanced accuracy at the validation-optimised threshold~$t^*$ (probability cutoff
> maximising balanced accuracy on the held-out validation set; Section~\ref{sec:evaluation})"

---

### M2. `tab:heads` bold rows ambiguous — replaced with per-head symbols ✅ FIXED

**Location:** `tab:heads` caption and table body; H3 running text

**Problem:** Bold rows marked LSTM $L^*$ but caption only said "Bold rows: saturation
context~$L^*$" — ambiguous since Transformer is the primary head. For sex and BMI, bold
rows did not match Transformer $L^*$. Also, the age bold row (80 min) was incorrectly
marking the LSTM/Transformer reversal point, not the true $L^*$ (120 min for both heads).

**Fix applied:**
- Removed all bold formatting from table rows.
- Added superscript markers in the $L$ column: `$^{\ast}$` = LSTM $L^*$, `$^{\dagger}$` = Transformer $L^*$.
- Per-task markers: Sleep eff 240 min ($^{\ast\dagger}$); Apnea 120 min ($^{\ast\dagger}$);
  Sex LSTM 120 min ($^{\ast}$) + Transformer 240 min ($^{\dagger}$);
  Age 120 min ($^{\ast\dagger}$); BMI LSTM 10 min ($^{\ast}$) + Transformer 240 min ($^{\dagger}$).
- Updated caption to define the markers.
- Updated running text: removed "highlighted in bold" phrasing; updated sex Adv. reference
  from 120 min to Transformer $L^*$ = 240 min (+0.092); updated Transformer-leads-LSTM
  margin for sex from +0.033 (at LSTM $L^*$) to +0.053 (at Transformer $L^*$ = 240 min).
- Removed erroneous "($L^*$)" from the age reversal sentence (80 min is the reversal
  point, not $L^*$; age $L^* = 120$ min for both heads).

---

### M3. Training vs inference `$K$` distinction was commented out ✅ FIXED

**Location:** H2 section (lines 394–399, now active)

**Problem:** Block explaining that $K$ is inference-time aggregation, distinct from
training's $w{=}5$ overlapping windows, was commented out.

**Fix applied:** Uncommented and lightly rewritten as active prose:
> "$K$ here denotes the inference-time aggregation count: non-overlapping windows averaged
> per subject at prediction time, swept post-hoc from stored predictions with no additional
> GPU cost. This is distinct from the training protocol, which fixed $w{=}5$ overlapping
> windows per subject per epoch."

---

### M4. `$K_{\max} = 996$` for depression unexplained ✅ FIXED

**Location:** `tab:saturation`, Depression row footnote

**Problem:** Depression row showed $K_{\max} = 996$ with no explanation; looked like a typo.

**Fix applied:** Uncommented and expanded footnote:
> "For depression, $L^*{=}30$~s (no context benefit), so both blocks are identical and
> $\Delta{=}0$; $K_{\max}{\approx}996$ reflects the ${\approx}996$ non-overlapping
> 30-s windows in a typical 8-hour recording."

---

## MINOR — Small gaps that a careful reader would notice

---

### m1. Study overview omitted inference aggregation `$K$` ✅ FIXED

**Location:** Study overview paragraph

**Fix applied:** Added sentence after the training context sentence:
> "At inference, predictions are aggregated across $K$ non-overlapping windows
> ($K$ swept from 1 to a subject-specific maximum $K_{\max}$; Section~\ref{sec:evaluation})."

---

### m2. `tab:saturation` "---" entries not explained in that table's caption ✅ FIXED

**Location:** `tab:saturation` caption

**Fix applied:** Added to the caption:
> "---: $K{=}5 > K_{\max}$ at this $L^*$ (unavailable)."

---

### m3. "Supplementary Figs." abbreviation ✅ FIXED

**Location:** Modality ablation section, line 897 (renumbered)

**Fix applied:** Changed "Supplementary~Figs.~S-14 and~S-15" to
"Supplementary~Figures~S-14 and~S-15".

---

### m4. "Prediction stability" undefined ✅ FIXED

**Location:** Modality ablation section, last sentence

**Fix applied:** Added inline gloss:
> "per-subject prediction stability (consistency of individual predicted probabilities
> across repeated inference windows)"

---

## Cross-reference review ✅ CLEAN

Verified all `\ref{}`, `\eqref{}` calls in Results and Methods:
- All figure labels (`fig:preprocessing`, `fig:kvsk`, `fig:heatmap`, `fig:iso_main`,
  `fig:waterfall`, `fig:prcurves`, `fig:sweep`) exist and are referenced correctly.
- All table labels (`tab:sweep`, `tab:saturation`, `tab:heads`, `tab:isocompute`,
  `tab:modality`, `tab:tasks`) exist and are referenced correctly.
- All section labels exist (`sec:datasets`, `sec:preprocessing`, `sec:tasks`,
  `sec:sleepfm_encoder`, `sec:evaluation`, `sec:training`, `sec:heads`,
  `sec:channel_comparison`, `sec:modality_ablation`, `sec:sweep`, etc.).
- Results → Methods references all point to the correct methodology sections.
- Methods → Results references are limited to `fig:preprocessing` (appropriate).
- Discussion → Results references are `fig:waterfall` and `tab:heads` (appropriate
  back-references).
- No broken or misleading cross-references found.

**Note:** `Supplementary~Figure~S-X` in the overview remains a placeholder — update
the hardcoded "S-X" once the SleepFM architecture figure is added to the supplementary.
