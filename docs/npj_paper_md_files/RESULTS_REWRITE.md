# Results Section Rewrite (2026-08-22, updated 2026-08-23)

**Formatting note (2026-08-23):** this file was rendering with large
stretches of text crossed out on GitHub. Cause: GitHub's Markdown treats
`~` as a strikethrough delimiter, and this file's body text is LaTeX
source, which uses `~` constantly as a non-breaking space (`Table~1`,
`30~s`, etc.) — GitHub was pairing up unrelated tildes across whole
paragraphs. Fix applied below: every paste-ready LaTeX section is now
wrapped in a fenced code block (which GitHub never applies inline
Markdown formatting to, so the `~` characters display literally and stay
byte-for-byte copy-paste-safe for `npj_main.tex`). The commentary/notes
sections below aren't meant to be pasted anywhere, so their few stray
tildes were just reworded away instead.

**2026-08-23 update**: incorporated a second round of your friend's
annotated-PDF comments. Three changes, all applied below: (1) the
scaling-law motivating sentence ("we then asked...") was reworded from a
question into a direct statement of practical motivation; (2) "iso-budget
substitutability" was dropped as a named label in the H2 figure
walkthrough (and its one back-reference in the Extended Data paragraph)
in favor of the plain-language description that was already sitting next
to it — "context-irreplaceable" and "context-dependent saturation speed"
were left alone since they're self-explanatory on their own, not compound
jargon; (3) the scaling-law paragraph was moved out of the H1 section and
into a new closing paragraph at the end of H4, right after the waterfall
gain-decomposition, where "is the gap worth its cost" is a fully
quantified question rather than an abstract one. See the Verification log
at the end for what was rechecked after this move.

**Second 2026-08-23 update**: a third round of comments, now integrated.
(1) H3 is reframed around *why* architecture should matter for context
handling (LSTM's sequential bottleneck vs. Transformer's direct
attention, moved forward from Discussion into the section's opening),
and the short-context Transformer-over-LSTM reversal is now explicitly
tied to that same mechanism rather than left unexplained. (2) The
sleep-efficiency context-irreplaceable finding now gets its practical "so
what" once, at its first mention in H2 (pointing forward to H4, where
"worth the cost" is fully quantified), instead of being re-explained at
its later Extended Data confirmation, which now just states that the same
ceiling holds for both heads. (3) The apnea Transformer-vs-LSTM
K-divergence sentence was investigated against the actual collected
results (`NSRR-tools/results/collected/phase0_v3/analysis.csv`) — the
underlying numbers check out exactly, but the interpretive claim didn't
actually explain anything, so it's commented out in place (not deleted)
with a note on what analysis would be needed to write a real explanation.
See the Verification log for the full number check.

**Third 2026-08-23 update**: fourth round, all in the H3 section.
(1) Title changed from "Temporal heads gain context-dependent advantage"
to "Architecture advantage grows with context length" — the old one
buried the actual finding behind a weak "gain...advantage" construction.
(2)–(3) H3 restructured so the confirm-verdict — sequence modelling wins
only when context does, and MeanPool is already good enough when it
doesn't — now *opens* the section instead of appearing at the end after a
block of numbers. This time nothing was cut further: the same
per-task numbers that survived the second-round trim are all still here,
just reordered so the surprising findings (the short-context LSTM-beats-
Transformer reversal, and BMI's architecture-barely-matters result) get
the emphasis instead of the expected "more context helps more" pattern.
(4) The Extended Data FLOPs paragraph, cut too hard in the second round,
is restored close to its original depth (both tasks' compute-range
percentages, the $1280\times$/$512\times$ per-step cost gap, the
quadratic-attention explanation) but reorganized into three short
paragraphs instead of one dense block, ending on the bridge into H4 that
your friend flagged as worth keeping.

**Not yet applied to `npj_main.tex`.** This is a full rewrite of the
Results section (currently lines 256 to 1099 or so — the paper has been
edited since, so treat this as approximate), for you to manually paste in
paragraph-by-paragraph so you can see exactly what changed. Nothing in
`npj_main.tex` has been touched.

## Why this rewrite

Your friend's annotated-PDF comments (repeated across several Results
subsections) were: prose spends too many words re-describing table
contents instead of interpreting them; too many numbers packed into
parentheses, one set per task, making the reader do the pattern-extraction
work the prose should be doing; tables/figures should carry the numeric
detail, prose should carry the finding and its meaning.

**Rule applied throughout**: each paragraph now opens with the
finding/pattern in plain language, cites at most one or two headline
numbers as evidence (usually the two extremes — largest and smallest
effect), and points to the table/figure for the rest. Nested
parentheticals explaining how to read a table (what a dash means, why a
column is unavailable at some contexts) were left in table captions,
where they already mostly lived, not repeated in body prose. Main-text
tables/figures are introduced by name ("Table 1 shows...", "as shown in
Figure 2...") rather than dropped in as a bare trailing citation — per
your instruction, only secondary/supplementary references stay in the
shorter parenthetical style.

**What was NOT changed**: the actual findings, directions, and
interpretations — including every physiological-mechanism sentence (why
apnea depends on respiratory channels, why sex depends on cardiac
channels, etc.), the H1–H4 confirm/reject verdicts, the scaling-law
result, the robustness checks, and the deployment-guidance conclusions.
These are preserved, just decluttered of redundant numeric listing
around them.

**Error found and fixed during verification** (see the Verification log
at the end): the current text states age-group's context gain as
$\Delta = {+}0.051$; cross-checked against both Table 1 (`tab:sweep`:
30-s $K_{\max}$ = 0.854, $L^*{=}120$min $K_{\max}$ = 0.902) and Table 2
(`tab:saturation`, which explicitly lists $\Delta = {+}0.048$ for age
group), the correct value is **$\Delta = {+}0.048$**, not $+0.051$. Fixed
below. This is the kind of number that's easy to mistype once and then
copy forward — worth double-checking in the current `.tex` too.

---

## Study overview

*(Kept close to the current version — this section is already a clean
factual walkthrough, not narrative-heavy, so it wasn't the target of your
friend's comments. Only light polish.)*

```tex
Four cohorts from the National Sleep Research Resource (NSRR)~\cite{nsrr}
were used: the Sleep Heart Health Study (SHHS)~\cite{shhs}, the
Osteoporotic Fractures in Men Sleep Study (MrOS)~\cite{mros}, the Apnea
Positive Pressure Long-term Efficacy Study (APPLES)~\cite{apples}, and the
Stanford Technology Analytics and Genomics in Sleep study
(STAGES)~\cite{stages}, comprising approximately 16,000 overnight PSG
recordings from adult participants across community and clinic settings
(Methods, \nameref{sec:datasets}). Seven prediction tasks were evaluated,
covering binary and multi-class labels derived from clinical instruments
and PSG-derived indices (Table~\ref{tab:tasks}; Methods,
\nameref{sec:tasks}). Five tasks (sleep efficiency, apnea, sex, age
group, and BMI) constitute the primary analysis; OSA severity and
depression screening are secondary analyses with smaller available
cohorts ($N_{\text{test}} < 250$), and their results should be
interpreted with corresponding caution.

Raw PSG signals were preprocessed uniformly across all cohorts via channel
harmonisation, bandpass filtering, resampling to 128~Hz, and z-score
normalisation, as shown in Figure~\ref{fig:preprocessing}a (Methods,
\nameref{sec:preprocessing}). Preprocessed signals were then passed to a
frozen SleepFM foundation model encoder, which maps each 5-s multichannel
patch to a 512-dimensional embedding, as shown in
Figure~\ref{fig:preprocessing}b (Supplementary~Figure~S-1; Methods,
\nameref{sec:sleepfm_encoder}). Three lightweight sequence classification
heads (Bi-LSTM, Transformer, and MeanPool) were trained on top of the
frozen encoder at six context lengths ($L \in \{30$~s, 10, 40, 80, 120,
240~min$\}$), with subjects partitioned 70/15/15 into training,
validation, and test sets, held fixed across all conditions (Methods,
\nameref{sec:preprocessing}). At inference, predictions are aggregated
across $K$ non-overlapping windows, swept from 1 to a subject-specific
maximum $K_{\max}$ (Methods, \nameref{sec:evaluation}). Full descriptions
of the cohorts, signal preprocessing pipeline, model architectures,
training configuration, and experimental design are provided in the
Methods section.

In the following sections, we work through H1–H4 in turn, followed by
secondary analyses. Unless otherwise stated, all primary results use the
Transformer head on the reduced-channel configuration (see Methods for
channel details).
```

---

## Context-length saturation is task-specific (H1)

```tex
Longer context improved performance across all five primary tasks (H1):
AUROC increased with $L$ in every case. The size of the gain and the
saturation point $L^*$ (defined as the smallest $L$ within 0.005 AUROC of
peak; Methods equation~\eqref{eq:lstar}) varied substantially across tasks
(Supplementary~Figure~S-4).

Table~\ref{tab:sweep} shows AUROC at every combination of training
context~$L$ and inference aggregation~$K$ ($K{=}1$, $K{=}5$, $K{=}K_{\max}$)
for all seven tasks, with each task's saturation context~$L^*$ marked in
bold. Table~\ref{tab:saturation} condenses this into a direct comparison
between the 30-s baseline and the saturation context, making both the
context gain (H1) and the aggregation gain (H2) readable at a glance.
Supplementary~Figure~S-6 translates the same results into a clinical
threshold map, showing the minimum context length each task needs to
first exceed a given AUROC target.

Context sensitivity varied more than fourfold across the five primary
tasks (all figures below at $K{=}K_{\max}$). Sleep efficiency depended on
context most strongly of any task: AUROC rose from 0.707 at 30~s to
0.831 at 240~min ($\Delta = {+}0.124$), still climbing at the longest
context evaluated, with no sign of saturation. Apnea detection showed the
second-largest gain ($\Delta = {+}0.103$, saturating at $L^*{=}120$~min),
followed by sex classification ($\Delta = {+}0.079$, $L^*{=}240$~min) and
age-group prediction ($\Delta = {+}0.048$, $L^*{=}120$~min). BMI
classification sat at the opposite extreme, gaining only
$\Delta = {+}0.030$ across the entire sweep ($L^*{=}240$~min nominally,
though the gain is negligible), demonstrating that not every clinical
label requires long temporal context. The LSTM head showed the same
ranking with consistently smaller gains (Supplementary~Tables~S-XI,
S-XII, and~S-XIII).

Among the smaller-cohort secondary tasks, depression screening
($N_{\text{test}} = 229$) showed no context benefit for the Transformer:
AUROC actually decreased monotonically beyond 30~s ($L^*{=}30$~s,
$\Delta = {+}0.000$), while the LSTM showed a small, non-monotonic peak at
10~min. Both patterns are most likely explained by sampling variability
in a small test set rather than a genuine effect. OSA severity
classification, by contrast, did benefit from longer context, saturating
at $L^*{=}80$~min for the Transformer (0.888, $\Delta = {+}0.098$) and
$L^*{=}40$~min for the LSTM; though this result, too, should be read
cautiously given its small test set ($N_{\text{test}} = 161$). Both
secondary tasks are broken down further, across all three heads, in
Supplementary~Table~S-XV.

To confirm these gains reflect real effects rather than sampling noise, we
computed 95\% bootstrap confidence intervals for AUROC at each task's
saturation context (Supplementary~Table~S-XIV; Methods,
\nameref{sec:evaluation}). For the large-test-set primary tasks, intervals
were narrow, confirming the observed gains are not sampling artefacts. Sex
classification at 240~min, for example, spanned just [0.894, 0.925]. For
the small-test-set secondary tasks, intervals were substantially wider
(${\sim}0.18$–$0.20$ AUROC units), so individual point estimates carry more uncertainty. Balanced accuracy at the
validation-optimised threshold and standard accuracy for all binary tasks
and heads are provided in the same supplementary table.

As a robustness check, we verified that each task's saturation pattern
held when the pooled model's test predictions were examined separately
within each contributing cohort (Supplementary~Section~S-IX;
Supplementary~Table~S-VIII; Supplementary~Figure~S-3). The qualitative
curve \emph{shape} was consistent across cohorts spanning a
${\sim}7.5\times$ size range for four of the five multi-cohort tasks, even
though absolute AUROC varied between cohorts (for example, APPLES apnea
ran 6--9 points above SHHS throughout, at the same rise-then-plateau
shape). Sleep efficiency was the only exception, and even there the
divergence was confined to a single cohort: SHHS and APPLES both continued
rising through 240~min, matching the pooled no-saturation pattern, while
MrOS alone plateaued by 120~min.

H1 is confirmed: context length is a primary performance driver, with
Transformer saturation points ranging from 120~min (age and apnea) to
240~min (sleep efficiency and sex); BMI formally reaches $L^*{=}240$~min
but with a negligible gain. We next examine how quickly performance
saturates with the number of inference windows at a fixed~$L$ (H2).
```

---

## Inference aggregation saturates quickly (H2)

```tex
Inference aggregation saturated after a small number of windows (H2).
Figure~\ref{fig:kvsk} shows AUROC versus inference window count~$K$ for
three representative tasks chosen to illustrate distinct saturation
behaviours (Supplementary~Figure~S-7 extends this to all five primary
tasks).

The three panels of Figure~\ref{fig:kvsk} reveal three qualitatively
different regimes. For age prediction (panel~a), context length and
inference aggregation became interchangeable once the context is long
enough: curves for $L \in \{80, 120, 240\}$~min converged tightly at
$K_{\max}$ (AUROC 0.900–0.905). The shorter contexts ($\leq$40~min) 
plateaued below this ceiling (0.85–0.88 at $K_{\max}$) and could not be 
traded off in the same way. Sleep
efficiency (panel~b) shows the opposite, context-irreplaceable regime:
the curves stayed separated, each context reaching a higher ceiling than
aggregation at a shorter one could match, though the two shortest, 30~s and
10~min, were an exception and plateaued together.
Sex classification (panel~c) shows a third regime: the number of windows
needed to saturate shrank dramatically as context grew, from roughly
$K \approx 1000$ at 30~s to just $K = 2$ at 240~min, reflecting how a
longer context captures more discriminative signal per window.

Table~\ref{tab:saturation} provides a compact cross-task summary: at each
task's saturation context~$L^*$, the gap between $K{=}1$ and $K{=}K_{\max}$
closed within a few windows. Where more than five windows fit
(40--80~min), five already recovered over 99\% of the full-night
($K{=}K_{\max}$) AUROC; at the longest contexts (120--240~min), only two
to four windows fit in a night, and those few were themselves already at
saturation. H2 is confirmed: a handful of inference windows is
practically sufficient at any $L \geq 40$~min.

At short contexts, aggregation helps only up to a ceiling set by the
context length itself, so it cannot recover the benefit of training on a
longer context. For sex classification, maximal aggregation at 30~s
(0.832 at $K{=}K_{\max}$) still fell below the 0.893 reached at 80~min
with only five windows (Supplementary~Table~S-V provides the full
$K$-grid; Supplementary~Table~S-IV compares mean-probability and
majority-vote aggregation for all tasks). Whether closing this gap by
training on a longer context is worth the added cost is quantified later,
in H4. With a few aggregation windows established as sufficient, the next
question is whether the sequence-head architecture changes how much
discriminative signal each context window yields (H3).

Extended Data Figure~\ref{fig:extkagg} repeats the $K$-aggregation
analysis with both sequence heads rather than the Transformer alone,
testing whether the regimes above hold for the LSTM as well. It shows the
three tasks whose two-head behaviour is most distinct: sex classification
and sleep efficiency, carried over from the main figure, and apnea
detection, whose head difference appears only once both heads are
compared. In sex classification, the number of windows needed to saturate
shrank as context grew for both heads, and the Transformer additionally
led the LSTM at every $K$. In sleep efficiency, the context-irreplaceable
pattern also held for both heads: even aggregating to $K_{\max}$ at a
short context still fell well short of a longer context, for either the
LSTM or the Transformer. Apnea detection reveals what a single-head figure
cannot: the two heads were nearly tied at $K{=}1$ (LSTM 0.645, Transformer
0.656) but the Transformer pulled progressively ahead as more windows were
aggregated, reaching a $+0.033$ lead by $K_{\max}$ (0.792 vs.\ 0.825).
Supplementary~Figure~S-21 shows the remaining tasks (age, BMI, OSA),
whose patterns are already established elsewhere in the main text.
```

---

## Architecture advantage grows with context length (H3)

```tex
Sequence modelling increasingly outperformed mean-pooling as context
grew, but only for tasks that benefited from longer context in the first
place (H3). The reason is architectural: mean-pooling averages the
per-window embeddings and discards their temporal arrangement, whereas the
LSTM and Transformer can model how the signal is distributed across the
recording. For a task whose signal lives in that temporal structure, this
advantage grows as the context lengthens; for a task where it does not,
mean-pooling is already good enough and the added complexity buys little.
To test this, we compared all three heads (LSTM, Transformer, and MeanPool) across all six context lengths and every task. Table~\ref{tab:heads} shows this comparison, with each head's own saturation context marked in the $L$ column.

For context-sensitive tasks, the Transformer's edge over MeanPool grew
monotonically with context length, from a small margin at 30~s to as much
as $+0.093$ AUROC at saturation for apnea, with sex classification close
behind and sleep efficiency somewhat lower ($+0.071$). The LSTM showed the same growth pattern,
confirming this is a property of temporal modelling in general, not of one
head. BMI is the clear counterexample: the Transformer's advantage over
MeanPool never exceeded $+0.031$ AUROC at any context length. With almost
no context-driven gain to capture in the first place ($\Delta = {+}0.030$
across the full sweep), there is no temporal structure for a sequence
model to exploit, so mean-pooling already captures nearly all the
available signal.

Between the two sequence heads, the Transformer further outperformed the
LSTM at long context, by as much as $+0.053$ AUROC for sex at its
saturation point, reflecting how the two handle long range: self-attention
connects any two patches directly, while the LSTM propagates information
step by step and can lose long-range dependencies as the sequence grows.
This edge is context-dependent, and at short contexts it even reversed:
the LSTM matched or exceeded the Transformer for BMI, apnea, and age at
30~s, again for BMI and age at 40~min, and for sleep efficiency at 10~min,
because there is little long-range structure for attention to exploit yet.
Sex classification is the only task where the Transformer led at every
context. The Transformer's advantage over the LSTM was not a fixed gap
present from 30~s onward. It emerged and increased only as context grew.
Table~\ref{tab:heads} reports the full per-task, per-context AUROC for all
three heads.

The two small-cohort secondary tasks mirrored the same split seen among
the primary tasks. OSA severity behaved like the context-sensitive tasks:
MeanPool was slightly ahead at short contexts, but the Transformer
overtook it from 40~min on and led by $+0.048$ at 80~min. It is consistent with
longer respiratory patterns in the signal that short windows miss.
Depression's behaviour was similar to BMI, showing no consistent advantage for either
sequence head at any context, its signal already captured in short-window
statistics (Supplementary~Table~S-XV).

Extended Data Figure~\ref{fig:extcompute} confirms this architecture
ranking is not an artifact of a single context length. For two tasks with
a strong and persistent Transformer advantage, sex and sleep efficiency,
it plots test AUROC against total training compute, measured in FLOPs as
defined in Methods, \nameref{sec:heads}, with a power-law fit per head. The head ordering
held across each head's compute range in both cases. For sex, the
Transformer reached 89.2\% AUROC at its highest compute, against 84.1\%
for the LSTM and 80.4\% for MeanPool; for sleep efficiency, the same
ordering held at 80.0\%, 76.6\%, and 74.0\%, with the Transformer's AUROC
rising most steeply with compute, as expected for the most
context-sensitive task.

This accuracy gap comes with a large cost difference. MeanPool's per-step
training cost scales only with the input embedding dimension, while the
LSTM and Transformer additionally pay for their hidden-state dimension
and, for the Transformer, the feedforward block and self-attention score
computation. At matched context length, MeanPool needs roughly
$1280\times$ less compute per step than the LSTM, and at least
$512\times$ less than the Transformer. This gap widens further with
context length as the Transformer's quadratic self-attention term comes
to dominate. Because the heads occupy different compute ranges, the figure
compares each within its own range rather than at matched compute, so it
is not a per-FLOP efficiency comparison. Supplementary~Figure~S-23 shows
the four remaining tasks, age, BMI, apnea, and OSA, where a similar 
pattern holds with smaller head separation.

The lower cost of Mean-pool is why it stays competitive for tasks
like BMI, where the extra accuracy a heavier architecture acheives is small.
This previews the resource trade-off the next section quantifies
directly: whether the accuracy gained from a more expensive head, or a
longer context, is actually worth what it costs.
```

---

## Iso-budget analysis: context versus aggregation (H4)

```tex
With aggregation adding little beyond a few inference windows (H2) and
temporal heads extracting more signal per window than simple averaging
(H3), a practical resource-allocation question follows: given a fixed
amount of overnight signal, is it better to train a longer-context model
that aggregates fewer windows, or a shorter-context model that aggregates
more? This fixed amount is the signal budget, context length $L$
times the number of aggregation windows $K$, a total recording time in
minutes rather than a measure of compute. At a fixed budget, the best
split between context and aggregation depended on both the budget and the
task (H4). We examine this in two steps: first, the full
context-by-aggregation grid locates where each strategy wins; second, the
Pareto frontier identifies the minimum budget each task needs.
```

### Budget-dependent crossover between context and aggregation

```tex
Figure~\ref{fig:heatmap} shows AUROC across the full $(L, K)$ grid for four
representative tasks, with context length $L$ down the rows, window count $K$ across the
columns, and each cell coloured by AUROC. Configurations on the same dashed
diagonal share a signal budget, so reading along one contour shows how
AUROC changes when a fixed budget is spent on longer context rather than on
more aggregation. Along the small-budget contours ($L \times K \leq
80$~min), AUROC peaked at short context with many windows, where aggregating
short windows matched or exceeded a single longer one. Along the
large-budget contours ($L \times K \geq 240$~min), the best configuration
depended on the task. Therefore, AUROC at a fixed budget does not
depend on the total budget alone but also on how it is split. 
This proved a partial rejection of H4. 
Table~\ref{tab:isocompute} lists the exact best AUROC and its optimal
$(L, K)$ at every budget for all five main tasks.

The best way to spend the budget split the tasks into three groups. Sex
and age favoured the longest affordable context at high budgets: at the
240-min budget, sex reached AUROC~0.898 with a single 240-min window, well
above the 0.851 the 120-min budget reached with three 40-min windows, so
the gain came from spending a larger budget on one longer window rather
than on more aggregation. Apnea and sleep efficiency instead favoured a
moderate-to-long context combined with several windows: apnea's best
strategy at the 240-min budget was three 80-min windows (0.822), beating
both a shorter, more-aggregated alternative and a single 240-min window,
and it still favoured multiple windows at the largest budget tested, so its
crossover to a single long window lies beyond 480~min. Sleep efficiency's
short-context aggregation saturated at a hard ceiling near 0.707 that only
a genuinely longer training context could break past. BMI gained little
from either lever, staying between 0.747 and 0.775 across every budget,
consistent with its context-insensitivity established above.
Supplementary~Figure~S-8 gives the heatmaps for the tasks
Figure~\ref{fig:heatmap} omits. Age showed the similar long-context dominance
as sex, while the two secondary tasks diverged. Depression was near-flat
across the grid and the highest performance values for each length 
achieved with one inference window. OSA severity was weak at the shortest context but rose sharply by 40--80~min.
```

### Pareto frontier and minimum deployment cost

```tex
Figure~\ref{fig:iso_main} illustrates AUROC against total budget for two
representative tasks. Each faded line is one training context traced across
aggregation counts. The solid frontier connects the best AUROC
reachable at each budget, labelled with the context that achieves it. For
sex classification (panel~a), the best strategy changed
with budget: the 30-s context with many windows won at small budgets, but
the 240-min context overtook it beyond about 240~min; so aggregation
substituted for context only up to that threshold. Sleep efficiency
(panel~b) was the opposite, with the longest context
holding the frontier at nearly every budget. So context length was
irreplaceable for this task. 
Supplementary~Figure~S-9 extends this
to the remaining tasks: age showed the same budget-dependent change as sex,
apnea was best at a moderate 80-min context across most budgets, BMI stayed
flat, and the two secondary tasks gave noisy frontiers.
Supplementary~Figure~S-10 reframes this as a cost question: the minimum
signal budget needed at inference to reach each AUROC target. 
A deployment decision grid is shown in Supplementary~Figure~S-11, which
makes it easier to visualize, for a given recording budget, the best
achievable AUROC and the $(L, K)$ configuration that reaches it.

Figure~\ref{fig:waterfall} breaks the total AUROC gain over a minimal
baseline (MeanPool, 30~s, $K{=}1$) into three additive steps:
inference-time aggregation, longer training context, and better head architecture.
The two representative tasks shown are opposites. 
For BMI, the performance gain came
mostly from architecture, with context adding very little. 
This was consistent with its iso-budget flatness we saw before. 
For sleep efficiency, context length
drove most of the gain, roughly 80\% of the total, confirming
longer training contexts as the primary driver for this task.
Supplementary~Figure~S-12 gives the same decomposition for sex, age, apnea,
depression, and OSA severity.
```

---

## Long-context performance follows a predictable scaling law

```tex
Sweeping to the longest contexts is expensive. the six context lengths
here span three orders of magnitude in training cost. Ideally, a task's
gain from long context could be predicted without running the expensive
training at all. For every task, AUROC rose with context length
along a smooth power-law curve that flattened toward a ceiling (Methods
equation~\eqref{eq:powerlaw}). Language-model scaling laws use the same
power-law form to predict a large model's performance from smaller, cheaper
runs. Here, the scaling axis is context length rather than model size.
Because that
shape is regular, we fit the curve to the four cheapest contexts (30~s to
80~min) and used it to predict the two most expensive, holding 120 and
240~min out of the fit. The predictions matched the held-out values to
within 1.4 AUROC points on average (Supplementary~Figure~S-5;
Supplementary~Section~S-XVII.A). 
Between the five main tasks, sleep efficiency was the one with slightly higher
error. This was expected: the AUROC for this task was still
rising with context at 240~min. Thus, the saturating fit underpredicted its actual AUROC, consistent with its context-irreplaceable behaviour.
For the other tasks, though, the runs using cheap short-context
closely anticipated the expensive long-context ones. This is an early sign that
scaling-law extrapolation extends beyond language modelling.

```

---

## Precision-recall validation

```tex
To confirm that the AUROC gains reported above are not artifacts of a
single classification threshold, Figure~\ref{fig:prcurves} shows
precision-recall curves at four context lengths for two representative
binary tasks. Apnea detection (panel~a) shows the
largest improvement in average precision (AP) among the primary tasks,
rising from 0.755 at 30~s to 0.865 at 120~min, with gains spread across
the full recall range rather than concentrated at extreme operating
points. The curve is nearly saturated by 120~min. Sleep efficiency and sex
show similar full-recall-range improvement pattern
(Supplementary~Figure~S-13). BMI (panel~b)
shows the opposite: near-flat AP across every context tested, with curves
that nearly overlap. This confirms its context-insensitivity holds in
precision-recall space as well as in AUROC. 
```

---

## Longer context improves prediction consistency, not just accuracy

```tex
Supplementary~Figure~S-20 is a heatmap for sex, BMI, sleep efficiency,
and apnea. The rows correspond to 300 randomly-drawn test subjects, grouped by their true
label (negative subjects together, positive subjects together, split by
a dashed line) and, within each group, ordered from most stable to most
variable prediction. Each column is a context length. Cell colour shows
the model's predicted probability that the subject is positive: pale
means uncertain, close to a coin flip; deep red means confidently
positive; deep blue means confidently negative.

For sex, sleep efficiency, and apnea, cells start out pale and mixed at
30~s, then deepen and become uniform within each true-label group as
context grows: the positive group turns a consistent deep red, the
negative group a consistent deep blue. The model becomes both more
confident and more correct with longer context. A handful of rows in the
negative group stay red at every context length. These are subjects the
model confidently misclassifies as positive no matter how much context
it receives. BMI shows little change. Its cells stay less confident
at every context length, matching its low context sensitivity.

Extended Data Figure~\ref{fig:extvariance} looks at this more closely,
for sex and sleep efficiency. For every subject, it takes that subject's
predicted probability in each inference window and computes how much it
varies across windows: a low value means the model gives nearly the same
answer every window, and a high value means the answer swings around.
Subjects are then split into two groups, correctly and incorrectly
classified at 240~min. Each group's spread of variance values is
drawn as a violin. A wide violin means that group covers a broad range of
variance, a narrow violin concentrated near the bottom means most
subjects in that group are highly consistent.
For sex, the two violins overlap at 30~s. By 240~min, the correct-subject
violin has collapsed to near zero, while the incorrect-subject violin
stays wide. Longer context therefore makes the model highly and
consistently confident specifically for the subjects it gets right.
Sleep efficiency moves in the same direction but does not reach the same
state: its correct-subject violin also narrows with context but never
collapses, staying moderately wide even at 240~min, so correctly
classified subjects retain some uncertainty across windows.
Supplementary~Figure~S-24 shows the violin plot for the rest of the tasks.
% remaining tasks apnea, depression,
% and OSA follow the same pattern. BMI shows almost no separation between
% the two groups; and age reverses it, with correctly classified subjects
% becoming as variable as incorrect ones at long context.
```

---

## Context needs do not follow task difficulty

```tex
How much a task gains from longer context does not follow from how difficult
the task is. Supplementary~Figure~S-14 shows each task's
context sensitivity, the AUROC it gains from 30~s to its best context,
against its baseline AUROC at 30~s. Age and sex were the easiest tasks at
30~s but gained only moderately, while sleep efficiency and apnea were among the
hardest and gained the most. BMI was about as hard as apnea yet gains almost
nothing, so a task's starting accuracy said little about how much context
would help it. The same pattern of task context-sensitivity vs difficulty 
was observed for both LSTM and Transformer. 
```

## Aggregate context-length scaling

```tex
Supplementary~Figure~S-16 summarises context sweeping with one curve per head,
each averaged over the seven tasks. Panel~a plots the mean gain in AUROC over
the 30~s baseline against context length, with a $\pm$1 SD band. 
All three heads have a positive slope. Thus on average across tasks longer
context improves AUROC, showing that temporal context carries real predictive signal.
The Transformer scales fastest, about $+0.79$ AUROC points per doubling,
roughly twice the LSTM's $+0.35$ or MeanPool's $+0.41$. The LSTM and MeanPool slopes are nearly identical here, within one SD. On this averaged metric, sequence modelling and plain averaging
look alike while they are not task by task. The LSTM beats MeanPool at long
context in every main task. The
wide SD shadow around the curves is itself the point: tasks scale so 
differently that any single averaged curve hides the task-specific saturation 
the rest of the paper relies on.
```

## Channel count expansion

```tex
Supplementary~Figure~S-17 overlays the saturation curves for each task across two channel sets using the Transformer head: the reduced 7--8 channels (the default configuration for all analyses in this paper) and the full ${\leq}23$ channels, which takes full advantage of SleepFM's channel capability (Methods, \nameref{sec:channel_comparison}). For apnea, BMI, and sex, the full set consistently improves AUROC across the sweep, with gains of approximately $0.03$--$0.05$ for apnea and BMI and a smaller gain for sex. In contrast, sleep efficiency and age show little change, while depression and OSA severity fluctuate in both directions. Given the small test sets for the latter two tasks ($N < 250$), these differences are more likely to reflect sampling variability than a systematic channel effect.

The full channel set roughly triples the number of channels and, consequently, the compute and memory required to encode each recording. This cost is justified primarily for the cardiorespiratory tasks: the reduced set already captures respiration and cardiac activity through a single airflow channel and a single cardiac channel, whereas the added effort belts, oximetry, and cardiac lead provide information that improves apnea, BMI, and sex prediction. Sleep efficiency and age gain little because the brain-activity channels relevant to these tasks are already included in the reduced set. Thus, the benefit of additional channels is task-dependent and arises when the reduced set omits channels containing information relevant to the prediction task.

```

---

## Modality group ablation

```tex
The modality-group ablation used the LSTM head with reduced-channel input at $L = 120$~min. For each task, the \textit{All-mod} condition (all four signal groups active) served as the reference. 25 experiments were performed in total: five main tasks $\times$ five ablation conditions (modality groups and conditions defined in Methods, \nameref{sec:modality_ablation}). Table~\ref{tab:modality} reports the complete results, with Supplementary Figures~S-18 and~S-19 providing bar-chart and radar-chart visualisations, respectively.

The ablation reveals task-dependent differences in the value of each modality. Apnea was most sensitive to the respiratory channel removal, which resulted in the largest single-group drop ($\Delta = {-}0.057$). Performance deteriorated further when only brain-activity channels were retained, producing the weakest condition overall ($\Delta = {-}0.103$). This pattern is consistent with the physiological basis of the task: apnea is identified through the occurrence and frequency of respiratory events, making respiratory signals more directly relevant than brain waves.

For sleep efficiency, brain-activity channels were the most important modality group ($\Delta = {-}0.083$). On the contrary, retaining brain-activity channels alone nearly matched the full-modality baseline, consistent with sleep efficiency being closely tied to EEG-based sleep staging. Cardiorespiratory channels alone performed worst ($\Delta = {-}0.111$), further indicating that the predictive signal for this task is predominantly neural rather than cardiorespiratory.

Sex classification showed a different pattern, with cardiac channels contributing slightly more than brain-activity channels ($\Delta = {-}0.074$ versus $-0.069$). This was also evident when considering each modality in isolation: cardiorespiratory channels alone reached higher AUROC than brain-activity channels alone. This is consistent with prior work showing that deep neural networks can predict sex directly from ECG waveforms with high accuracy, leveraging sex-related differences in ventricular rate, PR and QRS durations, QTc, and ST-segment morphology~\cite{siegersma2022}. SleepFM's cardiac representations plausibly capture some of this same sex-specific electrophysiological signal.

Age-group prediction also relied primarily on brain-activity channels, with their removal producing the largest performance drop ($\Delta = {-}0.046$). Removing either respiratory or cardiac channels had little effect. However, brain-activity channels alone did not fully recover the performance of the complete model. This 
suggested that age-related information was captured predominantly, but not exclusively, by brain activity.

BMI followed the same brain-activity-dominant pattern as age. The
cardiorespiratory-only condition performed worst ($\Delta = {-}0.081$), inferring that cardiorespiratory information alone cannot replace brain activity.
The results also point to a small role for EMG. Removing only brain
activity leaves EMG active and reaches 0.721 AUROC, but the
cardiorespiratory-only condition also zeroes EMG and drops further to
0.675. This fits a broader pattern in chin-EMG-based sleep classification:
one study found accuracy fell from 74\% in normal-weight subjects to
65--70\% in overweight and obese subjects~\cite{rehman2025}, so
muscle-activity channels may carry some BMI-relevant signal.
```

---

## Full-file re-review (2026-08-23)

Re-read the whole document end to end after the H3 restructuring, section
by section, checking two things: (1) that no section lost content it
shouldn't have, relative to what survived the second-round trim, and
(2) that every section follows the agreed structure (lead with the
finding, cite one or two headline numbers, point to the table/figure for
the rest; verdict/confirm statement stated once, not repeated).

- **Study overview**: unchanged, factual, no issue.
- **H1**: structure intact — opens with the finding, Table 1/Table 2 are
  properly introduced by name, per-task evidence capped at one number
  each, closes on the H1-confirmed verdict once. No content lost relative
  to the second-round version.
- **H2**: structure intact. The one change this round was adding the
  sleep-efficiency forward-pointer to H4 and lightening its Extended Data
  repeat — both already reviewed in the prior verification pass.
- **H3**: substantially restructured this round (see above) — verdict now
  leads, FLOPs paragraph restored to near-original depth but reorganized
  into three shorter paragraphs instead of one dense block. Checked that
  every number present in the second-round version is still present here:
  $+0.093$ (apnea), $+0.053$ (sex, Transformer-over-LSTM), $+0.031$ (BMI
  ceiling), the four short-context reversal tasks (BMI/apnea/age at 30 s,
  BMI/age at 40 min), OSA/depression small-N contrast, and now also the
  restored 89.2/84.1/80.4\% and 80.0/76.6/74.0\% compute-range figures and
  the $1280\times$/$512\times$ cost-multiplier numbers that Round 2 had
  cut — all present. One structural note: because the H3-confirmed
  verdict now opens the section, it is *not* separately restated at the
  end (Round 2's version had it appear once, at the end; this round moved
  it, it does not appear twice) — this is intentional, matching the "say
  it once" rule applied everywhere else, not an oversight.
- **H4**: unchanged this round, structure already reviewed in the prior
  pass (budget-crossover findings capped per task, Pareto/waterfall
  sections lead with pattern before numbers, scaling-law paragraph closes
  the section per the second-round move).
- **Precision-recall validation, Cross-task context sensitivity,
  Aggregate context-length scaling, Channel count expansion**: all four
  are short, supplement-driven sections with no numeric-litany problem to
  begin with; unchanged, still consistent with the rest.
- **Modality group ablation**: unchanged this round. Structure already
  matches the rule (each task's finding stated with its physiological
  interpretation attached, one headline $\Delta$ each, table carries the
  rest).

No section was found to need further trimming or restoration beyond the
H3 fixes made this round.

---

## Fifth-round review (2026-08-24) — completeness + cross-paper consistency

Full pass specifically checking (a) whether any *finding or interpretation*
(as opposed to a redundant number) was dropped in the earlier decluttering,
and (b) whether every per-task claim agrees across text, tables, and
figures. Three real issues found and fixed in the rewrite; one broader
tension flagged that also lives in the current `.tex`.

**Fixed in the rewrite:**

1. **Dropped finding restored (H3).** The original short-context
   Transformer-over-LSTM reversal list was: BMI/apnea/age at 30 s, BMI/age
   at 40 min, **and sleep efficiency at 10 min** (LSTM 0.717 vs. Transformer
   0.711, confirmed in Table 3). The earlier rewrite dropped the sleep
   efficiency case. That left an incompleteness, because the next sentence
   claims "sex is the only task where the Transformer leads at every
   context" — which is only true if every *other* task (sleep efficiency
   included) reverses somewhere. Restored the sleep-efficiency 10 min case
   so the enumeration actually backs the "sex is the only exception" claim.

2. **Cross-section inconsistency fixed (H3 vs. H1).** H3 said BMI
   "saturation occurs at just 10 min," but H1 (twice) and Tables 1–2 give
   BMI's Transformer $L^*$ as **240 min** (nominal, negligible gain). The
   10 min figure is the **LSTM's** $L^*$ (Table 3 marks BMI: LSTM $L^*$
   10 min, Transformer $L^*$ 240 min), so citing it inside a sentence about
   *the Transformer's* advantage contradicted H1. Reworded to drop the
   specific saturation point and instead say there is almost no
   context-driven gain to capture in the first place ($\Delta = {+}0.030$),
   which is the actual reason architecture doesn't help BMI and is
   consistent with H1. **Note for the live `.tex`:** the original text has
   this same "For BMI, where saturation occurs at 10 min" phrasing (and the
   Methods modality-ablation subsection also refers to "BMI's 10-min
   saturation $L^*$"). These aren't wrong for the LSTM, but they read as
   contradicting the Transformer-primary $L^*{=}240$ min stated in H1 —
   worth making consistent in the paper (either always qualify "BMI's LSTM
   $L^*$ of 10 min" or reword as done here).

3. **Imprecise claim tightened (H2 Extended Data).** The rewrite had said
   sleep efficiency's context-irreplaceable regime means "the same ceiling
   applies regardless of which head is used." The ceilings are not actually
   identical across heads (e.g. 40 min: LSTM 0.731 vs. Transformer 0.760).
   The real point — which the original conveyed with those numbers — is
   that *within each head*, short-context aggregation can't reach the
   long-context level. Reworded to state that per-head pattern instead of
   implying an identical ceiling.

**Deliberately-dropped numbers confirmed acceptable** (these are the
"declutter" removals the whole rewrite was for, not lost findings; the
tables/figures still carry them): the apnea and sleep-efficiency bootstrap
CIs in H1 (kept only the sex example); the "+0.056 at L=120min" sex example
in H2 Extended Data; per-task from→to Adv. ranges in H3 (kept apnea +0.093
and sex +0.053 as the two anchors); the age "both heads beat MeanPool at
80 min (0.890/0.900/0.843)" illustration in H3 (the general claim covers
it); various ΔAP numbers in Precision-Recall; and the exact 480-min-budget
AUROC values in H4.

**One dropped interpretation worth a second look (author's call, not
changed):** the original H4 apnea sentence ended "...suggesting the
crossover for this task lies above 480 min" — a small forward-looking
insight (apnea would likely keep benefiting past the tested budget range)
that the rewrite dropped along with the surrounding numbers. Not restored
because it's mildly speculative and tied to the specific budget figures
that were decluttered, but flagging it in case you want it back.

**Cross-paper consistency (text ↔ tables ↔ figures) — checked, no
contradictions beyond the BMI-$L^*$ item above:** verified each of the five
primary tasks tells a single coherent story across H1 (saturation), H2
(aggregation), H3 (architecture), H4 (budget), Precision-Recall, modality
ablation, and the variance figure. Sleep efficiency (context-hungry,
BAS/neural-driven, monotonic AP, weaker confidence-sharpening), apnea
(L*=120, respiratory-driven, largest AP gain, Transformer-favouring),
sex (context-hungry, cardiac-driven, sharpest confidence-sharpening,
only always-Transformer task), age (L*=120, BAS-driven, the variance
*exception*), and BMI (context-insensitive, modality-interaction,
architecture-doesn't-help) are each internally consistent across all
sections. Directions of every $\Delta$ and every task ranking match the
tables.

---

## Sixth-round audit (2026-08-24) — citation completeness + supp-figure number check

Full audit prompted by the per-cohort paragraph problem (a real analysis
that was described vaguely and cited nothing). Went through every
supplementary figure/table/section reference in the rewrite and checked
each number against the authoritative resolved values in
`npj_supplementary.aux` (not the hand-typed Contents list, which can be
stale). Two genuine citation **bugs** found, both of which also exist in
the live `npj_main.tex` and should be fixed there too:

1. **Modality figures mis-numbered (also wrong in `npj_main.tex` line
   1043–1044).** The modality-ablation paragraph cited "Supplementary
   Figures S-16 and S-17 ... bar chart and radar chart." But per the
   `.aux`, **S-16 is Aggregate Context-Length Scaling** and **S-17 is
   Channel Count Expansion** — two entirely different figures. The
   modality bar chart is **S-18** (`fig:supp-ablation`) and the radar is
   **S-19** (`fig:supp-radar`). Fixed to S-18/S-19 in the rewrite.

2. **Extrapolation reference off by one (also wrong in `npj_main.tex`
   lines 455 and 2092).** The scaling-law paragraph cited "Supplementary
   Section S-XVIII.A." The extrapolation subsection
   (`sec:supp-extrapolation`) actually resolves to **S-XVII.A**, and there
   is also a dedicated figure for it, **S-5** (`fig:supp-extrapolation`),
   that was not cited at all. Fixed to cite both Figure S-5 and Section
   S-XVII.A.

Missing-citation gaps fixed:

3. **Per-cohort robustness paragraph** now cites Supplementary Section
   S-IX, Table S-VIII, and Figure S-3 (previously cited nothing;
   verified S-IX and S-3 against the `.aux`). The vague MrOS wording that
   read as contradicting the sleep-efficiency headline was also fixed (see
   Fifth-round note and the main answer).

4. **H1 secondary tasks (depression, OSA)** now point to Supplementary
   Table S-XV for the per-head breakdown. (The OSA *label definition* was
   deliberately **not** pointed to the supplement, because it lives in the
   main paper — Table `tab:tasks` and Methods "Clinical Prediction Tasks";
   an earlier draft of this pointer wrongly attributed it to S-VII and was
   corrected.)

Everything else verified **correct** against the `.aux`: S-1, S-4, S-6,
S-7, S-9, S-10, S-11, S-12, S-13, S-14, S-15, S-16, S-17, S-20, S-21,
S-23, S-24, and the section numbers S-IX and S-XV all match their resolved
values. No other reference in the rewrite is mis-numbered.

**Note on method:** these numbers were checked against the compiled
`.aux`, which reflects the supplementary as it currently compiles. If the
supplementary is edited (sections/figures added, removed, or reordered)
the numbers can shift again; re-run the `.aux` check before final
submission rather than trusting this list.

---

## Verification log

Numbers and claims in this rewrite were checked against Tables 1–4
(`tab:sweep`, `tab:saturation`, `tab:heads`, `tab:isocompute`) and
`tab:modality` as they currently stand in `npj_main.tex`, across multiple
re-read passes:

- **Fixed**: age-group context gain, $\Delta$, corrected from the current
  text's $+0.051$ to the table-verified $+0.048$ (see note near the top
  of this file).
- **Fixed (own error, caught on pass 2)**: the H3 paragraph originally
  claimed the Transformer-over-MeanPool advantage grew "most dramatically
  for sleep efficiency." Checking Table 3: sleep efficiency's advantage at
  $L^*$ is $+0.071$, but apnea's is $+0.093$ and sex's is $+0.092$ — sleep
  efficiency is actually the *smallest* of the three at saturation, not
  the largest. Reworded to correctly cite apnea ($+0.093$) as the largest,
  with sleep efficiency and sex named as "similarly large" rather than
  singled out.
- **Fixed (pre-existing error in `npj_main.tex` itself, not introduced by
  this rewrite)**: the current text (around line 767) describes
  age-group's 240-min-budget result as "0.893 at 240-min, $K{=}2$."
  Table 4 (`tab:isocompute`) actually lists this configuration as
  **120m, $K{=}2$** (120 times 2 equals 240, so the budget total is
  right, but the context length is 120 min, not 240). This rewrite does
  not cite age-group's specific $(L,K)$ figure at all, so it doesn't
  repeat the error — **but that original sentence in the current `.tex`
  should be corrected separately** ("240-min, $K{=}2$" to "120-min,
  $K{=}2$") since it's a factual mismatch with its own table.
- **Fixed (own near-error, caught on pass 2)**: the sex-classification
  budget-crossover paragraph originally implied 0.898 (240m, $K{=}1$) and
  0.851 (40m, $K{=}3$) were both "at the same total budget," mirroring
  how the current `.tex` phrases it. But 40 times 3 is 120, not 240 —
  0.851 is Table 4's 120-minute-budget answer, not a second option
  *within* the 240-minute budget. Reworded to correctly attribute each
  number to its own budget row (240-min budget leads to 0.898; 120-min
  budget leads to 0.851), avoiding the same ambiguity present in the
  current `.tex`'s phrasing of this comparison.
- **Checked and confirmed correct as originally stated**: sleep efficiency
  ($\Delta{=}{+}0.124$, $L^*{=}240$m), apnea ($\Delta{=}{+}0.103$,
  $L^*{=}120$m), sex ($\Delta{=}{+}0.079$, $L^*{=}240$m), BMI
  ($\Delta{=}{+}0.030$, $L^*{=}240$m nominal) — all match Table 2 exactly.
  Sex 30-s $K{=}1$-to-$K_{\max}$ (0.695 to 0.832) matches Table 1. Table 3
  Transformer-over-MeanPool figures ($+0.013$ to $+0.071$ sleep
  efficiency, $+0.093$ apnea, $+0.092$ sex) and the sex-classification
  Transformer-over-LSTM gap ($+0.053$ at $L^*$) all match Table 3 exactly.
  Table 4's 240-min-budget row (sex 0.898 at 240m/K1; apnea 0.822 at
  80m/K3) and 120-min-budget row (sex 0.851 at 40m/K3) both match,
  correctly attributed above. Modality-ablation deltas (apnea RESP
  $-0.057$; sleep efficiency BAS $-0.083$; sex EKG $-0.074$ vs. BAS
  $-0.069$; age BAS $-0.046$; BMI cardio-only $-0.081$) all match
  `tab:modality` exactly.
- **Restored this round, re-verified against the original `.tex` text**:
  the H3 FLOPs paragraph's compute-range percentages (sex: Transformer
  89.2\%, LSTM 84.1\%, MeanPool 80.4\%; sleep efficiency: Transformer
  80.0\%, LSTM 76.6\%, MeanPool 74.0\%) and per-step cost multipliers
  ($1280\times$ MeanPool-vs-LSTM, $512\times$ MeanPool-vs-Transformer) —
  these are the exact figures from the pre-rewrite original text, not new
  numbers, restored after being cut too aggressively in the second round.
- **Not independently re-derived**: supplementary figure/table
  characterizations (S-4 through S-24, and supplementary tables S-IV,
  S-V, S-XI through S-XV, S-XVIII) were carried forward from the
  existing, previously-verified main-text descriptions rather than
  re-checked against the supplementary file itself in this pass — these
  were already audited for accuracy in an earlier session (the "full
  supplementary audit" phase). If any of those supplementary
  figures/tables have changed since then, those specific sentences should
  be spot-checked before pasting in.
- **One more recommendation, not a factual issue**: consider
  double-checking the live `.tex` for the same age-group $\Delta = 0.051$
  typo in any *other* location it might appear — I grepped for it and it
  only occurs once (the spot fixed here), but it's worth a second look if
  you edit the Discussion's per-task walkthrough too, since copy-paste
  errors like this one sometimes propagate.
- **Re-check after moving the scaling-law paragraph to H4**: the
  paragraph's own numbers (four cheapest contexts $\leq 80$ min,
  predicting 120/240 min to within 1.4 AUROC points, Supplementary
  Section S-XVIII.A) are unchanged from the original — only their
  surrounding motivation and location moved, not the claim itself, so no
  new number was introduced by the move. The new opening sentence
  ("this is exactly the gap...large for sleep efficiency...negligible for
  BMI") reuses the waterfall paragraph's own already-verified contrast
  (BMI: architecture dominates, context near 0\%; sleep efficiency:
  context roughly 80\% of total gain) rather than asserting a new
  comparison, so it's grounded in the immediately preceding,
  already-checked sentence rather than a fresh claim.
- **Apnea K-divergence sentence investigated directly against collected
  results** (`NSRR-tools/results/collected/phase0_v3/analysis.csv`,
  `apnea_binary`, context=40m, both heads, `mean_prob_auroc` column). Full
  K-sweep pulled and checked: LSTM/Transformer AUROC at $K{=}1$ is
  0.6446/0.6556 (paper says 0.645/0.656 — matches); at $K{=}12$,
  0.7920/0.8231 (paper's "0.792" for LSTM matches exactly); at
  $K{=}16$ through $25$ (roughly $K_{\max}$), Transformer reaches
  0.8248 to 0.8249 (paper's "0.825" matches). The gap grows from 0.011 at
  $K{=}1$ to roughly 0.032 to 0.034 by $K$ around 8 to 12 and then
  plateaus — a real, roughly monotonic trend (one small dip at $K{=}6$),
  not sampling noise. **Conclusion: the factual claim is fully verified
  and now stated with its numbers in the main text.** The
  *interpretation* ("indicating its architectural advantage is only
  fully expressed once several windows are aggregated") was a different
  matter — checked and found to just restate the observed pattern rather
  than explain it, so per the author's decision it's commented out in
  place rather than deleted, with a note on what analysis (per-window
  prediction-variance comparison between heads, requiring raw prediction
  parquets that exist only on the cluster, not locally) would be needed
  to write a real one.
