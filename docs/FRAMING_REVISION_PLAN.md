# Framing & Positioning Revision Plan — Abstract, Introduction, Discussion

**Location note:** this file discusses and quotes `npj_main.tex` /
`npj_supplementary.tex`, which live in the separate `npj_digital_medicine_submission`
repo at `/Users/boshra/NSRR-workspace/npj_digital_medicine_submission/`. It was
moved here (2026-07-31) so it stays git-tracked without being visible in that repo,
which the supervisor has access to. Line numbers below were accurate as of the move
and will drift as the paper is edited further — treat them as approximate.

Written in response to supervisor feedback: the paper currently reads as "we tried
different lengths and measured accuracy" rather than as a study with a clear,
consequential, professionally-framed motivation. This is a **positioning** problem,
not a **results** problem — every analysis needed to support a stronger story
(iso-compute frontier, deployment budgets, compute cost, extrapolation) already
exists in the paper or supplementary. The fix is to change what gets said first,
in what words, and in what order — not to run new experiments.

This document does not edit `npj_main.tex` / `npj_supplementary.tex` directly.
It is a set of concrete, line-referenced suggestions plus drafted replacement text
for you to adapt and paste in.

**Update:** the supervisor's first-round written comments have been received and
incorporated. Section 10 (near the end of this document) is a dedicated point-by-point
response to those comments — including two abstract drafts the supervisor provided,
a question about whether the H1–H4/contributions-list format matches npj convention,
and three substantive scientific questions (encoder generalizability, training-signal
exposure across context lengths, and the 240-min context cap). One of the supervisor's
questions (the token-budget sensitivity experiment) surfaced what looks like a real,
pre-existing factual gap in the current draft, not just a framing issue — see §10.4.
Sections 5 and 6.3 below now carry short pointers to §10 where the supervisor's
comments supersede or refine the original suggestion.

**Round 1 implemented (2026-07-31):** the user approved all of the above and asked
for it to be implemented directly in `npj_main.tex`, with one substantive pushback
on §10.3 (whether the frozen-encoder dependence is really a "limitation") and two
concrete decisions on §10.4 (no token-budget experiment was ever run) and §10.5
(consolidate the 240-min cap to exactly two reasons: cohort consistency, GPU memory).
Every section below now ends with an **"Implemented (round 1)"** note recording what
actually changed in `npj_main.tex` and its current state. `npj_main.tex` was
recompiled cleanly after all changes (39 pages, no new errors/warnings, no undefined
references).

---

## 1. Diagnosis: why it currently reads as "just a comparison"

1. **The motivating stakes are never stated up front.** The Introduction (line 111
   onward) opens with what PSG *is* and what prior work did, but never says, in
   plain terms, *why a reader should care about context length* beyond "it hasn't
   been studied." There is no sentence anywhere in the Introduction that says
   "getting this wrong costs you X" before the Results prove it.
2. **The cost/compute story is real but hidden in Results and Supplementary, not
   claimed as a contribution.** The iso-compute analysis (`sec:results_isocompute`,
   line 688), the Pareto/deployment-cost framing (line 815), the compute-scaling
   figure (Extended Data Fig. 2, line 657), and the supplementary deployment grid
   (`sec:supp-deployment`) already answer "how do I spend a fixed compute/recording
   budget optimally?" — a resource-allocation question, not a length comparison.
   None of this vocabulary appears in the Abstract or Introduction's framing of the
   problem; it only shows up once the reader is already deep in Results.
3. **"Length" is used as the headline word almost everywhere**, including in
   sentences that are actually about a design/engineering decision (how much signal
   budget, GPU memory, and wall-clock time to allocate per subject). "Length" reads
   as a passive, descriptive measurement; "temporal context" or "context budget"
   reads as a variable someone actively *chooses* under constraints — which is the
   real story.
4. **The contributions list (line 205) undersells the paper.** It lists
   "characterization" and "an iso-compute analysis showing X and Y are not
   equivalent" — accurate, but phrased as an observation, not as a decision-support
   result. It also omits the extrapolation finding (line 417–423: cheap short-context
   runs forecast expensive long-context AUROC to within ~1.4 points) — a genuinely
   "advanced-sounding," scaling-laws-style result that is currently buried mid-Results
   and never mentioned in Abstract, Introduction, or Discussion.
5. **The Discussion's practical payoff is present but late and diluted.** The
   sentence that comes closest to your supervisor's ask — "a lightweight model
   trained on brief windows with cheap test-time aggregation may closely approximate
   the performance ceiling at substantially lower GPU memory cost" — is one clause
   inside a long paragraph at line 1135–1137, not a headline claim.

**Bottom line:** the paper has already done the work to sound like a resource-allocation
/ compute-efficiency study. It just doesn't *say so* until the reader is well past the
Abstract.

**Implemented (round 1):** all five diagnosed gaps are now addressed directly in
`npj_main.tex` — see §5–§7 below for the specific edits. Point 4 (missing
extrapolation finding) is now stated in the Abstract, the Introduction's
contributions paragraph, and the Discussion's new deployment-guidance paragraph.
Point 5 (buried practical payoff) is now a full paragraph near the top of Discussion
with concrete deployment numbers, not a single clause.

---

## 2. The core reframe

Stop narrating this as: *"We compared model performance at six different context
lengths."*

Start narrating this as: *"Every PSG prediction model implicitly commits to a
temporal context budget — how many minutes of an overnight recording it consumes per
prediction — and that choice is not free: it fixes GPU memory footprint (attention
scales quadratically with context), training wall-clock time (up to ~12× longer at
the longest context we tested — 653 vs. 54 minutes for the same task, Methods/Supp
Table S-XIII), and inference cost, while an unjustified default risks either paying
for capacity a task does not need or silently under-provisioning a task that does.
We treat context length as a first-class, task-specific resource-allocation
variable and derive the cost-performance frontier that lets a practitioner choose
it deliberately rather than by convention."*

This reframe is not a rewrite of the science — it's the existing iso-compute /
deployment-budget / compute-scaling material, promoted from "secondary analysis" to
"the point of the paper," and stated in the first two paragraphs instead of the last
two sections.

**Implemented (round 1):** this exact reframe now opens the Introduction. A new
paragraph was inserted after the first background paragraph stating that every PSG
model "implicitly commits to a temporal context budget," citing the quadratic
attention/GPU-memory cost and the ~12$\times$ training-time ratio (653 vs. 54 min,
Supplementary Table S-XIII) as concrete evidence this is not free. The Discussion's
opening paragraph now carries a matching sentence, and the Conclusion's opening and
closing sentences were rewritten around the same "resource-allocation variable, not
implementation detail" framing.

---

## 3. Terminology swap

Apply this consistently in Abstract, Introduction, and Discussion prose (not
necessarily in Methods/Results where `context length ($L$)` is precise mathematical
notation tied to the formalism — keep that as-is, it's correct and unambiguous
there).

| Instead of (bare, passive) | Say (active, framed as a decision) |
|---|---|
| "context length" (as a standalone noun in narrative prose) | "temporal context," "temporal context budget," "observation window budget" |
| "we studied/compared different lengths" | "we characterize the cost-performance frontier of temporal context," "we quantify how much temporal context each task requires and at what cost" |
| "context length affects performance" | "temporal context is a consequential, non-free design variable" |
| "aggregation vs. long context" | "the context–aggregation resource-allocation trade-off," "the compute-budget allocation problem" |
| "sweep across six context lengths" | "a systematic characterization of the temporal-context/compute frontier" (keep "sweep" only in Methods, where it is a defined protocol term) |
| "our findings show X saturates" | "our findings quantify the point of diminishing returns" / "the minimum sufficient temporal budget" |

Do **not** eliminate "context length" entirely — `$L$` is your defined variable and
npj readers doing quantitative work will want the precise term. The point is to lead
each section with the framed language, then use precise notation once the reader is
oriented, exactly as good ML papers do with "compute-optimal" (Chinchilla) framing
before dropping into `N`, `D`, `C` notation.

**Implemented (round 1):** applied throughout the new/rewritten Abstract,
Introduction, and Discussion prose — "temporal context," "context budget," and
"cost-performance frontier" are now the lead terms, with `$L$`/`context length`
retained as precise notation once introduced. Keywords (line ~85) were also updated:
`temporal context, computational efficiency, deep learning, foundation models,
polysomnography, sleep` (previously `context length, deep learning, foundation
models, polysomnography, sleep, temporal modelling`).

---

## 4. Title (line 49–50)

Current: *"Context Efficiency in Overnight Polysomnography: Task-Specific Saturation
Points for Foundation Model-Based Clinical Prediction."*

This is already decent — "Context Efficiency" is the right instinct and should stay.
"Saturation Points" is the weak half; it's descriptive/statistical rather than
consequential. Two directions, pick one:

- **Minimal edit** (keeps structure, swaps second half toward decision-relevance):
  *"Context Efficiency in Overnight Polysomnography: Task-Specific Cost–Performance
  Frontiers for Foundation Model-Based Clinical Prediction."*
- **Leads harder with the resource framing**:
  *"How Much of the Night Do You Need? Task-Specific Temporal Budgeting for
  Foundation Model-Based Polysomnography Prediction."*

Either is a small change; do not over-invest here relative to Abstract/Intro/Discussion.

**Implemented (round 1): deferred, not changed.** The user's round-1 instructions
focused on Abstract/Introduction/Discussion content; the title was left as-is
(`Context Efficiency in Overnight Polysomnography: Task-Specific Saturation Points
for Foundation Model-Based Clinical Prediction`). Still an open, low-cost option for
a later pass — the "Minimal edit" direction above is the more conservative of the
two and would take one line change whenever this is revisited.

---

## 5. Abstract (line 87–104)

> **Superseded by §10.1.** The supervisor independently sent two draft abstracts that
> converge with the diagnosis below (stakes-first framing, "temporal context" not
> "length"). §10.1 evaluates both, recommends adopting a trimmed version of the
> supervisor's second draft as the primary base, and folds in the extrapolation
> finding this section originally proposed. Read this section for the reasoning, but
> use §10.1's text as the actual draft to work from.

The current abstract (the "longer but better version," already the active one) is
factually complete but narrates in the diagnosed "comparison" register: it opens with
"PSG encodes markers... yet models have treated context length as a fixed
implementation detail" (good instinct, but immediately drops into "we present a
systematic study" — passive/descriptive) and lists three *patterns* rather than
leading with *why the patterns matter* or *what a reader should do differently*.

**Suggested structure** (4 moves, in order):
1. One sentence: PSG prediction models must commit to a temporal context, and that
   choice has real compute/deployment cost that is currently unexamined (stakes).
2. One sentence: what was done (keep this tight — cohorts, tasks, encoder, sweep).
3. 2–3 sentences: the frontier/cost findings, not just the saturation findings —
   lead with the resource-allocation result (iso-compute, aggregation-substitution
   limits, and the extrapolation/forecasting result), and only then the task-specific
   saturation pattern.
4. One closing sentence: the actionable payoff (a decision framework, not just "a
   characterization").

**Draft** (adapt/trim to the 150-word limit — this draft runs a bit long and is meant
as raw material, not a drop-in replacement):

> Polysomnography (PSG)-based prediction models must commit to a temporal context —
> how much of an overnight recording each prediction consumes — yet this choice is
> treated as a fixed implementation detail even though it fixes training compute,
> GPU memory, and inference cost. We reframe temporal context as a task-specific
> resource-allocation variable and derive its cost-performance frontier across seven
> clinical and demographic prediction tasks in ~16,000 subjects from four NSRR
> cohorts, using a frozen SleepFM encoder and three sequence heads swept across six
> context budgets (30 s to 240 min) and up to 50 inference-time aggregation windows.
> Sufficient context was sharply task-specific, from negligible benefit beyond a few
> minutes (BMI) to continuous gains through 240 min (sleep efficiency, sex). Multi-window
> aggregation only partially substituted for long-context training at matched
> compute, and cheap short-context runs forecast expensive long-context performance
> to within ~1 AUROC point for most tasks. These results turn temporal context from
> an arbitrary default into an actionable, task-specific budgeting decision for
> clinical PSG model deployment.

Notes:
- This pulls the extrapolation/forecasting finding (currently absent from the
  abstract) into the headline result set — it's the single most "advanced-sounding"
  finding already in the paper and costs nothing to add.
- "an actionable, task-specific budgeting decision" as the closing clause directly
  answers "why is this important in practice," which is exactly what your supervisor
  flagged as missing.
- Trim by cutting the "using a frozen SleepFM encoder..." clause down, or merging
  sentences 2–3, to hit ≤150 words.

**Implemented (round 1):** the user rejected both this draft and the supervisor's
own drafts as final text ("not the best... rewrite a nice abstract") and asked for a
fresh synthesis. The abstract actually shipped is a new draft, not this section's or
§10.1's — see §10.1's end-of-section note for the adopted text, exact word count
(150/150), and what it kept from each source. `npj_main.tex` lines ~68–83.

---

## 6. Introduction (line 111–217)

### 6.1 Add a "stakes" paragraph early

Right now paragraph 1 (line 114–129) is pure background (what PSG is, what it
encodes) and paragraph 2 (line 131–145) is a literature survey. Nowhere before the
literature review does the reader learn *why context length is a problem worth
solving*, in concrete terms. Insert a short paragraph — either as a new paragraph 2
(before the SeqSleepNet/IITNet literature survey) or folded into the end of
paragraph 1 — that states the practical cost explicitly. This is the single highest-leverage
edit in the whole document.

**Draft insertion**, using only numbers already established elsewhere in the paper
(Supp. Table S-XIII; Methods `sec:heads`; the $O(N^2)$ attention/FlashAttention
discussion at line 1722):

> This choice is not free. Transformer-style sequence heads scale attention cost
> quadratically with the number of input patches, so training compute and GPU memory
> grow sharply with temporal context; in our own experiments, training the same
> architecture on the same task took roughly 12$\times$ longer at the longest context
> tested than at the shortest (Supplementary Table~S-XIII). Recording duration also
> bounds how much signal is available for inference-time aggregation, and both axes
> compound at deployment scale, where every additional minute of committed context
> multiplies compute and memory cost across an entire clinical population rather than
> a single subject. A context length chosen by convention rather than evidence
> therefore risks two distinct failure modes: over-provisioning tasks that saturate
> early, wasting compute and memory that could serve more subjects or more tasks per
> GPU, or under-provisioning tasks that need substantially more context, silently
> capping achievable performance without the practitioner ever finding out. Despite
> this, temporal context has received essentially no systematic treatment outside
> sleep staging (below), leaving both failure modes unquantified for the broader
> class of clinical and demographic prediction tasks PSG foundation models now
> target.

Place this either right after the current paragraph 1's last sentence (line 129,
"...or process each 30-second segment in isolation") or as a standalone paragraph
before the SeqSleepNet paragraph. It gives the literature survey that follows a
reason to exist ("here's what's been tried, and none of it treats this as a resource
question").

**Implemented (round 1):** inserted essentially this paragraph verbatim (trimmed
slightly) right after "...or process each 30-second segment in isolation," before
the SeqSleepNet/IITNet literature paragraph. Kept the 12$\times$ figure and the
Supplementary Table S-XIII citation; removed the "(below)" aside and the `sec:heads`
cross-reference since the FlashAttention/memory mechanism is explained later in
Methods, not needed at first mention. `npj_main.tex`, new paragraph after line 111.

### 6.2 Strengthen, don't just state, the two "open questions" paragraphs (line 159–176)

These paragraphs are already close to the right frame — line 176 even says "has
direct consequences for deployment in resource-constrained settings" — but that
sentence is a throwaway at the end of a paragraph rather than the paragraph's thesis.
Restructure so each of these two paragraphs *opens* with the resource-allocation
framing and *then* states the technical question, rather than the reverse. E.g., open
paragraph 3 (line 159) with something like: *"The practical cost of getting this
wrong compounds for patient-level clinical prediction..."* rather than *"The
context-length dependence of clinical and patient-level prediction... is therefore
less understood."*

**Implemented (round 1):** both paragraphs restructured as suggested. The
patient-level-prediction paragraph now opens "The practical cost of an unexamined
default compounds for patient-level clinical prediction..."; the aggregation
paragraph now opens "A second open question, with direct consequences for
deployment under compute or recording-duration constraints, is whether
inference-time aggregation can substitute for longer training context..." (the old
trailing "has direct consequences for deployment..." clause was removed from the
end since it's now the opening). `npj_main.tex`, the two paragraphs immediately
before "We address these two open questions...".

### 6.3 Contributions list (line 205–216)

> **See §10.2 first.** The supervisor asked whether the H1–H4 itemized hypothesis
> list and this numbered contributions list actually match npj Digital Medicine's
> house style. Checked against npj's own submission guidelines (§10.2): the
> Introduction is required to run *without subheadings*, and Nature-portfolio
> Introductions are conventionally continuous narrative prose. Itemized display
> lists of hypotheses/contributions are not a hard rule violation, but they are
> stylistically atypical for this journal and read more like a CS-conference paper.
> §10.2 recommends narrativizing both lists; the reworded contributions list below
> should be treated as an intermediate step (content only), with §10.2's guidance
> applied on top of it before finalizing.

Currently:
```
1. First systematic, cross-task characterization...
2. Empirical task-specific saturation profiles...
3. An iso-compute analysis showing long-context training and multi-window
   aggregation are not equivalent...
4. A modality ablation...
```

Suggested reordering and rewording — lead with the decision-support framing, and add
the missing extrapolation contribution:

```
1. A cost-performance frontier for temporal context in clinical PSG prediction:
   the first systematic, cross-task characterization of how much temporal context
   each of seven prediction tasks requires, across over 10,000 subjects from four
   independent cohorts.
2. An iso-compute analysis quantifying when inference-time aggregation of short
   windows can substitute for long-context training at matched compute budget,
   and when it structurally cannot — directly informing deployment under
   compute or recording-duration constraints.
3. Evidence that context-length requirements are forecastable at low cost:
   a power-law fit using only the four cheapest context budgets predicts
   performance at the two most expensive ones to within ~1.4 AUROC points,
   suggesting the full expensive sweep is not always necessary to characterize
   a new task.
4. A modality ablation linking task-specific temporal requirements to the
   underlying physiological signal group (brain, respiratory, cardiac, muscle)
   each task depends on.
```

(Item 3 is new to the contributions list — it currently exists only as one sentence
mid-Results, line 417–423 — and item order changed to lead with the frontier framing.
Item numbering/wording should be tuned to match final phrasing.)

**Implemented (round 1):** converted to narrative prose per §10.2 (not a numbered
list — see that section), using this content near-verbatim as "fourfold" prose:
"The contributions of this work are fourfold. First, a cost-performance frontier for
temporal context... Second, an iso-compute analysis... Third, evidence that these
requirements are forecastable at low cost... Fourth, a modality ablation...".
`npj_main.tex`, paragraph immediately after the H1–H4 narrative paragraph.

### 6.4 H1–H4 list (line 190–201) — revised per §10.2

Original guidance here was "keep as-is." **Revised after the supervisor's question
(§10.2):** the *content* of H1–H4 is fine and precisely stated, and the H-labels
should stay as internal cross-reference shorthand (used later in Methods/Results
per the existing `CLAUDE.md` convention). What should change is the *display
format* — convert the bulleted itemize block into a narrative paragraph, consistent
with npj's "Introduction without subheadings" convention and with the paper's own
existing Results-section rule of putting the finding before the H-label, not the
reverse (`CLAUDE.md`, "Hypothesis framing convention"). See §10.2 for the concrete
before/after.

**Implemented (round 1):** converted to narrative prose using near-exactly §10.2's
before/after example, folded into the same paragraph as the reframed frozen-encoder
sentence (see §10.3's note): "...so the trainable sequence heads isolate what
temporal integration over these representations contributes as context length
grows... This design lets us ask, for each task, whether AUROC rises with context
length and saturates at a task-specific threshold $L^*$ (**H1**); whether...". All
four H-labels retained inline exactly as specified. `npj_main.tex`, paragraph
starting "The encoder is frozen throughout...".

---

## 7. Discussion (line 1048–1331)

### 7.1 Opening paragraph (line 1050–1077)

Currently opens with a critique of the literature's treatment of context length as
"an implementation detail." Good instinct, but it's framed as a historiographical
observation, not a cost argument. Add one sentence near the start that names the
cost explicitly, so the Discussion's opening mirrors the Introduction's new stakes
paragraph (Section 6.1 above) — this bookending is what makes the paper feel like it
has a thesis rather than a list of findings. E.g., after the existing first sentence,
add: *"This is not merely an unexamined design choice: because attention-based
sequence heads scale super-linearly with context and GPU memory is a hard constraint
in most training and deployment settings, an unexamined default context length is
also an unexamined compute and memory commitment."*

**Implemented (round 1):** inserted verbatim as the second sentence of the
Discussion's opening paragraph, right after "...treated as an implementation detail
rather than an experimental variable." `npj_main.tex`, `sec:discussion` opening.

### 7.2 Pull the deployment-budget material forward from Supplementary into Discussion

The supplementary sections "Minimum Inference Budget to Reach Target AUROC"
(`sec:supp-mincost`) and "Deployment Scenario Grid" (`sec:supp-deployment`) contain
exactly the kind of concrete, actionable, "advanced" statements your supervisor is
asking for — and they currently live only in the supplementary, never summarized in
the Discussion. Add a short paragraph (or extend the existing iso-compute paragraph
at line 1117–1146) that states 2–3 of these numbers directly in the Discussion body:

**Draft addition**, placed after the existing iso-compute paragraph (after line 1146,
before the architecture-comparison paragraph):

> These trade-offs translate directly into deployment guidance. Reaching AUROC
> $\geq$0.85 required at least 120 min of context for every main task evaluated; no
> amount of inference-time aggregation from 30-s windows closed this gap, confirming
> that some performance thresholds are simply unreachable below a task-specific
> minimum context regardless of how many short windows are averaged
> (Supplementary~Figure~S-10). Conversely, for tasks such as sex and age-group
> classification, a 30-min total recording budget was sufficient to reach the same
> threshold when paired with many short-context inference windows rather than a
> single long one (Supplementary~Figure~S-11), a materially cheaper deployment
> configuration than defaulting to full-night context. The optimal allocation
> therefore shifts systematically with the available budget: many cheap windows at
> small budgets, a single long-context pass once the budget is large enough to
> afford it, exactly the crossover behaviour captured by the iso-compute contours
> above (Figure~\ref{fig:heatmap}).

This directly answers "why is it important in practice" with numbers, not just
qualitative language, and reuses figures already in the supplementary — no new
analysis required.

**Implemented (round 1):** added as a new paragraph immediately after the existing
K-aggregation paragraph and before the architecture-comparison paragraph, combined
with §7.3's extrapolation sentence into one paragraph (see §7.3's note — they read
better together than as two separate additions). One number was corrected from this
draft for accuracy: "required at least 120 min of context for every main task
evaluated" was too strong given BMI never reaches AUROC 0.85 at any budget; the
shipped text reads "required training at $L \geq 120$~min for every task that could
reach it at all (sex, age group, apnea, sleep efficiency)... and BMI could not reach
this target at any budget, consistent with its low overall ceiling." `npj_main.tex`,
new paragraph after "...remains to be seen." and before "The choice of
sequence-modelling architecture...".

### 7.3 Promote the "cheap predicts expensive" result into the Discussion narrative

The extrapolation finding (line 417–423 in Results) is currently never referenced
again in Discussion. Add one or two sentences, ideally in the same new paragraph as
7.2 or immediately after, tying it to the practical framing: *"This cost-performance
frontier is itself cheap to estimate: fitting the frontier from only the four
least expensive context budgets predicted performance at the two most expensive
budgets to within ~1.4 AUROC points on average (Results), suggesting that
characterizing a new task's temporal-context requirements need not require running
the full, most expensive sweep."* This is the sentence that most directly answers
your supervisor's "sound more advanced" note — it reframes the paper's own
methodology as compute-efficient, not just its findings.

**Implemented (round 1):** merged into §7.2's new paragraph as its closing sentence,
essentially verbatim (see §7.2's note for exact placement). `npj_main.tex`, last
sentence of the new deployment-guidance paragraph.

### 7.4 Closing paragraphs (line 1287–1331, including merged Conclusion)

The final paragraphs already gesture at the right framing ("context length is a
consequential and task-specific design parameter... whose interaction with
inference-time aggregation strategy has direct implications for compute
efficiency," line 1325–1328) but this appears only in the very last paragraph,
diluted among five other closing thoughts. Suggested: move a version of this
sentence — or a strengthened version of it — up so it is one of the first two
sentences of the Conclusion (line 1311), not the last paragraph's second-to-last
clause. Consider closing the entire paper on the frontier/decision-framework framing
rather than the more generic "provide a concrete empirical baseline for future
work" (line 1329–1331), which undersells relative to everything above it. E.g., end
on: *"...providing practitioners a way to choose temporal context deliberately,
against a quantified cost, rather than by convention."*

**Implemented (round 1):** both edits made as suggested. Conclusion's opening
sentence now reads "This paper establishes temporal context as a task-specific
resource-allocation variable in PSG-based clinical prediction, not merely an
implementation detail, characterising its cost-performance frontier by sweeping six
context lengths..."; the closing paragraph now ends "...give practitioners a way to
choose temporal context deliberately, against a quantified performance and compute
cost, rather than by convention, and provide a concrete empirical baseline for
future work..." (kept the original "empirical baseline for future work" clause
rather than fully replacing it, since it's still true and useful, just no longer the
*only* closing idea). `npj_main.tex`, `sec:conclusion` opening and closing
paragraphs. A third limitation-side addition not originally scoped in this
subsection was also made here: see §10.3's and §10.5's notes for a new sentence
tying the 240-min cap to the sleep-efficiency/sex non-saturation limitation.

---

## 8. Additional levers to manoeuvre on

Beyond the three sections requested, these are lower-effort, high-payoff places to
reinforce the same reframe without touching Results/Methods substance:

1. **Keywords** (line 106): currently `context length, deep learning, foundation
   models, polysomnography, sleep, temporal modelling`. Consider adding
   `compute efficiency` or `resource allocation` — keywords are indexed and shape how
   the paper is discovered/categorized, and currently signal "temporal modelling
   study" rather than "efficiency/deployment study."
2. **Figure/section headers already do some of this work — extend it.** Section
   headers like "Pareto frontier and minimum deployment cost" (line 815) and
   "Budget-dependent crossover between context and aggregation" (line 699) are
   exactly the right register. The Abstract/Intro/Discussion should sound like they
   were written by the same person who titled those sections — right now there's a
   noticeable register gap between "we present a systematic study" (Abstract) and
   "minimum deployment cost" (Results headers). Match the Abstract/Intro/Discussion
   prose register to the existing Results section headers, not the other way around.
3. **Consider explicitly invoking the scaling-laws analogy once, in Discussion.**
   The `markdown/SCALING_LAW_ANALYSIS_IDEAS.md` file (Part 2) already identifies
   close precedent — Chinchilla-style compute-optimal allocation, and Montgomery et
   al. 2025's joint context/compute scaling laws for LLMs — as the right academic
   frame for exactly this paper's iso-compute + extrapolation results. A single
   sentence in Discussion naming this connection explicitly (e.g., "this
   iso-compute characterization is a domain-specific analogue of compute-optimal
   allocation studies in large language models, where task performance is jointly a
   function of context length and total compute") would signal to reviewers that
   the paper is intentionally positioned within a recognized, "advanced" line of ML
   research rather than presenting an isolated ablation study. Cite Hoffmann et al.
   (Chinchilla) and/or Montgomery et al. 2510.14919 if added — both are already
   vetted as relevant in that file.
4. **Avoid self-deprecating or purely descriptive verbs.** Search for "we present a
   systematic study," "we investigate," "we examine" as sentence openers in
   Abstract/Intro/Discussion and replace with verbs that assert a contribution:
   "we characterize," "we quantify," "we establish," "we derive." This is a small,
   mechanical pass but changes the register throughout.
5. **Do not touch Methods or Results prose for this pass.** Their register ("Pareto
   frontier," "minimum deployment cost," "iso-compute contour") is already correct
   and should be the *target* register for the rest of the paper, not something that
   needs fixing itself.

**Implemented (round 1):** item 1 (keywords) done — see §3's note. Items 2 and 4
(register-matching, verb choice) were applied throughout while writing the
Abstract/Introduction/Discussion edits above rather than as a separate pass; no
"we present a systematic study"-style openers remain in the rewritten text. Item 3
(scaling-laws sentence) **not done this round** — left as a genuine optional add-on,
since it wasn't part of the user's explicit round-1 ask and works fine on its own as
a future micro-edit to the Discussion's Chinchilla-adjacent paragraph. Item 5 was
*not* fully honored: the user's round-1 request explicitly included Methods fixes
(§10.4, §10.5), which were correctness fixes rather than register changes — see
those sections' notes; this is a deliberate, justified exception, not a slip.

---

## 9. What NOT to change

- No new experiments, figures, or numbers are needed — every number suggested above
  (12$\times$ training-time ratio, ≥120-min threshold for 0.85 AUROC, 30-min budget
  sufficiency for sex/age, ~1.4-point extrapolation error) already exists in the
  paper or supplementary and is cited above with its source location.
- Do not alter the H1–H4 *labels or content*, or the acronym-definition ordering
  documented in `CLAUDE.md` — both are correct and independently maintained
  conventions. The one change in scope, per §10.2, is *display format*: convert the
  itemized H1–H4 block and the contributions list from bulleted display lists into
  narrative prose, keeping the same labels and content as inline parenthetical
  references.
- Do not remove "context length ($L$)" as precise notation in Methods/Results —
  only shift the narrative register in Abstract/Introduction/Discussion prose per
  Section 3's terminology table.

**Implemented (round 1):** all three constraints honored — no new experiments/figures
were run or created; `context length ($L$)` notation is untouched in Methods/Results;
H1–H4 content and labels are unchanged (only their display format changed, per §10.2).
Verified via `git diff --stat` after all edits: only `npj_main.tex` and its compiled
`npj_main.pdf` changed, `npj_supplementary.tex` untouched.

---

## 10. Supervisor's initial comments — assessment and answers

The supervisor sent four rounds of feedback: (1) two draft abstracts, (2) a question
about whether the H1–H4/contributions-list format matches the journal's convention,
and (3) three substantive scientific questions. Each is addressed below. Not every
comment is accepted as-is — where a comment doesn't hold up or would weaken the
paper, that's stated plainly with reasoning, per your request.

### 10.1 Abstract — assessment of the two supplied drafts

Both drafts move in the same direction as §5's diagnosis (lead with stakes, use
"temporal context" not "length"), which is a good independent confirmation that the
reframe is right. Assessment of each:

- **Draft 1** ("Overnight polysomnography (PSG) captures complex physiological
  dynamics... optimal temporal scope required... remains uncharacterized.") is a
  strong single opening sentence — tighter and more "sounds advanced" than anything
  in the current abstract or in §5's draft. Its weakness is that it's only an
  opening line, not a full abstract; "optimal temporal scope to resolve distinct
  clinical phenotypes" is an excellent hook but says nothing yet about cost/compute,
  which was the other half of the supervisor's own separate verbal feedback
  (efficiency, GPU budget, practical deployment). Recommend using this as the
  **opening sentence**, not the whole abstract.
- **Draft 2** is a complete, well-structured abstract and is the stronger candidate
  overall: it states the gap ("the amount of temporal context required... remains
  largely unexplored"), states why it matters ("essential for balancing predictive
  performance and computational efficiency... insight into temporal organization of
  the underlying physiological processes" — this is exactly the practical-importance
  framing that was missing), states what was done, and states three findings in
  order of importance. This is a better structure than the current active abstract
  and than §5's original draft.
  - **Problem: length.** Draft 2 is **280 words** (counted directly), nearly double
    npj's confirmed 150-word hard limit for an unstructured abstract (verified
    against npj's own submission guidelines — see §10.2 below, same source). It
    cannot be submitted as-is and needs roughly a 45% cut.
  - **Missing:** the extrapolation/forecasting finding (§5's original suggestion,
    Results line 417–423 — cheap short-context runs predict expensive long-context
    AUROC to within ~1.4 points) is still absent from both supervisor drafts. This
    is the single most "advanced-sounding," least "simple comparison"-sounding
    result already in the paper and costs nothing to add; recommend folding it in
    if space allows after trimming, even as a half-sentence.

**Recommended synthesis** (Draft 1's opening sentence + Draft 2's structure,
trimmed to fit ≤150 words, extrapolation finding folded into the third finding as a
clause):

> Overnight polysomnography (PSG) captures complex physiological dynamics across
> hours of sleep, yet the temporal context a predictive model needs to resolve
> distinct clinical phenotypes remains uncharacterized, leaving performance,
> compute cost, and deployment feasibility unbalanced by default rather than by
> design. We present the first systematic characterization of temporal context
> requirements across seven clinical and demographic prediction tasks in ~16,000
> subjects from four NSRR cohorts (SHHS, MrOS, APPLES, STAGES), using the
> pre-trained SleepFM encoder and three sequence-modeling strategies (Bi-LSTM,
> Transformer, and non-temporal pooling) across temporal contexts from 30 s to 240
> min. Requirements were sharply task-dependent: BMI classification gained little
> beyond a few minutes, apnea and age-group prediction plateaued near 120 min,
> while sleep efficiency and sex classification kept improving through the longest
> contexts tested. Aggregating many short segments partially substituted for long
> context on simpler tasks but not on context-dependent ones, and a fit using only
> the four cheapest context budgets forecast the two most expensive to within ~1.4
> AUROC points. These results establish temporal context as a task-specific design
> variable that must be set deliberately to balance clinical performance against
> computational cost.

(This runs close to 150 words; trim the "using the pre-trained..." clause first if
still over.)

**Implemented (round 1) — final abstract, superseding this draft too.** The user
read this synthesis and judged it "not the best," asked for a genuinely better
rewrite rather than a patch of either the supervisor's or this draft, while keeping
the direction (stakes-first, "temporal context," balance performance against
compute cost). The abstract actually shipped is a new, tighter synthesis, exactly
150/150 words:

> Overnight polysomnography (PSG) captures complex physiological dynamics across
> hours of sleep, yet the temporal context a predictive model requires to resolve
> clinical phenotypes remains uncharacterized, despite being essential for balancing
> predictive performance against computational cost. We present the first
> systematic characterization of temporal context requirements across seven
> clinical and demographic tasks in ~16,000 subjects from four NSRR cohorts, using
> the pre-trained SleepFM encoder and three sequence-modeling heads across contexts
> from 30 seconds to 240 minutes. Requirements were sharply task-dependent:
> negligible benefit beyond a few minutes for body-mass index, but continuous gains
> through 240 minutes for sleep efficiency and sex. Aggregating short segments only
> partially substituted for long context on context-sensitive tasks, and a frontier
> fit from the four cheapest context budgets predicted the two most expensive
> within roughly 1.4 AUROC points. These results establish temporal context as a
> task-specific design variable that must be set deliberately in future
> sleep-analysis models.

What changed relative to the two drafts above, and why:
- Kept Draft 1's opening clause ("captures complex physiological dynamics across
  hours of sleep") almost verbatim — it *is* the strongest hook available and both
  the supervisor and this document's own §5 independently converged on wanting it.
- Kept Draft 2's "balancing predictive performance against computational cost"
  clause (the supervisor's own words) as the practical-stakes payoff of sentence 1,
  rather than deferring it to the closing sentence as the earlier §10.1 synthesis
  did — this way the "why it matters" answer arrives before the reader has invested
  in reading the rest of the abstract, not after.
- Cut architecture names (Bi-LSTM/Transformer spelled out), cohort acronyms
  (SHHS/MrOS/APPLES/STAGES), and one of the four saturation examples (apnea,
  age-group) that both drafts included — all recoverable in the Introduction, and
  none of them changes what a reader takes away from the abstract. This bought back
  the ~25 words needed to state the extrapolation finding precisely (with the exact
  ~1.4-point number, not a vague "about one point") instead of dropping it.
- Precision fix versus the earlier §10.1 synthesis: "within ~1.4 AUROC points" (the
  real, verified number from Results) rather than a rounded "~1 AUROC point."
- Deliberately did **not** try to also restate the temporal-head-advantage finding
  (Transformer > LSTM > MeanPool) that both supervisor drafts led with as finding
  three — at 150 words there was room for either that finding or the
  extrapolation/forecasting result, and the extrapolation result is the one that
  actually answers "why does this sound like more than a simple comparison," which
  was the whole point of this exercise. The head-comparison finding is still fully
  covered in the Introduction's H3 sentence and the Discussion.

`npj_main.tex` lines ~68–83 (`\abstract{...}`); keywords updated on the same block
per §3's note. Word count verified with `wc -w` before insertion (150 words exactly,
counted the same way npj's own limit is conventionally checked).

### 10.2 Is the H1–H4/contributions-list format standard for this journal?

**Checked directly against npj Digital Medicine's own submission guidelines**
(fetched 2026-07-31): the guidelines explicitly state the manuscript **"must
include an Introduction without subheadings, Results with subheadings, a
Discussion without subheadings or separate limitations or conclusions sections,
and a Methods section with subheadings."** They also confirm the abstract "must
not exceed 150 words" and "must be unstructured and contain no subheadings" — used
to validate the word-count claim in §10.1 above.

**Answer to "do other papers mention hypotheses first": generally, no, not in this
itemized-list form.** An itemized, labeled H1–H4 hypothesis block and a separately
numbered "contributions of this work" list are not literal rule violations (neither
is technically a "subheading"), but they are not the house style either. This
pattern — explicit numbered hypotheses (H1, H2, ...) reused as cross-reference
labels, plus a bulleted "our contributions are" list — is a convention borrowed from
**CS/ML conference papers (NeurIPS, ICML-style)** and, separately, from
**pre-registered clinical-trial or psychology papers** with primary/secondary
endpoints. Nature-portfolio research articles, including npj Digital Medicine,
almost always motivate a study and state what it contributes in **continuous
narrative prose**, ending the Introduction with a paragraph (not a list) describing
the study design and its main advances. Given the "Introduction without
subheadings" rule is explicit and enforced, a heavily itemized Introduction is a
mismatch with the journal's actual register, and is plausibly part of why the paper
currently reads like an engineering ablation study rather than a clinical-audience
research article — the same underlying issue the supervisor's first comment raised
about the abstract.

**Recommendation:** convert both the H1–H4 itemize block (line 190–201) and the
contributions itemize list (line 205–216) into flowing paragraphs. Keep the H1–H4
*labels* (bolded inline, e.g., "we therefore test whether performance saturates at
a task-specific context length (**H1**)...") so that later parenthetical references
in Methods/Results (already documented in `CLAUDE.md`'s "Hypothesis framing
convention") still resolve, but drop the itemize environment itself. Example
transformation:

**Before** (current, line 190–201):
```
This design tests four pre-specified hypotheses:
\begin{itemize}
  \item[\textbf{H1.}] \textbf{Context saturation.} AUROC increases with L and
  saturates at a task-specific threshold L*...
  \item[\textbf{H2.}] \textbf{Aggregation saturation.} ...
  \item[\textbf{H3.}] \textbf{Temporal head advantage.} ...
  \item[\textbf{H4.}] \textbf{Aggregation substitution.} ...
\end{itemize}
```

**After** (narrative, same content and labels, no display list):
```
This design lets us ask, for each task, whether performance rises with context
length and saturates at a task-specific threshold L* (H1); whether, for a fixed
trained model, performance saturates after a small number of inference windows K
(H2); whether temporal heads (LSTM, Transformer) outperform non-temporal MeanPool
at long context but not short (H3); and whether, at an equal total inference
budget (L x K held constant), performance depends only on the budget or also on
how it is split between context and aggregation count (H4).
```

Apply the same treatment to the contributions list (§6.3's reworded content, minus
the itemize environment — turn the four numbered items into one closing paragraph
of the Introduction).

**One caveat, not a reason to reject this:** this is a stylistic/register judgment,
not a hard compliance issue — the current itemized format will not get the paper
desk-rejected on a technicality. But it is a legitimate, evidence-backed answer to
the supervisor's question, and it reinforces the exact concern about the paper
sounding like "a simple comparison" rather than an advanced, integrated argument:
narrative prose forces the four questions to read as one coherent study design
rather than four independent checklist items.

**Implemented (round 1):** both conversions done exactly as recommended — see §6.3's
and §6.4's implementation notes for the shipped text. The H1–H4 labels are preserved
inline and bolded (`\textbf{H1}` etc.); the itemize/enumerate environments are gone
from the Introduction entirely. `npj_main.tex` compiles cleanly with no broken
cross-references to H1–H4 elsewhere in Methods/Results (checked: those references
use plain `H1`/`H2`/etc. text, not LaTeX `\ref{}` to the removed list items, so
removing the itemize environment didn't break anything).

### 10.3 Are saturation points intrinsic to sleep physiology, or an artifact of the frozen SleepFM encoder? Were other encoders tested?

**Factual answer: no, other frozen encoders (e.g., OSF, SleepMaMi, SleepFounder) or
a fine-tuned/unfrozen encoder were not tested.** Only the frozen, pre-trained
SleepFM encoder was used throughout (Introduction line 178, 188; Methods
`sec:sleepfm_encoder`). This is a fair and important question, and the paper
**already has a partial answer in the Discussion's limitations paragraph** (line
1250–1253): *"The saturation profiles are specific to the SleepFM encoder, trained
on its particular corpus with a patch-level contrastive objective; a different
pretraining strategy, backbone capacity, or data distribution would likely alter
the balance between what the patch-level representation already encodes and what
the downstream head must learn through temporal integration."* So the honest
caveat exists — it is just buried in the fourth-from-last paragraph of the
Discussion, never mentioned in the Abstract or Introduction, and not framed sharply
enough to preempt exactly the question the supervisor asked.

**This is worth accepting and acting on, in two parts:**

1. **Reframe the freezing decision as a deliberate, defensible trade-off, not an
   oversight.** The paper already states the *reason* for freezing (Introduction
   line 188: "Freezing the encoder isolates context length as the sole variable")
   — this is a real methodological strength: it cleanly attributes the saturation
   curves to context length rather than confounding them with encoder fine-tuning
   dynamics or capacity differences. Make this trade-off explicit in the same
   breath as the limitation: freezing buys internal validity (clean attribution to
   context length) at the cost of external validity (unknown whether saturation
   points transfer to a different encoder or to a fine-tuned model). Right now the
   paper states the benefit (Introduction) and the cost (Discussion) forty pages
   apart, with nothing connecting them.
2. **Sharpen and relocate the caveat.** Add a explicit clause early — ideally in the
   Abstract itself, and again as the *opening*, not closing, sentence of Discussion's
   limitations material — naming specific alternative encoders not evaluated
   (OSF~\cite{osf}, SleepMaMi~\cite{sleepmami}, SleepFounder~\cite{sleepfounder} —
   all already cited elsewhere in the paper for other reasons) and stating plainly
   that this study cannot distinguish "intrinsic to sleep physiology" from
   "intrinsic to SleepFM's specific patch-level contrastive representation."
   Suggested Abstract clause (fits the trimmed draft in §10.1 with minor
   compression): *"...under a fixed, frozen SleepFM representation..."* inserted at
   first mention of the encoder. Suggested Discussion opening for this paragraph:
   *"Because the encoder was frozen and only one pretraining strategy (SleepFM's
   patch-level contrastive objective) was evaluated, these saturation profiles
   cannot be attributed to sleep physiology independent of the representation used
   to access it; whether a differently pretrained or fine-tuned encoder would shift
   the saturation points is an open, testable question this study does not
   answer."*

**Not recommended:** actually running new experiments with alternative frozen
encoders before this submission. That would be a substantial new study (repeating
the full sweep for each additional encoder), not a framing fix, and is better
scoped as explicit future work named in Discussion rather than attempted under
submission time pressure. Say so directly in the limitations text rather than
avoiding the question.

**User pushback, received before implementation:** the user pushed back on framing
this as a "limitation" at all, offering their own model: SleepFM is functionally a
*tokenizer* — it encodes each few-second segment once, exactly as any sequence
model's front end must, and the trainable sequence heads are what actually learn the
inter-segment (inter-epoch) relationships that produce the saturation curves. Under
that view, freezing the encoder isn't a compromise specific to this study; it's what
every sequence model does, so there's no special "FM limitation" to apologize for.
Asked for a frank assessment of whether this holds.

**Frank assessment, as requested:**

*Where the user is right, and the paper should sound confident about it:* the
frozen-backbone-plus-trainable-head design is a completely standard, well-precedented
methodology, not a compromise. It's the same paradigm as linear-probing a frozen
BERT or ViT, and its purpose here is exactly right: isolating what the *sequence
heads* learn about temporal integration from what the *encoder* already knows,
which is precisely the variable this paper is trying to study. This is not something
to hedge or apologize for, and the original Discussion paragraph's apologetic tone
undersold a genuinely sound design choice. The rewritten paragraph (below) leads
with this, in the user's own framing, as its first and dominant point.

*Where the tokenizer analogy doesn't fully hold, and is worth being honest about:*
a text tokenizer (e.g., byte-pair encoding) is a near-lossless, fixed, objective-
agnostic *segmentation* rule — it decides where to cut, not what survives the cut.
SleepFM's frozen encoder is not that: it is a *learned, lossy compressor*, trained
with a specific patch-level contrastive objective that determined what physiological
information gets packed into each 512-dimensional patch embedding and what gets
discarded before the sequence head ever sees it. That means each task's observed
saturation point $L^*$ is shaped jointly by (a) the true timescale over which the
task's physiological signal actually unfolds, which is encoder-independent, and (b)
what SleepFM's specific contrastive pretraining happened to preserve within a single
patch versus left recoverable only by integrating across patches, which is
encoder-dependent. A differently pretrained encoder (a masked-reconstruction
objective, a supervised objective, or a wearable-oriented objective) could plausibly
pack information differently per patch and shift where a given task's $L^*$ falls,
even if the general phenomenon (temporal integration matters more for some tasks
than others) replicated. This is a real, unresolved scope boundary, not a flaw in
the design and not something the tokenizer framing dissolves — it is exactly the
kind of claim ("these numbers are sleep-physiology-intrinsic") the study cannot make
on its own.

**Net verdict:** the user is right that this should not read as an apologetic
limitation, and right that the frozen-backbone design is defensible and standard.
It would be overclaiming, though, to say the encoder choice has *no* bearing on the
specific numbers reported (as opposed to the general phenomenon) — that stronger
claim does not hold. **Resolution implemented:** confident probing-paradigm framing
as the dominant tone, matching the user's instinct, with exactly one precise,
non-apologetic sentence retained on scope: it is the specific $L^*$ *values*, not
the qualitative pattern of task-dependent saturation, whose encoder-invariance is
untested. Shipped paragraph (`npj_main.tex`, "Several factors constrain..."
paragraph, Discussion):

> Several factors constrain how broadly the present findings should be generalised.
> The encoder was frozen throughout so that the sequence heads' temporal
> integration, not encoder fine-tuning dynamics, is what these saturation curves
> measure, the standard frozen-backbone probing paradigm used to isolate a
> downstream component's contribution. This choice buys internal validity at a
> cost: because only one pretraining objective, SleepFM's patch-level contrastive
> loss, was evaluated, the reported saturation thresholds cannot be attributed to
> sleep physiology independent of the representation used to access it. A
> differently pretrained or fine-tuned encoder might pack information differently
> across patches, shifting where a given task's $L^*$ falls even if the qualitative
> phenomenon, that temporal integration matters more for some tasks than others,
> replicated; testing this directly against alternative frozen encoders (OSF,
> SleepMaMi, SleepFounder) or a fine-tuned backbone is a natural next step this
> design does not itself resolve. [Continues with the 240-min-cap tie-in from
> §10.5, then the original cohort/small-$N$/context-gap sentences, unchanged.]

Also strengthened to match: the Introduction's "Freezing the encoder isolates
context length as the sole variable" sentence (previously a single throwaway line)
now reads "The encoder is frozen throughout: each 5-second segment is reduced once
to a fixed representation, the role a tokenizer or patch embedding plays in any
sequence model, so the trainable sequence heads isolate what temporal integration
over these representations contributes as context length grows, independent of
encoder fine-tuning dynamics" — the user's own framing, stated with confidence,
right where the design choice is first introduced, forty pages before the honest
scope-boundary caveat in Discussion rather than disconnected from it.

### 10.4 How was greater total signal exposure at long context handled? (token-budget schedule)

**What the design actually does, explained plainly:** every subject contributes
exactly $w=5$ training windows per epoch, regardless of context length $L$. This
equalizes the *number of gradient updates per subject per epoch* across all six
context lengths — the stated "fairness criterion for an unconfounded context-length
comparison" (main text line 1770–1772). It does **not** equalize the *total amount
of signal (patches/tokens) seen per subject per epoch* — a 240-min window contains
2,880 patches per draw versus 6 patches for a 30-s window, so the 240-min model
sees roughly 480x more raw signal per epoch than the 30-s model, by design. The
paper's own text names the alternative that *would* equalize total signal exposure
— a "token-budget schedule" where the number of windows drawn is inversely
proportional to context length ($w \propto 1/L$) — and rejects it because at 240
min it collapses to a single window per subject per epoch, "too few for stable
gradient estimation" (main text line 1782–1785; Supp `sec:supp-kbudget`, line
518–523).

**Is this a reasonable design choice?** Yes, and it is the more defensible of the
two options: the chosen $w=5$ scheme controls for the confound most directly tied
to *optimization* (gradient-update count, which affects convergence and training
stability) rather than the confound tied to *total information available*, which is
unavoidable once context length itself is the independent variable being tested —
a longer context window necessarily contains more raw signal; that is what "longer
context" means. Attempting to equalize total signal exposure across context lengths
would either starve long-context training (1 window/epoch, as noted) or force
short-context models to see many more windows/epoch than long-context models,
reintroducing a different confound (short-context models get vastly more gradient
steps). There is no scheme that removes both confounds simultaneously; $w=5$ is a
principled choice of which one to control for, not an oversight.

**Where this needs a real fix, not just clearer prose:** the main text (line
1786–1788) states, *"A supplementary sensitivity experiment is described in
Supplementary Section~S-III that tests whether fixing $w$ at five is equivalent to
a token-budget schedule at all context lengths above 10~min."* This claim does
**not check out**:
- Supplementary Section S-III is **"Foundation Encoder and Downstream Heads"**
  (`sec:supp-encoder`, per the Supplementary Contents page) — a section about the
  SleepFM encoder architecture, unrelated to window sampling.
- The section that *is* about this topic, S-V "Training Window Sampling"
  (`sec:supp-kbudget`), contains only the **rationale/argument** for choosing
  $w{=}5$ over a token-budget schedule (quoted above) — it does not report an
  actual empirical sensitivity experiment with results comparing the two schedules
  at context lengths above 10 min.
- A repo-wide search of both `.tex` files found no other section containing such an
  experiment under any name.

**This is very likely why the supervisor "did not understand the token-budget
schedule experiment"** — the paper cites results that, as currently written, do not
exist in the document. This needs one of two fixes before submission, and it's a
factual-accuracy issue independent of the framing work in the rest of this document:
1. **If the experiment was actually run** (check the NSRR-tools codebase /
   `scaling_law_ideas/` or training logs for anything comparing fixed-$w$ vs.
   token-budget schedules) — write it up properly in Supplementary Section S-V with
   real numbers, and fix the main-text cross-reference from "S-III" to "S-V" (or
   wherever the write-up lands).
2. **If it was not run** — remove or rephrase the main-text sentence so it does not
   claim an empirical test that isn't backed by content. It can be honestly
   softened to something like: *"An alternative token-budget schedule ($w \propto
   1/L$) would equalize total signal exposure but reduces to a single window per
   subject per epoch at 240 min (Supplementary Section~S-V); this was judged
   insufficient for stable gradient estimation and not pursued further."* — which
   is accurate to what the supplement currently contains and does not overclaim.

Given the effort to actually run a clean sensitivity experiment (retrain a subset
of context lengths under both schedules) is nontrivial this close to submission,
option 2 (rephrase to match existing content) is the lower-risk path unless the
experiment already exists somewhere unwritten.

**User confirmation, received before implementation:** the user confirmed directly
that no token-budget experiment was ever run. It was something considered as a
plausible alternative to $w{=}5$, but rejected once its unfairness in the
gradient-update sense (option 2's reasoning above) was understood — i.e., option 2
was the correct path, not option 1. The user pointed at
`NSRR-tools/docs/*.md` for the project's own prior reasoning on this rather than
reasoning about it from scratch again.

**Grounding found in the codebase docs**, confirming and extending option 2's
argument: `docs/EXPERIMENTS_GUIDE.md` states plainly, *"K=5 fixed is fair in the
gradient-update sense, which is the right criterion for comparing context
lengths... Token budget is fair in the information sense — but it gives the 30s
model 160× more gradient updates per epoch, introducing a different confound
(higher effective learning rate, stronger regularisation through repetition)."*
The same file adds a point not yet in the paper anywhere: *"With sufficient epochs
and random window sampling, the K=5 model still covers most of the 30s window
space over training — K=5 controls the per-epoch exposure, not the total data seen
across the full training run."* `docs/TRAINING_PROTOCOL_FIXES.md` frames the
distinction even more sharply: a token-budget schedule *"answers a different
question: 'what is the optimal compute allocation across context lengths?' — not
'does context length improve performance?'"* `docs/PAPER_PLAN.md` independently
confirms the experiment status: *"token-budget sensitivity ablation... has not been
run"* (multiple entries, e.g. line 456). This is exactly the "aspirational TODO
that never got run, then the paper text describing it as done was never walked
back" pattern §10.4 above suspected.

**Implemented (round 1):** took option 2, using the codebase docs' own reasoning
(not just the paper's existing argument) so the fix is more complete than the
original suggestion above, not merely a rephrase. The false "supplementary
sensitivity experiment... Section S-III" sentence is gone. Replacement text in
`npj_main.tex` (`sec:training`, "Window sampling for training and validation"):

> An alternative token-budget schedule ($w \propto 1/L$, keeping total signal per
> epoch constant) was considered and rejected. It equalises information exposure
> but not gradient-update count: at 30 s it would give the model ~160 times more
> gradient updates per epoch than at 240 min, trading the confound it removes
> (unequal total signal per epoch) for a larger one (unequal optimisation exposure,
> effectively a different learning-rate and regularisation regime at each context
> length). At 240 min it also collapses to a single window per subject per epoch,
> too few for a stable gradient estimate, and more fundamentally it answers a
> different question, the compute-optimal allocation across context lengths
> (Supplementary Section S-V), rather than whether context length itself improves
> performance under matched optimisation. The fixed-$w$ protocol keeps
> gradient-update count matched across context lengths at the cost of each subject
> contributing more raw signal per epoch at long context than at short; over the
> full training run, however, random start-position sampling means the
> short-context model still visits most of its available window space across
> epochs, partially offsetting this asymmetry outside any single epoch.

This now cites Supplementary Section S-V correctly (not the wrong "S-III"), claims
no experiment that doesn't exist, and adds the "over the full training run" mitigant
from `EXPERIMENTS_GUIDE.md` that neither the original paper text nor this document's
first-pass suggestion included. Supplementary Section S-V (`sec:supp-kbudget`) was
left unchanged — it was already accurate and made no overclaim, so there was nothing
to fix there.

### 10.5 Why was the sweep capped at 240 minutes, given some tasks (sleep efficiency, sex) had not saturated?

**The paper currently gives three different explanations for the 240-min cap,
scattered across three locations, and they are not fully consistent with each
other as written:**

1. **Main text, Methods `sec:sweep` (line 1864–1866):** context lengths "span
   approximately three decades on a logarithmic scale... with the upper end
   matching the typical duration of an overnight PSG recording." This is the
   *only* explanation a reader of the main paper alone encounters — and it is the
   vaguest and least accurate of the three: a typical overnight PSG recording is
   roughly 6–8 hours (360–480 min), so 240 min (4 h) is only about half to
   two-thirds of a typical night, not really "matching" it.
2. **Supplementary S-IV (`sec:supp-clgrid`, line 470–472):** "240 min is the
   minimum post-filter recording length across all datasets" — i.e., the real
   binding constraint is a **data-availability / cohort-consistency** one: it's the
   longest context length at which every subject in every one of the four cohorts,
   after the consistency filter, still has enough recording to contribute a window,
   keeping the sample composition identical across all six context lengths for a
   fair comparison. This is the precise, defensible reason, but it only appears in
   the supplementary.
3. **Main text Methods, TransformerHead paragraph (line 1734–1737):** FlashAttention
   "was essential for training at $N=2{,}880$ patches (240-min context) within a
   10 GB GPU memory budget at batch size 32" — implying **hardware/memory** was
   also close to a binding constraint at 240 min, independent of the data-availability
   reason.

**These three are not necessarily contradictory** (240 min could simultaneously be
the data-availability ceiling *and* close to the GPU-memory ceiling *and* a rough
fraction of a typical night), but the paper never says so — it reads, especially
from the main text alone, as if the cap were a loose approximation to "a full
night," when the actual precise constraint (data availability, per the
supplementary) is a different and more rigorous reason that just isn't stated
where the reader needs it.

**Recommendation:**
1. **Consolidate into one clear statement in main-text Methods** (`sec:sweep`),
   replacing or supplementing the vague "matching the typical duration" sentence
   with the precise reason: 240 min was the longest context length at which the
   full four-cohort sample, after the consistency filter, remained available at
   every context length in the sweep, keeping the compared samples identical in
   composition rather than shrinking as context length grows. If the GPU-memory
   constraint was also independently binding around the same value, say so in the
   same sentence rather than leaving it as a separate, seemingly unrelated fact
   forty pages earlier in Methods.
2. **Connect this explicitly to the honest limitation already in Discussion.**
   Sleep efficiency and sex classification were still improving at 240 min (no
   plateau observed) — the paper already reports this as a finding, but doesn't
   explicitly tie it back to *why* the sweep stopped there. Add one sentence, near
   wherever the 240-min cap is explained in Methods or first flagged as a
   limitation in Discussion: *"Because 240 min was capped by [data availability /
   hardware / both], the true saturation point $L^*$ for sleep efficiency and sex
   classification, both still rising at the longest tested context, may exceed 240
   min; extending the cohort filter or the compute budget to test longer contexts
   is a direct next step."* This turns an implicit gap into an explicit,
   well-reasoned limitation — exactly the kind of thing a careful reviewer (or
   supervisor) will otherwise ask about.
3. **Do not claim monotonic improvement would continue indefinitely.** Keep the
   framing scoped to "may exceed the tested range," not "would keep improving
   forever" — physiologically implausible and not supported by the data.

**User decision, received before implementation:** the user resolved the
three-explanations ambiguity directly rather than leaving it as "possibly both":
consolidate to exactly two reasons, (1) cohort consistency — not choosing a context
so long it would filter out a large proportion of subjects, and (2) the exponential
(quadratic-attention) memory requirement, especially for the Transformer, where even
240 min was hard to fit. The vaguer "matches a typical overnight recording" framing
(item 1 in the three-explanations list above) was dropped rather than kept as a
third, softer justification.

**Implemented (round 1):** `sec:sweep`'s sentence rewritten to state exactly these
two reasons, replacing the vague duration claim entirely:

> The upper end of 240 min reflects two binding constraints rather than an attempt
> to approximate a full night: it is the longest context at which every subject
> across all four cohorts still satisfies the recording-length consistency filter
> (Section~sec:datasets), so extending it further would have disproportionately
> excluded cohorts with shorter recordings and shrunk the sample compared across
> context lengths; and, for the Transformer head, it is close to the practical
> memory ceiling of a single accelerator even with Flash Attention's linear-memory
> attention kernel, whose cost still scales with the number of input patches
> (Section~sec:heads).

Recommendation 2 (tie the cap to the sleep-efficiency/sex non-saturation
limitation) was also implemented, folded into the same Discussion paragraph as
§10.3's encoder-scope caveat rather than as a separate addition (they're adjacent
"how far can these specific numbers be trusted" points, and read better together):

> Because 240 min was capped by the longest context at which every cohort retained
> its full subject count after the consistency filter (Methods, Section sec:sweep),
> sleep efficiency and sex classification, both still rising at the longest tested
> context, may have a true $L^*$ beyond 240 min; extending the cohort filter or the
> compute budget to test longer contexts is a direct next step.

Recommendation 3 honored — no claim of indefinite improvement, scoped to "may have a
true $L^*$ beyond 240 min." `npj_main.tex`: `sec:sweep` (Methods) and the Discussion
limitations paragraph (same one edited in §10.3).

---

## 11. Suggested order of operations

1. ✅ **Fix the token-budget cross-reference/claim (§10.4)** — this is a factual-accuracy
   issue, not a style choice, and should be resolved before anything else below:
   either locate and write up the real experiment, or rephrase the sentence to match
   what the supplement actually contains.
2. ✅ Draft and slot in the Introduction "stakes" paragraph (Section 6.1) — highest
   leverage, unblocks the rest of the reframe.
3. ✅ Rewrite the Abstract — shipped a fresh synthesis rather than §10.1's draft;
   see §10.1's implementation note for the final text and why it diverged.
4. ✅ Narrativize the H1–H4 list and the contributions list per §10.2, using §6.3's
   reworded content as the raw material.
5. ✅ Add the Discussion deployment-budget paragraph and extrapolation callback
   (Sections 7.2–7.3) — implemented as one combined paragraph.
6. ✅ Add the encoder-dependence caveat (§10.3) to the Abstract and relocate/sharpen it
   at the top of its Discussion paragraph — implemented as a probing-paradigm
   reframe per the user's tokenizer argument, not the original apologetic framing;
   see §10.3's note for the full assessment. Not added to the Abstract itself (no
   room at 150 words; judged lower priority than the extrapolation finding).
7. ✅ Consolidate the 240-min cap explanation into main-text Methods and link it to the
   sleep-efficiency/sex limitation (§10.5) — consolidated to exactly the two reasons
   the user specified (cohort consistency, GPU memory), not all three originally
   found in the paper.
8. ✅ Tighten the Discussion opening and closing paragraphs (Sections 7.1, 7.4).
9. ✅ Sweep for terminology (Section 3) and verb choice (Section 8.4) — applied inline
   while writing each edit above, not as a separate mechanical pass.
10. ◻ Not done this round: title tweak (Section 4), scaling-laws sentence (Section
    8.3). ✅ Keywords (part of Section 8.1) done. All three are genuine optional
    follow-ups, not oversights — see §4's and §8's implementation notes.

**All of §10's user-specific items (§10.1–§10.5) are also complete** — see each
subsection's "Implemented (round 1)" note for exact shipped text, and §1–§9's notes
for the base reframe. `npj_main.tex` recompiles cleanly (39 pages, two pdflatex
passes, no undefined references, no new warnings beyond pre-existing benign ones).
Only `npj_main.tex` changed; `npj_supplementary.tex` was read for verification but
not edited, since every fix landed in the main text.

**Update on item 10 above:** the scaling-laws sentence marked "not done this round"
was picked up and substantially expanded in Round 2 (§12 below) — it's no longer an
open item.

---

## 12. Round 2 (2026-08-03) — clarity, tone, and scaling-laws emphasis

### Trigger

Two independent signals prompted this round:
1. Several people the user asked to read the round-1 abstract (§10.1's shipped
   150-word version) said it was **unclear and vague**.
2. This echoes, rather than contradicts, the supervisor's original complaint
   (§10 above): a short, heavily compressed abstract built mostly from
   meta-language ("cost-performance frontier," "task-specific design variable")
   can satisfy "sounds advanced" while still failing "is understandable," because
   there isn't enough room left for concrete, graspable specifics. The fix for
   both complaints turned out to be the same direction: more concrete content, not
   less — just organized so it doesn't read as a bare comparison.

The user's instructions for this round: rewrite the abstract starting from the
**pre-round-1 "longer but better" draft** (restored as a comment in the file
between rounds — see the note in `npj_main.tex`'s abstract block), explicitly
allowing it to exceed the 150-word limit for now ("later I will shorten it");
review the Introduction for overclaiming and stiff/bureaucratic phrasing
(specifically flagged: "address" and words like it), aiming for a lighter,
more investigative tone; and substantially increase the visibility of the
context-length extrapolation analysis (§10.1/§5's "cheap-predicts-expensive"
finding) across Introduction, Methods, *and* Results — not just Results — framed
explicitly as adapting scaling-law methodology from the LLM domain to this one,
in professional language, rather than left implicit.

### 12.1 Abstract — rewritten again, this time optimized for clarity over brevity

**Diagnosis of why the round-1 abstract read as vague despite being accurate:**
it led with meta-claims ("temporal context... remains uncharacterized," "an
unexamined computational commitment") before any concrete finding appeared, and
by the time it got to findings, the 150-word budget had already forced maximal
compression (e.g., "negligible benefit beyond a few minutes for body-mass index"
with no other task examples, apnea/age-group dropped entirely to make room for
the extrapolation clause). A reader skimming it gets the *register* of an
important-sounding paper without enough *content* to know what was actually
found.

**Implemented:** rebuilt the abstract using the restored pre-round-1 draft's
concrete, enumerated structure ("Four consistent patterns emerge. First...
Second... Third... Finally...") as the skeleton, keeping the practical-stakes
opening from round 1 (compute/memory cost framing) but restoring the specific
per-task examples (BMI, apnea, age-group, sleep efficiency, sex) and the
temporal-head-advantage finding that the 150-word version had cut, and adding a
new fourth finding for the scaling-law extrapolation result, explicitly framed as
"borrowing an approach common in large language model research." Deliberately
left over the word limit (currently ~350 words) per the user's explicit
instruction; a code comment above the abstract in `npj_main.tex` flags this and
says to trim before final submission. Shipped text is in `npj_main.tex`'s
`\abstract{...}` block (lines ~74–99 as of this round); not reproduced in full
here since it will keep changing, but every sentence maps onto a specific,
already-verified number elsewhere in this document or the paper itself.

### 12.2 Introduction — tone and overclaiming pass

**"Address" and words like it:** searched the full document for "address" as a
verb describing what *this paper* does (not describing what *prior work failed
to do*, which is a different, accurate use). Found and fixed:
- "We address these two open questions by using the frozen SleepFM encoder..."
  → "We explore these two questions using the frozen SleepFM encoder..."
- "Results address H1–H4 in order; full experimental formalisations are in
  Methods" (end of Introduction) → "Results work through H1–H4 in turn; full
  experimental formalisations are in Methods"
- "Results address H1–H4 in order, followed by secondary analyses." (start of
  Results) → "We work through H1–H4 in turn below, followed by secondary
  analyses." (deliberately varied from the Introduction's phrasing so the two
  don't read as a copy-pasted repeat of each other, which was its own minor
  flatness problem)
- Discussion: "...questions the field has not previously addressed in a
  controlled, multi-task setting." → "...questions the field has not previously
  examined in a controlled, multi-task setting." (not in the Introduction, but
  the same word/pattern, fixed for consistency since it's the same issue)

**Overclaiming pass:** the clearest instance found was **"first"** stacking up
across the Abstract and the Introduction's contributions paragraph — "the first
systematic characterization" (abstract) and "the first systematic, cross-task
characterization" (contributions item 1) — an absolute primacy claim repeated
twice in close proximity, which is exactly the kind of thing a skeptical reader
(or reviewer aware of adjacent work) can push back on. Fixed by dropping "the
first" from the contributions item ("a systematic, cross-task characterization"),
letting the existing, already-hedged "To our knowledge, no study asks..." (kept
as-is, appropriately qualified) carry the novelty claim instead of an
unqualified "first." Did **not** find other significant overclaiming beyond this
— checked "establish," "demonstrate," "confirm," "novel," and "groundbreaking"-
adjacent language; remaining instances (e.g., "H1 is confirmed") are tied to the
paper's own pre-registered-hypothesis convention (`CLAUDE.md`) and are accurate,
formal confirmations against a stated criterion, not rhetorical overclaiming, so
left unchanged.

### 12.3 Scaling-laws emphasis — Introduction, Methods, and Results

**The ask:** don't let the context-length extrapolation analysis live only as a
sentence or two in Results; make it a visible thread through the paper, framed
explicitly as bringing scaling-law thinking from the LLM domain into PSG/clinical
prediction for (as far as this study can tell) the first time, phrased with
curiosity rather than as a settled claim.

**Literature grounding used** (from `markdown/SCALING_LAW_ANALYSIS_IDEAS.md`,
Part 2, verified with fresh author-list lookups before citing, since that file
only had "et al." shorthand): Hoffmann, J. et al., *Training Compute-Optimal
Large Language Models* (Chinchilla), NeurIPS 35 (2022) — the canonical
small-scale-predicts-large-scale result; Montgomery, K. et al., *Predicting Task
Performance with Context-aware Scaling Laws*, arXiv:2510.14919 (2025) — the
closest existing precedent, jointly modeling context length and compute for
LLMs; and, for an honest contrast rather than an inflated novelty claim, Zhang,
S. et al., *Exploring Scaling Laws for EHR Foundation Models*, arXiv:2505.22964
(2025), cited to acknowledge that scaling-law thinking is already being explored
for *other* clinical data modalities, so the paper's claim is narrowly scoped to
*context length specifically*, not "scaling laws in healthcare" broadly. All
three added to `npj_main.tex`'s bibliography (keys `chinchilla`, `montgomery2025`,
`ehrscaling`), positioned after the cohort citations (`stages`) to roughly match
where they're first cited in reading order.

**Introduction:** added a new paragraph, between the H1–H4 paragraph and the
contributions paragraph, that motivates the extrapolation analysis explicitly as
a scaling-laws import: states that expensive context lengths are also the most
costly to train, that LLM research has shown performance follows fittable
compute/context relationships, notes the EHR-scaling contrast above, and states
plainly "to our knowledge, [this is] untested for physiological or clinical
prediction models," then poses it as a question ("We were curious whether it
would [hold]...") rather than a settled finding. The contributions paragraph's
third item was reworded from a bare statement of the numeric result into "an
early test of whether scaling-law-style extrapolation, adapted from large
language model research, transfers to this setting," so the framing is present
at the point of first mention, not just in a later methods/results callback.

**Methods:** added a new subsubsection, "Context-Length Extrapolation
(Scaling-Law Analysis)," in `sec:evaluation` (Evaluation Protocol), between
"Saturation Threshold $L^*$" and "Confidence Intervals." States the LLM-research
motivation with the same two citations, gives the power-law functional form as a
numbered equation (`\eqref{eq:powerlaw}`, $\text{AUROC}(L) = c - aL^{-b}$),
describes the four-cheap-points-predict-two-expensive-points held-out design, and
points to the sensitivity check (three-point and two-point refits) and full
results in Supplementary Section S-XVIII.A rather than duplicating the
supplementary's numbers. This methodology previously existed only in the
supplementary; the main text now describes it as its own procedure rather than
leaving a reader to infer it from a single Results sentence.

**Results:** split the extrapolation finding out of the paragraph it was fused
into (a cross-cohort robustness check, an unrelated analysis it happened to sit
next to) and gave it its own paragraph with an explicit framing sentence: "We
then asked a different kind of question, closer in spirit to scaling-law studies
of large language models than to a standard robustness check..." The paragraph
now cross-references the new Methods equation (`Methods~\eqref{eq:powerlaw}`)
and closes with an interpretive sentence ("an encouraging early sign for
extending scaling-law-style reasoning beyond language models") instead of
dropping straight into "H1 is confirmed" with no acknowledgment that a
qualitatively different kind of claim was just made.

**Discussion (not explicitly requested, done as a low-risk extension):** the
existing deployment-guidance paragraph (added in round 1, §7.2/§7.3) already
described the extrapolation result in practical terms; added one closing
sentence there naming the LLM-scaling-law parallel explicitly with the same two
citations, so a reader who only reads Discussion (skipping Methods/Results)
still encounters the framing once. This is the "scaling-laws sentence" item that
round 1's §8.3 had flagged as optional and deferred — now done, substantially
expanded beyond a single sentence.

### 12.4 What was deliberately not done this round

- **Abstract not trimmed to 150 words** — explicit user instruction to leave it
  long for now; flagged in a code comment in `npj_main.tex` as a pre-submission
  TODO.
- **No new experiments** — the extrapolation analysis, its numbers, and its
  sensitivity results were already computed and validated in round 1 and earlier
  sessions (`scaling_law_ideas/idea_b_context_extrapolation.py` in NSRR-tools);
  this round is entirely a presentation/framing change, consistent with the
  original diagnosis in §1 that this is a positioning problem, not a results
  problem.
- **Supplementary (`npj_supplementary.tex`) not edited** — Section S-XVIII.A
  already contains the full extrapolation results and sensitivity analysis
  accurately; the main text now references it more prominently rather than
  duplicating its content.
- **Did not soften "H1 is confirmed" or similar hypothesis-confirmation
  language** — judged accurate and convention-bound (`CLAUDE.md`'s hypothesis
  framing rules), not overclaiming, per §12.2's assessment above.

### 12.5 Verification

`npj_main.tex` recompiled cleanly after each of the four edits in this round
(abstract; introduction; methods+results+discussion; this document), two
pdflatex passes each time, no undefined references, no new warnings beyond the
pre-existing benign hyperref ones. Four separate commits were made in the
`npj_digital_medicine_submission` repo, one per edit, per the user's "commit
after each step" instruction; this document was updated and committed
separately in this repo (`NSRR-tools`) as the fifth and final step.
