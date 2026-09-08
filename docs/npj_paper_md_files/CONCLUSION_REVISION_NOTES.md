# Conclusion Rewrite + "Not a Universal Recipe" Clarity Audit (2026-08-22)

**Not yet implemented.** The user is currently applying a separate, earlier
round of small wording fixes (Intro/Discussion cost-vs-context balance,
iso-compute→iso-budget rename, epoch/segment/patch terminology cleanup — all
already applied directly to `npj_main.tex`) and asked for this round to be
written up here instead of implemented immediately, so the two rounds don't
collide. Line numbers below are current as of this write-up
(`npj_main.tex`, ~2500 lines total after those other edits) and will drift —
re-grep the quoted phrases before editing rather than trusting the numbers.

## 1. Trigger

The user flagged that the Conclusion (end of Discussion, currently starting
around line 1412 with "This paper establishes temporal context as a
task-specific resource-allocation variable...") reads as "cheap" and
"only comparison-based" — i.e., it sounds like the paper's contribution is
"we compared some context lengths and measured cost," rather than the
actual, more valuable point. The user also shared, verbatim, how they
explained the paper's actual goal to their supervisor (in response to his
question "what is the optimal context length for each task?"):

> "...My goal is not to provide a general recipe that says 'context length X
> is optimal for task Y.' Rather, I first want to show that using only a few
> seconds to a few minutes of physiological data (as is common in many sleep
> studies) is not necessarily sufficient for all tasks. Then, I want to say
> that for some tasks, training on short contexts and simply aggregating
> predictions during inference is enough, while for others, longer contexts
> are necessary. The idea is to understand when the additional compute and
> memory required for long-context training are actually justified. I also
> want to show, similar to findings in the LLM literature, that it may be
> possible to predict long-context performance from experiments with shorter
> contexts (through our scaling-law experiments), allowing us to estimate
> whether the expected performance gain is worth the additional
> computational cost. I tried to emphasize that these findings are not
> meant to be general guidelines, but rather observations under the specific
> experimental settings we used, and I described those settings in detail to
> ensure reproducibility."

Four distinct points are packed into that explanation, and the current
Conclusion captures only fragments of them:
1. Short context (seconds-to-minutes) is not automatically sufficient —
   this is a real, non-obvious finding, not a comparison-study footnote.
2. Tasks split into two regimes: short-context-plus-aggregation is enough
   for some, long-context training is required for others. The point of
   measuring this is to know **when the added compute/memory cost is
   justified** — cost matters, but as the answer to a question, not as the
   question itself.
3. Scaling-law-style extrapolation can predict expensive-context
   performance from cheap-context runs — a practical tool, not just a
   methodological curiosity.
4. None of this is a universal recipe. It is a reproducible measurement
   under one encoder/cohort/protocol, described in enough detail that
   others can repeat it under their own settings.

The current Conclusion states results but leads with "resource-allocation
variable" / "cost-performance frontier" as the paper's self-identity (point
2's *conclusion* without point 1's *finding*), and doesn't restate points 3
or 4 at all — even though point 4 already exists elsewhere, buried at the
very end of the Discussion body (see §3 below). A Conclusion is the part
most readers and reviewers weight most heavily on a second pass, so it's
worth having it carry the full argument, not just the cost half of it.

## 2. Suggested Conclusion rewrite

Replace the two Conclusion paragraphs (currently ~line 1412–1436, "This
paper establishes..." through "...alternative encoder architectures.")
with:

> This paper shows that the amount of overnight recording a clinical
> prediction task requires is not fixed or universal. For some tasks, a few
> minutes of signal, aggregated appropriately at inference, matches the
> performance of training on hours of recording; for others, only
> substantially longer context closes a persistent gap that no amount of
> aggregation recovers. Because longer context also costs more to train
> and deploy, this distinction determines when that added compute and
> memory investment is actually justified, rather than assumed by
> convention. Sweeping six context lengths across seven clinical tasks and
> four cohorts with a frozen foundation model encoder makes this
> task-by-task distinction concrete and measurable. Saturation behaviour
> was task-specific: tasks grounded in full-night physiology continued to
> improve through the longest contexts evaluated, tasks driven by periodic
> or cumulative phenomena saturated at intermediate contexts, and
> metabolic targets showed minimal context benefit across the sweep.
> Inference-time aggregation partially recovered performance at matched
> signal budgets but could not substitute for long-context training on
> context-sensitive tasks; in our experiments, a small number of inference
> windows captured most of the available aggregation benefit. Temporal
> heads gained increasing advantage over mean-pooling as context grew, and
> the Transformer generally outperformed the LSTM, with both gaps widest
> on the most context-sensitive tasks.
>
> Taken together, these findings establish temporal context as a
> consequential, task-specific design variable in PSG model development,
> one whose appropriate setting cannot be assumed a priori and whose
> interaction with inference-time aggregation has direct, quantifiable
> implications for compute efficiency. Because the most informative
> context lengths are also the most expensive to evaluate, we further show
> that scaling-law-style extrapolation, borrowed from language-model
> research, can predict performance at these expensive settings from
> cheaper, shorter ones — offering a practical way to estimate whether a
> longer context is worth its cost before training it. These are not
> universal recommendations: the saturation points reported here reflect
> this specific encoder, these cohorts, and this training protocol,
> described in full so that they can be reproduced, tested, and extended
> under different settings. What we offer instead is a methodology and a
> concrete empirical baseline: a way to measure, for a new task and
> setting, whether more context is worth its cost, rather than defaulting
> to convention.

What changed and why:
- **Paragraph 1** now opens with the finding (context need is not fixed,
  splits into two regimes) instead of the label ("resource-allocation
  variable"). Cost is still there — "because longer context also costs
  more... this distinction determines when that added investment is
  justified" — but as the payoff of the finding, not the finding's name.
  Everything after "Sweeping six context lengths..." is your existing text,
  unchanged.
- **Paragraph 2** keeps your existing first sentence (task-specific design
  variable, compute efficiency) almost as-is, then adds two new sentences
  that were missing entirely: the scaling-law capability restated as a
  standalone contribution, and an explicit "these are not universal
  recommendations" statement — almost a direct paraphrase of what you told
  your supervisor — before closing with "a methodology... rather than
  defaulting to convention," which reframes the paper's contribution as a
  reusable measurement approach, not a per-task answer key.

## 3. Clarity audit: does the rest of the paper already avoid the "best length per task" reading?

Re-read the Abstract, Introduction, and Discussion specifically checking
whether a reader could come away thinking the paper's goal is "tells you
the optimal context length for task Y." Good news first: **most of it
already avoids this**, and doesn't need touching:

- **Abstract**: describes the four patterns descriptively ("body mass index
  classification gains almost nothing beyond a few minutes... sleep
  efficiency and sex classification continue to improve through the
  longest context evaluated") rather than prescriptively, and closes with
  "a practical basis for choosing context length deliberately instead of by
  convention" — a decision-tool framing, not an answer-lookup. Fine as is.
- **H1** (~line 222): "AUROC rises with context length and saturates at a
  **task-specific** threshold $L^*$" — already explicit that the saturation
  point is per-task, not universal. Fine as is.
- **Intro's "this choice is not free" paragraph** (~line 150): already
  frames the risk as two-sided — over-provisioning *and*
  under-provisioning depending on the task — which is consistent with "it
  depends on the task," not "there's one right answer." Fine as is.
- **Discussion's limitations paragraph** (~line 1340, "The encoder was
  frozen throughout..."): already states directly that "a differently
  pretrained or fine-tuned encoder might pack information differently
  across patches, shifting where a given task's $L^*$ falls" — i.e.,
  saturation points are conditional on the encoder, not universal
  constants. Fine as is.
- **Discussion's closing paragraph before the Conclusion** (~line 1389,
  "More broadly, the observations reported here depend on data scale,
  cohort composition..."): this is where point 4 (not a universal recipe)
  is actually already stated clearly — "What this and similar studies can
  contribute is a transparent account of the conditions under which
  results were obtained, so that subsequent work can reproduce findings in
  new settings..." This is good, existing text. **The problem isn't that
  this caveat is missing from the paper — it's that it's buried at the very
  end of Discussion and never echoed in the Conclusion**, which is why the
  rewrite in §2 restates it there too.

Two spots worth a small tweak, not a rewrite:

1. **Contribution 1** (Intro, ~line 244): *"a systematic, multi-cohort
   characterization of how much temporal context each prediction task
   requires, establishing a cost-performance frontier for deploying these
   models in practice."* On its own, out of context, "how much context each
   task requires" can be misread as "we determined the required length per
   task" (an answer), rather than "we measured how this varies" (a
   diagnostic). Suggested tweak — add a scope qualifier: *"...a systematic,
   multi-cohort characterization of how context requirements vary across
   seven clinical prediction tasks under a fixed encoder and training
   protocol, establishing a cost-performance frontier..."* The added clause
   ("under a fixed encoder and training protocol") does the same job as
   H1's "task-specific" — it signals the finding is conditional, without
   weakening the claim.
2. **Optional, lower priority**: consider one sentence near the end of the
   Introduction (after the four contributions, ~line 254) that states the
   scope caveat early rather than only at the very end of Discussion — e.g.
   *"These patterns are intended as evidence that temporal context
   requirements are measurable and task-dependent, not as a universal
   recipe; Discussion returns to this scope in detail."* This is optional
   because the caveat does already exist twice (Discussion's limitations
   paragraph, and now the Conclusion after §2's rewrite) — adding a third,
   earlier instance is a judgement call about how much a first-time reader
   needs to be told this before reaching Results, not a correctness issue.

## 4. Summary of what to add/remove/edit, and where

| Location (approx. current line) | Action | Priority |
|---|---|---|
| Conclusion, ~1412–1436 | Replace both paragraphs with §2's rewrite | High — this was the reported problem |
| Contribution 1, Intro ~244 | Add "...under a fixed encoder and training protocol" scope qualifier | Medium — real ambiguity, small fix |
| End of Introduction, ~254 | Optionally add one forward-pointing scope-caveat sentence | Low — optional, redundant with existing caveats elsewhere |
| Abstract | No change | — already descriptive, not prescriptive |
| H1 statement, ~222 | No change | — already says "task-specific" |
| Intro "not free" paragraph, ~150 | No change | — already two-sided (over/under-provisioning) |
| Discussion limitations paragraph, ~1340 | No change | — already states encoder-conditionality |
| Discussion closing paragraph, ~1389 | No change | — already states the "not a universal recipe" point; the fix is that the Conclusion now echoes it, not that this paragraph needs editing |
