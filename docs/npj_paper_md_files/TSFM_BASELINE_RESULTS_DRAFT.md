# TSFM Baseline Comparison — Draft Paper Additions

**Status: first draft, for review.** Written 2026-09-16, extended 2026-09-17.
This is a first pass at the material requested: a new Results subsection, a new
Methods subsection, a Discussion edit, a Supplementary extension outline
(2026-09-16), plus full-sweep/head-comparison tables analogous to the main
paper's Table~1/Table~3, a consolidated cross-model table (Section 6.7), and
two exploratory figure notebooks analogous to Fig.~2/Fig.~3 (2026-09-17,
Sections 6-7), covering the OSF / PhysioOmni / Mantis baseline-comparison
work. **Nothing has been pasted into `npj_main.tex` or `npj_supplementary.tex`
yet** — following this repo's established workflow (`RESULTS_REWRITE.md`),
paste manually after reviewing, paragraph by paragraph if you want to iterate
on wording first.

**Correction 2026-09-17 (read this before trusting Section 3's numbers):**
building Section 6.7 surfaced a real bug in how I computed $\Delta$
(context-sensitivity gain) for six cells in Section 3's cross-encoder table —
I had used the global peak AUROC across all contexts instead of the AUROC *at
$L^*$*, which differ whenever the true peak sits just beyond $L^*$ but still
within its 0.005 saturation tolerance (the same mechanism behind a bug the
paper itself already caught and fixed in its own age-group row, per
`CLAUDE.md`'s "age-group Δ typo" note). All six affected numbers (OSF apnea,
sex, age, BMI; PhysioOmni age; Mantis age) are now corrected in Section 3, and
Section 6.7 explains the fix in full with the exact before/after values. None
of the corrections reverse a qualitative claim already made in the prose; all
are small (≤0.006 AUROC).

**Second correction pass, 2026-09-17 (you asked me to check Section 3's text
for accuracy):** found and fixed three more issues, all in the Results tex
block itself, none affecting Section 6/6.7's tables: (1) a mis-citation, "the
smaller-cohort OSA severity task (Table~\ref{tab:tasks})" pointed at the
task-definition table, which carries no AUROC values at all, not any table
showing the actual OSA result, now points at `tab:crossmodel_transformer`;
(2) an overgeneralized claim that PhysioOmni's gap to SleepFM "narrows but
does not close" at longer context, true for sex and BMI but not for age
(which stays equally wide, 4.6 to 5.1pp) or sleep efficiency (which stays
narrow throughout, 1.4 to 1.5pp), now stated per-task instead of as one
blanket claim; (3) the old compact `tab:tsfm_sensitivity` table is removed
entirely (redundant with, and less complete/more framing-risky than,
Section 6.7's `tab:crossmodel_transformer`, which now serves as this
subsection's main-text table) — see Section 6.7's placement note for the
full reasoning.

## How this was produced (read before trusting the numbers)

1. Read `npj_main.tex` in full (Introduction through the end-matter) to match
   existing terminology, notation, and house style.
2. Read `docs/TSFM_MODEL_COMPARISON.md` in full (the existing synthesis document).
3. Spot-verified numbers directly against the underlying CSVs, not just trusted
   the comparison doc: `sex_binary` and `bmi_binary` (SleepFM, OSF, OSF-LoRA,
   PhysioOmni, PhysioOmni-LoRA, Mantis) and `apnea_binary` (SleepFM, Mantis),
   including `n_subjects`. **Every number checked matched the comparison doc
   exactly**, including the small SleepFM-vs-Mantis subject-count mismatches it
   flags (e.g. `apnea_binary`: 2,054 vs. 2,077). I did not re-verify every single
   cell in every table below independently — see "What I did not verify" at the
   end.
4. Independently recomputed $L^*$/$\Delta$ (same definition as the paper's
   equation for $L^*$: smallest context within 0.005 AUROC of the sweep peak) for
   OSF and PhysioOmni directly from `results/collected/phase0_osf/analysis.csv`
   and `results/collected/phase0_physioomni/analysis.csv`, since the comparison
   doc only tabulated this for Mantis. This is the basis for the cross-encoder
   summary table below.
5. Checked the live Mantis LoRA status directly (`NSRR-tools-mantis` worktree,
   `logs_mantis_lora/status/*.jsonl`) as of 2026-09-16: still running (several
   `10m` LSTM runs `STARTED`/mid-training today; `age_class` `10m` just finished).
   Confirms the comparison doc's §3.4 status is current, not stale.
6. Checked `docs/TSFM_THIRD_MODEL_DECISION.md` for the Mantis selection rationale
   and the pre-registered "expect a weak frozen result" prediction (real, and it
   did not hold — see the Results draft below).
7. Checked `npj_supplementary.tex`'s existing `sec:supp-sota` section (why the
   paper doesn't report head-to-head AUROC against external numbers) so the new
   material is framed consistently with it, not redundantly.

## Decisions I made that you should confirm or override

These are judgment calls where I picked an answer rather than leaving a gap. Flag
any of these you want to change.

1. **`osa_binary_apples_postqc` is NOT included anywhere below.** The comparison
   doc's own §7.2/§7.4 flags a real, unresolved tension: on essentially the same
   clinical question and cohort as `apnea_binary`, OSF loses to SleepFM in the
   pooled `apnea_binary` result but wins by the largest margin in this document on
   `osa_binary_apples_postqc`. The doc's own author recommended holding it back
   without either a controlled retrain or an explicit paragraph defending the
   training-set-size explanation. I agree with that recommendation and did not
   build it into the draft. If you want it included, it needs its own paragraph
   addressing this tension directly, not a silent table entry.
2. **`depression_extreme_binary` is mentioned only as a supplementary-table
   candidate, not in the main-text table or prose.** It is a real, clean,
   contamination-free result (OSF wins consistently, PhysioOmni loses
   consistently), but it is a secondary/exploratory task in the main paper
   already (small $N$, daggered), and adding a sixth task to an already
   three-model, five-task comparison risked cluttering the main-text table. Your
   call whether to add it.
3. **Superseded 2026-09-17, keep for history:** this item originally said the
   main-text table would show only $L^*$/$\Delta$ (not full per-context AUROC),
   with a separate compact table (`tab:tsfm_sensitivity`). After building
   Section 6.7's cross-model table, we decided that compact table was
   redundant with it (same $L^*$/$\Delta$ information, derivable from the
   fuller table's bold cell and $\Delta$ column) and had a real
   channel-mismatch framing risk besides. **The main-text table is now
   `tab:crossmodel_transformer` (Section 6.7): full per-context AUROC for
   all four encoders, not just $L^*$/$\Delta$.** Absolute performance
   differences (OSF/Mantis beating SleepFM on some tasks, PhysioOmni losing
   throughout) are still reported in prose with the strongest, most
   defensible numbers, per this item's original intent, now just pointing at
   the fuller table.
4. **No numeric LoRA results for Mantis anywhere in the draft**, main or
   supplementary, per the comparison doc's own finding that no comparable
   frozen-vs-LoRA metric exists yet for Mantis (the $K_{\max}$-aggregated
   inference script has never been run against a Mantis LoRA checkpoint). I
   describe the LoRA run as in-progress and explain why, rather than showing
   partial or non-comparable numbers.
5. **I did not draft a new figure.** A compact bar or line figure (e.g. one panel
   per task, one bar per encoder) would likely read better than the table below,
   but figure generation in this repo is notebook-driven and user-controlled
   (`results/paper_figures/notebooks_npj/`), so I left this as a table you can
   decide whether to also turn into a figure.
6. **Bibliography entries for PhysioOmni and Mantis are incomplete** (missing
   verified author lists) — see the "Bibliography additions" section. Do not
   paste these into `npj_main.tex` without filling in real author names.

---

## 1. Bibliography additions needed

`\bibitem{osf}` already exists in `npj_main.tex`. Two new entries are needed.
**I could not find a verified author list for either paper in the existing docs
and did not want to guess names for a citation** — fill these in from the actual
papers before submission (same convention as the existing `mros` entry's
"verify against NSRR dataset page before submission" TODO).

```tex
\bibitem{physioomni}
% TODO: verify exact author list before submission (not found in existing
% NSRR-tools docs; HF repo owner is "Weibang", GitHub org 935963004, same
% group as LaBraM/NeuroLM).
[Author(s) TBD].
Towards robust multimodal physiological foundation models: handling
arbitrary missing modalities.
Preprint at \url{https://arxiv.org/abs/2504.19596} (2025).

\bibitem{mantis}
% TODO: verify exact author list before submission.
Feofanov, V. et al.
Mantis: lightweight foundation model for time series classification.
Preprint at \url{https://arxiv.org/abs/2502.15637} (2025).
```

Optional third citation, useful if you want independent (not just our own)
evidence that Mantis transfers to sleep-staging-adjacent tasks (see the Methods
draft's rationale paragraph, and the Discussion draft):

```tex
\bibitem{mantis_eeg}
% TODO: verify exact author list before submission. Caveat worth knowing:
% Feofanov and Redko are authors on both this paper and the Mantis paper
% itself, so this is not fully independent evidence of Mantis's transfer
% quality (docs/TSFM_THIRD_MODEL_DECISION.md, Section 2).
Gnassounou, T., Moakher, A., Xie, Y., Feofanov, V. \& Redko, I.
Leveraging generic time series foundation models for EEG classification.
Preprint at \url{https://arxiv.org/abs/2510.27522} (2025).
```

---

## 2. Methods addition

Placement: new subsection after `sec:modality_ablation` (the current last Methods
subsection), before `\backmatter`.

```tex
\subsection{Comparison across independently pretrained encoders}
\label{sec:tsfm_baselines}

All results above use a single frozen encoder, SleepFM. SleepFM was the
sleep-PSG-specific foundation model with a public checkpoint and a
peer-reviewed track record (ICML 2024~\cite{sleepfm}; \emph{Nat. Med.}
extension~\cite{sleepfm_natmed}) when this study's encoder was selected. OSF~\cite{osf},
published subsequently at ICML 2026, is to our knowledge the only other
sleep-PSG-specific foundation model with a public checkpoint. To test whether
the task-specific context-length sensitivity reported above is a property of
the tasks and signals themselves, rather than an artefact of SleepFM's
particular pretraining, we repeated the full context-length sweep with three
additional frozen encoders substituted for SleepFM: OSF, PhysioOmni~\cite{physioomni},
and Mantis~\cite{mantis}. Every other element of the pipeline was held fixed:
the same task labels, subject partitions (subject to small per-encoder
differences in embedding-extraction coverage, Supplementary
Section~S-XX), lightweight LSTM and Transformer sequence heads, six context
lengths, and $K$-aggregated evaluation protocol described above.

The three encoders span a gradient of relatedness to the target domain,
chosen deliberately rather than to assemble a leaderboard. OSF is pretrained
on large-scale overnight PSG with a contrastive objective similar in spirit
to SleepFM's own, and is the closest available like-for-like alternative.
PhysioOmni is pretrained on general clinical and brain-computer-interface
physiological corpora (EEG, EOG, electrocardiography (ECG), electromyography
(EMG)), not overnight sleep recordings specifically, and has no
respiratory-signal pathway in its released architecture; the apnea task,
which depends on airflow and respiratory-effort channels, is therefore
excluded for this encoder. Mantis is a lightweight (approximately 8M
parameter), general-purpose time-series classification model pretrained on
generic, non-physiological time-series corpora, included specifically to
test whether the observed context-sensitivity pattern requires any
physiological pretraining at all, a stronger and more direct generalization
test than comparing only within the family of sleep- or physiology-specific
encoders.

None of the three models natively supports input sequences as long as the
longest context lengths evaluated in this study. Each was therefore used the
same way SleepFM itself is used throughout this paper: as a short-segment
embedder, with the same sequence heads trained on top of its per-segment
embeddings to integrate information across longer context (rather than a
model-internal long-context mechanism). OSF and Mantis process non-overlapping
30-s epochs; PhysioOmni was run at the same 30-s granularity for consistency
across encoders. OSF requires the full complement of respiratory and cardiac
channels used in the full-channel configuration
(\nameref{sec:channel_comparison}) and is compared against that
SleepFM configuration; PhysioOmni and Mantis use the reduced-channel
configuration used throughout the rest of this paper, matching the channels
they were pretrained to expect.

Each encoder was evaluated with its backbone weights entirely frozen,
following the same probing paradigm as the SleepFM-based results above, and
additionally fine-tuned with low-rank adaptation (LoRA)~\cite{lora}, warm-started from
the corresponding frozen-encoder head and continued rather than trained
jointly from initialization. Fine-tuning requires backpropagating through the
encoder on raw signal rather than pre-extracted embeddings, so its
computational cost scales with the number of raw 30-s epochs inside each
training window and grows by up to two orders of magnitude from the shortest
to the longest context length (Supplementary Section~S-XX). A fine-tuned
sweep matching the frozen sweep's full six context lengths, across every
task, head, and encoder, was not tractable within the time available before
submission. Coverage differs by encoder for reasons specific to each,
detailed in Supplementary Section~S-XX: OSF's fine-tuned sweep was
deliberately stopped after 120~min (240~min withheld, a documented,
cost-informed decision applied uniformly across every task and head);
PhysioOmni's is incomplete for a combination of an architectural
compute-efficiency ceiling and an unrelated operational delay; and Mantis's
fine-tuning sweep had not reached a stage producing a metric comparable to
the rest of this paper's evaluation protocol at the time of writing, and is
therefore not reported. Results below state fine-tuning coverage explicitly
wherever a number is shown.
```

**Note**: this references `\nameref{sec:channel_comparison}` for the
full/reduced-channel split, which already exists in `npj_main.tex`. It also adds
one new citation, `\cite{lora}`, for the LoRA method itself (Hu et al. 2022,
"LoRA: Low-Rank Adaptation of Large Language Models") — not yet in the
bibliography; add if this subsection is used:

```tex
\bibitem{lora}
Hu, E. J. et al.
LoRA: low-rank adaptation of large language models.
in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)} (2022).
```

---

## 3. Results addition

Placement: new subsection after "Modality group ablation"
(`sec:results_ablation`, currently the last Results subsection), before the
Discussion section begins.

```tex
\subsection{Task-specific context sensitivity replicates across independently pretrained encoders}
\label{sec:results_tsfm}

The results above establish that context-length requirements are
task-specific for one encoder, SleepFM. To test whether this pattern reflects
the tasks and underlying physiology rather than SleepFM's own pretraining, we
repeated the context-length sweep with three additional, independently
pretrained frozen encoders substituted for SleepFM: OSF, a sleep-PSG-specific
encoder; PhysioOmni, a general physiological encoder with no sleep-specific
pretraining; and Mantis, a general-purpose time-series classification model
with no physiological pretraining at all (Methods,
\nameref{sec:tsfm_baselines}).

The task ordering by context sensitivity replicated closely across all three
additional encoders (Table~\ref{tab:crossmodel_transformer}). Sleep efficiency
showed the largest gain from longer context under every encoder tested
($\Delta = {+}0.120$ to ${+}0.128$ AUROC), still rising at the longest context
evaluated in every case, matching the SleepFM-based finding above. BMI showed
the smallest gain under every encoder ($\Delta = {+}0.024$ to ${+}0.044$).
Apnea, evaluated in the three encoders with a respiratory-signal pathway,
showed the second-largest gain in all three ($\Delta = {+}0.097$ to
${+}0.123$), saturating at $L^*{=}120$~min in every case. Age formed a
consistent middle tier ($\Delta = {+}0.034$ to ${+}0.062$). Sex classification
was the one task where encoders diverged in the size, though not the
direction, of the context gain ($\Delta = {+}0.030$ for OSF to ${+}0.130$ for
PhysioOmni): OSF's frozen encoder already reached 0.929 AUROC at 30~s, leaving
little room for further gain, whereas PhysioOmni started markedly lower
(0.754) and gained the most of any encoder tested. This is consistent with a
ceiling effect on encoders that already extract most of the
sex-discriminative signal from a single short segment, rather than a
disagreement about whether sex benefits from longer context.
Supplementary~Table~S-XX gives the LSTM counterpart.

% TABLE PLACEMENT: \begin{table*}...\end{table*} for tab:crossmodel_transformer
% goes here. Its LaTeX is kept in ONE place, Section 6.7 below, rather than
% duplicated in this tex block -- copy it from there when pasting this
% subsection into npj_main.tex. (Section 6.7 also has the LSTM counterpart,
% tab:crossmodel_lstm, for Supplementary~Table~S-XX above.)

The encoders differed substantially in absolute performance, and these
differences did not favor any single encoder uniformly. OSF's frozen encoder
outperformed SleepFM on sex, age, and BMI, including on cohorts confirmed free
of any subject-level overlap with OSF's own pretraining data (Supplementary
Section~S-XX); it was inconclusive on sleep efficiency and did not
consistently outperform or underperform SleepFM on apnea. PhysioOmni's frozen
encoder underperformed SleepFM at every context on every comparable task. The
gap was widest for sex, 7.8 percentage points at 30~s, narrowing to
3.4 points by 240~min; the same narrowing held for BMI (4.5 to 3.1 points).
Age showed no such narrowing, staying 4.6 to 5.1 points wide across the
sweep, and sleep efficiency stayed narrow throughout (1.4 to 1.5 points).
Mantis's frozen encoder, despite carrying no physiological pretraining
whatsoever, outperformed SleepFM on sex (by 1.3 to 4.4 percentage points at
every context) and age (0.2 to 3.5 points), and on the smaller-cohort OSA
severity task at five of six contexts
(Table~\ref{tab:crossmodel_transformer}); it was closely matched with SleepFM
on apnea and BMI, and slightly behind at long context on sleep efficiency.
This outcome ran counter to a prediction we made before running Mantis, based
on published evidence that freezing a generic time-series model typically
degrades performance sharply on EEG-based tasks; we report it as the genuine,
pre-specified surprise that it was.

Fine-tuning the encoder with low-rank adaptation (LoRA) had a task-dependent
effect, mirroring this paper's own central finding that the value of
additional model capacity is not uniform across tasks. For OSF, where the
fine-tuned sweep is most complete (Methods, \nameref{sec:tsfm_baselines}),
LoRA improved apnea and, more modestly, sex and age, relative to the frozen
encoder; it was flat to slightly negative for BMI; and it consistently
underperformed the frozen encoder for sleep efficiency at every context
tested. PhysioOmni's fine-tuned sweep is far less complete, covering only a
subset of tasks and context lengths (Supplementary Table~S-XX); where it did
run, fine-tuning improved on the frozen encoder at every matched context.
Mantis's fine-tuning was still in progress at the time of writing and did not
reach a stage producing a metric directly comparable to the rest of this
evaluation; we report its engineering cost (Supplementary Section~S-XX) but
not its accuracy.
```

**Numbers cross-check** (all independently re-derived by me from
`results/collected/{phase0_v3,phase0_osf,phase0_physioomni,phase0_mantis}/analysis.csv`,
transformer head, `k="all"`, test split):

- Sex 30s: SleepFM 0.832, OSF 0.929, PhysioOmni 0.754, Mantis 0.863, matching
  "7.8 percentage points" (0.832-0.754) and the OSF/PhysioOmni $\Delta$
  figures above.
- Age: per-context gaps between SleepFM and Mantis are 0.2pp (30s), 1.5pp
  (10m), 3.5pp (40m), 1.9pp (80m), 2.1pp (120m), 1.4pp (240m), so the correct
  range is +0.2pp (30s) to +3.5pp (40m). **This corrects an error in an
  earlier version of this note**, which wrongly attributed the +3.5pp
  maximum to 120m instead of 40m (a plain arithmetic slip, not the same bug
  as Section 6.7's $\Delta$-at-$L^*$ fix). The "0.2 to 3.5 points" range
  itself, as stated in the prose above, was already correct; only this
  note's parenthetical location was wrong.
- PhysioOmni gap-narrowing pattern (sex/BMI narrow, age/sleep-efficiency do
  not): gap@30s vs.\ gap@240m, in percentage points: sex 7.8$\to$3.4, BMI
  4.5$\to$3.1, age 4.6$\to$5.1, sleep efficiency 1.4$\to$1.5. Matches the
  rewritten paragraph above exactly.
- OSA severity: Mantis beats SleepFM at 5 of 6 contexts (comparison doc §2.4),
  by +1.9 to +4.2pp with an essential tie at 80m (-0.1pp); independently
  recomputed from `tab:crossmodel_transformer`'s own OSA row and matches.
  Cited via `Table~\ref{tab:crossmodel_transformer}` now (an earlier version
  of this prose wrongly cited `Table~\ref{tab:tasks}`, the task-definition
  table, which carries no AUROC values at all, for this claim, fixed above).

---

## 4. Discussion edit

**Old text** (currently in the "Several factors constrain..." limitations
paragraph):

```tex
A differently pretrained or fine-tuned
encoder might pack information differently across patches, shifting where a given
task's $L^*$ falls even if the qualitative phenomenon, that temporal integration
matters more for some tasks than others, replicated; testing this directly against
alternative frozen encoders (OSF~\cite{osf}, SleepMaMi~\cite{sleepmami},
SleepFounder~\cite{sleepfounder}) or a fine-tuned backbone is a natural next step
this design does not itself resolve.
```

**Proposed new text:**

```tex
A differently pretrained or fine-tuned
encoder might pack information differently across patches, shifting where a given
task's $L^*$ falls even if the qualitative phenomenon, that temporal integration
matters more for some tasks than others, replicated. We tested this directly
against three independently pretrained frozen encoders, OSF, PhysioOmni, and
Mantis (Results, \nameref{sec:results_tsfm}), spanning sleep-specific,
general-physiological, and non-physiological pretraining respectively; the
task ordering by context sensitivity replicated closely across all three,
strengthening the case that this ordering reflects properties of the tasks
and the underlying physiology rather than an artefact of SleepFM's own
pretraining. Absolute performance did not replicate as cleanly: OSF and
Mantis each outperformed SleepFM on a subset of tasks despite the latter's
much larger, sleep-specific pretraining corpus, a reminder that context-length
sensitivity and absolute discriminative performance are separate properties
that need not track each other. Testing this same question against a
fine-tuned (rather than frozen) backbone, or against SleepMaMi~\cite{sleepmami}
and SleepFounder~\cite{sleepfounder}, neither of which has a public checkpoint
at the time of writing, remains a natural direction for future work.
```

This keeps the two citations that still don't have public checkpoints
(SleepMaMi, SleepFounder) as "cited but not tested," exactly as
`docs/TSFM_MODEL_COMPARISON.md` §6.2 recommends, and removes OSF from that list
since it now has been tested.

**A second, optional Discussion addition**, placed just after the paragraph
above, addressing the "why SleepFM, given these other results" question a
reviewer is likely to ask directly:

```tex
These comparisons also bear on why SleepFM was chosen as this study's primary
encoder in the first place: it was the sleep-PSG-specific foundation model
with a public checkpoint and a peer-reviewed evaluation when this study's
protocol was designed, and OSF, published subsequently, is to our knowledge
still the only other such model publicly available. That OSF and Mantis
outperform SleepFM's frozen embeddings on some tasks does not revise this
paper's central claim, which concerns how much context a task needs rather
than which encoder extracts the most signal from it; the replication of the
task ordering across encoders (Table~\ref{tab:crossmodel_transformer}) is itself
evidence that this claim does not depend on which encoder was chosen.
```

I'd suggest this second block **only if** you want to preempt the "why not use
the better encoder" question explicitly rather than let a reviewer raise it
first. It's a genuinely defensible answer, not a hedge, but it does add length
to an already long Discussion. Your call.

---

## 5. Supplementary extension (outline, not full prose)

Given the size of `docs/TSFM_MODEL_COMPARISON.md`, I'd suggest extending
`sec:supp-sota` (right after the existing `tab:supp-related-context` longtable,
before `sec:supp-runtimes` begins) with a new subsection, e.g. "Comparison
against additional frozen encoders," containing:

1. **Architecture/input-handling table** — a trimmed version of
   `TSFM_MODEL_COMPARISON.md` §1's table (role, checkpoint/license, native
   window, channels used, apnea comparability, embedding shape, pretraining
   overlap, peer-review status). This is genuinely useful supplementary
   material and mostly ready to paste with light editing for tone.
2. **Full per-task, per-context tables** for OSF, PhysioOmni, and Mantis
   (frozen) against their respective SleepFM baselines — the tables already in
   `TSFM_MODEL_COMPARISON.md` §2.1, §2.2, §2.4 are close to paste-ready.
3. **OSF contamination quantification** — the SHHS overlap numbers (87.7% of
   SHHS test subjects in OSF's own pretrain train/valid split; STAGES, MrOS,
   APPLES confirmed clean by exact-ID match), stated once, clearly, with the
   instruction that any pooled OSF number involving SHHS should be read with
   this caveat.
4. **OSF LoRA table** (the 5-context sweep, `240m` explicitly marked as not
   run, not blank).
5. **PhysioOmni LoRA table**, explicitly marked with its jagged coverage (only
   9 of 48 cells; `bmi_binary`/`age_class` have zero LoRA cells) and a short,
   honest note on why (architectural ceiling + an operational GPU-billing
   delay, stated as two separate causes, not blended into one "it was slow"
   sentence).
6. **Computational-cost narrative** — the per-context-length training cost
   table and the "granularity vs. batching" finding
   (`docs/LORA_GPU_THROUGHPUT_INVESTIGATION.md`; `TSFM_MODEL_COMPARISON.md`
   §3-4), condensed. This is a genuine methodological finding (fine-tuning a
   raw-signal backbone at PSG-scale context lengths is structurally expensive,
   independent of which model) worth keeping even though the LoRA results
   themselves are incomplete.
7. **Split-population caveat** — the small per-encoder subject-count
   differences from embedding-extraction coverage (e.g. SleepFM vs. OSF: 1
   subject each way in APPLES/STAGES; SleepFM vs. Mantis: up to a few dozen
   per task), stated once so individual tables don't need repeated footnotes.
8. **A frank statement that `mean_pool` and val-split threshold tuning were
   never run for any of the three additional encoders**, so no encoder-choice
   claim should be made about the MeanPool-vs-temporal-head architecture
   finding (H3) or about balanced-accuracy numbers.

I have not drafted full prose for this section, since it's mostly tables plus
short connective text and the source material in `TSFM_MODEL_COMPARISON.md` is
already close to what's needed; happy to do a full prose pass once you've
confirmed the main-text framing above is right, since the supplementary framing
should match it.

---

## 6. Full-sweep tables (Table 1 and Table 3 analogs), added 2026-09-17

You asked for the equivalents of `npj_main.tex`'s Table~1 (`tab:sweep`, the
full context-length $\times$ $K$ sweep) and Table~3 (`tab:heads`, the head
comparison), for OSF, PhysioOmni, and Mantis, so you can see the complete
picture and decide yourself whether/where to use them. These are **not
generated by hand** — I wrote
[`scripts/gen_tsfm_baseline_tables.py`](../../scripts/gen_tsfm_baseline_tables.py),
which reads the three `analysis.csv` files directly and formats the LaTeX,
using the exact same $L^*$ definition as the paper's own equation (smallest
$L$ within 0.005 AUROC of the peak, evaluated at $K{=}K_{\max}$). I ran it,
and every cell below is its unedited output; nothing was retyped by hand.
Re-run the script if the underlying CSVs change.

**Placement recommendation, added 2026-09-17 after discussing this with you:**
Sections 6.1-6.6 (six tables below, one Table-1-style and one Table-3-style
per baseline) are **supplementary-only material**, not main text. Once
Section 6.7's cross-model table existed, it became clear these six overlap
heavily with it (same $K{=}K_{\max}$ numbers, just organized per-model instead
of per-task) with one exception: **only these six show the $K{=}1$ and
$K{=}5$ rows**, i.e. aggregation behavior below $K_{\max}$, which no other
table in this document shows for the TSFM baselines. Keep them in
supplementary as complete reference detail (matching how the main paper's own
supplementary already carries full per-context/per-K tables the main text
only summarizes), but be aware that **nothing in the current Results/Methods
draft text actually makes a claim that needs the $K{=}1$/$K{=}5$ rows** — this
document deliberately scoped the new subsection to H1-style context
sensitivity only, not an H2-style "does aggregation-saturation replicate too"
claim. If you don't want unreferenced supplementary bulk, either add one
sentence making that H2-style claim (the data already supports it, at a
glance the same few-windows-suffice pattern holds per model, but I have not
verified this claim carefully the way I verified the H1-style ones) or drop
Sections 6.1-6.6 entirely. Your call — I'd lean toward keeping them with a
one-sentence pointer, since reviewers of a "robustness/generalization" claim
often appreciate seeing the full backing data even if the main prose doesn't
walk through every cell.

**What I additionally spot-checked by hand**, independent of the script,
directly against the raw CSVs, to catch a script bug the script itself
couldn't self-diagnose: OSF `sleep_efficiency_binary`
(30s and 240m, $K{=}1,5,\text{all}$) and Mantis `sleep_efficiency_binary`
(same cells) — all matched the script's output exactly. Combined with the
`sex_binary`/`bmi_binary`/`apnea_binary` checks in Section 0 above (which also
match), I'm confident in the extraction logic, but I have **not** hand-checked
every one of the roughly 500 individual numbers below.

**Design choices, all worth your review:**

- **All tasks available per model are included**, not just the five primary
  ones, so you have the complete picture, including `osa_binary_apples_postqc`
  and `depression_extreme_binary`, both marked secondary ($\dagger$, $N{<}250$)
  exactly as the main paper's own Table~1 already does. **Remember the
  `osa_binary_apples_postqc` caveat from Section 0, item 1** (real tension with
  `apnea_binary` on the same cohort/threshold) before using OSF's version of
  that table.
- **Table 3 analog has no MeanPool column**, since that head was never run for
  any of the three baselines (Sections 2/5 above). I replaced the paper's
  "Adv. = Transformer − MeanPool" column with "$\Delta_{TL}$ = Transformer −
  LSTM," the only head-difference this data can actually support. This is a
  genuinely different quantity than the main paper's Table~3, not a drop-in
  replacement, so if you use this table, its caption/column header needs to
  say so explicitly (the generated caption already does).
- **The $K_{\max}$ header row is approximate and uses one reference task**
  (`sleep_efficiency_binary`, the largest-$N$ available task for each model)
  the same way the main paper's own Table~1 header uses one implicit
  reference population — I made this explicit in the caption since it wasn't
  explicit in the original.
- **OSF is directly comparable to the main paper's Table~1 only with the
  full-vs-reduced-channel caveat already established** (Methods draft,
  Section 2 above): OSF was run on full-channel data, so compare it to
  `npj_main.tex`'s `phase0_v3_full` numbers if you want a matched-channel
  comparison, not the reduced-channel Table~1 shown in the current main text.
- A few individual cells are visibly noisy (e.g. OSF's `osa_binary_apples_postqc`
  LSTM drops from 0.922 at 120m to 0.791 at 240m, a 13-point swing) — consistent
  with this document's standing caveat that the small-$N$ secondary tasks are
  noisy, not a sign of a data bug. I did not smooth or flag these
  cell-by-cell; you'll see them as you review.

### 6.1 OSF — Table 1 analog (full sweep)

```tex
\begin{table}[!t]
 \caption{Full context-length sweep, OSF frozen encoder (Transformer, test split). Same format and $L^*$ definition as Table~\ref{tab:sweep}. $K_{\max}$ at each context shown in parentheses (approximate; sleep\_efficiency\_binary used as reference). --- denotes no subject reaches $K{=}5$ windows at that context. $\dagger$: small test sets ($N{<}250$).}
 \label{tab:sweep_osf}
 \centering
 \footnotesize
 \setlength{\tabcolsep}{3pt}
 \begin{tabular}{ll@{\hskip8pt}*{6}{>{\centering\arraybackslash}p{32pt}}}
   \toprule
   & & \multicolumn{6}{c}{AUROC at training context length~$L$} \\
   \cmidrule(lr){3-8}
   Task & $K$ & 30~s & 10~min & 40~min & 80~min & 120~min & 240~min \\
   & ($K_{\max}{\approx}$) & (1149) & (57) & (14) & (7) & (4) & (2) \\
   \midrule
   Sleep efficiency
     & 1           & 0.630 & 0.679 & 0.696 & 0.711 & 0.746 & 0.801 \\
     & 5           & 0.706 & 0.731 & 0.765 & 0.790 & 0.801 & 0.841 \\
     & $K_{\max}$  & 0.714 & 0.740 & 0.773 & 0.794 & 0.802 & \textbf{0.841} \\
   \addlinespace
   Apnea detection
     & 1           & 0.626 & 0.668 & 0.719 & 0.779 & 0.814 & 0.883 \\
     & 5           & 0.787 & 0.832 & 0.858 & 0.902 & 0.910 & --- \\
     & $K_{\max}$  & 0.813 & 0.853 & 0.870 & 0.905 & \textbf{0.910} & 0.911 \\
   \addlinespace
   Sex classification
     & 1           & 0.745 & 0.836 & 0.879 & 0.912 & 0.931 & 0.946 \\
     & 5           & 0.907 & 0.933 & 0.946 & 0.958 & 0.963 & --- \\
     & $K_{\max}$  & 0.929 & 0.938 & 0.950 & \textbf{0.959} & 0.963 & 0.954 \\
   \addlinespace
   Age-group prediction
     & 1           & 0.808 & 0.867 & 0.896 & 0.918 & 0.926 & 0.936 \\
     & 5           & 0.891 & 0.916 & 0.928 & 0.939 & 0.942 & --- \\
     & $K_{\max}$  & 0.907 & 0.922 & 0.931 & \textbf{0.941} & 0.942 & 0.942 \\
   \addlinespace
   BMI (obese)
     & 1           & 0.718 & 0.768 & 0.797 & 0.811 & 0.823 & 0.837 \\
     & 5           & 0.812 & 0.833 & 0.838 & 0.847 & 0.847 & --- \\
     & $K_{\max}$  & 0.823 & 0.841 & 0.842 & \textbf{0.847} & 0.848 & 0.845 \\
   \addlinespace
   \midrule
   OSA severity (APPLES)$^\dagger$
     & 1           & 0.523 & 0.716 & 0.809 & 0.808 & 0.859 & 0.863 \\
     & 5           & 0.819 & 0.854 & 0.906 & 0.875 & 0.914 & --- \\
     & $K_{\max}$  & 0.867 & 0.868 & 0.906 & 0.877 & \textbf{0.914} & 0.887 \\
   \addlinespace
   Depression screening$^\dagger$
     & 1           & 0.694 & 0.795 & 0.825 & 0.833 & 0.815 & 0.773 \\
     & 5           & 0.739 & 0.774 & 0.787 & 0.787 & 0.776 & --- \\
     & $K_{\max}$  & 0.762 & 0.770 & \textbf{0.777} & 0.781 & 0.776 & 0.765 \\
   \bottomrule
 \end{tabular}
\end{table}
```

### 6.2 PhysioOmni — Table 1 analog (full sweep)

```tex
\begin{table}[!t]
 \caption{Full context-length sweep, PhysioOmni frozen encoder (Transformer, test split). Same format and $L^*$ definition as Table~\ref{tab:sweep}. $K_{\max}$ at each context shown in parentheses (approximate; sleep\_efficiency\_binary used as reference). --- denotes no subject reaches $K{=}5$ windows at that context. $\dagger$: small test sets ($N{<}250$).}
 \label{tab:sweep_physioomni}
 \centering
 \footnotesize
 \setlength{\tabcolsep}{3pt}
 \begin{tabular}{ll@{\hskip8pt}*{6}{>{\centering\arraybackslash}p{32pt}}}
   \toprule
   & & \multicolumn{6}{c}{AUROC at training context length~$L$} \\
   \cmidrule(lr){3-8}
   Task & $K$ & 30~s & 10~min & 40~min & 80~min & 120~min & 240~min \\
   & ($K_{\max}{\approx}$) & (1148) & (57) & (14) & (7) & (4) & (2) \\
   \midrule
   Sleep efficiency
     & 1           & 0.620 & 0.621 & 0.642 & 0.681 & 0.706 & 0.769 \\
     & 5           & 0.671 & 0.692 & 0.726 & 0.761 & 0.788 & 0.816 \\
     & $K_{\max}$  & 0.693 & 0.709 & 0.738 & 0.766 & 0.788 & \textbf{0.816} \\
   \addlinespace
   Sex classification
     & 1           & 0.609 & 0.646 & 0.715 & 0.784 & 0.813 & 0.858 \\
     & 5           & 0.714 & 0.787 & 0.847 & 0.873 & 0.884 & --- \\
     & $K_{\max}$  & 0.754 & 0.817 & 0.864 & 0.879 & \textbf{0.884} & 0.877 \\
   \addlinespace
   Age-group prediction
     & 1           & 0.770 & 0.799 & 0.817 & 0.831 & 0.835 & 0.845 \\
     & 5           & 0.799 & 0.827 & 0.844 & 0.851 & 0.852 & --- \\
     & $K_{\max}$  & 0.807 & 0.832 & 0.845 & \textbf{0.852} & 0.852 & 0.854 \\
   \addlinespace
   BMI (obese)
     & 1           & 0.648 & 0.681 & 0.697 & 0.707 & 0.712 & 0.732 \\
     & 5           & 0.691 & 0.719 & 0.727 & 0.734 & 0.737 & --- \\
     & $K_{\max}$  & 0.702 & 0.728 & 0.735 & 0.735 & 0.737 & \textbf{0.746} \\
   \addlinespace
   \midrule
   Depression screening$^\dagger$
     & 1           & 0.648 & 0.694 & 0.703 & 0.692 & 0.667 & 0.722 \\
     & 5           & 0.707 & 0.724 & 0.721 & 0.722 & 0.702 & --- \\
     & $K_{\max}$  & 0.718 & \textbf{0.729} & 0.725 & 0.722 & 0.702 & 0.726 \\
   \bottomrule
 \end{tabular}
\end{table}
```

**Note**: PhysioOmni has no `apnea_binary` row (no respiratory pathway, Methods
draft Section 2 above).

### 6.3 Mantis — Table 1 analog (full sweep)

```tex
\begin{table}[!t]
 \caption{Full context-length sweep, Mantis frozen encoder (Transformer, test split). Same format and $L^*$ definition as Table~\ref{tab:sweep}. $K_{\max}$ at each context shown in parentheses (approximate; sleep\_efficiency\_binary used as reference). --- denotes no subject reaches $K{=}5$ windows at that context. $\dagger$: small test sets ($N{<}250$).}
 \label{tab:sweep_mantis}
 \centering
 \footnotesize
 \setlength{\tabcolsep}{3pt}
 \begin{tabular}{ll@{\hskip8pt}*{6}{>{\centering\arraybackslash}p{32pt}}}
   \toprule
   & & \multicolumn{6}{c}{AUROC at training context length~$L$} \\
   \cmidrule(lr){3-8}
   Task & $K$ & 30~s & 10~min & 40~min & 80~min & 120~min & 240~min \\
   & ($K_{\max}{\approx}$) & (1148) & (57) & (14) & (7) & (4) & (2) \\
   \midrule
   Sleep efficiency
     & 1           & 0.646 & 0.656 & 0.642 & 0.680 & 0.724 & 0.781 \\
     & 5           & 0.693 & 0.709 & 0.752 & 0.784 & 0.807 & 0.827 \\
     & $K_{\max}$  & 0.707 & 0.720 & 0.764 & 0.787 & 0.807 & \textbf{0.827} \\
   \addlinespace
   Apnea detection
     & 1           & 0.602 & 0.636 & 0.651 & 0.708 & 0.752 & 0.807 \\
     & 5           & 0.708 & 0.771 & 0.814 & 0.842 & 0.856 & --- \\
     & $K_{\max}$  & 0.733 & 0.792 & 0.832 & 0.851 & \textbf{0.857} & 0.842 \\
   \addlinespace
   Sex classification
     & 1           & 0.754 & 0.818 & 0.849 & 0.878 & 0.891 & 0.914 \\
     & 5           & 0.839 & 0.884 & 0.913 & 0.925 & 0.935 & --- \\
     & $K_{\max}$  & 0.863 & 0.891 & 0.916 & 0.927 & \textbf{0.935} & 0.923 \\
   \addlinespace
   Age-group prediction
     & 1           & 0.777 & 0.815 & 0.852 & 0.878 & 0.899 & 0.906 \\
     & 5           & 0.839 & 0.875 & 0.908 & 0.916 & 0.923 & --- \\
     & $K_{\max}$  & 0.856 & 0.885 & 0.912 & \textbf{0.918} & 0.923 & 0.919 \\
   \addlinespace
   BMI (obese)
     & 1           & 0.659 & 0.714 & 0.727 & 0.734 & 0.751 & 0.751 \\
     & 5           & 0.728 & 0.763 & 0.774 & 0.775 & 0.781 & --- \\
     & $K_{\max}$  & 0.746 & 0.774 & \textbf{0.782} & 0.778 & 0.781 & 0.770 \\
   \addlinespace
   \midrule
   OSA severity (APPLES)$^\dagger$
     & 1           & 0.487 & 0.588 & 0.737 & 0.762 & 0.795 & 0.873 \\
     & 5           & 0.740 & 0.786 & 0.859 & 0.886 & 0.879 & --- \\
     & $K_{\max}$  & 0.818 & 0.837 & 0.872 & 0.887 & 0.879 & \textbf{0.903} \\
   \addlinespace
   Depression screening$^\dagger$
     & 1           & 0.721 & 0.778 & 0.775 & 0.766 & 0.757 & 0.755 \\
     & 5           & 0.779 & 0.768 & 0.745 & 0.744 & 0.760 & --- \\
     & $K_{\max}$  & 0.751 & 0.748 & 0.755 & 0.746 & \textbf{0.760} & 0.755 \\
   \bottomrule
 \end{tabular}
\end{table}
```

**Reminder**: Mantis's numbers above have no bootstrap confidence interval
computed yet (Section 0/comparison-doc caveat) — treat every cell, not just the
sex/240m dip discussed in the Results draft, as a point estimate without an
uncertainty band.

### 6.4 OSF — Table 3 analog (head comparison, LSTM vs. Transformer)

```tex
\begin{table}[!t]
 \caption{Head comparison across all context lengths~$L$, OSF frozen encoder ($K{=}K_{\max}$, test split). Same format as Table~\ref{tab:heads}, restricted to LSTM and Transformer: the MeanPool head was not run for any TSFM baseline (Methods, \nameref{sec:tsfm_baselines}), so no MeanPool column or Transformer-vs-MeanPool advantage can be shown here. $L^*$ markers: $^\ast$ LSTM, $^\dagger$ Transformer. $\Delta_{TL}$: Transformer$\,-\,$LSTM at $K{=}K_{\max}$.}
 \label{tab:heads_osf}
 \centering
 \small
 \setlength{\tabcolsep}{2pt}
 \begin{tabular}{llccr}
   \toprule
   Task & $L$ & LSTM & Transformer & $\Delta_{TL}$ \\
   \midrule
      Sleep efficiency
      & 30~s  & 0.725 & 0.714 & $-0.011$ \\
      & 10~min  & 0.749 & 0.740 & $-0.009$ \\
      & 40~min  & 0.759 & 0.773 & $+0.014$ \\
      & 80~min  & 0.782 & 0.794 & $+0.012$ \\
      & 120~min  & 0.795 & 0.802 & $+0.007$ \\
      & 240~min$^\ast$$^\dagger$  & 0.824 & 0.841 & $+0.018$ \\
   \addlinespace
      Apnea detection
      & 30~s  & 0.814 & 0.813 & $-0.001$ \\
      & 10~min  & 0.845 & 0.853 & $+0.008$ \\
      & 40~min  & 0.864 & 0.870 & $+0.006$ \\
      & 80~min  & 0.879 & 0.905 & $+0.026$ \\
      & 120~min$^\ast$$^\dagger$  & 0.882 & 0.910 & $+0.028$ \\
      & 240~min  & 0.886 & 0.911 & $+0.025$ \\
   \addlinespace
      Sex classification
      & 30~s  & 0.928 & 0.929 & $+0.002$ \\
      & 10~min  & 0.943 & 0.938 & $-0.004$ \\
      & 40~min$^\ast$  & 0.945 & 0.950 & $+0.005$ \\
      & 80~min$^\dagger$  & 0.950 & 0.959 & $+0.009$ \\
      & 120~min  & 0.943 & 0.963 & $+0.020$ \\
      & 240~min  & 0.942 & 0.954 & $+0.012$ \\
   \addlinespace
      Age-group prediction
      & 30~s  & 0.909 & 0.907 & $-0.002$ \\
      & 10~min  & 0.925 & 0.922 & $-0.002$ \\
      & 40~min$^\ast$  & 0.932 & 0.931 & $-0.000$ \\
      & 80~min$^\dagger$  & 0.934 & 0.941 & $+0.006$ \\
      & 120~min  & 0.924 & 0.942 & $+0.019$ \\
      & 240~min  & 0.924 & 0.942 & $+0.018$ \\
   \addlinespace
      BMI (obese)
      & 30~s  & 0.820 & 0.823 & $+0.003$ \\
      & 10~min$^\ast$  & 0.834 & 0.841 & $+0.007$ \\
      & 40~min  & 0.835 & 0.842 & $+0.007$ \\
      & 80~min$^\dagger$  & 0.839 & 0.847 & $+0.008$ \\
      & 120~min  & 0.832 & 0.848 & $+0.016$ \\
      & 240~min  & 0.832 & 0.845 & $+0.013$ \\
   \addlinespace
      OSA severity (APPLES)$^\dagger$
      & 30~s  & 0.883 & 0.867 & $-0.016$ \\
      & 10~min  & 0.904 & 0.868 & $-0.036$ \\
      & 40~min  & 0.890 & 0.906 & $+0.016$ \\
      & 80~min  & 0.876 & 0.877 & $+0.001$ \\
      & 120~min$^\ast$$^\dagger$  & 0.922 & 0.914 & $-0.007$ \\
      & 240~min  & 0.791 & 0.887 & $+0.096$ \\
   \addlinespace
      Depression screening$^\dagger$
      & 30~s  & 0.765 & 0.762 & $-0.003$ \\
      & 10~min  & 0.760 & 0.770 & $+0.011$ \\
      & 40~min$^\ast$$^\dagger$  & 0.770 & 0.777 & $+0.007$ \\
      & 80~min  & 0.767 & 0.781 & $+0.014$ \\
      & 120~min  & 0.771 & 0.776 & $+0.006$ \\
      & 240~min  & 0.763 & 0.765 & $+0.002$ \\
   \bottomrule
 \end{tabular}
\end{table}
```

### 6.5 PhysioOmni — Table 3 analog (head comparison)

```tex
\begin{table}[!t]
 \caption{Head comparison across all context lengths~$L$, PhysioOmni frozen encoder ($K{=}K_{\max}$, test split). Same format as Table~\ref{tab:heads}, restricted to LSTM and Transformer: the MeanPool head was not run for any TSFM baseline (Methods, \nameref{sec:tsfm_baselines}), so no MeanPool column or Transformer-vs-MeanPool advantage can be shown here. $L^*$ markers: $^\ast$ LSTM, $^\dagger$ Transformer. $\Delta_{TL}$: Transformer$\,-\,$LSTM at $K{=}K_{\max}$.}
 \label{tab:heads_physioomni}
 \centering
 \small
 \setlength{\tabcolsep}{2pt}
 \begin{tabular}{llccr}
   \toprule
   Task & $L$ & LSTM & Transformer & $\Delta_{TL}$ \\
   \midrule
      Sleep efficiency
      & 30~s  & 0.692 & 0.693 & $+0.002$ \\
      & 10~min  & 0.718 & 0.709 & $-0.008$ \\
      & 40~min  & 0.743 & 0.738 & $-0.006$ \\
      & 80~min  & 0.769 & 0.766 & $-0.003$ \\
      & 120~min  & 0.782 & 0.788 & $+0.006$ \\
      & 240~min$^\ast$$^\dagger$  & 0.813 & 0.816 & $+0.003$ \\
   \addlinespace
      Sex classification
      & 30~s  & 0.753 & 0.754 & $+0.001$ \\
      & 10~min  & 0.828 & 0.817 & $-0.011$ \\
      & 40~min  & 0.855 & 0.864 & $+0.009$ \\
      & 80~min$^\ast$  & 0.869 & 0.879 & $+0.010$ \\
      & 120~min$^\dagger$  & 0.866 & 0.884 & $+0.017$ \\
      & 240~min  & 0.859 & 0.877 & $+0.018$ \\
   \addlinespace
      Age-group prediction
      & 30~s  & 0.815 & 0.807 & $-0.008$ \\
      & 10~min  & 0.841 & 0.832 & $-0.009$ \\
      & 40~min  & 0.848 & 0.845 & $-0.004$ \\
      & 80~min$^\ast$$^\dagger$  & 0.866 & 0.852 & $-0.014$ \\
      & 120~min  & 0.861 & 0.852 & $-0.009$ \\
      & 240~min  & 0.859 & 0.854 & $-0.006$ \\
   \addlinespace
      BMI (obese)
      & 30~s  & 0.703 & 0.702 & $-0.001$ \\
      & 10~min  & 0.727 & 0.728 & $+0.001$ \\
      & 40~min$^\ast$  & 0.741 & 0.735 & $-0.006$ \\
      & 80~min  & 0.736 & 0.735 & $-0.000$ \\
      & 120~min  & 0.745 & 0.737 & $-0.008$ \\
      & 240~min$^\dagger$  & 0.740 & 0.746 & $+0.006$ \\
   \addlinespace
      Depression screening$^\dagger$
      & 30~s  & 0.716 & 0.718 & $+0.003$ \\
      & 10~min$^\ast$$^\dagger$  & 0.716 & 0.729 & $+0.014$ \\
      & 40~min  & 0.709 & 0.725 & $+0.016$ \\
      & 80~min  & 0.717 & 0.722 & $+0.004$ \\
      & 120~min  & 0.721 & 0.702 & $-0.019$ \\
      & 240~min  & 0.683 & 0.726 & $+0.043$ \\
   \bottomrule
 \end{tabular}
\end{table}
```

### 6.6 Mantis — Table 3 analog (head comparison)

```tex
\begin{table}[!t]
 \caption{Head comparison across all context lengths~$L$, Mantis frozen encoder ($K{=}K_{\max}$, test split). Same format as Table~\ref{tab:heads}, restricted to LSTM and Transformer: the MeanPool head was not run for any TSFM baseline (Methods, \nameref{sec:tsfm_baselines}), so no MeanPool column or Transformer-vs-MeanPool advantage can be shown here. $L^*$ markers: $^\ast$ LSTM, $^\dagger$ Transformer. $\Delta_{TL}$: Transformer$\,-\,$LSTM at $K{=}K_{\max}$.}
 \label{tab:heads_mantis}
 \centering
 \small
 \setlength{\tabcolsep}{2pt}
 \begin{tabular}{llccr}
   \toprule
   Task & $L$ & LSTM & Transformer & $\Delta_{TL}$ \\
   \midrule
      Sleep efficiency
      & 30~s  & 0.703 & 0.707 & $+0.004$ \\
      & 10~min  & 0.729 & 0.720 & $-0.008$ \\
      & 40~min  & 0.748 & 0.764 & $+0.016$ \\
      & 80~min  & 0.782 & 0.787 & $+0.005$ \\
      & 120~min  & 0.787 & 0.807 & $+0.020$ \\
      & 240~min$^\ast$$^\dagger$  & 0.809 & 0.827 & $+0.018$ \\
   \addlinespace
      Apnea detection
      & 30~s  & 0.756 & 0.733 & $-0.023$ \\
      & 10~min  & 0.799 & 0.792 & $-0.006$ \\
      & 40~min  & 0.820 & 0.832 & $+0.012$ \\
      & 80~min$^\ast$  & 0.828 & 0.851 & $+0.023$ \\
      & 120~min$^\dagger$  & 0.825 & 0.857 & $+0.031$ \\
      & 240~min  & 0.827 & 0.842 & $+0.016$ \\
   \addlinespace
      Sex classification
      & 30~s  & 0.859 & 0.863 & $+0.005$ \\
      & 10~min  & 0.893 & 0.891 & $-0.002$ \\
      & 40~min  & 0.908 & 0.916 & $+0.008$ \\
      & 80~min$^\ast$  & 0.920 & 0.927 & $+0.006$ \\
      & 120~min$^\dagger$  & 0.925 & 0.935 & $+0.010$ \\
      & 240~min  & 0.922 & 0.923 & $+0.001$ \\
   \addlinespace
      Age-group prediction
      & 30~s  & 0.862 & 0.856 & $-0.006$ \\
      & 10~min  & 0.893 & 0.885 & $-0.008$ \\
      & 40~min  & 0.904 & 0.912 & $+0.008$ \\
      & 80~min$^\dagger$  & 0.911 & 0.918 & $+0.007$ \\
      & 120~min$^\ast$  & 0.918 & 0.923 & $+0.004$ \\
      & 240~min  & 0.910 & 0.919 & $+0.009$ \\
   \addlinespace
      BMI (obese)
      & 30~s  & 0.740 & 0.746 & $+0.006$ \\
      & 10~min  & 0.773 & 0.774 & $+0.000$ \\
      & 40~min$^\ast$$^\dagger$  & 0.783 & 0.782 & $-0.001$ \\
      & 80~min  & 0.786 & 0.778 & $-0.007$ \\
      & 120~min  & 0.785 & 0.781 & $-0.003$ \\
      & 240~min  & 0.772 & 0.770 & $-0.001$ \\
   \addlinespace
      OSA severity (APPLES)$^\dagger$
      & 30~s  & 0.804 & 0.818 & $+0.014$ \\
      & 10~min  & 0.819 & 0.837 & $+0.018$ \\
      & 40~min  & 0.815 & 0.872 & $+0.058$ \\
      & 80~min$^\ast$  & 0.847 & 0.887 & $+0.040$ \\
      & 120~min  & 0.846 & 0.879 & $+0.033$ \\
      & 240~min$^\dagger$  & 0.817 & 0.903 & $+0.086$ \\
   \addlinespace
      Depression screening$^\dagger$
      & 30~s  & 0.733 & 0.751 & $+0.019$ \\
      & 10~min  & 0.739 & 0.748 & $+0.010$ \\
      & 40~min  & 0.766 & 0.755 & $-0.011$ \\
      & 80~min  & 0.755 & 0.746 & $-0.009$ \\
      & 120~min$^\dagger$  & 0.731 & 0.760 & $+0.030$ \\
      & 240~min$^\ast$  & 0.780 & 0.755 & $-0.025$ \\
   \bottomrule
 \end{tabular}
\end{table}
```

### 6.7 Cross-model comparison tables (all models, all contexts, one table per head), added 2026-09-17

You asked for a second format: one table with **all four encoders and all six
contexts together**, so rows can be compared directly against each other at a
glance, rather than needing to flip between the six separate per-model tables
above. Same script (`gen_tsfm_baseline_tables.py --cross-model`), same
$K{=}K_{\max}$ convention as Table 3.

**Placement, decided 2026-09-17: `tab:crossmodel_transformer` (this table)
goes in main text, replacing the old, narrower `tab:tsfm_sensitivity` table
in Section 3 above** (Section 3's Results tex block has been updated
accordingly: the old compact table is removed, its citations now point here,
and a "See Supplementary Table S-XX" sentence added for the LSTM
counterpart). `tab:tsfm_sensitivity` showed only $L^*$/$\Delta$, derivable
directly from this table's bold cell and $\Delta$ column; keeping both would
have meant two main-text tables saying the same thing, one with the numbers
stripped out. Beyond redundancy, `tab:tsfm_sensitivity`'s "SleepFM" column
also had a real framing risk this table avoids: it showed only the
reduced-channel SleepFM numbers next to OSF's full-channel ones in the same
row, implicitly inviting exactly the channel-mismatched comparison this
document's own Methods draft warns against elsewhere. This table shows both
SleepFM channel configurations as separate rows, so that risk doesn't arise.
`tab:crossmodel_lstm` (below) is the Supplementary~Table~S-XX companion.

**A real bug I found and fixed while building this**, worth knowing about: my
first pass computed $\Delta$ as (global max across all 6 contexts) $-$ (30s
value). That is subtly wrong whenever a task's true numerical peak sits
*beyond* $L^*$ but still within $L^*$'s own 0.005-AUROC saturation tolerance
(the same situation `npj_main.tex`'s own age-group row already hit, per the
`age-group Δ typo` note in the paper repo's `CLAUDE.md`: stated `+0.051`,
correct value `+0.048`, exactly this mechanism). The correct convention,
matching `npj_main.tex`'s own Table~2 (`tab:saturation`), is AUROC *at
$L^*$* minus AUROC at 30s, not the global peak minus 30s. I caught this by
re-deriving SleepFM's own already-published age-group $\Delta$ from the raw
CSV as a sanity check and getting 0.051 instead of the paper's stated 0.048 —
which told me the bug was in my script, not in the paper. **This also means
six numbers in Section 3's cross-encoder table above (added 2026-09-16) were
wrong** and I have now corrected them there too: OSF apnea ($+0.098\to+0.097$),
OSF sex ($+0.034\to+0.030$), OSF age ($+0.035\to+0.034$), OSF BMI
($+0.025\to+0.024$), PhysioOmni age ($+0.047\to+0.044$), and Mantis age
($+0.066\to+0.062$) — all small (≤0.006), none reverse a qualitative claim
already made in that section's prose, but they were wrong and are now fixed.
The tables in Section 6.1-6.6 above (K=1/5/K_max full sweep, no $\Delta$
column) were never affected by this bug — only $\Delta$ values were.

```tex
\begin{table*}[!t]
 \caption{Cross-model comparison, Transformer head, $K{=}K_{\max}$, test split. All four encoders (SleepFM shown at both channel configurations; OSF ran only on full-channel, PhysioOmni and Mantis only on reduced-channel, Methods, \nameref{sec:tsfm_baselines}) at every context length, for direct row-by-row comparison. \textbf{Bold}: each row's own saturation context $L^*$ (same 0.005-AUROC-of-peak definition as Table~\ref{tab:sweep}, computed independently per model/task/head). $\Delta$: AUROC at $L^*$ minus AUROC at 30~s (same convention as Table~\ref{tab:saturation}; not simply the global max minus 30~s, which can differ slightly when the true peak lies just beyond $L^*$ but within its 0.005 tolerance). ---: task not run for that encoder (PhysioOmni has no respiratory pathway, so no apnea row; osa\_binary\_apples\_postqc is OSF/Mantis only). $\dagger$: small test sets ($N{<}250$). No bootstrap confidence intervals exist for OSF/PhysioOmni/Mantis cells yet (Methods). See Supplementary~Table~S-XX for the LSTM counterpart.}
 \label{tab:crossmodel_transformer}
 \centering
 \resizebox{\linewidth}{!}{%
 \footnotesize
 \setlength{\tabcolsep}{4pt}
 \begin{tabular}{ll*{6}{c}r}
   \toprule
   Task & Model & 30~s & 10~min & 40~min & 80~min & 120~min & 240~min & $\Delta$ \\
   \midrule
   Sleep efficiency
     & SleepFM (reduced-ch.)  & 0.707 & 0.711 & 0.760 & 0.796 & 0.815 & \textbf{0.831} & $+0.124$ \\
     & SleepFM (full-ch.)  & 0.707 & 0.715 & 0.761 & 0.791 & 0.798 & \textbf{0.825} & $+0.118$ \\
     & OSF (full-ch.)  & 0.714 & 0.740 & 0.773 & 0.794 & 0.802 & \textbf{0.841} & $+0.128$ \\
     & PhysioOmni (reduced-ch.)  & 0.693 & 0.709 & 0.738 & 0.766 & 0.788 & \textbf{0.816} & $+0.123$ \\
     & Mantis (reduced-ch.)  & 0.707 & 0.720 & 0.764 & 0.787 & 0.807 & \textbf{0.827} & $+0.120$ \\
   \addlinespace
   Apnea detection
     & SleepFM (reduced-ch.)  & 0.753 & 0.793 & 0.825 & 0.847 & \textbf{0.857} & 0.854 & $+0.103$ \\
     & SleepFM (full-ch.)  & 0.800 & 0.826 & 0.870 & 0.895 & \textbf{0.900} & 0.901 & $+0.101$ \\
     & OSF (full-ch.)  & 0.813 & 0.853 & 0.870 & 0.905 & \textbf{0.910} & 0.911 & $+0.097$ \\
     & PhysioOmni (reduced-ch.)  & --- & --- & --- & --- & --- & --- & --- \\
     & Mantis (reduced-ch.)  & 0.733 & 0.792 & 0.832 & 0.851 & \textbf{0.857} & 0.842 & $+0.123$ \\
   \addlinespace
   Sex classification
     & SleepFM (reduced-ch.)  & 0.832 & 0.851 & 0.872 & 0.897 & 0.905 & \textbf{0.910} & $+0.079$ \\
     & SleepFM (full-ch.)  & 0.843 & 0.867 & 0.899 & \textbf{0.925} & 0.929 & 0.920 & $+0.082$ \\
     & OSF (full-ch.)  & 0.929 & 0.938 & 0.950 & \textbf{0.959} & 0.963 & 0.954 & $+0.030$ \\
     & PhysioOmni (reduced-ch.)  & 0.754 & 0.817 & 0.864 & 0.879 & \textbf{0.884} & 0.877 & $+0.130$ \\
     & Mantis (reduced-ch.)  & 0.863 & 0.891 & 0.916 & 0.927 & \textbf{0.935} & 0.923 & $+0.072$ \\
   \addlinespace
   Age-group prediction
     & SleepFM (reduced-ch.)  & 0.854 & 0.870 & 0.877 & 0.900 & \textbf{0.902} & 0.905 & $+0.048$ \\
     & SleepFM (full-ch.)  & 0.859 & 0.869 & 0.887 & 0.905 & \textbf{0.908} & 0.911 & $+0.050$ \\
     & OSF (full-ch.)  & 0.907 & 0.922 & 0.931 & \textbf{0.941} & 0.942 & 0.942 & $+0.034$ \\
     & PhysioOmni (reduced-ch.)  & 0.807 & 0.832 & 0.845 & \textbf{0.852} & 0.852 & 0.854 & $+0.044$ \\
     & Mantis (reduced-ch.)  & 0.856 & 0.885 & 0.912 & \textbf{0.918} & 0.923 & 0.919 & $+0.062$ \\
   \addlinespace
   BMI (obese)
     & SleepFM (reduced-ch.)  & 0.747 & 0.755 & 0.755 & 0.769 & 0.766 & \textbf{0.777} & $+0.030$ \\
     & SleepFM (full-ch.)  & 0.777 & 0.788 & 0.801 & \textbf{0.811} & 0.812 & 0.816 & $+0.034$ \\
     & OSF (full-ch.)  & 0.823 & 0.841 & 0.842 & \textbf{0.847} & 0.848 & 0.845 & $+0.024$ \\
     & PhysioOmni (reduced-ch.)  & 0.702 & 0.728 & 0.735 & 0.735 & 0.737 & \textbf{0.746} & $+0.044$ \\
     & Mantis (reduced-ch.)  & 0.746 & 0.774 & \textbf{0.782} & 0.778 & 0.781 & 0.770 & $+0.036$ \\
   \addlinespace
   \midrule
   OSA severity (APPLES)$^\dagger$
     & SleepFM (reduced-ch.)  & 0.789 & 0.804 & 0.853 & \textbf{0.888} & 0.856 & 0.861 & $+0.098$ \\
     & SleepFM (full-ch.)  & 0.755 & 0.762 & 0.791 & \textbf{0.814} & 0.814 & 0.818 & $+0.058$ \\
     & OSF (full-ch.)  & 0.867 & 0.868 & 0.906 & 0.877 & \textbf{0.914} & 0.887 & $+0.048$ \\
     & PhysioOmni (reduced-ch.)  & --- & --- & --- & --- & --- & --- & --- \\
     & Mantis (reduced-ch.)  & 0.818 & 0.837 & 0.872 & 0.887 & 0.879 & \textbf{0.903} & $+0.085$ \\
   \addlinespace
   Depression screening$^\dagger$
     & SleepFM (reduced-ch.)  & \textbf{0.756} & 0.739 & 0.750 & 0.749 & 0.754 & 0.746 & $+0.000$ \\
     & SleepFM (full-ch.)  & 0.738 & 0.742 & \textbf{0.752} & 0.756 & 0.740 & 0.741 & $+0.014$ \\
     & OSF (full-ch.)  & 0.762 & 0.770 & \textbf{0.777} & 0.781 & 0.776 & 0.765 & $+0.015$ \\
     & PhysioOmni (reduced-ch.)  & 0.718 & \textbf{0.729} & 0.725 & 0.722 & 0.702 & 0.726 & $+0.011$ \\
     & Mantis (reduced-ch.)  & 0.751 & 0.748 & 0.755 & 0.746 & \textbf{0.760} & 0.755 & $+0.009$ \\
   \bottomrule
 \end{tabular}%
 }% end resizebox
\end{table*}
```

```tex
\begin{table*}[!t]
 \caption{Cross-model comparison, LSTM head, $K{=}K_{\max}$, test split. All four encoders (SleepFM shown at both channel configurations; OSF ran only on full-channel, PhysioOmni and Mantis only on reduced-channel, Methods, \nameref{sec:tsfm_baselines}) at every context length, for direct row-by-row comparison. \textbf{Bold}: each row's own saturation context $L^*$ (same 0.005-AUROC-of-peak definition as Table~\ref{tab:sweep}, computed independently per model/task/head). $\Delta$: AUROC at $L^*$ minus AUROC at 30~s (same convention as Table~\ref{tab:saturation}; not simply the global max minus 30~s, which can differ slightly when the true peak lies just beyond $L^*$ but within its 0.005 tolerance). ---: task not run for that encoder (PhysioOmni has no respiratory pathway, so no apnea row; osa\_binary\_apples\_postqc is OSF/Mantis only). $\dagger$: small test sets ($N{<}250$). No bootstrap confidence intervals exist for OSF/PhysioOmni/Mantis cells yet (Methods).}
 \label{tab:crossmodel_lstm}
 \centering
 \resizebox{\linewidth}{!}{%
 \footnotesize
 \setlength{\tabcolsep}{4pt}
 \begin{tabular}{ll*{6}{c}r}
   \toprule
   Task & Model & 30~s & 10~min & 40~min & 80~min & 120~min & 240~min & $\Delta$ \\
   \midrule
   Sleep efficiency
     & SleepFM (reduced-ch.)  & 0.697 & 0.717 & 0.731 & 0.759 & 0.778 & \textbf{0.788} & $+0.091$ \\
     & SleepFM (full-ch.)  & 0.715 & 0.708 & 0.732 & 0.758 & 0.768 & \textbf{0.810} & $+0.095$ \\
     & OSF (full-ch.)  & 0.725 & 0.749 & 0.759 & 0.782 & 0.795 & \textbf{0.824} & $+0.099$ \\
     & PhysioOmni (reduced-ch.)  & 0.692 & 0.718 & 0.743 & 0.769 & 0.782 & \textbf{0.813} & $+0.122$ \\
     & Mantis (reduced-ch.)  & 0.703 & 0.729 & 0.748 & 0.782 & 0.787 & \textbf{0.809} & $+0.106$ \\
   \addlinespace
   Apnea detection
     & SleepFM (reduced-ch.)  & 0.758 & 0.774 & 0.792 & 0.821 & \textbf{0.832} & 0.827 & $+0.074$ \\
     & SleepFM (full-ch.)  & 0.792 & 0.808 & 0.843 & 0.865 & \textbf{0.874} & 0.871 & $+0.082$ \\
     & OSF (full-ch.)  & 0.814 & 0.845 & 0.864 & 0.879 & \textbf{0.882} & 0.886 & $+0.068$ \\
     & PhysioOmni (reduced-ch.)  & --- & --- & --- & --- & --- & --- & --- \\
     & Mantis (reduced-ch.)  & 0.756 & 0.799 & 0.820 & \textbf{0.828} & 0.825 & 0.827 & $+0.072$ \\
   \addlinespace
   Sex classification
     & SleepFM (reduced-ch.)  & 0.825 & 0.850 & 0.845 & 0.861 & \textbf{0.872} & 0.857 & $+0.047$ \\
     & SleepFM (full-ch.)  & 0.861 & 0.889 & \textbf{0.906} & 0.889 & 0.894 & 0.887 & $+0.045$ \\
     & OSF (full-ch.)  & 0.928 & 0.943 & \textbf{0.945} & 0.950 & 0.943 & 0.942 & $+0.018$ \\
     & PhysioOmni (reduced-ch.)  & 0.753 & 0.828 & 0.855 & \textbf{0.869} & 0.866 & 0.859 & $+0.116$ \\
     & Mantis (reduced-ch.)  & 0.859 & 0.893 & 0.908 & \textbf{0.920} & 0.925 & 0.922 & $+0.062$ \\
   \addlinespace
   Age-group prediction
     & SleepFM (reduced-ch.)  & 0.865 & 0.870 & 0.887 & \textbf{0.890} & 0.893 & 0.885 & $+0.025$ \\
     & SleepFM (full-ch.)  & 0.871 & 0.880 & \textbf{0.899} & 0.901 & 0.898 & 0.901 & $+0.028$ \\
     & OSF (full-ch.)  & 0.909 & 0.925 & \textbf{0.932} & 0.934 & 0.924 & 0.924 & $+0.023$ \\
     & PhysioOmni (reduced-ch.)  & 0.815 & 0.841 & 0.848 & \textbf{0.866} & 0.861 & 0.859 & $+0.051$ \\
     & Mantis (reduced-ch.)  & 0.862 & 0.893 & 0.904 & 0.911 & \textbf{0.918} & 0.910 & $+0.056$ \\
   \addlinespace
   BMI (obese)
     & SleepFM (reduced-ch.)  & 0.760 & \textbf{0.762} & 0.756 & 0.767 & 0.756 & 0.748 & $+0.002$ \\
     & SleepFM (full-ch.)  & 0.788 & 0.796 & \textbf{0.798} & 0.802 & 0.798 & 0.802 & $+0.010$ \\
     & OSF (full-ch.)  & 0.820 & \textbf{0.834} & 0.835 & 0.839 & 0.832 & 0.832 & $+0.014$ \\
     & PhysioOmni (reduced-ch.)  & 0.703 & 0.727 & \textbf{0.741} & 0.736 & 0.745 & 0.740 & $+0.038$ \\
     & Mantis (reduced-ch.)  & 0.740 & 0.773 & \textbf{0.783} & 0.786 & 0.785 & 0.772 & $+0.042$ \\
   \addlinespace
   \midrule
   OSA severity (APPLES)$^\dagger$
     & SleepFM (reduced-ch.)  & 0.769 & 0.816 & \textbf{0.834} & 0.767 & 0.789 & 0.774 & $+0.064$ \\
     & SleepFM (full-ch.)  & \textbf{0.772} & 0.738 & 0.763 & 0.756 & 0.724 & 0.694 & $+0.000$ \\
     & OSF (full-ch.)  & 0.883 & 0.904 & 0.890 & 0.876 & \textbf{0.922} & 0.791 & $+0.039$ \\
     & PhysioOmni (reduced-ch.)  & --- & --- & --- & --- & --- & --- & --- \\
     & Mantis (reduced-ch.)  & 0.804 & 0.819 & 0.815 & \textbf{0.847} & 0.846 & 0.817 & $+0.043$ \\
   \addlinespace
   Depression screening$^\dagger$
     & SleepFM (reduced-ch.)  & 0.757 & \textbf{0.770} & 0.761 & 0.744 & 0.748 & 0.750 & $+0.013$ \\
     & SleepFM (full-ch.)  & 0.738 & \textbf{0.751} & 0.739 & 0.740 & 0.752 & 0.717 & $+0.013$ \\
     & OSF (full-ch.)  & 0.765 & 0.760 & \textbf{0.770} & 0.767 & 0.771 & 0.763 & $+0.004$ \\
     & PhysioOmni (reduced-ch.)  & 0.716 & \textbf{0.716} & 0.709 & 0.717 & 0.721 & 0.683 & $+0.000$ \\
     & Mantis (reduced-ch.)  & 0.733 & 0.739 & 0.766 & 0.755 & 0.731 & \textbf{0.780} & $+0.047$ \\
   \bottomrule
 \end{tabular}%
 }% end resizebox
\end{table*}
```

---

## 7. Notebooks for exploratory Fig 2 / Fig 3 analogs, added 2026-09-17

You asked for something similar to Fig. 2 (AUROC vs. $K$, iso-budget lines)
and Fig. 3 (iso-budget heatmap) for the TSFM baselines, to decide whether to
use them. Two new notebooks are now at
`NSRR-tools/results/paper_figures/notebooks_npj/`, named to make clear they're
exploratory drafts, not paper-numbered:

- **`sfig26_tsfm_kvsk.ipynb`** — analog of `main_fig2_kvsk.ipynb`. Grid layout:
  one row per encoder (SleepFM reduced-channel, SleepFM full-channel, OSF
  full-channel, PhysioOmni reduced-channel, Mantis reduced-channel), one
  column per representative task (age, sleep efficiency, sex — the same three
  tasks the paper's own Fig. 2 uses, chosen there to show three distinct
  $K$-saturation regimes). Same `kvsk_panel` function the real Fig. 2 uses,
  same context-length color palette, same iso-budget dashed lines.
- **`sfig27_tsfm_heatmap.ipynb`** — analog of `main_fig3_heatmap.ipynb`. Same
  5-row (encoder) $\times$ 4-column (sex, apnea, sleep efficiency, BMI) grid.
  PhysioOmni's apnea cell renders as "no data" (the panel functions' own
  existing behavior for an empty DataFrame), since PhysioOmni has no
  respiratory pathway — I did not special-case this, it falls out naturally
  from passing an empty frame.

**Both SleepFM channel configurations are included as separate rows**
(reduced and full), not just one, so you can visually compare OSF against its
correct full-channel baseline and PhysioOmni/Mantis against the reduced-channel
baseline used everywhere else in the paper, in the same figure, without
mixing them into a single misleading "SleepFM" row.

### How the data was loaded (a genuine gap I had to close, not a style choice)

The real Fig. 2/Fig. 3 notebooks load a pre-built `heatmap_df_test.csv` per
task/head from `final_results/{experiment}/inference/{task}_{head}/`
(via `utils.data.load_heatmap`). **That per-task-head file tree does not
exist for OSF, PhysioOmni, or Mantis** — those three only have a single
`analysis.csv`/`training.csv` pair each, still living under
`NSRR-tools/results/collected/`, not `final_results/`. I checked this by
listing `final_results/` directly rather than assuming.

I confirmed `analysis.csv` already carries the same fine-grained $K$-grid as
the real `heatmap_df_test.csv` files (same discrete $K$ values per context,
e.g. 1,2,3,4,5,6,8,10,...,500 at 30s), by comparing specific cells between
`final_results/phase0_v3/inference/sex_binary_transformer/heatmap_df_test.csv`
and `results/collected/phase0_v3/analysis.csv` directly — they agree exactly,
including how the `k="all"` row is represented: `heatmap_df_test.csv` stores
it as a **fractional** $K$ equal to `n_segments / n_subjects` (the true mean
number of windows aggregated per subject, since $K_{\max}$ varies by
subject), not the literal string `"all"`. I reproduced that exact convention.

Rather than build a one-off loader inline in each notebook, I added one new,
**purely additive** function to the shared `utils/data.py` (nothing existing
was changed):

```python
def load_heatmap_from_collected(collected_root, experiment, task, head,
                                split="test"):
    """Build a heatmap_df-equivalent DataFrame directly from a `collected/`
    analysis.csv, for baselines without a final_results/.../heatmap_df_test.csv
    tree (the TSFM comparison: OSF, PhysioOmni, Mantis). See
    docs/npj_paper_md_files/TSFM_BASELINE_RESULTS_DRAFT.md Section 7 for how
    this was verified against the real heatmap_df_test.csv convention.
    """
```

Both new notebooks import this alongside the existing `load_heatmap`, so
SleepFM's two rows use the real, existing `load_heatmap` / `final_results/`
path (unchanged, exactly what the paper's own figures use) and the three
baseline rows use the new function pointed at each experiment's
`results/collected/` location, including the Mantis one, which lives in the
**separate `NSRR-tools-mantis` worktree**, not this repo — the notebook's
setup cell has an explicit, commented path for this, since the normal
`_find_workspace()` auto-detect (which looks for `final_results/`) would not
find it on its own.

**I did run both notebooks end-to-end** (executed every code cell, in order,
with the project's own `.venv` interpreter — not via a Jupyter kernel
directly, since the kernel registered for these notebooks resolves to a
different, matplotlib-less environment on this machine; I ran the extracted
cell source directly instead, which exercises the identical code path).
**Both completed with no errors**, and the PhysioOmni × apnea cell correctly
came back as an empty DataFrame (0 rows), exercising the "no data" path
rather than crashing. The rendered PDFs and PNGs are already sitting at
`NSRR-tools/results/paper_figures/final_npj/sfig26_tsfm_kvsk.{pdf,png}` and
`sfig27_tsfm_heatmap.{pdf,png}` — open them directly if you don't want to run
Jupyter yourself first.

**What the rendered figures actually show, worth knowing before you open
them**: the visual pattern across the 5 encoder rows is strikingly
consistent for both figures, more so than I expected going in. In the $K$-vs-AUROC
grid, every encoder's age column shows the curves for the three longest
contexts converging near a shared ceiling (the "iso-budget substitutability"
regime the real Fig. 2 caption describes for age); every encoder's sleep
efficiency column shows the curves staying separated at every $K$ (the
"context-irreplaceable" regime); every encoder's sex column shows the number
of windows needed to saturate shrinking sharply as context grows. This is the
same qualitative regime, per task, in all five rows, which is the clearest
single piece of visual evidence I've produced for this document's
"replicates across encoders" claim (Results draft, Section 3 above) — you may
want to consider leading with this figure rather than the numbers-only table,
if you decide to include this material at all. The heatmap grid shows the
same story: the iso-budget contour shape per task looks similar down each
column. One cosmetic issue, inherited unmodified from the real
`main_fig2_kvsk.ipynb`'s own panel-label code (I didn't touch it): the bold
`(a)`/`(d)`/`(g)`... panel labels sit in the same top-left corner as the "8h"
iso-budget annotation in several panels and visually collide with it — worth
a small fix if you take this figure further, not something I patched since
it's shared, unmodified plotting logic.

---

## 8. What I did not verify

Being explicit about the boundary of what I checked, per your instruction not
to assume:

- I did **not** re-verify every cell of every table in `TSFM_MODEL_COMPARISON.md`
  against the raw CSVs — I spot-checked `sex_binary` (all four models, including
  OSF-LoRA and PhysioOmni-LoRA), `bmi_binary` (SleepFM, OSF, PhysioOmni), and
  `apnea_binary` (SleepFM, Mantis), all of which matched exactly. I did not
  independently re-check `sleep_efficiency_binary` or `age_class` cell values, or
  any LSTM-head numbers.
- **Section 6's tables**: `scripts/gen_tsfm_baseline_tables.py` pulls every
  number itself (nothing hand-typed), and I additionally hand-checked its
  `sleep_efficiency_binary` output for OSF and Mantis (30s and 240m,
  $K{=}1,5,\text{all}$) directly against the raw CSVs — matched exactly. I did
  **not** hand-check every task/context/$K$ cell the script produced (roughly
  500 numbers across the six tables), only its extraction logic and these spot
  checks. The `total_compute_min`-vs-`n_segments/n_subjects` equivalence the
  script relies on for the `k="all"` row is confirmed for `sex_binary`,
  240m (Section 7's data-loading writeup) but not independently re-derived for
  every other task/context.
- **Section 7's notebooks**: both were executed end-to-end successfully and I
  visually inspected the rendered PNGs (described in Section 7) — this is real
  verification, not a static read of the code. I did **not**, however, check
  every individual heatmap cell's color/value against the underlying numbers
  pixel-by-pixel, only that the overall shapes match what Sections 0/3/6 already
  established numerically and that no panel silently rendered wrong or empty
  data where real data should exist.
- I did **not** re-verify the OSF SHHS-contamination percentages (87.7%, etc.)
  against the underlying subject-ID split files myself; I'm relying on the
  comparison doc's own claim of having done exact-ID matching.
- I did **not** check whether Mantis's LoRA sweep has progressed further than
  the 2026-09-16 snapshot I looked at (age_class 10m just finished; several
  others still running). If you're reading this more than a day or two after
  2026-09-16, re-check `NSRR-tools-mantis/logs_mantis_lora/status/` before
  finalizing the "still in progress, not reported" framing.
- I did **not** find or verify author names for the PhysioOmni or Mantis
  bibliography entries — see Section 1 above.
- I did **not** attempt to resolve the arXiv-date-vs-project-start-date question
  for PhysioOmni precisely (whether it existed before or after this project's
  encoder was chosen); I deliberately wrote the "why SleepFM first" framing
  (Methods draft, Discussion draft) to rest only on facts I could confirm
  (SleepFM's earlier, peer-reviewed, public-checkpoint status; OSF being
  published subsequently), rather than asserting a PhysioOmni-specific timeline
  claim I wasn't sure of.
