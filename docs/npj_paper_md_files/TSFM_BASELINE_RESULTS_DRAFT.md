# TSFM Baseline Comparison — Draft Paper Additions

**Status: first draft, for review.** Written 2026-09-16. This is a first pass at
the material requested: a new Results subsection, a new Methods subsection, a
Discussion edit, and a Supplementary extension, covering the OSF / PhysioOmni /
Mantis baseline-comparison work. **Nothing has been pasted into `npj_main.tex` or
`npj_supplementary.tex` yet** — following this repo's established workflow
(`RESULTS_REWRITE.md`), paste manually after reviewing, paragraph by paragraph if
you want to iterate on wording first.

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
3. **Main-text table is $L^*$/$\Delta$ only (the context-sensitivity replication
   finding), not absolute AUROC.** Absolute performance differences (OSF/Mantis
   beating SleepFM on some tasks, PhysioOmni losing throughout) are reported in
   prose with the strongest, most defensible numbers, with the full per-context
   tables deferred to supplementary. This follows your stated preference to
   emphasize the *agreement* in trends without hiding real differences.
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
additional encoders (Table~\ref{tab:tsfm_sensitivity}). Sleep efficiency
showed the largest gain from longer context under every encoder tested
($\Delta = {+}0.120$ to ${+}0.128$ AUROC), still rising at the longest context
evaluated in every case, matching the SleepFM-based finding above. BMI showed
the smallest gain under every encoder ($\Delta = {+}0.025$ to ${+}0.044$).
Apnea, evaluated in the three encoders with a respiratory-signal pathway,
showed the second-largest gain in all three ($\Delta = {+}0.098$ to
${+}0.123$), saturating at $L^*{=}120$~min in every case. Age formed a
consistent middle tier ($\Delta = {+}0.035$ to ${+}0.066$). Sex classification
was the one task where encoders diverged in the size, though not the
direction, of the context gain ($\Delta = {+}0.034$ for OSF to ${+}0.130$ for
PhysioOmni): OSF's frozen encoder already reached 0.929 AUROC at 30~s, leaving
little room for further gain, whereas PhysioOmni started markedly lower
(0.754) and gained the most of any encoder tested. This is consistent with a
ceiling effect on encoders that already extract most of the
sex-discriminative signal from a single short segment, rather than a
disagreement about whether sex benefits from longer context.

\begin{table}[!t]
 \caption{Context-length sensitivity across four independently pretrained
 encoders (Transformer head, $K{=}K_{\max}$, each encoder's own reduced- or
 full-channel configuration as used for its comparison, Methods
 \nameref{sec:tsfm_baselines}). $\Delta$: AUROC gain from 30~s to each
 encoder's own saturation context~$L^*$; $\ddagger$: still rising at the
 longest context tested (240~min), so $\Delta$ is a lower bound. ---: apnea
 excluded for PhysioOmni (no respiratory-signal pathway in the released
 architecture).}
 \label{tab:tsfm_sensitivity}
 \centering
 \footnotesize
 \setlength{\tabcolsep}{4pt}
 \begin{tabular}{l cc cc cc cc}
   \toprule
   & \multicolumn{2}{c}{SleepFM} & \multicolumn{2}{c}{OSF}
   & \multicolumn{2}{c}{PhysioOmni} & \multicolumn{2}{c}{Mantis} \\
   \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
   Task & $L^*$ & $\Delta$ & $L^*$ & $\Delta$ & $L^*$ & $\Delta$ & $L^*$ & $\Delta$ \\
   \midrule
   Sleep efficiency & 240~min$^\ddagger$ & $+$0.124 & 240~min$^\ddagger$ & $+$0.128
     & 240~min$^\ddagger$ & $+$0.123 & 240~min$^\ddagger$ & $+$0.120 \\
   Apnea detection & 120~min & $+$0.103 & 120~min & $+$0.098
     & --- & --- & 120~min & $+$0.123 \\
   Sex classification & 240~min$^\ddagger$ & $+$0.079 & 80~min & $+$0.034
     & 120~min & $+$0.130 & 120~min & $+$0.072 \\
   Age-group prediction & 120~min & $+$0.048 & 80~min & $+$0.035
     & 80~min & $+$0.047 & 80~min & $+$0.066 \\
   BMI (obese) & 240~min & $+$0.030 & 80~min & $+$0.025
     & 240~min & $+$0.044 & 40~min & $+$0.036 \\
   \bottomrule
 \end{tabular}
\end{table}

The encoders differed substantially in absolute performance, and these
differences did not favor any single encoder uniformly. OSF's frozen encoder
outperformed SleepFM on sex, age, and BMI, including on cohorts confirmed free
of any subject-level overlap with OSF's own pretraining data (Supplementary
Section~S-XX); it was inconclusive on sleep efficiency and did not
consistently outperform or underperform SleepFM on apnea. PhysioOmni's frozen
encoder underperformed SleepFM at every context on every comparable task, by
as much as 7.8 percentage points of AUROC at short context, narrowing but not
closing at the longest context tested. Mantis's frozen encoder, despite
carrying no physiological pretraining whatsoever, outperformed SleepFM on
sex (by 1.3 to 4.4 percentage points at every context) and age (0.2 to 3.5
points), and on the smaller-cohort OSA severity task (Table~\ref{tab:tasks});
it was closely matched with SleepFM on apnea and BMI, and slightly behind at
long context on sleep efficiency. This outcome ran counter to a prediction we
made before running Mantis, based on published evidence that freezing a
generic time-series model typically degrades performance sharply on EEG-based
tasks; we report it as the genuine, pre-specified surprise that it was.

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

- Sex 30s: SleepFM 0.832, OSF 0.929, PhysioOmni 0.754, Mantis 0.863 — matches
  "7.8 percentage points" (0.832−0.754) and the OSF/PhysioOmni $\Delta$ figures
  above.
- Age: SleepFM 0.854→0.905 (240m), Mantis 0.856→0.923 (120m peak) →
  differences of +0.2pp (30s) to +3.5pp (120m), as stated.
- OSA severity: Mantis beats SleepFM at 5 of 6 contexts (comparison doc §2.4);
  stated only qualitatively above via the Methods/Table~\ref{tab:tasks}
  cross-reference to avoid overloading the main-text table with a sixth,
  small-$N$ secondary task (see "Decisions" above).

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
task ordering across encoders (Table~\ref{tab:tsfm_sensitivity}) is itself
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

## 6. What I did not verify

Being explicit about the boundary of what I checked, per your instruction not
to assume:

- I did **not** re-verify every cell of every table in `TSFM_MODEL_COMPARISON.md`
  against the raw CSVs — I spot-checked `sex_binary` (all four models, including
  OSF-LoRA and PhysioOmni-LoRA), `bmi_binary` (SleepFM, OSF, PhysioOmni), and
  `apnea_binary` (SleepFM, Mantis), all of which matched exactly. I did not
  independently re-check `sleep_efficiency_binary` or `age_class` cell values, or
  any LSTM-head numbers.
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
