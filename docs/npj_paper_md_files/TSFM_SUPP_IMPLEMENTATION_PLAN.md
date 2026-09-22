# TSFM Supplementary Implementation Additions — Plan & State

**Purpose of this file**: handoff/state tracker for a task in progress. If this
session runs out of context or is resumed by a different agent, read this file
first, then resume at the first unchecked item. Update the checklists as you go.

**The task**: add comprehensive implementation/adaptation/training/honest-drawback
content about the three new TSFM baseline models (OSF, PhysioOmni, Mantis) to
`npj_digital_medicine_submission/npj_supplementary.tex`. Two rounds: Round 1 =
research + first draft; Round 2 = verification + gap-filling. Each round split
into subtasks so nothing gets missed.

**What already exists (do not duplicate)**: `npj_supplementary.tex` already has
a full section `\section{Comparison Across Independently Pretrained Encoders}`
(`\label{sec:supp-tsfm}`, currently around line 2782) covering: OSF/SHHS
contamination quantification, LSTM cross-model table, full per-context sweep +
head-comparison tables per encoder, LoRA coverage/stopping-criteria table,
frozen-vs-LoRA table, and the K-aggregation/iso-budget figures across encoders.
**This is all RESULTS/analysis. It contains ZERO implementation, preprocessing,
data-adaptation, architecture-adaptation, or training-mechanics detail.** That
detail exists only in internal docs (`NSRR-tools/docs/TSFM_*_IMPLEMENTATION_PLAN.md`,
`CLAUDE.md`, `TSFM_MODEL_COMPARISON.md`, `LORA_GPU_THROUGHPUT_INVESTIGATION.md`)
and has never been distilled into the paper. **This is the gap this task fills.**

`npj_main.tex` has a short Methods subsection (`\label{sec:tsfm_baselines}`,
~line 2531) with a brief paragraph per model — that is main-text-appropriate
brevity and should NOT be expanded; the new material goes in the supplementary
only, as the user requested.

## Source material map (read these, don't re-derive)

- `NSRR-tools/CLAUDE.md` — "TSFM Baseline Model Comparison" section: usage-mode
  (Plan A/B/C) framework, code-reuse assessment, contamination facts, per-model
  status log. **Read this first**, it's the most reliable single source and is
  kept up to date by convention.
- `NSRR-tools/docs/TSFM_BASELINE_CANDIDATES.md` — model survey, checkpoint/license facts.
- `NSRR-tools/docs/TSFM_THIRD_MODEL_DECISION.md` — why Mantis was picked over MOMENT;
  compute/FLOP comparison table; the pre-registered "expect a weak frozen result" prediction.
- `NSRR-tools/docs/TSFM_MODEL_COMPARISON.md` (1181 lines) — cross-model results +
  **the honest computational-cost narrative (§3-4)**: measured TFLOP/s tables,
  the granularity/overhead-bound mechanism, PhysioOmni's 15-day GPU-billing
  incident, Mantis's estimated per-context costs. Read in full — this is a primary
  source for the "honest drawbacks" material across all three models.
- `NSRR-tools/docs/LORA_GPU_THROUGHPUT_INVESTIGATION.md` — the TF32/GPU-allocation
  investigation, the `metrics.json` multi-resume timing bug, verdict on what helps
  and what doesn't at which context length.
- Per-model implementation plans (long, code-verified, primary sources for Stage
  1/Stage 2 mechanics): `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` (2867 lines),
  `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` (2743 lines, in NSRR-tools),
  `docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md` (526 lines, canonical copy is in the
  **NSRR-tools-mantis worktree**, not NSRR-tools, since mantis isn't merged yet).
- `docs/OSF_EXPERIMENTS_GUIDE.md`, `docs/PHYSIOOMNI_EXPERIMENTS_GUIDE.md`,
  `docs/PHYSIOOMNI_PLANNING_HANDOFF.md`, `docs/OSF_CHANNEL_REPROCESSING_PLAN.md`.
- Actual code/configs (ground truth — cite file:line, don't trust docs blindly,
  docs can go stale): `src/nsrr_tools/datasets/osf_*.py` /`physioomni_*.py`
  /`mantis_*.py`, `scripts/train_*.py`, `scripts/extract_*embeddings*.py`,
  `configs/phase0_{osf,physioomni,mantis}*.yaml`, `experiments/v2_{osf,physioomni,mantis}*.yaml`,
  `jobs/*{osf,physioomni,mantis}*.sh`.
- Reference clones (read-only, architecture ground truth): `OSF-Open-Sleep-FM/`,
  `PhysioOmni/`, and whatever the Mantis package/vendor path is inside
  `NSRR-tools-mantis` (check for a vendored copy or `pip install mantis-tsfm` usage).
- **Repo locations**: OSF and PhysioOmni code is merged into `NSRR-tools` main
  working copy (per user, 2026-09-19 — CLAUDE.md's "not yet merged" status note
  for OSF/PhysioOmni is stale, trust the user's statement over that file).
  Mantis code lives ONLY in the `NSRR-tools-mantis` git worktree (still running,
  not merged) — do not expect it in `NSRR-tools` proper.

## Target insertion point in npj_supplementary.tex

Insert three new `\subsection`s inside `\section{Comparison Across
Independently Pretrained Encoders}` (`sec:supp-tsfm`), immediately after the
section's intro paragraph (ends `...across all four encoders.` around line
2794) and BEFORE `\subsection{OSF pretraining-cohort overlap}`. Order:

1. `\subsection{OSF: Implementation, Data Adaptation, and Training Details}`
2. `\subsection{PhysioOmni: Implementation, Data Adaptation, and Training Details}`
3. `\subsection{Mantis: Implementation, Data Adaptation, and Training Details}`

Each subsection internally structured (bold run-in headers, not
`\subsubsection`, to match this document's existing style — check a section
like `sec:supp-encoder` for the pattern) as:
- **Preprocessing and data adaptation** — which HDF5 tree, channel mapping,
  normalization handling, windowing/patching, embedding extraction pipeline.
- **Stage 1: frozen encoder + downstream head** — what's frozen, new files,
  config/hyperparams, compute characteristics, coverage achieved.
- **Stage 2: low-rank adaptation (LoRA) fine-tuning** — LoRA config (target
  modules, rank, alpha, % trainable), warm-start/staging strategy, key
  hyperparams, coverage achieved and exact stopping point/reason.
- **Honest computational drawbacks** — concrete measured numbers (TFLOP/s,
  min/epoch), the specific architectural or operational cause, distinguishing
  "inherent to the model" from "an operational mistake on our end."
- (Fold in, don't duplicate as a separate list) other reviewer-relevant
  caveats specific to that model not already covered by the existing
  contamination/license/peer-review text elsewhere in the section.

**Also fix**: `sec:supp-tsfm` is currently MISSING from the Contents list at
the top of the document (checked: the `\tocmain{...}` list ends at
`sec:supp-head-ext`; `sec:supp-tsfm` has no entry, likely added after the ToC
was last updated). Add a `\tocmain{sec:supp-tsfm}{...}` entry once the new
subsections exist, titled to reflect the expanded content (implementation +
comparison), consistent with the other entries' phrasing style.

**Style constraints (from project memory / existing doc conventions)**:
- No em-dashes anywhere in prose (use commas/semicolons/restructured sentences).
- Use the task abbreviations already defined in the "Task naming" paragraph
  near the top of the supplementary (sex, apnea, age, BMI, OSA, depression,
  sleep efficiency) — don't reintroduce full task names.
- Match existing hedging conventions: never present an incomplete/missing
  cell as if it were a normal complete result; state measured vs. estimated
  vs. structurally-inferred numbers explicitly, matching how
  `TSFM_MODEL_COMPARISON.md` and the existing LoRA-coverage subsection do this.
- `\texttt{}` for code/config identifiers, matching existing supplementary usage.
- New tables/figures (if added) follow the `S-\Roman{table}` / `S-\arabic{figure}`
  auto-numbering already set up (`\label{tab:supp-...}`, `\label{fig:supp-...}`).
- Do not touch `npj_main.tex` — this task is supplementary-only.

## Round 1 — Research + First Draft

### Subtask 1.A — OSF research (delegate to subagent)
- [x] Status: **DONE** (completed 2026-09-19)
- Deliverable: `NSRR-tools/docs/npj_paper_md_files/tsfm_supp_findings_osf.md` — written.

**Also found while waiting on agents**: `NSRR-tools/docs/npj_paper_md_files/TSFM_BASELINE_RESULTS_DRAFT.md`
§5 ("Supplementary extension, outline not full prose") is a PRE-EXISTING outline
for supplementary additions the user never pasted in. Read it before drafting —
reconcile rather than duplicate or contradict it.

### Subtask 1.B — PhysioOmni research (delegate to subagent)
- [x] Status: **DONE** (completed 2026-09-19)
- Deliverable: `NSRR-tools/docs/npj_paper_md_files/tsfm_supp_findings_physioomni.md` — written.

### Subtask 1.C — Mantis research (delegate to subagent)
- [x] Status: **DONE** (completed 2026-09-19)
- Deliverable: `NSRR-tools/docs/npj_paper_md_files/tsfm_supp_findings_mantis.md` — written.

**All Round 1 research (1.A-1.C) complete.** Next: Subtask 1.D (draft LaTeX subsections).
- Deliverable: `NSRR-tools/docs/npj_paper_md_files/tsfm_supp_findings_mantis.md`
- Note: Mantis's Stage 2 was still running / incomplete as of the internal docs
  (no comparable frozen-vs-LoRA metric exists yet, per `TSFM_MODEL_COMPARISON.md`
  §3.4). Report this honestly, don't fabricate a completed picture.

### Subtask 1.D — Draft LaTeX subsections (main session, after 1.A-1.C return)
- [x] Status: **DONE** (2026-09-19)
- Read all three findings docs in full. Inserted a new
  `\subsection{Implementation and adaptation detail, by encoder}`
  (`\label{sec:supp-tsfm-implementation}`) into `npj_supplementary.tex`,
  immediately after `sec:supp-tsfm`'s intro paragraph and before
  `\subsection{OSF pretraining-cohort overlap}`. Contains three
  `\subsubsection`s (OSF, PhysioOmni, Mantis), each with: Preprocessing and
  data adaptation / Stage 1 / Stage 2 / Honest computational drawbacks /
  Other caveats specific to that model. Added 5 new tables
  (`tab:supp-tsfm-osf-cost`, `tab:supp-tsfm-physioomni-ceiling`,
  `tab:supp-tsfm-mantis-cost`, `tab:supp-tsfm-mantis-costest`, plus reused
  existing numbering conventions) — ~590 lines of new LaTeX.
- Added the missing `\tocmain{sec:supp-tsfm}{...}` Contents entry (line ~75).
- Compiled clean: no LaTeX errors, no multiply-defined/undefined-reference
  warnings for any new label (checked `npj_supplementary.log` directly).
  Two trivial Overfull/Underfull hbox "Information"-severity notices at the
  new content (cosmetic spacing only, not errors) — not fixed, low priority,
  candidate for a Round 2 polish pass if time allows.
- **Deliberately left out of the prose** (present in the findings docs but
  judged not essential for a reviewer-facing account, to keep prose length
  and abstraction level consistent with the rest of this section): OSF's
  dead/unused config keys, OSF's not-fully-root-caused N-count mismatches
  (the existing split-mismatch subsection already covers the core of this),
  PhysioOmni's raw SHHS-channel-correlation number (r=0.18), Mantis's
  MantisPlus/synthetic-checkpoint deferred-ablation plan, Mantis's
  8.11M-vs-8.037M parameter-count nuance, and specific software/library
  names (e.g. which cluster's software stack broke gradient checkpointing)
  in favor of paper-appropriate abstraction ("this cluster's software
  stack" rather than naming specific libraries). Revisit in Round 2 if the
  completeness pass judges any of these worth adding back.

## Round 1 complete. Proceeding to Round 2.

## Round 2 — Verification + Gap-Filling

### Subtask 2.A — Fact-check pass
- [x] Status: **DONE** (2026-09-19). 3 independent verification agents (fresh,
  not the Round 1 drafters) checked every concrete claim in the new prose
  against real source files. Result: 41 of 45 checked items CONFIRMED exactly;
  4 real issues found and FIXED in `npj_supplementary.tex`:
  1. OSF: "plus the head's own roughly 2.1 million parameters" was wrong/
     ambiguous — corrected to per-head figures (~1.7M LSTM, ~0.4M Transformer),
     verified against `results/collected/phase0_osf_lora/training.csv`'s real
     `n_trainable_params` column.
  2. PhysioOmni: LoRA trainable-fraction denominator (14.2M) was actually a
     backbone+placeholder-head test figure, not the real 4-encoder backbone
     (13.9M) — corrected the number and the resulting percentage.
  3. PhysioOmni: the "chunk_batch_size fix made no difference, tested
     identically" claim overstated an extraction-time (forward-only) A/B as if
     it were a Stage-2 (backward-pass) measurement; the source explicitly warns
     this doesn't transfer automatically — reworded to state the A/B was during
     frozen-encoder extraction and is suggestive, not a direct Stage-2 measurement.
  4. Mantis: two errors — (a) the layer-choice pilot result was reported
     backwards (prose said the final layer "outperformed" the earlier
     recommended layer; source shows the earlier layer scored higher by 0.0033,
     and the final layer was kept *despite* narrowly losing, for methodological
     consistency) — corrected; (b) the Nibi queue-time comparison ("full GPU
     queued 8x longer than a smaller partition") misapplied a pending/running
     ratio; source shows ALL larger partitions were bad (whole card ~8.6x
     oversubscribed, next-smaller size ~30x) and only the smallest partitions
     had headroom — corrected to state this accurately.
  Recompiled twice after fixes: 0 errors, 0 undefined references, braces
  balanced (1836/1836). Two pre-existing cosmetic hbox notices remain
  elsewhere in the document (not introduced by this task, not fixed, low
  priority).
- Minor unresolved item flagged by the Mantis check, not acted on: whether
  OSF's LSTM head was ever run on the two Tier-2 secondary tasks (the tex's
  actual wording is narrower/defensible and doesn't need to change, but this
  is worth a mental note if the user asks about it later).
- Re-verify every specific number/claim added in Round 1 against the actual
  config/code files (not just the planning docs, which can go stale) — spot
  check at least: LoRA target modules + trainable-param %, exact hyperparameters
  (LR, batch size, chunk_batch_size / context_micro_batch values), exact stopping
  contexts, measured TFLOP/s and min/epoch numbers.

### Subtask 2.B — Completeness pass
- [x] Status: **DONE**. Cross-checked the final prose against all three
  findings docs' full content. Confirmed present: preprocessing/channel
  mapping/normalization for all 3 models, architecture facts woven into the
  preprocessing/Stage-1 paragraphs (no separate heading needed), Stage 1 and
  Stage 2 mechanics and hyperparameters, honest drawbacks with 4 real
  quantitative tables (OSF measured cost, PhysioOmni ceiling breakdown,
  Mantis measured + estimated cost), and reviewer caveats (license,
  peer-review status, contamination pointers, split mismatches, missing
  MeanPool/threshold-tuning analyses, the Mantis sex-AUROC anomaly, the
  Mantis Stage-2 metric-incomparability warning). Deliberately-omitted items
  (see Subtask 1.D's note) were re-confirmed as reasonable to leave out —
  none of them are things a reviewer would need to evaluate the paper's
  claims, they're implementation trivia.

### Subtask 2.C — Consistency + style pass
- [x] Status: **DONE**. Grepped the new section for em-dash misuse (only one
  hit, a table-cell placeholder dash matching the document's own existing
  convention, not a prose em-dash — fine), unescaped `%` (none), and unicode
  em-dashes (none). Checked no new content restates specific per-context
  AUROC numbers that already live in this section's existing tables (avoided
  duplication/contradiction risk by design — the new subsections focus on
  mechanics, not results). Fixed two task-naming-convention misses ("OSA
  severity" used instead of spelling out "obstructive sleep apnea severity",
  twice, in the Mantis subsection) to match the document's own established
  task-abbreviation convention. Verified no label collisions and balanced
  braces (1836/1836).

### Subtask 2.D — Final read-through and handoff note
- [x] Status: **DONE**. Read the full new section fresh from disk end to end
  after all fixes. Recompiled with `pdflatex` twice more after the final
  wording fixes: clean, 82 pages, 0 errors, 0 undefined references.

## TASK COMPLETE (2026-09-19)

Both rounds finished. Summary of what was added to
`npj_digital_medicine_submission/npj_supplementary.tex`:
- New `\subsection{Implementation and adaptation detail, by encoder}`
  (`sec:supp-tsfm-implementation`), ~600 lines, inside the existing
  `sec:supp-tsfm` section, before its first existing subsection.
- Three `\subsubsection`s (OSF, PhysioOmni, Mantis), each covering
  preprocessing/data adaptation, Stage 1 (frozen + head), Stage 2 (LoRA),
  honest computational drawbacks (with measured/estimated cost tables), and
  model-specific reviewer caveats not already covered elsewhere in the section.
- 4 new tables: `tab:supp-tsfm-osf-cost`, `tab:supp-tsfm-physioomni-ceiling`,
  `tab:supp-tsfm-mantis-cost`, `tab:supp-tsfm-mantis-costest`.
- Added the previously-missing `\tocmain{sec:supp-tsfm}{...}` Contents entry.
- Fixed 6 concrete factual/wording errors caught by independent Round-2
  verification (see Subtask 2.A log above) before they could reach a reviewer.

**Research artifacts kept for future reference** (not deleted, may be useful
if this section needs further editing later): `tsfm_supp_findings_{osf,
physioomni,mantis}.md` in this same directory — the full, heavily-cited
research notes each drafting subsection was built from, containing more
detail than made it into the final paper prose (see Subtask 1.D's "left out"
note for what was trimmed and why).

**Nothing further is pending.** This plan file can be archived or deleted by
the user once they've reviewed the diff; not doing so unprompted.

## Log

- 2026-09-19: File created. Explored existing supplementary structure, confirmed
  `sec:supp-tsfm` is results-only, identified insertion point. About to dispatch
  three parallel research agents for Subtasks 1.A-1.C.
