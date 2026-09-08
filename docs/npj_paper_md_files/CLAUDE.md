# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Keep this file updated.** Whenever a significant decision is made — new pending tasks, file structure changes, submission milestones, or constraint discoveries — update the relevant section here so future sessions start with accurate context.

---

## What This Paper Is About

This paper studies **how much overnight PSG temporal context is needed for near-optimal clinical prediction** using a frozen SleepFM foundation model encoder. The core contribution is a systematic sweep across six context lengths (L ∈ {30s, 10m, 40m, 80m, 120m, 240m}) and up to K=50 inference-time aggregation windows, evaluated on seven clinical tasks (sex, sleep efficiency, BMI, age group, apnea severity, sleep staging, depression) across four NSRR cohorts (SHHS, MrOS, APPLES, STAGES, ~16,000 subjects total).

Three lightweight sequence heads sit on top of the frozen encoder: **Bi-LSTM**, **Transformer**, and **MeanPool** baseline. The key findings are task-specific saturation — physiologically complex tasks (e.g., OSA) need 80–240 min context while simpler tasks saturate at 10–40 min — and that aggregating many short windows partially but not fully substitutes for a single long-context forward pass at matched total signal budgets (iso-compute analysis).

The four hypotheses driving the paper (exact wording matches `npj_main.tex` Introduction):
- **H1** — AUROC rises with context length and saturates at a task-specific threshold L*.
- **H2** — for a model trained at fixed L, AUROC saturates after aggregating only a small number of inference windows K.
- **H3** — temporal sequence heads (LSTM, Transformer) outperform the non-temporal MeanPool baseline at long context but not short.
- **H4** — at an equal total budget (L×K held constant), AUROC depends only on the total budget, or also on how it is split between context length L and aggregation count K (the iso-budget analysis).

(Earlier versions of this file mislabelled H2–H4; the list above is authoritative and matches the current Results subsection order: H1 saturation → H2 aggregation → H3 architecture → H4 iso-budget.)

The paper uses the **V3 protocol** exclusively: overlapping-window fixed-K training, non-overlapping stride-N inference. Only `phase0_v3` results are valid for any reported numbers; `phase0` and `phase0_v2` are stale (pre-overlapping-window fix) and must never be reported or referenced.

---

## ⭐ CURRENT ACTIVE WORK: Results Section Rewrite (`markdown/RESULTS_REWRITE.md`)

**Read `markdown/RESULTS_REWRITE.md` in full before doing anything with the Results
section.** This is the live, in-progress work.

### What it is
A collaborator returned an annotated PDF of the paper with comments that the
Results prose was too dense: too many numbers packed into parentheses, prose
re-describing table contents instead of interpreting them, coined jargon, etc.
In response we are producing a **full rewrite of the entire Results section** in
`markdown/RESULTS_REWRITE.md`. Every paste-ready paragraph is wrapped in a fenced
```` ```tex ```` code block. (The fences exist **only** to stop GitHub from
rendering LaTeX `~` non-breaking spaces as strikethrough — see the "Formatting
note" at the top of the file. When pasting into `npj_main.tex`, copy only the text
*between* the fence lines.)

### The workflow (important — this is how the user works)
The user improves the Results **one paragraph at a time, collaboratively**:
1. The user selects a sentence/paragraph and asks about it (vagueness, a
   suspected error, repetition, wording, etc.).
2. The agent investigates (checks the data/tables/figures if a number or claim is
   involved), explains, and proposes a revision **in `RESULTS_REWRITE.md`** — the
   agent edits *that* file, never `npj_main.tex`.
3. The user reviews the revised paragraph in the md file, then **manually pastes
   it into `npj_main.tex` themselves**.

So: **the agent edits `RESULTS_REWRITE.md` only. Never edit `npj_main.tex`'s
Results section directly** unless the user explicitly says to. `npj_main.tex` is
updated by the user by hand, paragraph by paragraph.

### Conventions established during this rewrite (apply them)
- **No em-dashes at all** (already a repo rule) — commas, semicolons, or split
  sentences.
- **Numbers in parentheses are fine** (e.g. `(0.777)`), but **no explanatory-text
  parentheticals** — if a parenthesis contains a clause/aside, restructure it into
  a sentence.
- **Lead each paragraph with the finding** in plain language; cite one or two
  headline numbers; point to the table/figure for the rest. Introduce main
  tables/figures by name ("Table 1 shows…"), not as a bare trailing citation.
- **No coined jargon labels.** Already removed: "iso-budget substitutability",
  "context-dependent saturation speed". Use plain description. ("context-irreplaceable"
  was kept — it's self-explanatory.)
- **"iso-compute" was globally renamed to "iso-budget"** (the L×K quantity is a
  signal/recording budget, not a FLOPs measure). "compute"/"FLOPs" now refers only
  to the genuine training-compute analysis.
- **Verify every number against source** before asserting it: task/head/context/K
  metrics in `NSRR-tools/results/collected/phase0_v3/analysis.csv`; the head table
  values in `npj_main.tex`'s `tab:heads`; **supplementary figure/section/table
  numbers against `npj_supplementary.aux`** (the hand-typed Contents list can be
  stale). Watch the K_max confound: `tab:heads` compares at K_max, and K_max
  *differs* by context (≈4 windows at 120 min, ≈2 at 240 min), so a cross-context
  K_max comparison is not a same-K comparison.

### Tense — a decision the user deferred to a later session
The Results currently mix tenses inconsistently between subsections (H1 is
past-for-findings, which is correct; H2/H3 drifted to present). The agreed target
convention is **past tense for the study's own findings** ("AUROC increased",
"the Transformer outperformed"), **present tense for figure/table pointers**
("Table 3 shows") **and for timeless mechanisms** ("self-attention connects any
two patches directly"). A dedicated tense-normalisation sweep across the whole
Results is **pending** and was explicitly postponed to a future session.

### Bugs found during the rewrite that ALSO exist in the live `npj_main.tex` (fix when editing there)
These were fixed in `RESULTS_REWRITE.md` but the user has not yet propagated all of
them into `npj_main.tex`:
- **Wrong supplementary figure numbers**: modality bar/radar cited as "S-16 and
  S-17" but should be **S-18 and S-19** (S-16 is Aggregate Scaling, S-17 is Channel
  Expansion). In live `.tex` around line 1043.
- **Wrong section number**: scaling-law extrapolation cited as "Section S-XVIII.A"
  but is **S-XVII.A** (and has figure **S-5**). In live `.tex` ~lines 455 and 2092.
- **age-group Δ typo**: stated `+0.051`, correct value is **`+0.048`** (per
  `tab:saturation`). In live `.tex` ~line 415.
- **H2 K=5 logic**: "K=5 is a practically optimal deployment choice whenever
  L ≥ 40 min" is impossible at L=120/240 where K_max is only 2–4. Reworded in the
  md to "a handful of windows…". Live `.tex` still has the overstated version.
- **Sleep-efficiency H2 imprecision**: "curves remain well-separated at all K, with
  each additional context step contributing 0.07–0.10 AUROC" is false at the short
  end (30 s and 10 min nearly overlap at ~0.71; the 30 s→10 min step is +0.004, not
  0.07–0.10). Live `.tex` still has it.
- **BMI L\* inconsistency**: H3/Methods call BMI's L\* "10 min" (that's the *LSTM's*
  L\*), but H1 and Tables 1–2 give BMI's Transformer L\* as **240 min**. Reconcile
  in the live `.tex`.
- **Age iso-budget config**: age's 240-min-budget result is labelled "240-min/K=2"
  in the live `.tex` (~line 767) but `tab:isocompute` lists it as **120m/K=2**.

### Status of the rewrite (as of 2026-08-25)
Paragraph-level polishing done and reviewed with the user through **H1, H2, H3**
(including a citation-completeness audit that fixed the S-number bugs above, and a
per-cohort robustness paragraph that had cited nothing and explained the MrOS
sleep-efficiency divergence in a way that read as contradicting the headline —
now fixed). **Still to do with the user, paragraph by paragraph: H4, Precision-Recall,
Cross-task sensitivity, Aggregate scaling, Channel expansion, Modality ablation**
(these have correct citations but have not been through the same collaborative
polish), plus the deferred **tense sweep**. The file's own "Fifth-round audit",
"Sixth-round audit", and "Verification log" sections at the bottom record exactly
what was checked and changed.

### Related in-progress rewrite docs (same manual-paste workflow)
- `markdown/CONCLUSION_REVISION_NOTES.md` — proposed Conclusion rewrite (leads with
  the finding, not the "cost-performance frontier" label; adds the "not a universal
  recipe" framing). Not yet all pasted into `.tex`.
- `NSRR-tools/docs/FRAMING_REVISION_PLAN.md` — Abstract/Intro/Discussion framing
  (see the dedicated section lower in this file).

---

## Submission History

**TBME (archived):** The paper was originally written and formatted for IEEE Transactions on Biomedical Engineering (TBME), double-column, using `ieeecolor2.cls`. All TBME source files are archived in `tbme_submission/` — they are read-only reference material and must not be edited.

**npj Digital Medicine (active):** We decided to submit to npj Digital Medicine instead. This requires a major reformat: single-column Nature-portfolio style using `sn-jnl.cls`, section order Introduction → Results → Discussion → Methods, ≤150-word unstructured abstract, and superscript numbered references. The active files are `npj_main.tex` and `npj_supplementary.tex`.

**npj guidelines consulted:**
- Content types: https://www.nature.com/npjdigitalmed/content-types
- Submission guidelines: https://www.nature.com/npjdigitalmed/for-authors-and-referees/submission-guidelines
- Guide to authors: https://www.nature.com/npjdigitalmed/for-authors-and-referees/guide-to-authors

**Full submission checklist and pending TODOs:** `NPJ_SUBMISSION_PLAN.md` in this directory. Always consult it before starting any submission-related task.

---

## File Structure

```
npj_main.tex          ← active main manuscript (npj format)
npj_supplementary.tex ← active supplementary (standalone article class)
npj_main.pdf, npj_supplementary.pdf ← compiled outputs (stay in repo root)
figures/              ← all main/ext/sfig figure PDFs + model.pdf (moved out of
                         repo root 2026-08-03 to keep root readable for collaborators)
sn-jnl.cls            ← Springer Nature document class (do not edit)
bst/sn-nature.bst     ← Nature-portfolio bibliography style
NPJ_SUBMISSION_PLAN.md ← detailed submission checklist and pending TODOs
CLAUDE.md             ← this file (keep updated)
tbme_submission/      ← archived TBME files (read-only reference)
  generic-color.tex   ← original TBME main source (content reference)
  supplementary.tex   ← original TBME supplementary source
  combined/           ← merged main+supp PDF (reference only, do not edit)
  main_fig*.pdf       ← all figure PDFs
  sfig*.pdf           ← all supplementary figure PDFs
```

---

## Compiling

```bash
# Main manuscript (two passes to resolve cross-references)
pdflatex npj_main.tex && pdflatex npj_main.tex

# Supplementary
pdflatex npj_supplementary.tex && pdflatex npj_supplementary.tex
```

Build artifacts (`.aux`, `.log`, `.fdb_latexmk`, `.fls`, `.synctex.gz`, `.out`) are gitignored — do not commit them.

**Git commit rule:** Never add `Co-Authored-By: Claude` (or any Claude authorship trailer) to commit messages. Write plain subject + optional body only.

### Required TeX packages (BasicTeX)

`sn-jnl.cls` requires packages not in BasicTeX by default. If compilation fails with "File not found":

```bash
sudo tlmgr install sttools wrapfig threeparttable vruler appendix bigfoot
```

---

## Document Class and Key Preamble Decisions

`npj_main.tex` uses:
```latex
\PassOptionsToPackage{hidelinks}{hyperref}  % must come before \documentclass
\documentclass[pdflatex,sn-nature]{sn-jnl}
\unnumbered  % suppress section numbers — Nature portfolio style
```

`\unnumbered` is a **command**, not a class option. `hidelinks` must be passed via `\PassOptionsToPackage` because `sn-jnl.cls` loads `hyperref` internally.

`npj_supplementary.tex` uses plain `\documentclass[12pt,a4paper]{article}` with `\geometry{margin=2.5cm}` and supplementary counter styles (`S-Roman` sections, `S-arabic` figures, `S-Roman` tables).

Both files use `\graphicspath{{./figures/}{./}}` — all figure PDFs live in `figures/`
(moved there 2026-08-03 so the repo root isn't cluttered with ~40 PDFs; `{./}` is kept
as a fallback search path). Do not point at `tbme_submission/`.

---

## Section Order (npj vs TBME)

npj Digital Medicine follows Nature-portfolio convention:

```
Abstract (≤150 words, unstructured) → Introduction → Results → Discussion → Methods → end-matter
```

The TBME original had Methods before Results. The npj file has already been reordered. The TBME Conclusion section is merged into Discussion.

---

## Content Constraints

- **Never edit files in `tbme_submission/`** — they are the reference archive.
- **Never edit `tbme_submission/combined/`** — combined file is a point-in-time snapshot, not maintained.
- All edits go to `npj_main.tex` and `npj_supplementary.tex` only.
- Only report results from `phase0_v3` experiments; `phase0` and `phase0_v2` are stale (pre-overlapping-window fix).
- Do not use em-dashes (`---`) in prose; use commas, semicolons, or restructured sentences.
- When referencing supplementary tables from the main paper, use hardcoded numbers (e.g., `Table~S-XIII`), not `\ref{}`, because they are in a separate file.
- All numbers in tables must be real, correct values. If a cell does not apply (e.g., K=5 for a subject with K_max < 5), use `---` not a placeholder number.

### Acronym convention

Every acronym must be defined at its **first occurrence in reading order** (Abstract → Introduction → Results → Discussion → Methods), as `Full Name (ACRONYM)`. Because npj puts Methods last, acronyms used earlier that are only formally defined in Methods must be expanded at that earlier point instead.

This invariant must be re-checked whenever:
- Sections are reordered or text is moved between sections.
- New content is added that introduces or moves an acronym's first occurrence.
- Any sentence containing an acronym's first use is edited or relocated.

**2026-08-26 audit:** the list below was previously stale (several entries named a location the acronym was never actually expanded at — notably BMI, OSA, and AUROC, which had never been formally expanded anywhere in the file despite being used dozens of times). A full pass verified every acronym's true first occurrence against the live file, fixed the ones that were used-before-defined or multiply/redundantly defined (the "AHI problem": AHI, MIG, LSTM/Bi-LSTM naming, EOG, EHR), and corrected this list to match. Pre-edit backups of both `.tex` files are kept locally at `npj_digital_medicine_submission/backups/*.pre-taskname-edit.tex` (git-ignored, for diffing if table formatting looks off after the task-name column-width changes).

The full list of acronyms currently defined in `npj_main.tex` and where (reading order):
- **PSG** — Abstract ("polysomnography (PSG)")
- **EEG** — Introduction ("electroencephalography (EEG)")
- **LSTM** — Abstract ("Long Short-Term Memory (LSTM)"); standardized this session — the paper previously mixed "Bi-LSTM" (Abstract/Study-Overview/Methods heading) with bare "LSTM" (used 48x elsewhere) as if they were different things. Now "LSTM" everywhere; "Bi-LSTM" only survives inside one commented-out line.
- **BMI** — Abstract ("body mass index (BMI)"), reinforced at first Results mention (Study Overview paragraph). Previously never formally expanded anywhere in the file.
- **OSA** — Results, Study Overview paragraph ("Obstructive sleep apnea (OSA) severity"). Previously never formally expanded anywhere; also previously collided with the *separate* primary "apnea" task, which was mislabeled "apnea severity" in 4 places and "OSA detection" in 3 places — both fixed (now "apnea detection"/"apnea").
- **NSRR / SHHS / MrOS / APPLES / STAGES** — Introduction cohort sentence (full names) — verified correct, unchanged.
- **AUROC** — Abstract, last sentence ("area under the receiver operating characteristic curve (AUROC)"). Previously never formally expanded anywhere despite ~100 uses.
- **AHI** — Results, Study Overview paragraph, inside the apnea task's definition ("apnea--hypopnea index (AHI)"). Previously used unexpanded in Table 1 and Discussion long before its one Discussion-only definition ~1000 lines later (the original "AHI problem"); now defined once, early, and the later Discussion mention just uses "AHI" bare.
- **EOG** — Results, Table 5 (`tab:modality`) footnote ("EEG + electrooculography, EOG"). Previously used 7x, never expanded anywhere.
- **EMG** — Results, Table 5 (`tab:modality`) footnote ("EMG: electromyography").
- **RESP / BAS / EKG** — Results, Table 5 (`tab:modality`) footnote (functional descriptions, e.g. "EKG: cardiac channels"), reinforced in the running text immediately after. These function as the paper's own modality-group category names, not classic word-abbreviations.
- **EHR** — Discussion ("electronic health record (EHR)-linked survival models"). Previously used unexpanded once; fixed.
- **AP** — Results, precision-recall section ("average precision (AP)") — verified correct, unchanged. (Note: bare "PR" is never actually used in `npj_main.tex` — only the spelled-out "precision-recall" — so it needs no acronym-list entry there; it *is* used as "PR" in the supplementary, see below.)
- **FIR** — Methods, filtering section ("finite impulse response (FIR)") — verified correct, unchanged.
- **ESS / PHQ-9** — Methods, tasks section (full instrument names) — verified correct.
- **BDI-II** — resolved 2026-08-26 (user confirmed): Table 1 previously said bare "BDI" in its footnote, Label-source column, and threshold definition, while Methods (`sec:tasks`) said "BDI-II". Table 1 now says "BDI-II" throughout, matching Methods. (Supplementary line ~175 still shows bare "BDI (`bditotalscore}`)" — that's a literal NSRR raw column-name annotation, not an instrument-name claim, left as-is.)
- **AASM / CPAP** — Methods, Table 1 footnote (full names) — verified correct.
- **REM** — Methods, Table 1 footnote ("REM: rapid eye movement sleep") — location corrected (previously misattributed to a "sleep staging section" that doesn't currently contain this definition).
- **CLS** — Methods, TransformerHead section ("classification (CLS) token") — verified correct.
- **GAD-7 / ISI** — **no longer present anywhere in the file.** The previous entry claiming these are defined in a "Discussion null-results paragraph" is stale; that paragraph either never existed in this form or was rewritten since. Do not reference GAD-7/ISI as paper content unless re-added deliberately.

**`npj_supplementary.tex` is audited separately** (it's a standalone document with its own reading order — supplementary materials conventionally accompany the main text, so *general* ML/clinical terms already defined in main.tex don't need re-defining there, but genuinely supplementary-only terms and internal supplementary-only inconsistencies do). Fixed this session: a new "Task naming" preamble note (right after the Contents section) formally maps every task's short form used throughout the supplement, plus `AHI`, `PHQ-9` (Patient Health Questionnaire-9), `MIG` (Multi-Instance GPU — was used unexpanded ~750 lines before its one existing definition, the same AHI-pattern bug), `BA` (balanced accuracy), and `CI` (confidence interval), all previously either undefined or defined only after already being used bare. `PR` (precision-recall) was already correctly expanded at its first live use. Checked and left alone as acceptable (relies on main.tex's definition, standard practice, or is a universally-recognized term not needing expansion): `FIR`, `ECG`/`EKG` (used consistently as distinct terms — EKG is the modality-group name, ECG/ECG-L/ECG-R are raw per-cohort lead identifiers, not a naming bug), `GPU`, `HDF5`, `XML`, `MNE` (a software library name, not a phrase acronym), `H100`/`MIG`-the-GPU-product-name.

### Hypothesis (H1–H4) framing convention

H1–H4 labels are kept as organisational bookmarks throughout the paper. The convention is:
- **Introduction**: H1–H4 defined once, formally, as an itemised list near the end of the study-design paragraph (before the contributions list).
- **Results subsection headings**: findings-first, no H-label in the heading (e.g., "Context-length saturation is task-specific", not "Context saturation (H1)").
- **Results opening sentences**: finding first, H-label in parentheses second (e.g., "Context-length sensitivity is confirmed across all five tasks (H1): …").
- **Methods sec:sweep**: brief back-reference to H1–H4 as defined in Introduction; do not repeat the full list.
- Do not add "Testing hypothesis Hx…" as a sentence opener anywhere.

---

## NSRR-Tools Codebase

The experiment and analysis codebase lives at `/Users/boshra/NSRR-workspace/NSRR-tools/`.

```
NSRR-tools/
  src/nsrr_tools/
    core/          ← signal_processor.py, annotation_processor.py, channel_mapper.py,
                      metadata_builder.py, modality_detector.py
    models/
      sequence_head.py  ← Bi-LSTM, Transformer, MeanPool head definitions
    datasets/      ← dataset loader classes
    targets/       ← target extraction utilities
    utils/
      config.py    ← experiment config loading
      mount_utils.py
  configs/
    phase0_v3_config.yaml        ← V3 training config (the active one)
    phase0_v3_config_full.yaml   ← full-channel variant
    phase0_v3_config_abl.yaml    ← modality ablation variant
    phase0_v3_config_staging.yaml
  experiments/
    v2_registry.yaml             ← experiment registry (fast-channel V3 baseline)
    v2_full_registry.yaml        ← full-channel registry
    v2_ablation_registry.yaml    ← ablation registry
  scripts/
    collect_results_v2.py  ← pulls results from cluster to local CSVs
  docs/
    EXPERIMENTS_GUIDE.md   ← **definitive reference** for the full pipeline
    RESULTS_COLLECTION.md  ← documents CSV schema and collection workflow
    TRAINING_PROTOCOL_FIXES.md  ← rationale for V3 protocol changes
    PAPER_PLAN.md          ← original paper plan and section outline
  results/
    collected/
      phase0_v3/           ← all valid result CSVs (only use these)
        analysis.csv       ← per-(task, head, context_length, k) test metrics
        training.csv       ← per-epoch train/val/test metrics
    paper_figures/
      notebooks/           ← one notebook per figure
      notebooks/utils/     ← shared helpers
```

**Always read `docs/EXPERIMENTS_GUIDE.md` first** when asked anything about training, inference, analysis, figure generation, or the experiment pipeline. It is the single authoritative reference.

---

## Results CSV Format

Results live in `NSRR-tools/results/collected/phase0_v3/`. Only files in `phase0_v3/` are valid for the paper.

**`analysis.csv`** — primary results table, one row per (task, head, run_tag, context_length, k, split):

| Column | Description |
|--------|-------------|
| `task` | e.g., `sex`, `apnea`, `sleep_efficiency`, `bmi`, `age_group`, `staging`, `depression` |
| `head` | `bilstm`, `transformer`, `meanpool` |
| `run_tag` | experiment identifier (fast/full/ablation variant) |
| `context_length` | training context in seconds (e.g., 30, 600, 2400, 4800, 7200, 14400) |
| `context_length_min` | same in minutes |
| `k` | number of inference windows aggregated (1, 5, 10, 20, 50, or `all`) |
| `split` | `test` (use this for paper numbers) |
| `mean_prob_auroc` | primary metric (area under ROC, averaged over subjects) |
| `CI_lower`, `CI_upper` | 95% bootstrap CI bounds |
| `n_subjects`, `n_segments` | count columns |
| `total_compute_min` | k × context_length_min (used for iso-compute analysis) |

**`training.csv`** — per-epoch training log, one row per (task, head, context_length, epoch):

| Column | Description |
|--------|-------------|
| `task`, `head`, `context_length` | same as above |
| `epoch` | training epoch number |
| `is_best_epoch` | boolean, True for the checkpoint used in inference |
| `train_auroc`, `val_auroc`, `test_auroc` | per-split AUROC at this epoch |

See `docs/RESULTS_COLLECTION.md` for how `collect_results_v2.py` pulls data from the cluster.

---

## Figure Inventory (`figures/` PDFs)

All figure PDFs live in `figures/` (moved out of the repo root 2026-08-03). The
notebooks use **TBME numbering** internally but save to **npj-numbered**
filenames; the generation/collection scripts still write to the repo root, so a
generated PDF must be moved into `figures/` before compiling (or update the
script's output path).

### Main figures

| PDF in `figures/` | Content | Notebook (TBME name) |
|---|---|---|
| `main_fig1_preprocessing.pdf` | Pipeline/preprocessing schematic | `main_fig1_placeholder.ipynb` |
| `main_fig2_kvsk.pdf` | K vs K iso-compute | `main_fig3_kvsk.ipynb` |
| `main_fig3_heatmap.pdf` | Iso-compute heatmap | `main_fig4_heatmap.ipynb` |
| `main_fig4_iso_main.pdf` | Iso-compute main curves | `main_fig5_iso_main.ipynb` |
| `main_fig5_waterfall.pdf` | Waterfall gain decomposition | `main_fig6_waterfall.ipynb` |
| `main_fig6_pr_curves.pdf` | Precision-recall curves | `main_fig7_pr_curves.ipynb` |
| `main_fig7_sweep.pdf` | Context-length sweep (moved last in npj) | no dedicated notebook (generated separately) |

Note: notebooks still use old TBME figure numbers in their filenames. The PDFs have been renamed to npj numbering. Do not rename the notebooks.

### Extended data figures (peer-reviewed, cited in main text)

These are copies of sfig17/19/20 with different filenames. The notebooks currently save only a single version. **Each needs to be split into two outputs** (ext_fig version with selected tasks, sfig version with remaining tasks).

| PDF | Content | Source notebook | Tasks in ext_fig | Tasks in sfig |
|---|---|---|---|---|
| `ext_fig1_k_aggregation.pdf` | K-aggregation curves (LSTM + Transformer) | `sfig17_k_aggregation.ipynb` | Sex, Sleep Eff., Apnea | Age, BMI, OSA |
| `ext_fig2_compute_scaling.pdf` | AUROC vs training FLOPs | `sfig19_compute_scaling.ipynb` | Sex, Sleep Eff. | Age, BMI, Apnea, OSA |
| `ext_fig3_variance_violins.pdf` | Within-subject std(prob) by correct/incorrect | `sfig20_variance_violins.ipynb` | Sex, Sleep Eff. | Age, Apnea, BMI, Depression, OSA |

**Important note on ext_fig2 (compute scaling):** MeanPool is ~640× cheaper per step than Transformer, so it occupies a lower total-FLOPs regime (10^7–10^11) than temporal heads (10^10–10^14). The three fit lines therefore cover different x-ranges and do not support a direct head-vs-head comparison at matched FLOPs. The figure shows that head hierarchy holds within each head's natural compute regime, not that Transformer is more efficient per FLOP. Consider extrapolating all fit lines to a common x-range before final submission.

**Important note on ext_fig3 (variance violins):** Age shows the OPPOSITE pattern to Sex — at long contexts, correctly classified age subjects have HIGHER within-subject variance than incorrectly classified ones. Do not describe Age as "similar to Sex."

### Supplementary figures (sfig1–sfig23)

All sfig PDFs live in `figures/`. Counterpart notebooks named `sfig<N>_<name>.ipynb`. sfig17, sfig19, sfig20 are the full-task versions of the three extended data figures above — after the notebooks are split, they will contain only the non-extended tasks.

### Placeholders in npj_main.tex

Three `% EXTENDED DATA FIGURE N PLACEHOLDER` comment blocks exist in `npj_main.tex` at the locations where ext_fig1, ext_fig2, ext_fig3 will be inserted once the PDFs are finalised. Replace each comment block with a proper `\begin{figure}` environment when ready.

---

## Figure Generation

All paper figure notebooks are at `NSRR-tools/results/paper_figures/notebooks/`.

**Shared utilities** at `notebooks/utils/`:
- `data.py` — data loading helpers (reads analysis.csv, filters splits, maps column names)
- `panels.py` — panel-level figure functions (the actual plotting logic per subfigure)
- `style.py` — visual styling (colors, fonts, rcParams for all figures)

When modifying a figure, edit the notebook first. If a change applies to all figures (e.g., colour palette, font size), edit `utils/style.py`. If a change applies to multiple panels of one figure, edit `utils/panels.py`.

---

## Main + Supplementary Fact-Check Review

An ongoing, section-by-section review is cross-checking every factual claim,
number, and table in `npj_main.tex` and `npj_supplementary.tex` against the
actual NSRR-tools code, configs, and `results/collected/phase0_v3/analysis.csv`
(not just re-reading prose for quality). This is a multi-session effort —
**the persistent tracking log lives outside this repo, at
`/Users/boshra/NSRR-workspace/SUPP_MAIN_REVIEW.md`**. It records what's been
verified, what was found wrong and fixed, what's still pending, and exactly
where to resume. Always read that file before starting or continuing a
supplementary/main review pass.

---

## Abstract/Introduction/Discussion Framing Revision

The paper's framing (why temporal context matters, cost/deployment stakes,
answers to specific supervisor questions on encoder generalizability, the
token-budget schedule, and the 240-min context cap) was revised starting
2026-07-31. **The planning/reasoning document for this lives outside this
repo, at `/Users/boshra/NSRR-workspace/NSRR-tools/docs/FRAMING_REVISION_PLAN.md`**
— moved there deliberately so it stays git-tracked without being visible in
this repo, which the supervisor has access to. It records the diagnosis, the
supervisor's exact comments, and an "Implemented (round 1)" note with the
final shipped text for every change actually made to `npj_main.tex`. Read it
before making further framing-related edits to the Abstract, Introduction, or
Discussion.

---

## TSFM Baseline Comparison (separate effort, in progress)

In response to supervisor feedback questioning the SleepFM choice and
asking for comparison against recent time-series foundation models, we're
adding real baseline comparisons (OSF, PhysioOmni, MOMENT selected first;
more candidates catalogued if wanted later). This work lives entirely in
`NSRR-tools`, not here — **see `NSRR-tools/CLAUDE.md`** ("TSFM Baseline
Model Comparison" section) **and `NSRR-tools/docs/TSFM_BASELINE_CANDIDATES.md`**
for the full plan, model selection reasoning, and code-verified findings.

**State as of 2026-08-05:** model selection done; all three candidate repos
cloned locally and code-verified (none support native long-context, all
will be used as short-segment embedders like SleepFM itself; PhysioOmni
excludes apnea — no respiratory pathway; OSF has cohort-specific
contamination risk to caveat). **No detailed step-by-step implementation
plan exists yet** — that's the next step, not started. Nothing has run on
the cluster yet; the three model repos are only cloned locally
(`/Users/boshra/NSRR-workspace/{OSF-Open-Sleep-FM,PhysioOmni,moment}`), not
on Compute Canada. An agent picking this up on the cluster should start at
`NSRR-tools/CLAUDE.md`'s "Cluster Execution Guidance" section.

This paper repo is referenced (read-only) for framing/consistency by that
effort, but no code or content here needs to change for it.

---

## Pending Work (as of 2026-07-28)

See `NPJ_SUBMISSION_PLAN.md` for the full checklist. Key items:

1. ~~Trim abstract~~ — **Done** (148 words)
2. ~~Fig. → Figure~~ — **Done** (41 occurrences)
3. ~~H1–H4 framing~~ — **Done**: defined in Introduction, findings-first in Results, paraphrased in Methods
4. ~~Reference format~~ — **Done**: Nature style, inline `\thebibliography`, `\setcitestyle{super}`
5. ~~Introduction completeness + Results→Methods refs~~ — **Done**
6. ~~Extended data figure selection~~ — **Done**: decisions in `EXT_FIGURES_TASK_SELECTION.md`
7. **Split notebooks**: `sfig17_k_aggregation.ipynb`, `sfig19_compute_scaling.ipynb`, `sfig20_variance_violins.ipynb` each need to produce two output PDFs — ext_fig version (selected tasks) and sfig version (remaining tasks). See Figure Inventory above for which tasks go where.
8. **Wire ext_fig figures into npj_main.tex**: replace the three placeholder comment blocks with `\begin{figure}` environments and interpretive text. Body text drafts are in `EXT_FIGURES_TASK_SELECTION.md`.
9. **Fix sfig16 caption**: currently says "apnea detection" but the figure shows multiple tasks (sex, BMI, sleep efficiency, apnea, others).
10. Add Statistics subsection to Methods; clarify pooled-cohort vs external validation
11. Add calibration limitation and subgroup/fairness limitation to Discussion
12. Remove remaining IEEE-specific commands if any (search for `\IEEEPARstart`)
13. Verify `mros` reference against NSRR dataset page (sleepdata.org) before submission
14. Fill in author names, affiliations, and emails (currently placeholder text)
15. Write cover letter
