# npj Digital Medicine Submission Plan

Source guidelines consulted:
- https://www.nature.com/npjdigitalmed/content-types
- https://www.nature.com/npjdigitalmed/for-authors-and-referees/submission-guidelines
- https://www.nature.com/npjdigitalmed/for-authors-and-referees/guide-to-authors

---

## 1. Article Type

Submit as **Article** (primary research, original empirical study).

- Word limit: **none stated** (aim for ~4,000–5,000 words main text, concise)
- Abstract: **150 words, unstructured** (single paragraph, no headings)
- References: ~60 (not strictly enforced, but keep tight)
- Figures: no stated maximum; aim for **5–6 main figures**

---

## 2. Repository / File Plan

```
npj_submission/
  npj_main.tex          ← new main manuscript (single-column, Nature style)
  npj_supplementary.tex ← supplementary information
  figures/              ← copies of figure PDFs (rename Fig → Figure)
  refs.bib              ← same bib, no changes needed

TBME_submission/        ← archived as-is on the TBME branch
```

Branch strategy:
- Keep `main` branch as TBME submission
- Create `npj` branch from `main`
- In `npj` branch: leave `TBME_submission/` untouched, create `npj_submission/` alongside it

---

## 3. LaTeX Template Changes

| Item | TBME | npj |
|---|---|---|
| Document class | `ieeecolor2` (2-column) | `article` or Nature's own template, **single column** |
| Column layout | 2-column | 1-column |
| Fonts | IEEE Helvetica/Times | Standard (Nature sets in production; use 12pt serif for submission) |
| Section numbering | `\section{I. Introduction}` Roman numerals | Plain `\section{Introduction}` — no numbers |
| Subsection numbering | `\subsection{A. Datasets}` letters | `\subsection{Datasets}` — no letters |
| Figure labels | `Fig.~\ref{}` | `Figure~\ref{}` (spell out fully) |
| Citation style | `\cite{}` → `[1]` inline IEEE | `\cite{}` → superscript `^1` (use `natbib` with `super` option or Nature bib style) |
| Reference format | IEEE bibliography style | Nature numbered style (author, title, journal, year, vol, pages) |
| Page size / margins | IEEE template | A4, 2.5 cm margins all sides (standard for Nature submission drafts) |
| `\IEEEmembership`, `\thanks` | Used | Remove; use standard `\author` + `\affil` |
| `\markboth`, `\journalname` | Used | Remove |

---

## 4. Structural Reorganisation

Nature-portfolio convention is: **Abstract → Introduction → Results → Discussion → Methods → end-matter**. Methods move to after Discussion. This is the most important structural change.

### Bridging the Introduction–Results gap (important)

Simply swapping Methods and Results would break the narrative: the reader would see "AUROC rose from 0.707 at 30 s to 0.831 at 240 min" without knowing what 30 s means, what K is, or what the four research questions are. The standard Nature-portfolio solution is **not** to move Methods content into Introduction — it is two lightweight additions:

**A. End the Introduction with a "Here, we..." paragraph (3–5 sentences)**
State informally: the four cohorts and approximate N, the SleepFM encoder and three heads, the six context lengths swept, the seven tasks, and the four research questions (without H1–H4 labels). Example framing: "We ask four questions: does performance saturate with context length, and at what task-specific point? Can inference-time aggregation substitute for long-context training at matched budgets? Do temporal sequence heads outperform simple averaging, and by how much? And what is the optimal trade-off between context length and aggregation count for a fixed compute budget?"

**B. Add a "Study design" first subsection of Results (~150 words + pipeline figure)**
This subsection (paired with the study design/pipeline figure as Figure 1) briefly names the cohorts, tasks, encoder, heads, and context levels before any numbers appear. This gives the reader exactly the orientation they need to interpret the subsequent subsections. It replaces the role that Methods normally plays when read first. The content is not detailed (no equations, no hyperparameters) — just a plain-language description of what was done.

After these two additions, subsequent Results subsections can go straight into findings. Remove "Testing hypothesis H1…" openers and replace with direct finding statements. The detailed Methods remain after Discussion as reference material.

### Current TBME order → npj order

```
TBME:                           npj:
  Abstract (structured)    →     Abstract (unstructured, 150 words)
  Introduction             →     [no heading] Introduction
  Related Work             →     (fold into Introduction)
  Methods                  →     Results
  Results                  →     Discussion
  Discussion               →     Methods
  Conclusion               →     (remove — fold into Discussion final paragraph)
  References               →     Data Availability
                                  Code Availability
                                  Acknowledgements
                                  Author Contributions
                                  Competing Interests
                                  References
```

Key points:
- **Introduction** in npj has **no section heading** — text flows directly after the abstract
- **Related Work** section (if present) should be folded into the Introduction, not a standalone section
- **Conclusion** section is removed; the final paragraph of Discussion serves this role
- **Methods** go after Discussion, numbered as a section but placed last before end-matter

---

## 5. Abstract Rewrite

Current abstract: structured with numbered points ("Three consistent patterns emerge. First…").

npj requires: **single unstructured paragraph, ≤150 words**, no sub-headings.

Suggested structure (all in one paragraph):
1. One sentence: clinical motivation and the open question (context length in PSG)
2. One–two sentences: what we did (study design, datasets, encoder, tasks)
3. Two–three sentences: what we found (three core patterns: task-specific saturation, aggregation trade-off, temporal head advantage)
4. One sentence: significance / take-home

Draft (to refine):
> Overnight polysomnography encodes markers of diverse clinical conditions, yet the effect of input
> context length on prediction performance has not been systematically studied. We swept six context
> lengths (30 s to 240 min) across seven clinical and demographic tasks in approximately 16,000 subjects
> from four cohorts, using a pre-trained frozen foundation model encoder and three sequence heads.
> Context-length saturation was task-specific: tasks grounded in full-night physiology improved
> continuously through the longest contexts, tasks driven by periodic phenomena saturated at
> intermediate lengths, and metabolic targets showed minimal context benefit. Inference-time
> aggregation of multiple short windows partially recovered performance at matched signal budgets
> but could not substitute for long-context training on context-sensitive tasks. Temporal sequence
> heads gained increasing advantage over mean-pooling as context grew. These findings establish
> context length as a consequential, task-specific design parameter in clinical PSG model development.

(~155 words — trim slightly to hit 150.)

---

## 6. Introduction Changes

- **Remove the section heading** (`\section{Introduction}` stays in LaTeX for structure but produces no printed heading in Nature style — check template)
- **Fold Related Work into Introduction**: the current separate Related Work section should become the second half of the Introduction (prior literature → gap → what we do)
- **Expand clinical framing**: npj readers include clinicians; open with why context length matters for clinical deployment, not just for model accuracy
- **End with explicit paragraph**: "Here we present…" or "In this study, we…" sentence clearly stating contribution, before transitioning to Results

---

## 7. Results Changes

- Move to **before Methods**
- Remove hypothesis labels H1–H4 from headings ("Testing hypothesis H1…") — replace with descriptive subsection headings:
  - e.g., "Context-length saturation is task-specific" instead of "Context-Length Saturation (H1)"
  - H4 section: "Inference-time aggregation partially substitutes for long-context training"
- Spell out "Figure" everywhere (`Figure~1` not `Fig.~1`)
- Keep subsections, but remove letter labels (A, B, C) from headings
- The ISO-compute subsubsections ("H4: budget-dependent crossover" etc.) → rename to plain descriptive phrases
- Main claims should be stated as findings, not hypothesis tests
- Tables: single-column format, caption above table (Nature convention), simplified — remove `\footnotesize`, `\setlength{\tabcolsep}`, etc. and use standard table formatting for the template

---

## 8. Discussion Changes

- Add a clear final paragraph that summarises the contribution and take-home message (replaces the removed Conclusion section)
- The existing conclusion can be trimmed and appended as the last paragraph of Discussion
- No changes to content are strictly required — the restructuring from previous editing sessions already improved it
- Ensure limitations paragraph is present and complete (it is)

---

## 9. Methods Changes

- Move to after Discussion
- Remove letter labels (A, B, C) from subsection headings
- Remove "III." etc. from any references to Methods sections within the text (since numbering changes)
- Add a **Statistics** subsection (if not already fully explicit) covering: bootstrap CI method, number of resamples, two-sided vs one-sided, significance thresholds — Nature expects this
- TRIPOD-AI reporting: see Section 11 below

---

## 10. Required End-Matter Sections (currently absent from TBME version)

Add the following sections in this order, after References:

### Data Availability
> The polysomnography datasets used in this study are available through the National Sleep Research Resource (NSRR) at sleepdata.org. Processed embeddings and derived target labels are available from the corresponding author upon reasonable request.

### Code Availability
> The analysis code is publicly available at https://github.com/boshrakh95/NSRR-tools.

### Acknowledgements
> (Move current `\thanks` funding statement here. Add any data access acknowledgements for NSRR/cohort-specific requirements.)

### Author Contributions
> Required. Format: "[Initials] did X. [Initials] did Y. All authors reviewed the manuscript."
> Use CRediT taxonomy: Conceptualization, Data curation, Formal analysis, Funding acquisition, Investigation, Methodology, Project administration, Resources, Software, Supervision, Validation, Visualization, Writing – original draft, Writing – review & editing.

### Competing Interests
> "The authors declare no competing interests." (or declare any if present)

### Ethics
> Human subjects statement already exists in `\thanks`; move here or to Methods.

---

## 11. Reporting Checklist: TRIPOD-AI

npj Digital Medicine requires a completed reporting checklist for AI/ML prediction studies.
**TRIPOD-AI** (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis — Artificial Intelligence extension) is the relevant checklist.

Key TRIPOD-AI items and current paper status:

| Item | Status | Action needed |
|---|---|---|
| Study objectives clearly stated | ✓ | None |
| Data sources described (cohorts, inclusion/exclusion) | ✓ | None |
| Outcome definition (task labels, thresholds) | ✓ | None |
| Predictors described (signal channels, preprocessing) | ✓ | None |
| Sample size justification | ✗ | Add brief note (or acknowledge as limitation) |
| Model development method described | ✓ | None |
| Model performance: discrimination (AUROC) | ✓ | None |
| Model performance: calibration | ✗ | Add calibration analysis OR explicitly acknowledge as a limitation |
| Internal validation / test set | ✓ (train/val/test split) | None |
| External validation | ✗ (4 cohorts pooled, not true external) | Clarify in Methods |
| Handling of missing data | Partial | Clarify in Methods |
| Fairness / subgroup analysis | ✗ | Add as limitation or brief analysis |
| Clinical utility / decision analysis | ✗ | Brief statement in Discussion |
| Intended clinical use context | Partial | Clarify in Discussion |
| Code availability | ✓ (GitHub) | Add to end-matter |

Priority action items: (1) add calibration note/limitation, (2) clarify pooled vs. external validation, (3) add subgroup/fairness statement as limitation, (4) complete and attach TRIPOD-AI checklist PDF at submission.

---

## 12. Figure and Table Formatting

### Figures
- Rename all `Fig.` → `Figure` in text and captions
- Figures must be **300 DPI minimum**, TIFF or PDF format (current PDFs should be fine)
- Captions must be **fully self-contained** — a reader should understand the figure without reading the text; expand current captions where they rely on text for context
- Multi-panel figures: label panels as **(a)**, **(b)** etc. (lowercase, bold) — current figures already do this
- Main text: aim for **5 main figures** (currently 5 main figures + 3 main tables — acceptable)
- Consider whether any main figures can be consolidated or moved to Extended Data

### Tables
- Caption **above** the table (Nature convention; TBME has caption above already — no change)
- Remove `\footnotesize` and `\setlength{\tabcolsep}` — use template defaults
- Simplify column headers; avoid abbreviations without definition in caption
- Consider whether Tab. I (full sweep) and Tab. II (saturation) can be merged into one main table to save space, with the other as Extended Data

### Supplementary → Extended Data + Supplementary Information
Nature distinguishes:
- **Extended Data**: peer-reviewed figures and tables directly supporting main conclusions; cited in main text as "Extended Data Figure 1"
- **Supplementary Information**: additional material, not peer-reviewed (methods details, extra experiments)

Current supplementary has ~16 figures and ~16 tables. Suggested split:
- **Extended Data** (promote to main submission): S-1 (saturation curves), S-3 (K vs K all tasks), S-4 (iso-compute extra tasks), S-5 (Pareto extra tasks), S-7 (deployment grid), S-8 (waterfall all tasks), S-XI/S-XII (LSTM sweep tables)
- **Supplementary Information**: remaining tables and figures

**Confirmed extended data choices (2026-07-27):** Three supplementary figures have been selected to move to extended data. The PDF files, generation notebooks, and supplementary `.tex` entries will be renamed/restructured first; only then will the main `.tex` be updated to include them with full interpretive text.

- **fig:supp-kagg (current S-17, sfig17\_k\_aggregation.pdf)** — AUROC vs inference-time K for all tasks at L ∈ {40, 120, 240 min}, both LSTM and Transformer. Direct empirical support for H2: K=5 captures >99% of full-night AUROC at L ≥ 40 min. Was already listed as old "S-3 (K vs K all tasks)" in the initial suggestion above.
- **fig:supp-variance (current S-20, sfig20\_variance\_violins.pdf)** — Within-subject prediction std(prob) distributions stratified by correct vs incorrect classification, all tasks, four context lengths. Shows that longer context produces lower variance for correctly-classified subjects and larger separation from incorrectly-classified ones. Quantitative, all-tasks, directly supports the prediction-stability claim.
- **fig:supp-compute (current S-19, sfig19\_compute\_scaling.pdf)** — Test AUROC vs total training FLOPs (power-law fits per head). Shows Transformer achieves higher AUROC at all compute levels with steepest scaling slope. Supports the "context efficiency / compute budget" framing of the paper and is relevant to practical deployment decisions. Note: the caption currently has a stale TBME reference ("Section III-G") that must be updated to the npj Methods section reference before promotion.

Also noted: **fig:supp-subject-stability (S-16) caption is incorrect** — it says "apnea detection" only, but the figure actually shows multiple tasks (sex, BMI, sleep efficiency, apnea, and potentially others). Caption must be corrected when that figure is next edited.

---

## 13. Reference Style

TBME uses `[1]` inline bracketed references with IEEE bibliography format.
npj uses **superscript numbered** references: `word^1` with Nature bibliography format.

Nature reference format (journal article):
```
Author, A. B., Author, C. D. & Author, E. F. Title of article. Journal Name vol, pages (year).
```

Action: Switch `\bibliographystyle` to `naturemag` or `unsrtnat`; update `\usepackage{cite}` to `\usepackage[super]{natbib}`. The `.bib` file entries themselves do not need changing.

---

## 14. Tone and Writing Style Changes

| Aspect | TBME | npj |
|---|---|---|
| Audience | Signal processing / biomedical engineers | Clinical informaticists, physicians, ML researchers — broader |
| Section tone | Technical, report-style | Narrative, story-driven |
| Hypothesis framing | "H1 is confirmed / rejected" | Omit; state findings as observations |
| Clinical framing | Secondary | Primary — open and close with clinical relevance |
| Jargon | IEEE-standard (AUROC, K_max notation) | Define all terms on first use; avoid acronym stacking |
| Abstract style | Structured 3-point list | Flowing single paragraph |

Specific changes:
- Remove "Testing hypothesis H1…" opener from each Results subsection; replace with a direct statement of what the result shows
- Expand first sentence of Introduction to frame clinical impact before getting technical
- The Discussion already has good narrative flow; no major changes needed

---

## 15. Submission Checklist Summary

At submission to npj Digital Medicine, the following must be included:

- [ ] Main manuscript (npj_main.tex / PDF): Article, single-column
- [ ] Supplementary Information file (npj_supplementary.tex / PDF)
- [ ] Cover letter (separate): explain fit to journal, significance, no overlap with other submissions, suggested reviewers
- [ ] TRIPOD-AI completed checklist (PDF, fill and upload)
- [ ] Data availability statement (in manuscript)
- [ ] Code availability statement (in manuscript)
- [ ] Author contributions (in manuscript, CRediT format)
- [ ] Competing interests statement (in manuscript)
- [ ] Ethics statement (in manuscript or Methods)
- [ ] All figure files (PDF or TIFF, 300 DPI)
- [ ] .bib file or formatted references

---

## 16. Priority Order for Implementation

When creating the npj branch and new files, apply changes in this order:

1. **Set up template**: new single-column LaTeX file, switch document class, set reference style
2. **Reorder sections**: Abstract → Intro → Results → Discussion → Methods → end-matter
3. **Abstract**: rewrite as 150-word unstructured paragraph
4. **Introduction**: fold Related Work, expand clinical framing, remove heading
5. **Results**: rename subsection headings (remove H1–H4), spell out Figure everywhere
6. **Methods**: move to end, clean up subsection labels, add Statistics subsection
7. **End-matter**: add Data/Code Availability, Author Contributions, Competing Interests
8. **Tables/Figures**: reformat for single-column, update captions to be self-contained
9. **References**: switch to superscript natbib, update bibliography style
10. **Supplementary**: split into Extended Data + Supplementary Information
11. **TRIPOD-AI**: complete checklist, add calibration limitation, subgroup limitation
12. **Final pass**: remove all IEEE-specific commands, spell-check, word count check

---

## 17. Implementation Status (as of 2026-07-24)

### Completed

- [x] **Steps 1–2 (template + section reorder)**
  - `tbme_submission/` subfolder created; all TBME files archived there
  - `sn-jnl.cls` and `bst/` copied from Springer Nature article template
  - `npj_main.tex` created with `\documentclass[pdflatex,sn-nature]{sn-jnl}` + `\unnumbered`
  - Section order: Introduction → Results → Discussion (+ Conclusion merged) → Methods
  - `npj_supplementary.tex` created as standalone single-column `article` class (12pt, A4, 2.5 cm margins)
  - `\graphicspath{{tbme_submission/}}` set in both files (figures served from archive folder)
  - All scientific content preserved verbatim from TBME source

- [x] **Step 3 (abstract)**
  - Trimmed from ~218 words to 148 words (≤150 limit met)
  - All three patterns preserved with specific examples (BMI negligible, sleep efficiency/sex → 240 min, apnea/age-group → 120 min)
  - Cohort names, model names, head names all retained
  - Source-code footnote removed from abstract (it belongs in Code Availability end-matter)

- [x] **Step 4 (Introduction–Results transition + hypothesis framing)**
  - Decision: keep H1–H4 labels as organisational bookmarks; introduce them formally in the
    Introduction so Results can reference them without confusion (no orientation problem)
  - H1–H4 list moved from Methods (sec:sweep) to Introduction (end of study-design paragraph);
    the specific context lengths $L \in \{30$s, 10, 40, 80, 120, 240 min$\}$ are now also
    stated explicitly in the Introduction
  - Introduction P6 now ends with full H1–H4 itemised list + "Results address H1–H4 in order;
    full experimental formalisations are in Methods (Section~\ref{sec:sweep})"
  - Added 3-sentence orientation paragraph at start of Results (reduced-channel note,
    LSTM results location, H1–H4 order)
  - Methods sec:sweep: H1–H4 list replaced with a concise 7-line paraphrase that refers
    back to the Introduction (no repetition)
  - Results subsection headings changed to findings-first (no H-label in heading):
    - "Context-Length Saturation" → "Context-length saturation is task-specific"
    - "Aggregation Saturation" → "Inference aggregation saturates quickly"
    - "Temporal Head Advantage" → "Temporal heads gain context-dependent advantage"
    - Subsubsection "H4: budget-dependent crossover…" → "Budget-dependent crossover between context and aggregation (H4)"
  - Results openers changed from "Testing hypothesis Hx (…)" to findings-first with H-label
    in parentheses: "Context-length sensitivity is confirmed across all five tasks (H1): …"
  - Pending sub-items from original Step 4 now resolved:
    - "Here, we…" paragraph: not needed as a separate paragraph — the H list in Introduction
      now serves this bridging function
    - "Study design" subsection in Results: replaced by 3-sentence orientation para above

- [x] **Step 5 (figure references + Results headings)**
  - Replaced all `Fig.~` → `Figure~` throughout `npj_main.tex` (41 occurrences)
  - H1–H4 subsection headings renamed (see Step 4 above)

- [x] **Step 9 (reference format)**
  - Confirmed: inline `\thebibliography` + `\bibitem` is accepted by Springer Nature / npj (preferred for submission)
  - `sn-jnl.cls [sn-nature]` automatically loads `natbib[numbers,sort&compress]`; added `\setcitestyle{super}` to preamble for Nature-style superscript citations
  - Reformatted all 33 `\bibitem` entries from IEEE to Nature style:
    - Authors: Last, F. M. & Last, F. M. (& before last; et al. for >6)
    - Title: sentence case, no quotes
    - Journal in `\emph{}`, volume in `\textbf{}`, pages–year in (year)
    - arXiv/medRxiv entries use `Preprint at \url{...} (year)` form
  - Fixed incorrect `mros` reference (was Ancoli-Israel 1999 insomnia survey; replaced with
    Blank et al. 2005 MrOS recruitment paper — verify against NSRR dataset page before submission)
  - No undefined citations; compiles cleanly to 33-page PDF

- [x] **Task 6 (Introduction completeness + Results→Methods parenthetical refs)**
  - Verified Introduction contains: task list, cohort names, $K_{\max}$ definition, SleepFM named, H1–H4 defined
  - Added $L^*$ formal definition ref in H1 opener: "(formally: smallest $L$ within 0.005 AUROC of peak; Methods~\eqref{eq:lstar})"
  - Added CI method ref in H1 CI sentence: "(subject-level resampling with 1,000 samples; see Methods Section~\ref{sec:evaluation})"
  - Added architecture ref in H3 opener: "(described in Methods Section~\ref{sec:heads})"
  - Added channel selection ref in Channel Count section: "(channel selection described in Methods Section~\ref{sec:channel_comparison})"
  - Added modality ablation ref in Modality Group Ablation section: "(modality groups and conditions defined in Methods Section~\ref{sec:modality_ablation})"
  - All five referenced labels (`sec:heads`, `sec:evaluation`, `eq:lstar`, `sec:channel_comparison`, `sec:modality_ablation`) confirmed present in Methods

- [x] **Acronym pass** — All acronyms expanded at first occurrence in reading order
  (Introduction → Results → Discussion → Methods). Full list maintained in CLAUDE.md
  under "Acronym convention".
  **Ongoing rule:** re-check the full list whenever sections are reordered or text
  containing an acronym's first use is moved or added. npj puts Methods last, so any
  acronym first used in Results or Discussion must be defined there, not deferred to
  Methods.

- [x] **Margin correction (twoside/binding-offset shift)**
  - `sn-jnl.cls` hardcodes `\LoadClass[twoside,fleqn]{article}` and geometry
    `bindingoffset=6mm`, shifting the text column ~15 mm left on odd pages and right
    on even pages; the class `oneside` option and a preamble `\geometry{bindingoffset=0mm}`
    call both have no effect because the class overrides them.
  - Fix: `\makeatletter \AtBeginDocument{ compute average of \oddsidemargin and
    \evensidemargin; set both equal } \makeatother` placed in the preamble. The hook
    fires after geometry's own `\AtBeginDocument` hook, so it correctly overwrites the
    asymmetric values. Uses `\divide\@tempdima by 2` (not `0.5\length`) as required by TeX.

- [x] **Table formatting — wide table centering and caption width (tab:tasks, tab:isocompute, tab:modality)**
  - All three `table*` environments had tabular content wider than `\textwidth` (461 pt,
    ~490 pt, ~430 pt respectively), causing overflow and the caption (at `\textwidth`)
    to appear narrower than the table.
  - Fix: wrapped each tabular in `\resizebox{\linewidth}{!}{...}`. Table scales to exactly
    `\linewidth`, matching the caption width and centering correctly once margins are equalized.

- [x] **Table formatting — spacing and width for tab:sweep and tab:saturation**
  - Both tables had `\tabcolsep=1pt` (very cramped) and footnotes using
    `p{0.95\linewidth}` / `p{0.95\columnwidth}`, which forced the table wider than the
    natural body content and pooled the excess stretch into the last column.
  - **tab:sweep**: `\tabcolsep` raised 1→3 pt; L-columns switched to
    `*{6}{>{\centering\arraybackslash}p{32pt}}` (all six equal-width at 32 pt);
    8 pt group separator (`@{\hskip8pt}`) added between label columns and L-columns;
    footnote changed to `p{0.85\linewidth}` so the table is content-width, not footnote-forced.
  - **tab:saturation**: `\tabcolsep` raised 1→3 pt; column spec changed to
    `{l@{\hskip8pt}ccc@{\hskip8pt}rc@{\hskip8pt}cccc}` adding 8 pt group separators
    between the four column blocks while keeping the Δ column at regular tabcolsep;
    footnote changed to `p{0.80\linewidth}` to match the natural body width and eliminate
    spurious stretch in the last column.
  - Manual tuning: `\tabcolsep` controls all-column spacing; `@{\hskip Xpt}` controls
    per-group-boundary spacing; `p{Xpt}` in the L-column spec controls equal column widths.

- [x] **Figure 1 (preprocessing pipeline) moved to Results**
  - `fig:preprocessing` float moved from its Methods location (between the normalisation
    paragraph and "Data Splitting") to the beginning of the Results section, placed right
    after the orientation paragraph.
  - Reference added to the orientation paragraph: "see Methods for channel details and
    `Figure~\ref{fig:preprocessing}` for the preprocessing and embedding pipeline".
  - The two existing text references in Methods (`Figure~\ref{fig:preprocessing}(a)` and
    `(b)`) are kept unchanged as back-references.

- [x] **Study overview subsection added at beginning of Results**
  - New `\subsection{Study overview}` (`\label{sec:overview}`) inserted between the
    orientation paragraph + figure and the first results subsection (H1).
  - Two short paragraphs: (1) four cohorts, ~16 k recordings, seven tasks with references
    to `sec:datasets` and `sec:tasks`/`tab:tasks`; (2) preprocessing pipeline →
    frozen SleepFM encoder (referencing `fig:preprocessing`(a,b) and
    `Supplementary~Figure~S-X`) → three heads × six context lengths → 70/15/15 split,
    with references to `sec:preprocessing` and `sec:sleepfm_encoder`.
  - Placeholder comment added in `npj_supplementary.tex` after "The Frozen SleepFM
    Encoder" subsection to mark where the full SleepFM architecture figure will be
    inserted; hardcoded label `S-X` in main text must be updated to the real number
    once the figure is added.

### Pending (next steps)

- [ ] **Step 6**: Add Statistics subsection to Methods; clarify pooled-cohort vs external validation
- [ ] **Step 10**: Decide which supplementary figures to promote to Extended Data
- [ ] **Step 11**: Add calibration limitation and subgroup/fairness limitation to Discussion
- [ ] **Step 12**: Remove remaining IEEE-specific commands (e.g., `\IEEEPARstart` stub if any remain)
- [ ] Verify `mros` reference against NSRR dataset page (sleepdata.org) before submission
- [ ] Fill in author names, affiliations, and emails (placeholders in both files)
- [ ] Write cover letter
