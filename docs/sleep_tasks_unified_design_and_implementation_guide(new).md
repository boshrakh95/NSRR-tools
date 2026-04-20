# Sleep Task Design, Label Audit, and Implementation Plan
## Complete consolidated note for coding-agent updates

This document merges the full reasoning and conclusions from the dataset/task audit across SHHS, APPLES, MrOS, and the earlier discussion about how recent sleep foundation model papers define tasks. It is intentionally detailed so a coding agent can update configs, dataset parsers, loaders, and experiment scripts without guessing.

---

# 1. Why this audit was necessary

The original issue was:

- some tasks such as sleep staging and AHI / apnea severity were learning reasonably
- many others such as anxiety, depression, rested in morning, etc. were near-random on validation/test while training accuracy was high

This pattern strongly suggests the main problem is not only model capacity, but also:

1. weak or noisy labels
2. inconsistent task definitions across datasets
3. mixing labels that are not actually equivalent
4. using difficult subjective tasks as if they were as “sleep-native” as AHI/staging
5. evaluating some tasks with suboptimal formulations (too many classes, arbitrary thresholds, etc.)

A major insight from the paper review is that recent sleep foundation model papers often succeed because they use:

- strong physiological labels (AHI, sleep staging, event detection)
- diagnosis / disease-history / EHR-style outcomes
- subject-level aggregation for subject-level tasks
- AUROC / AUPRC / C-index instead of raw accuracy alone
- large datasets

They are not usually doing the same thing as predicting noisy questionnaire thresholds from a small number of cohorts.

---

# 2. Important lessons from the papers

## 2.1 Not all “mental state” tasks in papers are like your PHQ/GAD tasks

The papers that report depression, anxiety, bipolar disorder, PTSD, etc. are often using:

- disease diagnoses
- disease histories
- longitudinal or survival-style outcomes
- clinically defined labels
- large pooled clinical cohorts

They are not the same thing as:

- PHQ-9 >= 10
- GAD-7 >= 10
- single-item morning restedness
- questionnaire thresholds from one cohort

This means your weaker results on anxiety/depression/restedness do not mean your context-length idea is wrong. It more likely means those targets are much noisier and weaker than AHI/staging.

## 2.2 Time-to-event only makes sense for true event-time labels

From the papers:

- SleepFM uses Cox PH style survival analysis for future disease prediction because they have diagnosis dates / event times / censoring.
- SleepFounder also reports future disease prediction with AUROC + C-index.
- SleepMaMi follows similar survival-style evaluation for disease-history outcomes.

That only makes sense when you have:
- event date
- censor date
- follow-up structure
- variable follow-up length

This does not apply to:
- AHI
- ESS
- PHQ
- GAD
- restedness
- ISI
- PSQI

For your currently available columns, the only clearly justified survival/time-to-event task is:
- SHHS CVD summary dataset

Everything else should stay as classification or regression.

## 2.3 Class imbalance in the papers

The papers handle imbalance mainly by:

- using better metrics: AUROC, AUPRC, Macro-F1, C-index
- sometimes using weighted loss explicitly (SleepMaMi clearly says weighted cross-entropy for imbalanced sleep staging / SDB segmentation)

So for implementation:
- do not rely on raw accuracy for skewed tasks
- report AUROC and AUPRC for binary imbalanced tasks
- report Macro-F1 and balanced accuracy for multiclass tasks
- weighted CE / weighted BCE is justified

---

# 3. Core rules for task design in this project

These are the rules the coding agent should follow:

## Rule 1: Do not merge datasets unless labels are truly comparable
Comparable means:
- same scale
- same clinical meaning
- same threshold meaning
- close enough measurement protocol

More data is not always better if label meaning drifts.

## Rule 2: Prefer strong sleep-native tasks first
Strongest tasks:
- AHI / OSA
- sleep staging
- ESS
- PSG-derived sleep efficiency
- validated scales like ISI / PSQI

Weakest tasks:
- single-item restedness
- generic fatigue items
- questionnaire-derived mood classes without careful filtering

## Rule 3: For subjective tasks, binary is preferred over multiclass
Especially for:
- ESS
- ISI
- PSQI
- restedness
- depression/anxiety derived from self-report

This reduces:
- label noise
- ambiguity
- class sparsity

## Rule 4: For depression/anxiety-like tasks, use extreme groups if possible
Example:
- depression low vs high, discard middle
- anxiety low vs high, discard middle

This is much cleaner than forcing 4 bins or a hard mild/moderate/severe multiclass problem with small data.

## Rule 5: Survival analysis only where the data actually supports it
Currently:
- SHHS CVD summary = yes
- everything else = no

---

# 4. Dataset-by-dataset analysis

## 4.1 SHHS

### Files reviewed
- shhs-cvd-summary-dataset-0.21.0.csv
- shhs2-dataset-0.21.0.csv
- shhs-interim-followup-dataset-0.21.0.csv
- shhs-harmonized-dataset-0.21.0.csv
- shhs1-dataset-0.21.0.csv
- SHHS variable dictionary

### Important confirmed variables

#### AHI
- nsrr_ahi_hp3r_aasm15
- meaning: harmonized AHI based on AASM-like modern criteria

This is the right AHI variable for SHHS.

#### ESS
- ess_s1
- ess_s2

These are valid ESS totals and good targets.

#### Restedness / morning sleep quality
- rest10
- ms204c

The dictionary indicates these are both 5-point morning survey items ranging from restless to restful or similar.

They are conceptually aligned enough to be used, but they remain weak labels.

#### CVD summary
From shhs-cvd-summary-dataset-0.21.0.csv, useful columns include:
- event indicators: mi, stroke, angina, chf, chd_death, cvd_death
- prevalent/history indicators: prev_mi, prev_stk, etc.
- dates: mi_date, stk_date, ang_date, chf_date, chd_dthdt, cvd_dthdt
- censor date: censdate
- derived summary: any_cvd

This file supports either:
- a crude binary classification task (any_cvd)
- or, better, a proper time-to-event setup

### SHHS conclusions

#### Strong tasks
- AHI / OSA
- ESS

#### Usable but weaker
- restedness (rest10, ms204c)

#### Survival-capable
- CVD outcomes from summary file

#### Recommendation
SHHS should contribute mainly to:
- merged AHI / OSA task
- merged ESS task
- optional SHHS-only survival branch
- optional exploratory restedness

---

## 4.2 APPLES

### Files reviewed
- baseline characteristics
- ESS/BP/BMI longitudinal data
- PVT longitudinal data
- WASI baseline data
- POMS / SAQLI longitudinal data
- apples-dataset-0.1.0.csv
- apples-harmonized-dataset-0.1.0.csv

### Important confirmed variables

#### AHI
- nsrr_ahi_chicago1999
- available in harmonized file
- used at visit 3 (diagnostic PSG visit), which is correct

This is not exactly the same definition as SHHS/MrOS AASM AHI, but it is still acceptable for within-task classification.

#### ESS
- esstotalscoreqc
- clean total ESS with QC

#### Depression
- bditotalscore
- valid BDI source
- but multiclass thresholding is likely too noisy

#### PSG-derived sleep quality
- sleepeffpsg
- wasoqcpsg

These are very useful and stronger than weak subjective items.

#### OSA severity
- osaseveritypostqc

This is a particularly strong clinically curated target and should be added.

#### Cognitive / vigilance performance
From PVT:
- MeanRT
- MedianRT
- LapsesRTge500
- others like errors, false starts, slope

These are strong objective outcome variables and likely better than weak mood/self-report tasks.

#### Other subjective items
Examples:
- RefreshInAMHP
- UnrestedDuringDaySHQ7e
- SleepyDuringDaySHQ7f
- FatiguedDuringDaySHQ7h

These can be used, but they are weaker than ESS / PSG metrics / PVT.

### APPLES conclusions

#### Strong tasks
- AHI / OSA
- ESS
- PSG sleep efficiency / WASO
- OSASeverityPostQC
- PVT-derived cognitive tasks

#### Medium
- BDI depression
- subjective refresh/restedness items

#### Recommendation
APPLES is your best dataset for:
- sleep physiology + behavior + cognition
- objective non-PSG downstream targets (PVT)

It should be used heavily for:
- OSA/AHI
- ESS
- sleep efficiency
- cognitive tasks
- optionally depression as an exploratory extreme-groups task

---

## 4.3 MrOS

### Files reviewed
- mros-visit1-dataset-0.6.0.csv
- mros-visit2-dataset-0.6.0.csv
- mros-visit1-harmonized-0.6.0.csv
- MrOS variable dictionary CSV

### Important confirmed variables

#### AHI
- nsrr_ahi_hp3r_aasm15
- only in visit 1 harmonized file

This is a strong AHI source, but visit 1 only.

#### ESS
- epepwort
- confirmed from dictionary as total ESS score

Very good.

#### ISI
- slisiscr
- confirmed as Insomnia Severity Index summary score
- slisicat also exists as a categorized version

This is a very strong validated target and should be enabled.

#### PSQI
- pqpsqi
- confirmed as Pittsburgh Sleep Quality Index summary score

This is a major positive finding. MrOS is your strongest dataset for validated sleep-quality scales.

#### Restedness / sleep quality item
- poxqual3
- likely a subjective quality/restedness item
- usable but weaker than ISI/PSQI

#### CVD history
- cvchd
- likely coronary heart disease history / prevalent disease, not survival-style future event prediction

This should not be treated like SHHS survival outcomes.

### MrOS conclusions

#### Strong tasks
- AHI / OSA
- ESS
- ISI
- PSQI

#### Weaker
- poxqual3
- cvchd

#### Recommendation
MrOS is your best dataset for validated subjective sleep scales:
- ESS
- ISI
- PSQI

This is much stronger than using generic restedness or anxiety/depression history flags.

---

# 5. Final unified task framework

## Tier 1 tasks (start here first)

These are the smallest set of strong, clean tasks that should be implemented first.

## Task A: OSA / AHI

### Why this task
- strongest physiological target
- directly tied to PSG
- supported across multiple datasets
- likely to show meaningful context-length effects

### Final formulations

#### A1. Binary OSA
- label 0: AHI < 15
- label 1: AHI >= 15

#### A2. 4-class OSA severity
- class 0: AHI < 5
- class 1: 5 <= AHI < 15
- class 2: 15 <= AHI < 30
- class 3: AHI >= 30

### Datasets and columns

#### SHHS
- file: shhs-harmonized-dataset-0.21.0.csv
- visit column: visitnumber
- label column: nsrr_ahi_hp3r_aasm15
- visits: both visits allowed as separate visit-level samples

#### MrOS
- file: mros-visit1-harmonized-0.6.0.csv
- label column: nsrr_ahi_hp3r_aasm15
- visits: visit 1 only
- important change: do not keep both visits for this task

#### APPLES
- file: apples-harmonized-dataset-0.1.0.csv
- visit column: visitn
- visit filter: 3
- label column: nsrr_ahi_chicago1999

### Merge decision
- MERGE = YES

### Why merging is allowed
Even though APPLES uses Chicago definition and SHHS/MrOS use harmonized AASM-like variables, this is still close enough for classification when used as a broad OSA severity signal.

### Metrics
- binary: AUROC, AUPRC, balanced accuracy
- multiclass: Macro-F1, balanced accuracy, kappa

### Notes to coding agent
Create both:
- osa_binary_15
- apnea_class_4

---

## Task B: Sleepiness (ESS)

### Why this task
- validated instrument
- same conceptual scale across datasets
- directly related to sleep burden
- much cleaner than restedness

### Final formulation

#### Main
- binary:
  - label 0: ESS <= 10
  - label 1: ESS >= 11

#### Optional secondary
- 3-class:
  - 0–10
  - 11–15
  - 16–24

### Datasets and columns

#### SHHS
- visit 1: ess_s1
- visit 2: ess_s2

#### MrOS
- epepwort

#### APPLES
- esstotalscoreqc

### Merge decision
- MERGE = YES

### Why merging is allowed
ESS is the same construct across cohorts and the same scoring scale.

### Metrics
- binary: AUROC, AUPRC
- optional 3-class: Macro-F1

### Notes to coding agent
Replace current multiclass ESS as the main task with binary ESS.
Keep 3-class ESS only as optional exploratory.

---

# 6. Tier 2 tasks (add next)

## Task C: Insomnia (ISI)

### Why this task
- validated clinical scale
- strong subjective sleep symptom target
- much better than vague insomnia flags or restedness

### Final formulation
- binary:
  - label 0: ISI < 15
  - label 1: ISI >= 15

### Datasets and columns

#### MrOS
- file: mros-visit2-dataset-0.6.0.csv
- column: slisiscr
- visit: visit 2 only

#### STAGES
- file: STAGES dataset file currently used
- column: isi_score

### Merge decision
- MERGE = NO

### Why not merge
Same conceptual instrument, but very different cohorts and likely different distributions. Keep dataset-specific first.

### Metrics
- AUROC, AUPRC

### Notes to coding agent
Current config incorrectly disables MrOS ISI because it assumed slisiscr was absent. It is present and confirmed by dictionary.
Enable this task.

---

## Task D: Sleep Quality (PSQI)

### Why this task
- one of the strongest validated subjective sleep quality measures
- stronger and more interpretable than single-item restedness

### Final formulation
- binary:
  - label 0: PSQI <= 5
  - label 1: PSQI > 5

### Dataset and column

#### MrOS only
- file: mros visits where pqpsqi exists
- column: pqpsqi

### Merge decision
- MERGE = NO
- only MrOS currently

### Metrics
- AUROC, AUPRC

### Notes to coding agent
Add a new task:
- psqi_binary

---

## Task E: PSG-derived sleep quality / sleep efficiency

### Why this task
- objective
- directly tied to sleep physiology
- more likely to learn than mood or subjective morning items

### Preferred formulation
- binary sleep efficiency:
  - label 0: sleep efficiency >= 85
  - label 1: sleep efficiency < 85

### Candidate datasets and columns

#### SHHS
- nsrr_ttleffsp_f1

#### APPLES
- sleepeffpsg
- wasoqcpsg can also be used as separate regression or binary task

#### MrOS
- poslpeff
- powaso

### Merge decision
- MERGE = OPTIONAL / CAUTIOUS
- best to start per-dataset, then test merged

### Why cautious
The exact derived definitions may differ slightly across datasets.

### Metrics
- binary: AUROC, AUPRC
- regression alternative: MAE, Pearson/Spearman

### Notes to coding agent
Add:
- sleep_efficiency_binary
Optionally later:
- waso_binary
- sleep_efficiency_regression

---

# 7. Tier 3 tasks (good but dataset-specific)

## Task F: Depression (extreme groups only)

### Why this is not Tier 1
- weaker signal from PSG
- different scales across datasets
- current multiclass design is too noisy

### Final formulation
Use extreme groups only, not full multiclass.

#### APPLES (BDI)
- low: bditotalscore <= 9
- high: bditotalscore >= 20
- drop middle range

#### STAGES (PHQ-9)
- low: phq_1000 <= 4
- high: phq_1000 >= 15
- drop middle range

### Merge decision
- MERGE = NO

### Why not merge
Different scales:
- BDI in APPLES
- PHQ-9 in STAGES

Do not mix them into one label space.

### Metrics
- AUROC, AUPRC

### Notes to coding agent
Current APPLES depression multiclass should not remain a main task.
Replace/add:
- depression_extreme

---

## Task G: Restedness / morning quality

### Why weak
- single-item self-report
- very noisy
- likely weakly linked to PSG
- not equivalent across datasets

### Final formulation
- binary:
  - positive/rested: >= 4
  - negative/not rested: <= 3

### Datasets and columns

#### SHHS
- visit 1: rest10
- visit 2: ms204c

#### MrOS
- poxqual3

#### APPLES
- RefreshInAMHP

### Merge decision
- MERGE = NO

### Why not merge
These items are not guaranteed to be the exact same question even if all are 1–5 style scales.

### Metrics
- AUROC, AUPRC

### Notes to coding agent
Keep as exploratory only:
- rested_binary
Do not use as flagship task for judging model quality.

---

# 8. Tier 4 tasks (specialized / bonus)

## Task H: SHHS cardiovascular disease survival

### Why important
This is the one place where time-to-event analysis is actually well justified and aligned with the papers.

### Use only if survival branch is added
This should be a separate experiment branch, not mixed into the initial context-length classification sweep.

### Candidate outcomes
From shhs-cvd-summary-dataset-0.21.0.csv:
- mi
- stroke
- angina
- chf
- chd_death
- cvd_death

### Required columns
- event indicator: one of the event columns above
- event time/date: corresponding date column
- censoring: censdate

### Merge decision
- MERGE = NO
- SHHS only

### Outputs / metrics
- C-index
- fixed-horizon AUROC if implemented

### Notes to coding agent
Add later:
- cvd_survival_*
This is optional and should not block initial experiments.

---

## Task I: APPLES cognitive / vigilance tasks

### Why strong
- objective
- sleep-relevant
- less noisy than mood items
- good for subject-level analysis

### Candidate variables
From PVT:
- MeanRT
- MedianRT
- LapsesRTge500

### Formulations
- regression: MeanRT
- binary classification:
  - high lapses vs low lapses
  - slow RT vs fast RT using quantiles

### Merge decision
- MERGE = NO
- APPLES only

### Notes to coding agent
Add later:
- pvt_meanrt_regression
- pvt_lapses_binary

Also consider APPLES-only:
- osa_severity_postqc from osaseveritypostqc

---

# 9. Final merging decisions across tasks

This section should drive the multi-dataset loader logic.

| Task | Merge datasets? | Reason |
|---|---|---|
| OSA / AHI | YES | same physiological construct, close enough definitions |
| ESS | YES | same validated scale |
| ISI | NO | dataset-specific, only MrOS + STAGES |
| PSQI | NO | currently only MrOS |
| Sleep efficiency | CAUTIOUS / optional | same broad construct, but derived definitions may differ |
| Depression | NO | BDI vs PHQ are different scales |
| Restedness | NO | different questions/items |
| CVD classification | NO | different meanings (any_cvd vs cvchd) |
| CVD survival | NO | SHHS only |
| PVT cognition | NO | APPLES only |

## Most important merging rule
Only AHI/OSA and ESS should definitely be merged from the start.

That means the coding pipeline should support:
- merged training for some tasks
- dataset-specific training for other tasks

---

# 10. Required config changes

## 10.1 Keep and refine existing tasks

### SHHS apnea_class
Keep:
- file: shhs-harmonized-dataset-0.21.0.csv
- column: nsrr_ahi_hp3r_aasm15

### APPLES apnea_class
Keep:
- file: apples-harmonized-dataset-0.1.0.csv
- column: nsrr_ahi_chicago1999
- visit filter: 3

### MrOS apnea_class
Keep, but modify:
- file: mros-visit1-harmonized-0.6.0.csv
- column: nsrr_ahi_hp3r_aasm15
- remove / disable keep_both_visits: true
- this task is visit 1 only

---

## 10.2 Add new OSA binary task
Add a task for every dataset with AHI:

### Name
- osa_binary

### Threshold
- AHI >= 15

### Dataset-specific columns
- SHHS: nsrr_ahi_hp3r_aasm15
- MrOS: nsrr_ahi_hp3r_aasm15
- APPLES: nsrr_ahi_chicago1999

---

## 10.3 Replace main sleepiness task with binary ESS
Current main sleepiness task should become binary.

### Name
- sleepiness_binary

### Threshold
- ESS >= 11

### Columns
- SHHS visit 1: ess_s1
- SHHS visit 2: ess_s2
- MrOS: epepwort
- APPLES: esstotalscoreqc

Keep 3-class ESS only as optional exploratory.

---

## 10.4 Fix MrOS insomnia task
Current config says:
- disabled because slisiscr absent

This is wrong.

### Correct task
- enable insomnia_binary
- file: mros-visit2-dataset-0.6.0.csv
- column: slisiscr
- threshold: >= 15

Also support STAGES:
- column: isi_score

Do not merge initially.

---

## 10.5 Add MrOS PSQI task
Add:
- psqi_binary
- file(s): MrOS visits where pqpsqi exists
- threshold: > 5

---

## 10.6 Add PSG sleep efficiency task
Suggested task:
- sleep_efficiency_binary

Threshold:
- efficiency < 85

Columns:
- SHHS: nsrr_ttleffsp_f1
- APPLES: sleepeffpsg
- MrOS: poslpeff

Start dataset-specific, optionally test merged later.

---

## 10.7 Replace APPLES depression multiclass with extreme-group version
Current APPLES:
- depression_class
- bditotalscore
- 4 classes

Suggested change:
- add/replace with depression_extreme
- low: bditotalscore <= 9
- high: bditotalscore >= 20
- discard middle

Similarly for STAGES:
- low: phq_1000 <= 4
- high: phq_1000 >= 15
- discard middle

Do not merge.

---

## 10.8 Keep restedness as exploratory only
Keep but lower priority:
- SHHS: rest10, ms204c
- MrOS: poxqual3
- APPLES: RefreshInAMHP

Use:
- >=4 as positive
- <=3 as negative

Do not merge.

---

## 10.9 Add APPLES-specific high-value tasks later
Add later:
- osa_severity_postqc from osaseveritypostqc
- pvt_meanrt_regression from MeanRT
- pvt_lapses_binary from LapsesRTge500

These are likely stronger than many weak questionnaire tasks.

---

# 11. Recommended task priority order

This is the order the coding agent should implement and enable.

## Phase 1 (must do first)
These are the smallest strong set for the paper’s first pass.

1. osa_binary
2. apnea_class_4
3. sleepiness_binary

These should be enough for the first context-length experiments.

## Phase 2
4. sleep_efficiency_binary
5. insomnia_binary

## Phase 3
6. psqi_binary
7. APPLES cognitive tasks (pvt_meanrt_regression, pvt_lapses_binary)
8. osa_severity_postqc

## Phase 4 (exploratory)
9. depression_extreme
10. rested_binary

## Phase 5 (optional advanced branch)
11. SHHS cvd_survival_*

---

# 12. Metrics and loss recommendations

## Binary tasks
Use:
- AUROC
- AUPRC
- balanced accuracy

Loss:
- weighted BCE or weighted CE

Recommended for:
- osa_binary
- sleepiness_binary
- insomnia_binary
- psqi_binary
- rested_binary
- depression_extreme

## Multiclass tasks
Use:
- Macro-F1
- balanced accuracy
- Cohen’s kappa (optional for AHI/staging)

Loss:
- weighted cross-entropy

Recommended for:
- apnea_class_4
- optional 3-class ESS

## Regression tasks
Use:
- MAE
- Pearson/Spearman

Recommended for:
- pvt_meanrt_regression
- optional sleep efficiency regression

## Survival tasks
Use:
- C-index
- optional fixed-horizon AUROC

Recommended for:
- SHHS CVD survival only

---

# 13. Final conclusions to preserve in code/design choices

1. The initial poor results on many subjective tasks are likely due more to label quality and task formulation than to encoder failure.
2. AHI and ESS are the safest cross-dataset merged tasks.
3. MrOS provides strong validated subjective sleep scales through ISI and PSQI and should be used for those.
4. APPLES provides valuable objective non-PSG targets via PVT and should not be reduced to only AHI and BDI.
5. Restedness should remain exploratory and should not drive architecture decisions.
6. Time-to-event analysis is useful only where true event/censoring information exists, currently SHHS CVD summary.
7. The codebase should support both:
   - merged multi-dataset tasks
   - dataset-specific tasks

---

# 14. Concrete implementation checklist for coding agent

## A. Update task config names / definitions
Add or update tasks:
- osa_binary
- apnea_class_4
- sleepiness_binary
- insomnia_binary
- psqi_binary
- sleep_efficiency_binary
- depression_extreme
- rested_binary
- later: pvt_meanrt_regression, pvt_lapses_binary, cvd_survival_*

## B. Fix dataset-specific visit handling
- SHHS AHI: both visits OK
- SHHS ESS: both visits OK
- MrOS AHI: visit 1 only
- MrOS ISI: visit 2 only
- APPLES AHI: visit 3 only
- APPLES BDI: baseline / visit 1 only
- APPLES ESS: PSG-matched visit already used

## C. Enable merged training only for
- OSA / AHI
- ESS

## D. Keep dataset-specific training for
- ISI
- PSQI
- depression
- restedness
- cognition
- CVD

## E. Add metric logic by task type
- binary -> AUROC/AUPRC
- multiclass -> Macro-F1
- regression -> MAE
- survival -> C-index

## F. Use weighted losses for imbalanced classification

---

# 15. Short version for immediate implementation

If only a minimal change set is possible right now, do this first:

1. Keep AHI task, but add osa_binary
2. Convert ESS main task to binary in SHHS / MrOS / APPLES
3. Enable MrOS slisiscr as insomnia binary
4. Add MrOS pqpsqi as PSQI binary
5. Stop treating restedness as a main task
6. Replace APPLES depression multiclass with extreme groups
7. Do not merge anything except AHI and ESS

---

End of document.

