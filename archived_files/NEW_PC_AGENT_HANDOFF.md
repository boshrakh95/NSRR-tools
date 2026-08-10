# Broad Project Handoff for New PC

## 1) What this project is (broader scope)
This is not only a target-extraction task. The broader objective is to execute the data and modeling roadmap described in your LaTeX writeups, while keeping outputs compatible across NSRR-tools, nocturn, and sleepfm-clinical.

Core direction:
- Build a unified PSG data pipeline across APPLES, SHHS, MrOS, STAGES
- Support objective 3 proposal experiments (context-length dependence + robustness to distribution shift)
- Produce extraction outputs that can feed downstream SleepFM/nocturn experiments

Primary planning sources:
- `/home/boshra95/NSRR-tools/data_preparation.tex`
- `/home/boshra95/NSRR-tools/proposal for objective3.tex`

The next agent must read those two files first before coding.

---

## 2) Mandatory read order for a new agent
Read these files in order:

1. High-level objective and phases:
   - `/home/boshra95/NSRR-tools/data_preparation.tex`
   - `/home/boshra95/NSRR-tools/proposal for objective3.tex`

2. Existing NSRR-tools implementation status:
   - `/home/boshra95/NSRR-tools/README.md`
   - `/home/boshra95/NSRR-tools/TARGET_EXTRACTION_IMPLEMENTATION_PLAN.md`
   - `/home/boshra95/NSRR-tools/IMPLEMENTATION_PLAN.md`
   - `/home/boshra95/NSRR-tools/PHASE2_IMPLEMENTATION_PLAN.md`

3. Dataset-specific notes and bug/adapter context:
   - `/home/boshra95/NSRR-tools/STAGES_DATA_NOTES.md`
   - `/home/boshra95/NSRR-tools/STAGES_ID_MATCHING_RESOLUTION.md`
   - `/home/boshra95/NSRR-tools/STAGES_ID_MISMATCH_DEBUG_GUIDE.md`
   - `/home/boshra95/NSRR-tools/CLASSIFICATION_TARGETS_ANALYSIS.md`
   - `/home/boshra95/NSRR-tools/DATASET_ADAPTERS_SUMMARY.md`

4. Ontology and label mappings (nocturn):
   - `/home/boshra95/nocturn/configs/ontology-core.yaml`
   - `/home/boshra95/nocturn/configs/ontology-datasets.yaml`
   - `/home/boshra95/nocturn/configs/datasets/apples.yaml`
   - `/home/boshra95/nocturn/configs/datasets/shhs.yaml`
   - `/home/boshra95/nocturn/configs/datasets/mros.yaml`
   - `/home/boshra95/nocturn/configs/datasets/stages.yaml`

5. SleepFM side processing/training references:
   - `/home/boshra95/sleepfm-clinical/README.md`
   - `/home/boshra95/sleepfm-clinical/DATA_PREPARATION_ANALYSIS.md`
   - `/home/boshra95/sleepfm-clinical/sleepfm/stages_preprocessing/README.md`
   - `/home/boshra95/sleepfm-clinical/sleepfm/stages_preprocessing/CHECKLIST.md`
   - `/home/boshra95/sleepfm-clinical/sleepfm/stages_preprocessing/IMPLEMENTATION_SUMMARY.md`
   - `/home/boshra95/sleepfm-clinical/sleepfm/stages_cognitive_prediction/README.md`

---

## 3) Current implementation status snapshot
Completed extraction outputs:
- APPLES: completed
- SHHS: completed
- MrOS: completed with dataset-specific caveats

Pending extraction outputs:
- STAGES (next)

Pending integration:
- Master targets file across all datasets
- Task-subject lookup json
- Validation report with class balance and missingness

Relevant local files already present:
- `/scratch/boshra95/psg/unified/targets/apples_targets.csv`
- `/scratch/boshra95/psg/unified/targets/shhs_targets.csv`
- `/scratch/boshra95/psg/unified/targets/mros_targets.csv`
- `/scratch/boshra95/psg/unified/metadata/unified_metadata.parquet`

---

## 4) Data layout (raw vs processed)
Raw datasets (source of truth):
- `/scratch/boshra95/nsrr_downloads/apples`
- `/scratch/boshra95/nsrr_downloads/shhs`
- `/scratch/boshra95/nsrr_downloads/mros`
- `/scratch/boshra95/nsrr_downloads/stages`

Processed PSG roots (project-side):
- `/scratch/boshra95/psg/apples`
- `/scratch/boshra95/psg/shhs`
- `/scratch/boshra95/psg/mros`
- `/scratch/boshra95/psg/stages`
- `/scratch/boshra95/psg/unified/metadata`
- `/scratch/boshra95/psg/unified/targets`

Additional STAGES-related storage seen:
- `/scratch/boshra95/stages/sleepfm_format`
- `/scratch/boshra95/stages/stages`

Rule:
- Never overwrite raw downloads.
- Write derived artifacts under `/scratch/boshra95/psg/...`.
- Keep unified metadata/targets as the canonical training index layer.

---

## 5) When to use each repo

### Use NSRR-tools for
- Dataset adapters and extraction scripts
- Unified metadata/target generation
- Cross-dataset harmonized tables
- Fast debugging and verification scripts

Key locations:
- `/home/boshra95/NSRR-tools/scripts`
- `/home/boshra95/NSRR-tools/configs`

### Use nocturn for
- Ontology-grounded column/file mapping
- Dataset-specific semantic definitions of tasks/labels
- Resolving ambiguity in task definitions before extraction changes

Key locations:
- `/home/boshra95/nocturn/configs/ontology-datasets.yaml`
- `/home/boshra95/nocturn/configs/datasets/*.yaml`

### Use sleepfm-clinical for
- SleepFM preprocessing/format compatibility
- STAGES conversion strategy and quality checks
- Downstream model training assumptions and expected data shape

Key locations:
- `/home/boshra95/sleepfm-clinical/sleepfm/stages_preprocessing`
- `/home/boshra95/sleepfm-clinical/sleepfm/stages_cognitive_prediction`

Decision rule for next agent:
- If question is "what does this label/column mean?" -> check nocturn first.
- If question is "how should data be formatted for SleepFM training?" -> check sleepfm-clinical.
- If question is "how to implement extraction/join/splits here?" -> do it in NSRR-tools.

---

## 6) Research plan alignment from tex files
The new agent must preserve this sequence from your LaTeX proposal:

1. Shared preparation layer (metadata, preprocessing, windows, masks, splits)
2. Phase 0A validation: manual context-length/coverage sweeps
3. Phase 0B validation: robustness stress tests (missing modalities, montage changes, artifacts, shifts)
4. Method experiments (Idea 1 / Idea 2, then robustness-focused directions)

Practical implication:
- Do not treat extraction as the endpoint.
- Extraction is the data-enablement stage for objective 3 experiments.

---

## 7) Known dataset-specific constraints already discovered
MrOS specifics already handled:
- No visit2 harmonized file for AHI
- Visit2 records kept with empty AHI/apnea fields by design
- `cvchd` used for CVD
- `slisiscr` absent, so insomnia disabled for MrOS in current config
- Explicit numeric conversion is required before thresholding for selected columns

Preserve these decisions unless user explicitly changes them.

---

## 8) Immediate next actions for continuation
1. Confirm current outputs still reproduce on new PC.
2. Complete STAGES extraction with ontology-confirmed columns and robust missing handling.
3. Build unified target master file and task subject lists.
4. Generate validation summary.
5. Start phase-0 style analyses aligned with `.tex` plan (context-length and robustness checks).

---

## 9) Environment and reliability notes
Preferred env for NSRR-tools scripts:
- `/home/boshra95/NSRR-tools/.venv`

Secondary env exists:
- `/home/boshra95/sleepfm_env`

VS Code caveat seen recently:
- Python/debugger extension activation instability occurred after extension updates.
- If debugging fails, verify extension versions on the new machine before blaming project code.

---

## 10) Acceptance criteria for the next agent run
Minimum acceptable handoff completion:
1. Agent demonstrates it has read the two `.tex` files and uses them as roadmap.
2. STAGES extraction completed and validated.
3. Master targets + task subject list generated.
4. Validation report created.
5. Agent clearly states where nocturn and sleepfm-clinical were used and why.
