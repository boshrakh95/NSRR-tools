# Prompt for New Agent (New PC Continuation)

You are continuing an existing multi-repo PSG project.  
Before writing code, you MUST read and follow the handoff and planning documents below.

## Mandatory first reads (in this order)
1. `/NSRR-tools/NEW_PC_AGENT_HANDOFF.md`
2. `/NSRR-tools/data_preparation.tex`
3. `/NSRR-tools/proposal for objective3.tex`

Then read implementation/context docs:
- `/NSRR-tools/TARGET_EXTRACTION_IMPLEMENTATION_PLAN.md`
- `/NSRR-tools/IMPLEMENTATION_PLAN.md`
- `/NSRR-tools/PHASE2_IMPLEMENTATION_PLAN.md`
- `/NSRR-tools/CLASSIFICATION_TARGETS_ANALYSIS.md`

And ontology/data-format references:
- `/nocturn/configs/ontology-core.yaml`
- `/nocturn/configs/ontology-datasets.yaml`
- `/nocturn/configs/datasets/apples.yaml`
- `/nocturn/configs/datasets/shhs.yaml`
- `/nocturn/configs/datasets/mros.yaml`
- `/nocturn/configs/datasets/stages.yaml`
- `/sleepfm-clinical/README.md`
- `/sleepfm-clinical/sleepfm/stages_preprocessing/README.md`
- `/sleepfm-clinical/sleepfm/stages_preprocessing/CHECKLIST.md`

## Operating constraints
- Treat the `.tex` files as the research source-of-truth (not optional notes).
- Extraction is only one stage; align work to the broader objective-3 roadmap.
- Do not change task definitions/threshold philosophy unless explicitly instructed.
- Preserve multi-visit handling conventions and existing schema compatibility.
- Our main repo is `NSRR-Tools`. Use `nocturn` for ontology/label semantics and `sleepfm-clinical` for SleepFM formatting assumptions.
- Keep raw data read-only; write derived outputs under `/scratch/boshra95/psg/...`.

## Current known status
- APPLES, SHHS, MrOS target extraction already completed.
- STAGES extraction + integration outputs remain pending.
- Existing outputs are under `/Users/boshra/cc_scratch/psg/unified/targets`.

## Required execution plan
After reading files, provide a short plan with these phases and then execute:
1. Repro/sanity check existing outputs
2. Complete STAGES extraction
3. Build unified master targets + task subject lists
4. Generate validation report
5. Start phase-0 analyses aligned to `.tex` (context-length and robustness validation setup)

## Required first response format
In your first response, include:
1. A 10–15 bullet summary of what you extracted from the two `.tex` files
2. A table mapping each immediate coding task to its source file(s)
3. Clear statement of what will be implemented now vs deferred

## Deliverables
1. Working STAGES extraction pipeline and debug config
2. Master targets artifact and task subject list artifact
3. Validation summary (coverage, class distribution, missingness)
4. Brief “research alignment” note showing how outputs enable objective-3 experiments

## Quality checks
- Verify column names against ontology configs before coding.
- Apply explicit numeric conversions before threshold logic where needed.
- Report dataset-specific caveats explicitly (do not hide missing-data constraints).
- Keep all changes minimal and consistent with existing project style.

Proceed now.

---
