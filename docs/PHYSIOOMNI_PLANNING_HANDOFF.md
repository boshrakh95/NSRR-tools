# PhysioOmni Planning Hand-off

Prompt to paste into a **new** Claude Code chat session (on this cluster,
in `/home/boshra95/NSRR-tools`) to review and rewrite the PhysioOmni
implementation plan before any code gets written. Planning only — no
branch, no implementation. Written 2026-08-17, while OSF's Stage 2
long-context sweep is still running on `osf-implementation`.

---

## Prompt

Review and substantially improve `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md`
in this repo (`NSRR-tools`). It was written 2026-08-13, before any real
implementation experience with this kind of model, and needs a fresh,
code-verified pass — not a rewrite from imagination.

**Read first, in this order:**
1. `CLAUDE.md` (repo root) — the "TSFM Baseline Model Comparison" section,
   especially the current "Status" subsection, which covers PhysioOmni's
   known caveats (apnea exclusion, license, channel coverage, the `/100`
   normalization quirk) and OSF's now-completed Stage 1 + in-progress
   Stage 2 as the closest working analog.
2. `docs/OSF_EXPERIMENTS_GUIDE.md` — the existing pipeline conventions
   (config/registry/job-script shape, `gen_commands_*.py` pattern) any new
   backbone should mirror.
3. `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` — OSF's actual implementation
   plan, Step 0 checklist, and Stage 1/Stage 2 design (LP-FT staging,
   warm-start-from-30s, `chunk_batch_size`, etc.). Treat this as a
   reference for the *shape* of a good plan, not a template to copy
   blindly — PhysioOmni's architecture, input format, and context-length
   ceiling (~8.5 min, not OSF's 30s) are different, so judge each OSF
   decision on its own merits before reusing it.
4. `docs/TSFM_BASELINE_CANDIDATES.md` — the per-model technical survey.
5. `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` — the existing draft to
   improve.

**Then investigate for real, don't assume:**
- The actual PhysioOmni codebase at `/home/boshra95/PhysioOmni` (read-only
  reference clone) — model architecture, checkpoint format/location,
  exact input tensor shape and normalization requirements, classification
  head structure, whether/how LoRA (`peft`) target modules apply, batch
  dimensions. Verify claims the existing plan doc makes against the code
  directly (e.g. grep for the actual layer types, don't trust a prior
  summary). Same rigor OSF's plan used: confirm with `grep`/direct reads,
  not by pattern-matching to OSF.
- OSF's actual implementation files in this repo — `scripts/train_osf_lora.py`,
  `scripts/train_osf_context_sweep.py` (or equivalent), the OSF dataset
  classes in `src/nsrr_tools/datasets/osf_*.py`, `experiments/v2_osf_lora_registry.yaml`,
  `configs/phase0_osf_lora_config.yaml`, `jobs/train_osf_lora_gpu.sh` — as
  a concrete reference for what PhysioOmni's own equivalents should look
  like structurally (Stage 1 frozen-embeddings + reused sequence heads,
  Stage 2 staged LoRA fine-tuning, config/registry/job-script conventions).

**Deliverable:** an improved `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md`
with:
- **Stage 1 (frozen backbone → embeddings → our existing sequence heads)
  fully detailed and cluster-runnable**, matching the level of detail in
  OSF's plan doc: confirmed channel mapping and the `/100`
  raw-amplitude-vs-z-scored normalization handling (checked per channel,
  not assumed uniform), checkpoint loading, exact new files needed (dataset
  class, extraction script, training script, config, registry, job
  scripts), and a Step 0 verification checklist to run before any real
  sweep.
- **Stage 2 (LoRA fine-tuning): a general/outline-level plan is enough for
  now** — target modules, staging approach — to be refined once Stage 1 is
  actually running, not before.

**Hard constraints:**
- Everything happens in **this repo** (`NSRR-tools`), not in the
  PhysioOmni GitHub repo.
- **Do not modify any OSF or SleepFM file** — no edits to
  `scripts/train_osf*.py`, `scripts/train_context_sweep.py`,
  `scripts/infer_subject_windows.py`, any `src/nsrr_tools/datasets/osf_*.py`
  or `context_window_dataset.py`, `configs/phase0_osf*.yaml`,
  `configs/phase0_v3*.yaml`, `experiments/v2_osf*.yaml`,
  `experiments/v2_registry.yaml`, any `jobs/*osf*.sh` or
  `*context_sweep*.sh`, `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`,
  `docs/OSF_EXPERIMENTS_GUIDE.md`, or the OSF-specific parts of
  `CLAUDE.md`. Everything PhysioOmni needs is a **new** file (own dataset
  class, own scripts, own config, own registry, own job scripts) — this is
  what let OSF's implementation coexist cleanly alongside the original
  SleepFM pipeline, and it's what will let a future
  `physioomni-implementation` branch (forked from `osf-implementation`)
  stay mergeable in both directions while OSF's own work keeps moving.
- **Do not create a branch. Do not write any implementation code. Do not
  run any training/cluster commands.** This pass only rewrites the plan
  doc. Branch creation and actual implementation happen in a follow-up
  prompt, after the plan is reviewed.
- Leave the changes **uncommitted** when done — summarize what changed and
  why, so it can be reviewed as a diff before anything is committed.
