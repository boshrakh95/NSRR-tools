# MOMENT_CLAUDE.md

**This file exists because `CLAUDE.md` is not safe to edit for live
status.** `CLAUDE.md` is tracked and shared across `osf-implementation`,
`physioomni-implementation`, and `moment-implementation`, and all of them
edit it independently — making it a permanent merge-conflict surface. The
same reasoning already produced `PHYSIOOMNI_CLAUDE.md`; this is MOMENT's
equivalent, scoped only to the `moment-implementation` branch /
`NSRR-tools-moment` worktree.

**⚠️ Unlike `CLAUDE.md`, this file is NOT auto-loaded into context.** A
session working on this branch should read it explicitly at the start of
any work here — don't assume its contents are already known.

For the actual technical plan, read
[`docs/TSFM_MOMENT_IMPLEMENTATION_PLAN.md`](docs/TSFM_MOMENT_IMPLEMENTATION_PLAN.md).
This file is the short, living "what's the current state" companion to that
plan, not a replacement for it.

> **Created 2026-08-22 from the `physioomni-implementation` branch** as an
> initial skeleton, so the MOMENT branch starts with structure rather than
> a blank page. Almost everything below is a placeholder to be filled in by
> the MOMENT session as real work happens. Keep it updated the same way
> `PHYSIOOMNI_CLAUDE.md` was: record what was actually verified, with
> dates, not what seemed likely.

---

## ⚠️ Hard constraints — these are not suggestions

### Only ever edit this worktree

| Directory | Branch | Owner |
|---|---|---|
| `/home/boshra95/NSRR-tools` | `osf-implementation` | OSF session |
| `/home/boshra95/NSRR-tools-omni` | `physioomni-implementation` | PhysioOmni session |
| **`/home/boshra95/NSRR-tools-moment`** | **`moment-implementation`** | **this session** |

**Write only inside `/home/boshra95/NSRR-tools-moment`, on the
`moment-implementation` branch.** Never write to the other worktrees; never
`git checkout` another branch here; never intend a change for another
branch. All worktrees share one underlying `.git`, so a stray write or
checkout in the wrong directory breaks another live session's work.

**Reading the other worktrees is encouraged** — OSF's and PhysioOmni's
finished code is the best structural reference available, and their real
failures (see the OOM warning below) are exactly what MOMENT should avoid
repeating. Read freely; write nowhere but here.

### Never touch files shared between branches — always create new ones

**Everything MOMENT needs must be a NEW, MOMENT-specific file or folder.**
This is what keeps an eventual merge to `main` tractable instead of a
conflict minefield, and it's how OSF and PhysioOmni have coexisted.

**Never edit**: `CLAUDE.md` (put status *here* instead), the SleepFM
pipeline (`scripts/gen_commands.py`, `train_context_sweep.py`,
`infer_subject_windows.py`, `src/nsrr_tools/datasets/context_window_dataset.py`),
anything `osf_*`/`train_osf*`/`infer_osf*`, anything `physioomni_*`, or any
existing config/registry/job script belonging to another model.

**Always create parallel `*_moment_*` files instead** —
`moment_context_window_dataset.py`, `train_moment_context_sweep.py`,
`configs/phase0_moment_config.yaml`, `experiments/v2_moment_registry.yaml`,
`jobs/*_moment_*.sh`, `logs_moment/`, and results under
`/scratch/boshra95/psg/unified/results/phase0_moment{,_lora}`.

**One genuine exception**: `src/nsrr_tools/models/sequence_head.py` is
dim-agnostic and reusable as-is (`input_dim` is just a constructor arg).
**Import it. Do not edit it.**

---

## ⚠️ Expect a Stage 2 (LoRA) OOM — plan for it, don't rediscover it

**Full detail: plan doc §4. Read that before designing Stage 2.** Short
version, because this has now bitten two implementations independently:

Stage 2 LoRA fine-tuning for both OSF and PhysioOmni processes **every raw
signal epoch through the full backbone**. Even when the backbone forward
pass is internally chunked (to limit how many epochs go through one forward
call), **all chunks' activations for a training batch must stay in memory
for the backward pass** — they're one autograd graph until a single shared
`backward()`. **So GPU memory scales with
`batch_size × raw-epochs-per-context-window`, not just the chunk size.** A
chunk-size knob bounds each forward call but does *not* reduce peak memory.

This caused real OOM failures for both models at longer contexts despite
working fine at short ones: **OSF failed at 40m and up; PhysioOmni failed
even at 10m, and at 40m/240m in some configurations.**

- **OSF's fix**: a per-context-length gradient-accumulation schedule
  (`context_micro_batch` in `experiments/v2_osf_lora_registry.yaml`, on
  `osf-implementation`), targeting roughly a **640-unit
  (`batch_size × context_length`) ceiling** while holding
  `effective_batch=32` via `accum_steps`.
- **PhysioOmni's fix** (more refined): a **head-aware** version of the same
  idea in its own registry
  (`experiments/v2_physioomni_lora_registry.yaml`), with **separate
  micro-batch schedules for `lstm` vs `transformer` heads**, since the
  transformer head's own O(N²) attention needs more conservative values
  than the LSTM's O(N).

**This is a general architectural problem, not specific to either model —
MOMENT should expect the same class of OOM at its own longer contexts.**
The exact numeric values will **not** transfer, since they depend on each
backbone's own memory footprint. But MOMENT's plan should **budget from the
start for the same fix pattern: a context-scaled, gradient-accumulation-based
micro-batch schedule, validated with a real GPU pilot at each context
length before trusting it for a full sweep** — rather than rediscovering
this OOM a third time from scratch.

Related, also learned the hard way (plan doc §4): prefer a **larger GPU
allocation** before gradient checkpointing (MIG partitions scale compute
*with* memory, so it's not a tradeoff; whole non-MIG H100s exist on Fir via
`--gpus=h100:1` and cost no extra queue time). Note MOMENT's own
`enable_gradient_checkpointing=True` default interacts with this.

---

## Environment

**Not built yet.** Plan: a dedicated `/home/boshra95/moment_env` venv.

Do **not** reuse `osf_env` or `physioomni_env` — each contains an
`nsrr_tools_src.pth` pointing at ONE worktree's `src/`, so reusing one would
silently import another branch's copy of `nsrr_tools`, and repointing it
would break that branch's live environment. PhysioOmni hit exactly this and
built its own venv for this reason.

To record here once built: Python version, torch version, package count,
and confirmation that `import nsrr_tools` resolves to
`/home/boshra95/NSRR-tools-moment/src`.

Known requirements (plan doc §3): `peft` and `accelerate` are **not**
declared MOMENT dependencies (imported ad hoc in its tutorial only) — pin
them explicitly. MOMENT requires `transformers>=4.54.1`; check that against
whatever base requirements file we build from.

## Checkpoint

**Not downloaded yet.** `AutonLab/MOMENT-1-{small,base,large}` on
HuggingFace; which one to use is an open question (plan doc §5.1). Record
the real byte size and parameter count here once downloaded — the repo's
own code/README does **not** state parameter counts, so don't quote the
paper's numbers as verified.

License: **MIT**, confirmed from the repo's `LICENSE` file. Cleanest of the
three baselines (PhysioOmni's is split GitHub/HF; OSF's has contamination
caveats instead).

## Reference materials

- Model repo: `/home/boshra95/moment` (read-only reference, never modified).
- `docs/TSFM_MOMENT_IMPLEMENTATION_PLAN.md` — the plan (skeleton).
- `docs/TSFM_BASELINE_CANDIDATES.md` §2.3 — code-verified MOMENT findings.
- `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`,
  `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` — the two finished plans to
  mirror. PhysioOmni's §15 (Stage 2 design) and checklist 2.6 (real
  failures + fixes) transfer most directly.
- `docs/EXPERIMENTS_GUIDE.md` — the SleepFM pipeline every baseline mirrors.

---

## Key facts to keep front-of-mind

- **Context length is 512 timesteps, hard-fixed** — patch 8, stride 8 → 64
  patches. At 128 Hz that's **≈4 seconds**, shorter than our own 5-second
  SleepFM patches. **Plan A and Plan C are both ruled out; Plan B only.**
- **How to map 512 timesteps onto our 30-second-epoch protocol is the
  central unsolved design question** of this whole integration (plan doc
  §5.2). It has no obvious right answer and determines much else.
- **MOMENT's own reference LoRA recipe is missing
  `modules_to_save=["head"]`** — as written it likely freezes the
  classification head. Fix this ourselves; don't copy the tutorial as-is.
- **`enable_gradient_checkpointing` defaults to `True`** — disable it for
  frozen-encoder Stage 1 runs (pure wasted compute with no backward pass
  through the backbone).
- **`n_channels` is not checkpoint-locked** — no architectural barrier to
  multi-channel input; it only sizes the fresh classification head.
- **RevIN normalization is applied internally, per-channel** — decide
  deliberately what to feed MOMENT, and verify empirically rather than
  assuming our existing z-scoring is right.
- **No contamination concern** (general time-series pretraining, not NSRR),
  unlike OSF's quantified SHHS overlap — but verify rather than assume.
- **MOMENT has no physiological pretraining at all.** Modest expectations
  are appropriate. It's a genuine general-vs-specialist test, and a loss is
  an informative result, not a failed experiment.

---

## Status log

Append dated entries here as work happens (newest last), the way
`PHYSIOOMNI_CLAUDE.md` records its phase status.

- **2026-08-22** — `MOMENT_CLAUDE.md` and
  `docs/TSFM_MOMENT_IMPLEMENTATION_PLAN.md` created as initial skeletons
  from the `physioomni-implementation` branch. No environment, no
  checkpoint, no code. Everything in the plan's §7 checklist is open; the
  MOMENT session's first task is reading the MOMENT repo in detail and
  expanding the plan into a real, cluster-runnable one.
